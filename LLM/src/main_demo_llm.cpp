//
//
//

#undef NDEBUG
#include "circuit.h"
#include "neuralNetwork.hpp"
#include "verifier.hpp"
#include "models.hpp"
#include "global_var.hpp"
#include <iostream>
#include <algorithm>
#include <cctype>
#include <cerrno>
#include <string>
#include <vector>
#include <sys/stat.h>
#include <sys/types.h>

#include "range_prover.hpp"
#include "hyrax_rp.hpp"
using namespace mcl::bn;
using namespace std;

// Questions
// 1. rng: how to use real huggingface model weights?
// 2. assert not pass:
//  2.1 range_prover.cpp: int log = 9; assert(range%log==0); // range = 16 or 32; 
//      (but range is 49 in push_back(NonlinearOpType::NonLinear, {{query_size, 49}}); -- why 49?)
//  2.2 hyrax.cpp: assert(p.y==p.x*p.a);
//  2.3 hyrax.cpp: assert(p.gamma==p.g*p.x+G*p.y);
// 3. timer: prover.cpp, void prover::sumcheckLassoInit: no prove_timer.start();

// Questions paper
// 1. GeLU: threshold

struct ModelConfig {
    const char *name;
    int depth;
    int headnum;
    int headdim;
    int attn_dim;
    int linear_dim;
};

const vector<ModelConfig>& supportedModels();

string normalizeModelName(const string& name)
{
    string normalized;
    normalized.reserve(name.size());
    for (unsigned char ch : name) {
        if (std::isalnum(ch))
            normalized.push_back(char(std::tolower(ch)));
    }
    return normalized;
}

const ModelConfig* findModelConfig(const string& model_name)
{
    string normalized_target = normalizeModelName(model_name);
    for (const auto& model : supportedModels()) {
        if (normalizeModelName(model.name) == normalized_target)
            return &model;
    }
    return nullptr;
}

const vector<ModelConfig>& supportedModels()
{
    static const vector<ModelConfig> models = {
        {"gpt2-small", 12, 12, 64, 768, 3072},
        {"gpt2-medium", 24, 16, 64, 1024, 4096},
        {"gpt2-large", 36, 20, 64, 1280, 5120},
        {"opt-125m", 12, 12, 64, 768, 3072},
        {"opt-350m", 24, 16, 64, 1024, 4096},
        {"opt-1.3b", 24, 32, 64, 2048, 8192},
        {"opt-2.7b", 32, 32, 80, 2560, 10240},
        {"opt-6.7b", 32, 32, 128, 4096, 16384},
        {"opt-13b", 40, 40, 128, 5120, 20480},
    };
    return models;
}

bool ensureDir(const string& path)
{
    if (path.empty())
        return false;
    if (path == ".")
        return true;

    size_t pos = 0;
    while (pos < path.size()) {
        pos = path.find('/', pos + 1);
        string sub = path.substr(0, pos);
        if (sub.empty())
            continue;
        if (::mkdir(sub.c_str(), 0775) != 0 && errno != EEXIST) {
            cerr << "Failed to create directory: " << sub << " errno=" << errno << '\n';
            return false;
        }
    }
    return true;
}

void ensureOutputDirsForModel(const string& model_name)
{
    const vector<string> debug_dirs = {
        model_name + "_initP1_V_mult",
        model_name + "_initP1_mult_array",
        model_name + "_initP2_V_mult",
        model_name + "_initP2_mult_array",
    };
    for (int squeeze_merge = 0; squeeze_merge <= 1; ++squeeze_merge) {
        string base = "../output/" + model_name + "/SqueezeMerge_" + std::to_string(squeeze_merge);
        ensureDir(base);
        for (const auto& dir : debug_dirs) {
            ensureDir(base + "/" + dir);
        }
    }
}

void ensureOutputDirsForSupportedModels()
{
    ensureDir("../output");
    for (const auto& model : supportedModels()) {
        ensureOutputDirsForModel(model.name);
    }
}

void printUsage(const char *argv0)
{
    cerr << "Usage: " << argv0 << " <model-name> [--skip-commit]\n";
    cerr << "Supported models: gpt2-small, gpt2-medium, gpt2-large, "
            "opt-125m, opt-350m, opt-1.3b, opt-2.7b, opt-6.7b, opt-13b\n";
    cerr << "Options:\n";
    cerr << "  --skip-commit        Skip weight/input commit phase\n";
    cerr << "  --debug-print, -d    Enable debug printing\n";
    cerr << "  --squeeze-merge[=N]  Set SqueezeMerge (0 or 1). If no value given, sets to 1\n";
}


int main(int argc, char **argv) 
{
    // initPairing(mcl::BN254);
    initPairing(mcl::BLS12_381);

    bool DEBUG_PRINT = false;
    int SqueezeMerge = 0;
    ensureOutputDirsForSupportedModels();

    string model_name = "gpt2-large";
    bool DEBUG_skip_commit = false;

    for (int i = 1; i < argc; ++i) {
        string arg = argv[i];
        if (arg == "--skip-commit") {
            DEBUG_skip_commit = true;
            cout << "Commit phase will be skipped." << endl;
        } else if (arg == "--debug-print" || arg == "-dp") {
            DEBUG_PRINT = true;
            cout << "Debug printing enabled." << endl;
        } else if (arg.rfind("--squeeze-merge", 0) == 0) {
            size_t eq = arg.find('=');
            if (eq != string::npos) {
                SqueezeMerge = atoi(arg.substr(eq + 1).c_str());
            } else if (i + 1 < argc && argv[i + 1][0] != '-') {
                SqueezeMerge = atoi(argv[++i]);
            } else {
                SqueezeMerge = 1; // default when flag present without value
            }
            cout << "SqueezeMerge=" << SqueezeMerge << endl;
        } else if (arg == "--help" || arg == "-h") {
            printUsage(argv[0]);
            return 0;
        } else {
            model_name = arg;
        }
    }
    const ModelConfig *model = findModelConfig(model_name);
    if (model == nullptr) {
        cerr << "Unsupported model name: " << model_name << '\n';
        printUsage(argv[0]);
        return 1;
    }
    cout << "Selected model: " << model->name << endl;
    cout << "DEBUG_PRINT=" << (DEBUG_PRINT ? "true" : "false") << endl;
    cout << "SqueezeMerge=" << SqueezeMerge << endl;
    cout << "DEBUG_skip_commit=" << (DEBUG_skip_commit ? "true" : "false") << endl;
    
    // range prover
    cout << "main_demo_llm range prover" << endl;
    range_prover range_prover(model->depth, model->headnum, model->headdim, model->attn_dim, model->linear_dim, 30, 32, 1);
    range_prover.init();
    range_prover.build();
    double range_prover_time = range_prover.prove();

    // gkr
    prover p;
    LLM nn(model->depth, model->headnum, model->headdim, model->attn_dim, model->linear_dim);
    cout << "main_demo_llm nn.create" << endl;
    nn.create(p, SqueezeMerge, DEBUG_PRINT, model->name, DEBUG_skip_commit); // nn.create(p, 0);
    verifier v(&p, p.C);  // proof_size is reseted by prover::init, so it only counts the proof size of GKR, not the range proof size or the input+weight commitment size
    v.setCommitDisabled(DEBUG_skip_commit);
    cout << "main_demo_llm v.range_prove" << endl;
    v.range_prove(range_prover_time);
    cout << "main_demo_llm v.prove" << endl;
    v.prove(32, DEBUG_PRINT, model->name, SqueezeMerge); // prove with 32 threads
    cout << "main_demo_llm finished" << endl;
    return 0;
}
