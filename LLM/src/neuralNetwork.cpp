//
// Created by 69029 on 3/16/2021.
//
#undef NDEBUG
#include "neuralNetwork.hpp"
#include "utils.hpp"
#include "global_var.hpp"
#include "prover.hpp"
#include <polynomial.h>
#include <circuit.h>
#include <iostream>
#include <cmath>
#include <iomanip>
#include <algorithm>


using std::cerr;
using std::endl;
using std::max;
using std::ifstream;
using std::ofstream;

ifstream in;
ifstream conf;
ofstream out;

const char *layerTypeName(layerType ty) {
    switch (ty) {
        case layerType::INPUT: return "INPUT";
        case layerType::FFT: return "FFT";
        case layerType::IFFT: return "IFFT";
        case layerType::ADD_BIAS: return "ADD_BIAS";
        case layerType::RELU: return "RELU";
        case layerType::Sqr: return "Sqr";
        case layerType::OPT_AVG_POOL: return "OPT_AVG_POOL";
        case layerType::MAX_POOL: return "MAX_POOL";
        case layerType::AVG_POOL: return "AVG_POOL";
        case layerType::DOT_PROD: return "DOT_PROD";
        case layerType::PADDING: return "PADDING";
        case layerType::FCONN: return "FCONN";
        case layerType::LAYER_NORM_1: return "LAYER_NORM_1";
        case layerType::LAYER_NORM_2: return "LAYER_NORM_2";
        case layerType::LAYER_NORM_3: return "LAYER_NORM_3";
        case layerType::GELU_1: return "GELU_1";
        case layerType::GELU_2: return "GELU_2";
        case layerType::GELU_3: return "GELU_3";
        case layerType::MHA_QK: return "MHA_QK";
        case layerType::SOFTMAX_1: return "SOFTMAX_1";
        case layerType::SOFTMAX_2: return "SOFTMAX_2";
        case layerType::SOFTMAX_3: return "SOFTMAX_3";
        case layerType::PLACE_HOLDER: return "PLACE_HOLDER";
        case layerType::RELU_CHECK: return "RELU_CHECK";
    }
    return "UNKNOWN";
}

void dumpLayerLog(const layeredCircuit &circuit, string filename="../output/gpt2-small_layers.log") {
    std::ofstream layers_log(filename, std::ios::out | std::ios::app);
    if (!layers_log.is_open()) {
        cerr << "failed to open " << filename << endl;
        return;
    }

    layers_log << std::left
               << std::setw(6) << "idx"
               << std::setw(10) << "size"
               << std::setw(14) << "zero_start"
               << std::setw(13) << "need_phase2"
               << std::setw(11) << "bit_len"
               << std::setw(11) << "max_bl_u"
               << std::setw(11) << "max_bl_v"
               << std::setw(11) << "bl_u[0]"
               << std::setw(11) << "bl_u[1]"
               << std::setw(11) << "bl_v[0]"
               << std::setw(11) << "bl_v[1]"
               << std::setw(10) << "size_u[0]"
               << std::setw(10) << "size_u[1]"
               << std::setw(10) << "size_v[0]"
               << std::setw(10) << "size_v[1]"
               << std::setw(15) << "uni_gates_size"
               << std::setw(15) << "bin_gates_size"
               << std::setw(13) << "uni_interval"
               << std::setw(13) << "bin_interval"
               << std::setw(14) << "ori_id_u_size"
               << std::setw(14) << "ori_id_v_size"
               << "ty" << '\n';

    for (int i = 0; i < circuit.size; ++i) {
        const auto &cur = circuit.circuit[i];
        layers_log << std::left
                   << std::setw(6) << i
                   << std::setw(10) << cur.size
                   << std::setw(14) << cur.zero_start_id
                   << std::setw(13) << (cur.need_phase2 ? 'T' : 'F')
                   << std::setw(11) << static_cast<int>(cur.bit_length)
                   << std::setw(11) << static_cast<int>(cur.max_bl_u)
                   << std::setw(11) << static_cast<int>(cur.max_bl_v)
                   << std::setw(11) << static_cast<int>(cur.bit_length_u[0])
                   << std::setw(11) << static_cast<int>(cur.bit_length_u[1])
                   << std::setw(11) << static_cast<int>(cur.bit_length_v[0])
                   << std::setw(11) << static_cast<int>(cur.bit_length_v[1])
                   << std::setw(10) << cur.size_u[0]
                   << std::setw(10) << cur.size_u[1]
                   << std::setw(10) << cur.size_v[0]
                   << std::setw(10) << cur.size_v[1]
                   << std::setw(15) << cur.uni_gates.size()
                   << std::setw(15) << cur.bin_gates.size()
                   << std::setw(13) << cur.uni_interval.size()
                   << std::setw(13) << cur.bin_interval.size()
                   << std::setw(14) << cur.ori_id_u.size()
                   << std::setw(14) << cur.ori_id_v.size()
                   << layerTypeName(cur.ty) << '\n';
    }
    layers_log.close();
    cout << "Dumped layer log to " << filename << endl;
}

namespace multi_max 
{
template<class T>
T max(T head) {
    return head;
}
template<class T, typename... Args>
T max(T head, Args... args) {
    T t = max<T>(args...);
    return (head > t)?head:t;
}
} // end of namespace

neuralNetwork::neuralNetwork(i64 psize_x, i64 psize_y, i64 pchannel, i64 pparallel, const string &i_filename,
                             const string &c_filename, const string &o_filename, bool i_llm,
                             int headnum, int headdim, int attn_dim, int linear_dim) :
        pic_size_x(psize_x), pic_size_y(psize_y), pic_channel(pchannel), pic_parallel(pparallel),
        SIZE(0), NCONV_FAST_SIZE(1), NCONV_SIZE(2), FFT_SIZE(5),
        AVE_POOL_SIZE(1), FC_SIZE(1), RELU_SIZE(2), act_ty(RELU_ACT),
        headnum(headnum), headdim(headdim), attn_dim(attn_dim), linear_dim(linear_dim) 
    {
        is_llm=i_llm;
        in.open(i_filename);
        if (!in.is_open()){
            cout << "neuralNetwork open no input file." << endl;
        }
        conf.open(c_filename);
        if (!conf.is_open()){
            cout << "neuralNetwork open no config file." << endl;
        }
}


void Out_group(Fr a,Fr b,Fr c)
{ 
    char flaga=' ',flagb=' ',flagc=' ';
    if (a>1e10)
    {
        flaga='-';
        a=-a;
    }
    
    if (b>1e10)
    {
        b=-b;
        flagb='-';
    }
    if (c>1e10)
    {
        flagc='-';
        c=-c;
    }
}

// input:   [data]
//          [[conv_kernel || relu_conv_bit_decmp]{sec.size()}[max_pool]{if maxPool}[pool_bit_decmp]]{conv_section.size()}
//          [fc_kernel || relu_fc_bit_decmp]
void neuralNetwork::initParam(prover &pr,int depth) 
{
    total_in_size = 0;
    total_para_size = 0;
    total_relu_in_size = 0;
    total_ave_in_size = 0;
    total_max_in_size = 0;
    // data
    const int padded_len = 1 << ceilPow2BitLength(len);
    const int padded_hidden = 1 << ceilPow2BitLength(attn_dim);
    i64 pos = static_cast<i64>(padded_len) * padded_hidden;

    const size_t fc_count = full_conn.size();
    pr.fc_row.resize(fc_count);
    pr.fc_col.resize(fc_count);
    pr.fc_start_id.resize(fc_count);
    pr.fc_input_row.resize(fc_count);
    pr.fc_input_col.resize(fc_count);
    pr.fc_input_id.resize(fc_count);
    pr.fc_real_row.resize(fc_count);
    pr.fc_real_col.resize(fc_count);
    pr.fc_real_input_row.resize(fc_count);
    pr.fc_real_input_col.resize(fc_count);
    mat_values.assign(fc_count, nullptr);

    const size_t ln_slots = std::max<size_t>(depth, 1);
    layer_norm_w_c.resize(ln_slots);
    layer_norm_w_e.resize(ln_slots);
    layer_norm_b_c.resize(ln_slots);
    layer_norm_b_e.resize(ln_slots);
    layer_norm_w_q_start.resize(ln_slots);
    layer_norm_b_q_start.resize(ln_slots);
    layer_norm_c1.resize(ln_slots);
    layer_norm_e1.resize(ln_slots);
    layer_norm_c2.resize(ln_slots);
    layer_norm_e2.resize(ln_slots);

    for (int i = 0; i < full_conn.size(); ++i) 
    {
        auto &fc = full_conn[i];
        refreshFCParam(fc);
        // fc_kernel (pr elements hardcode to size 100, used 0-47)
        pr.fc_real_input_row[i]=len;
        len=pr.fc_input_row[i]=1<<ceilPow2BitLength(len);
        pr.fc_real_input_col[i]=fc.channel_in;
        pr.fc_real_row[i]=fc.channel_in;
        pr.fc_input_col[i]=1<<ceilPow2BitLength(fc.channel_in);
        
        fc.channel_in=pr.fc_row[i]=1<<ceilPow2BitLength(fc.channel_in);
        pr.fc_real_col[i]=fc.channel_out;
        fc.channel_out=pr.fc_col[i]=1<<ceilPow2BitLength(fc.channel_out);
        
        fc.weight_start_id = pos;   // TODO calc FC pos
        pr.fc_start_id[i]=fc.weight_start_id;
        u32 para_size = pr.fc_row[i] * pr.fc_col[i];
        pos += para_size;
        total_para_size += para_size;
        //fc.bias_start_id = pos;
        //pos += channel_out;
        //total_para_size += channel_out;
        //fprintf(stderr, "full conn  bias   weight: %11lld%11lld\n", channel_out, total_para_size);
    }
    total_in_size = pos;
    vector<string> layers={"l1","l2","l3","fcon","round","MHA_QK","softmax*V","softmax*v","soft_end","fcon2","round2","l1","l2","l3","fcon3","round","gelu1","g2","g3","fcon4","round"};
    if (model_name.find("opt") != string::npos) {  // if model name contains "opt", use relu checker instead of gelu checker
        layers={"l1","l2","l3","fcon","round","MHA_QK","softmax*V","softmax*v","soft_end","fcon2","round2","l1","l2","l3","fcon3","round","relu_check","fcon4","round"};
    }
    SIZE=1+layers.size()*depth ;         //TODO here is very important to avoid strange memory errors
}
void neuralNetwork::merge_layer(prover &pr,i64 layer_id)
{
    // isolate and remove FCONN and RELU(rounding) layers
    int cntp=0;
    for(int i=0;i<layer_id;i++)
    {
        if( (int)pr.C.circuit[i].ty==4 || (int)pr.C.circuit[i].ty==11 || pr.C.circuit[i].ty==layerType::RELU_CHECK) // ty == RELU || ty == FCONN
        {
            ++cntp;
            layer v;
            v.ty=pr.C.circuit[i].ty;
            pr.C.circuit.push_back(v);
            swap(pr.C.circuit[i],pr.C.circuit[pr.C.circuit.size()-1]);
            vector<F> f;
            pr.val.push_back(f);
            swap(pr.val[i],pr.val[pr.val.size()-1]);
        }
    }
    vector<vector<F> >::iterator itf=pr.val.begin(); 
    int cnt2=0;
    for (vector<layer>::iterator it = pr.C.circuit.begin(); it <pr.C.circuit.end() ;) 
    {
        if ((int)it->ty==4 || (int)it->ty==11 || it->ty==layerType::RELU_CHECK) // ty == RELU || ty == FCONN
        {
            if (cnt2<cntp)
            {
                it = pr.C.circuit.erase(it);
                itf=pr.val.erase(itf);
                cnt2++;
            }
            else
            {
                ++it;
                ++itf;
            }
        } 
        else 
        {
            ++it;
            ++itf;
        }
    }
    // merge other layers
    int offset[5]={0,0,0,0,0};
    int lazy_offset[5]={0,0,0,0,0};
    for(int i=4;i<layer_id;i++) // classify each original layer i into 4 merged stages;
    { // for every gate output value in layer i, it appends that value into pr.val[os];
        auto t=pr.C.circuit[i].ty;
        int os=0;
        if(t==layerType::MHA_QK || t==layerType::GELU_1 || t==layerType::LAYER_NORM_1)
            os=1;
        else if(t==layerType::SOFTMAX_1 || t==layerType::GELU_2 || t==layerType::LAYER_NORM_2)
            os=2;
        else if(t==layerType::SOFTMAX_2 || t==layerType::GELU_3 || t==layerType::LAYER_NORM_3)
            os=3;
        else if(t==layerType::SOFTMAX_3)
            os=4;
        else
            break;
        assert(os!=0);
        assert(pr.val[i].size()==pr.C.circuit[i].size);  // #output of gate = #gate of layer
        
        for(int j=0;j<pr.C.circuit[i].size;j++)
        {
            pr.val[os].emplace_back(pr.val[i][j]);
        }
    }
    for(int i=1;i<4;i++) // records (1,2,3) layer's original unary and binary gates occupy the initial interval
    {
        auto t=pr.C.circuit[i].ty;
        int os=0, os1=0;
        if(t==layerType::MHA_QK || t==layerType::GELU_1 || t==layerType::LAYER_NORM_1)
            os=1;
        else if(t==layerType::SOFTMAX_1 || t==layerType::GELU_2 || t==layerType::LAYER_NORM_2)
            os=2;
        else if(t==layerType::SOFTMAX_2 || t==layerType::GELU_3 || t==layerType::LAYER_NORM_3)
            os=3;
        else if(t==layerType::SOFTMAX_3)
            os=4;
        if(os==i)
        {
            pr.C.circuit[os].uni_interval.emplace_back(make_pair(0,pr.C.circuit[i].uni_gates.size()));
            pr.C.circuit[os].bin_interval.emplace_back(make_pair(0,pr.C.circuit[i].bin_gates.size()));
        }
    }
    for(int i=4;i<layer_id;i++)
    {
        auto t=pr.C.circuit[i].ty; // i-th layer
        int os=0, os1=0;
        if(t==layerType::MHA_QK || t==layerType::GELU_1 || t==layerType::LAYER_NORM_1)
            os=1;
        else if(t==layerType::SOFTMAX_1 || t==layerType::GELU_2 || t==layerType::LAYER_NORM_2)
            os=2;
        else if(t==layerType::SOFTMAX_2 || t==layerType::GELU_3 || t==layerType::LAYER_NORM_3)
            os=3;
        else if(t==layerType::SOFTMAX_3)
            os=4;
        else
            break;
        assert(os!=0);
        
        auto t2=pr.C.circuit[i-1].ty; // (i-1)-th layer
        if(t2==layerType::MHA_QK || t2==layerType::GELU_1 || t2==layerType::LAYER_NORM_1)
            os1=1;
        else if(t2==layerType::SOFTMAX_1 || t2==layerType::GELU_2 || t2==layerType::LAYER_NORM_2)
            os1=2;
        else if(t2==layerType::SOFTMAX_2 || t2==layerType::GELU_3 || t2==layerType::LAYER_NORM_3)
            os1=3;
        else if (t2==layerType::SOFTMAX_3)
            os1=4;
        
        assert(i!=os);
        offset[os1]=offset[os]=0;
        for(int j=1;j<i;j++) // j-th layer before i-th layer;
        { // sums the sizes of all earlier j layers that belong to the same merged bucket os;
            auto tp=pr.C.circuit[j].ty;
            int tos=0;
            if(tp==layerType::MHA_QK || tp==layerType::GELU_1 || tp==layerType::LAYER_NORM_1)
                tos=1;
            else if(tp==layerType::SOFTMAX_1 || tp==layerType::GELU_2 || tp==layerType::LAYER_NORM_2)
                tos=2;
            else if(tp==layerType::SOFTMAX_2 || tp==layerType::GELU_3 || tp==layerType::LAYER_NORM_3)
                tos=3;
            else if(tp==layerType::SOFTMAX_3)
                tos=4;
            if(tos==os) // when append layer i into merged layer os, where do this layer's output gate indices start inside the big concatenated pr.val[os]
                offset[os]+=pr.C.circuit[j].size;
        }
        for(int j=1;j<i-1;j++)
        {
            auto tp=pr.C.circuit[j].ty;
            int tos=0;
            if(tp==layerType::MHA_QK || tp==layerType::GELU_1 || tp==layerType::LAYER_NORM_1)
                tos=1;
            else if(tp==layerType::SOFTMAX_1 || tp==layerType::GELU_2 || tp==layerType::LAYER_NORM_2)
                tos=2;
            else if(tp==layerType::SOFTMAX_2 || tp==layerType::GELU_3 || tp==layerType::LAYER_NORM_3)
                tos=3;
            else if(tp==layerType::SOFTMAX_3)
                tos=4;
            if(tos==os1) // where does the source live inside the concatenated merged bucket os1
                offset[os1]+=pr.C.circuit[j].size;
        }
        
        // the unary gates copied from original layer i will live in this contiguous subrange of the merged unary-gate vector
        pr.C.circuit[os].uni_interval.emplace_back(make_pair(pr.C.circuit[os].uni_gates.size(),pr.C.circuit[os].uni_gates.size()+pr.C.circuit[i].uni_gates.size()));
        
        for(auto g=pr.C.circuit[i].uni_gates.begin();g!=pr.C.circuit[i].uni_gates.end();g++)
        {
            if((int)g->lu==0)  // the input comes from the global auxiliary array
                pr.C.circuit[os].uni_gates.emplace_back(g->g+offset[os], g->u,0,g->sc);
            else  // the input comes from the previous layer (i-1)
            {
                pr.C.circuit[os].uni_gates.emplace_back(g->g+offset[os], g->u+offset[os1],os1,g->sc);
                assert(os==os1+1);
                assert(g->u<pr.C.circuit[i-1].size);
                assert(g->u+offset[os1]<pr.val[os1].size());
            }
        }
        
        // the binary gates copied from original layer i will live in this contiguous subrange of the merged binary-gate vector
        pr.C.circuit[os].bin_interval.emplace_back(make_pair(pr.C.circuit[os].bin_gates.size(),pr.C.circuit[os].bin_gates.size()+pr.C.circuit[i].bin_gates.size()));
        
        for(auto g=pr.C.circuit[i].bin_gates.begin();g!=pr.C.circuit[i].bin_gates.end();g++)
        {
            if((int)g->l==0)
                pr.C.circuit[os].bin_gates.emplace_back(g->g+offset[os],g->u,g->v,g->sc,g->l);
            else if((int)g->l==1)
                pr.C.circuit[os].bin_gates.emplace_back(g->g+offset[os],g->u+offset[os1],g->v+offset[os1],g->sc,g->l);
            else if((int)g->l==2)
                pr.C.circuit[os].bin_gates.emplace_back(g->g+offset[os],g->u+offset[os1],g->v,g->sc,g->l);
            assert(g->g+offset[os]<pr.val[os].size());
        }
    }
    pr.C.circuit.erase(pr.C.circuit.begin()+5,pr.C.circuit.begin()+layer_id-cntp);
    pr.val.erase(pr.val.begin()+5,pr.val.begin()+layer_id-cntp);

    
    pr.C.size=pr.C.circuit.size();
    for(int i=1;i<pr.C.size;i++)
    {
        initLayer(pr.C.circuit[i], pr.val[i].size(), pr.C.circuit[i].ty);
        if(pr.C.circuit[i].ty!=layerType::FCONN)
            checkNormalLayer(pr.C.circuit[i],i,pr.val);
    }
}
void neuralNetwork::create(prover &pr, bool merge, bool DEBUG_MERGE, std::string MODEL_NAME, bool DEBUG_disable_commit)
{
    model_name = MODEL_NAME;
    
    auto jmdbg_create_init_timer0 = std::chrono::high_resolution_clock::now();
    compute_e_table();  // prepare exp lookup table. proving becomes query (in,out) -> table    
    initParam(pr,layer_num);  // init FC params, layers and memory offset -> pr
    pr.C.init(Q_BIT_SIZE, SIZE);
    pr.val.resize(SIZE);  // Init GKR circuit container. empty layers, no gates yet.
    val = pr.val.begin();
    i64 layer_id = 0;
    inputLayer(pr.C.circuit[layer_id++]);  // circuit[0] init input layer with identity gates
    for (int i = 0; i < full_conn.size(); ++i) // load model weights, 12 layers * 4 full conn per layer = 48 FC
    {
        auto &fc = full_conn[i];
        refreshFCParam(fc);
        readFconWeight(fc.weight_start_id,pr.fc_real_row[i],pr.fc_real_col[i],i, DEBUG_disable_commit);
    }
    auto jmdbg_create_init_timer1 = std::chrono::high_resolution_clock::now();
    auto jmdbg_create_init_ms = std::chrono::duration_cast<std::chrono::microseconds>(jmdbg_create_init_timer1 - jmdbg_create_init_timer0).count();
    std::cout << "(jmTimer)neuralNetwork::create, compute_e_table, init space, load inputs weights, time=" << jmdbg_create_init_ms << " us" << std::endl; 

    auto jmdbg_create_commit_timer0 = std::chrono::high_resolution_clock::now();
    timer T;
    if (!DEBUG_disable_commit) {
        int logn = pr.C.circuit[0].bit_length;  // 28
        u64 n_sqrt = 1ULL << (logn - (logn >> 1));  // n^(1/2): 2^(logn-logn/2)=2^(28-14)=16384
        pr.gens.resize(n_sqrt);
        G1 base=gen_gi(pr.gens.data(),n_sqrt);  // [base, base*r, ..., base*r^(n_sqrt-1)]
        pr.gens.push_back(base);
        T.start();
        pr.commitInput(pr.gens,32,false);  //commit weight and input
        T.stop();
        pr.proof_size+= (1<<(pr.cc.l/2)) * G_BYTE_SIZE; // (Not counted in final proof size, reseted by prover::init) Hyrax returns an array Tk of length rownum 2^(l/2)
        cout<<"Model weight commit time: "<<T.elapse_sec()<<"s"<<endl;
    } else {
        pr.gens.clear();
        pr.commitInput(pr.gens,32,true);
        cout<<"Model weight commit time: skipped"<<endl;
    }
    auto jmdbg_create_commit_timer1 = std::chrono::high_resolution_clock::now();
    auto jmdbg_create_commit_ms = std::chrono::duration_cast<std::chrono::microseconds>(jmdbg_create_commit_timer1 - jmdbg_create_commit_timer0).count();
    std::cout << "(jmTimer)neuralNetwork::create, commit input disabled " << DEBUG_disable_commit << ", time=" << jmdbg_create_commit_ms << " us" << std::endl;

    cout<<"Start initiating circuit"<<endl;
    for (int i = 0; i < full_conn.size(); ++i) 
    {
        auto jmdbg_timer0 = std::chrono::high_resolution_clock::now();
        auto &fc = full_conn[i];
        refreshFCParam(fc);  // get fc.channel_in/out size
        if(i==0)
        {
            q_offset=0;
        }
        read_layer_norm(0);
        bool * sparsity=new bool[pr.fc_input_row[i]*pr.fc_input_col[i]]; // TODO(jm): memory leak
        memset(sparsity,0,sizeof(bool)*pr.fc_input_row[i]*pr.fc_input_col[i]);  // 32*1024=32768 bools
        int cnt=0;
        for(int j=0;j<pr.fc_real_input_row[i];j++)
            for(int k=0;k<pr.fc_real_input_col[i];k++)
            {
                ++cnt;
                sparsity[j*pr.fc_input_col[i]+k]=true;
            }
        if(i%4==0 || i%4==2)
        {
            ln_checker_layer1(pr.C.circuit[layer_id], layer_id,0,input_e,input_c,pr.fc_real_input_col[0],sparsity);
            ln_checker_layer2(pr.C.circuit[layer_id], layer_id,0,input_e,input_c,sparsity);
            ln_checker_layer3(pr.C.circuit[layer_id], layer_id,0,input_e,input_c,pr.fc_real_input_col[0],sparsity);
        }
        pr.fc_input_id[i]=q_offset;
        fullyConnLayer(pr.C.circuit[layer_id], layer_id, fc.weight_start_id,q_offset,0);
        int cx=7,ex=-8,cy=3,ey=-8;
        float c_A,e_A,c_B=1,e_B=-10,c_C=7,e_C=-8;
        c_A=input_c;
        e_A=input_e;
        
        roundLayer(pr.C.circuit[layer_id], layer_id,(float)c_A*c_B/c_C*pow(2,e_A+e_B-e_C));
        if(i%4==0)
        {
            multi_head_matrix_QK(pr.C.circuit[layer_id], layer_id);
            softmax_layer_1(pr.C.circuit[layer_id], layer_id,pow(2,e_C)*c_C,pow(2,e_C)*c_C,pow(2,e_C)*c_C,pow(1,-8));
            softmax_layer_2(pr.C.circuit[layer_id], layer_id,pow(2,e_C)*c_C,pow(2,e_C)*c_C,pow(2,e_C)*c_C,pow(1,-8));
            softmax_layer_3(pr.C.circuit[layer_id], layer_id,pow(2,e_C)*c_C,pow(2,e_C)*c_C,pow(2,e_C)*c_C,pow(1,-8));
        }
        if (i % 4 == 2)
        {
            std::string model_name_lower = MODEL_NAME;
            std::transform(model_name_lower.begin(), model_name_lower.end(), model_name_lower.begin(), [](unsigned char c){
                return std::tolower(c); 
            });
            bool use_relu = (model_name_lower.find("opt") != std::string::npos);

            if (use_relu) {
                relu_checker_layer(pr.C.circuit[layer_id], layer_id, pr.fc_real_col[i]);
            } else {
                gelu_checker_layer1(pr.C.circuit[layer_id], layer_id, pr.fc_real_col[i], -8,48, -8,217, -8,252, -8,615, ex,cx, ey,cy);
                gelu_checker_layer2(pr.C.circuit[layer_id], layer_id, pr.fc_real_col[i], -8,48, -8,217, -8,252, -8,615, ex,cx, ey,cy);
                gelu_checker_layer3(pr.C.circuit[layer_id], layer_id, pr.fc_real_col[i], -8,48, -8,217, -8,252, -8,615, ex,cx, ey,cy);
            }
        }
        auto jmdbg_timer1 = std::chrono::high_resolution_clock::now();
        auto jmdbg_ms1 = std::chrono::duration_cast<std::chrono::microseconds>(jmdbg_timer1 - jmdbg_timer0).count();
        std::cout << "(jmTimer)neuralNetwork::create, layer " << i << "/" << full_conn.size() << ", time=" << jmdbg_ms1 << " us" << std::endl;
    }

    std::ofstream layers_log;
    std::string output_file_log_name = "../output/" + MODEL_NAME + "/SqueezeMerge_" + std::to_string((int)merge) + "/" + MODEL_NAME + "_layers.log";
    if(DEBUG_MERGE)
    {
        layers_log.open(output_file_log_name, std::ios::out | std::ios::trunc);
        layers_log << "before merge:" << std::endl;
        layers_log.close();
        dumpLayerLog(pr.C, output_file_log_name);
    }

    auto jmdbg_create_merge_timer0 = std::chrono::high_resolution_clock::now();
    if(merge)
    {
        merge_layer(pr,layer_id);  // pr.C.circuit.size() becomes smaller
    }
    auto jmdbg_create_merge_timer1 = std::chrono::high_resolution_clock::now();
    auto jmdbg_create_merge_ms = std::chrono::duration_cast<std::chrono::microseconds>(jmdbg_create_merge_timer1 - jmdbg_create_merge_timer0).count();
    std::cout << "(jmTimer)neuralNetwork::create, merge_layer, merge enable=" << merge << ", time=" << jmdbg_create_merge_ms << " us" << std::endl;

    if(DEBUG_MERGE)
    {
        layers_log.open(output_file_log_name, std::ios::out | std::ios::app);
        layers_log << "after merge:" << std::endl;
        layers_log.close();
        dumpLayerLog(pr.C, output_file_log_name);
    }

    total_in_size += total_max_in_size + total_ave_in_size + total_relu_in_size;
    initLayer(pr.C.circuit[0], total_in_size, layerType::INPUT);  // Adjusts input layer size to include auxiliary, weights
    assert(total_in_size == pr.val[0].size());

    if(DEBUG_MERGE)
    {
        layers_log.open(output_file_log_name, std::ios::out | std::ios::app);
        layers_log << "adjust 0:" << std::endl;
        layers_log.close();
        dumpLayerLog(pr.C, output_file_log_name);
    }

    auto jmdbg_create_subset_timer0 = std::chrono::high_resolution_clock::now();
    pr.C.initSubset();
    auto jmdbg_create_subset_timer1 = std::chrono::high_resolution_clock::now();
    auto jmdbg_create_subset_ms = std::chrono::duration_cast<std::chrono::milliseconds>(jmdbg_create_subset_timer1 - jmdbg_create_subset_timer0).count();
    std::cout << "(jmTimer)neuralNetwork::create, initSubset, time=" << jmdbg_create_subset_ms << " ms" << std::endl;

    if(DEBUG_MERGE)
    {
        layers_log.open(output_file_log_name, std::ios::out | std::ios::app);
        layers_log << "after initSubset:" << std::endl;
        layers_log.close();
        dumpLayerLog(pr.C, output_file_log_name);
    }
    
    int cnt=0;
    for(int i=0;i<pr.C.size;i++)
        cnt+=pr.val[i].size();
    int bin=0,uni=0;
    for(int i=0;i<pr.C.size;i++)
    {
        bin+=pr.C.circuit[i].bin_gates.size();
        uni+=pr.C.circuit[i].uni_gates.size();
    }
    pr.mat_val=mat_values.data();  // Stores matrix values for later proof phases, Counts gates, witnesses, proof size
}

void neuralNetwork::inputLayer(layer &circuit) 
{
    initLayer(circuit, total_in_size, layerType::INPUT);
    for (i64 i = 0; i < total_in_size; ++i) 
        circuit.uni_gates.emplace_back(i, 0, 0, 1);

    calcInputLayer(circuit);
}


pair<int,int> search(double scale)
{
    double mindiff=1e9;
    int best_e,best_c;
    for(int e=-10;e<=10;e++)
    for(int c=1;c<=800;c++)
    {
        double s=pow(2,e)*c;
        if(abs(s-scale)<mindiff)
        {
            mindiff=abs(s-scale);
            best_e=e;
            best_c=c;
        }
    }
    return make_pair(best_e,best_c);
}

void neuralNetwork::read_layer_norm(int ln_id)
{
    //int layer_norm_w_c[30], layer_norm_w_e[30],layer_norm_b_c[30], layer_norm_b_e[30];
    //int layer_norm_w_q_start[30],layer_norm_b_q_start[30];
    i64 orgsize=val[0].size() ;
    val[0].resize(orgsize+2*channel_out); // inject gamma, beta of layer norm as aux input
    total_in_size +=2*channel_out;
    layer_norm_w_c[ln_id]=1;
    layer_norm_w_e[ln_id]=0;
    layer_norm_b_c[ln_id]=1;
    layer_norm_b_e[ln_id]=-8;
    layer_norm_w_q_start[ln_id]=orgsize;
    layer_norm_b_q_start[ln_id]=orgsize+channel_out;

    for(int i=0;i<channel_out;i++)
        val[0][i+layer_norm_w_q_start[ln_id]]=1;
    for(int i=0;i<channel_out;i++)
        val[0][i+layer_norm_b_q_start[ln_id]]=1;

}   
std::ostream& operator<<(std::ostream& os, __int128_t value) {
    if (value < 0) {
        os << '-';
        value = -value;
    }
    // save flags to restore them
    std::ios_base::fmtflags flags(os.flags());
    // set zero fill
    os << std::setfill('0') << std::setw(13);

    // 128-bit number has at most 39 digits,
    // so the below loop will run at most 3 times
    const int64_t modulus = 10000000000000; // 10**13
    do {
        int64_t val = value % modulus;
        value /= modulus;
        if (value == 0) {
            os.flags(flags);
            return os << val;
        }
        os << val;
    } while (1);
}
void neuralNetwork::ln_checker_layer1(layer &circuit, i64 &layer_id, int ln_id, int ey,int cy,int real_cn_in,bool* sparsity_map)
{
    
    int cw= layer_norm_w_c[ln_id]; //scale w  //TODO we need to fix all layer_norm value's read
    int ew= layer_norm_w_e[ln_id];
    int cb= layer_norm_b_c[ln_id];  //scale b
    int eb= layer_norm_b_e[ln_id];
    double sw=pow(2,ew)*cw,sb=pow(2,eb)*cb;
    double sy=pow(2,ey)*cy;
    int qw_off= layer_norm_w_q_start[ln_id];  // place of w vector
    int qb_off= layer_norm_b_q_start[ln_id];  // place of b vector
    
    int c1,e1,c2,e2;
    pair<int,int> S1,S2;
    S1=search(sw*sqrt(real_cn_in)/sy); // TODO: s1 is wrong, channel_out should be something else
    S2=search(sb/sy);
    e1=S1.first;
    c1=S1.second;
    e2=S2.first;
    c2=S2.second;
    layer_norm_c1[ln_id]=c1;
    layer_norm_e1[ln_id]=e1;
    layer_norm_c2[ln_id]=c2;
    layer_norm_e2[ln_id]=e2;
    int m=multi_max::max(1,-e1,-e2);
    i64 block_len = len* channel_in;
    i64 output_size=block_len+3*len;
    initLayer(circuit, output_size, layerType::LAYER_NORM_1);  //TODO: the output dim of the layer
    circuit.need_phase2=true;
    circuit.zero_start_id=block_len+2*len;
    i64 orgsize=val[0].size() ;
    val[0].resize(orgsize+10+block_len*2+4*len);// y(768*len), sum,B, sigma, delta1, delta2(768*len)
    for(int i=orgsize;i<val[0].size();i++)
        val[0][i]=0;
    ln_aux_start=orgsize; 
    val[0][orgsize]=1; 
    total_relu_in_size += 10+ block_len*2+4*len;
    vector<ll> qx(channel_in, 0), a(channel_in, 0);
    vector<ll> qw(channel_in, 0), qb(channel_in, 0);
    for (i64 co = 0; co < channel_in; ++co) 
    {
        qw[co]=convert(val[0][qw_off+co]);
        qb[co]=convert(val[0][qb_off+co]);
    }
    __int128 mn=1;
    for (i64 i = 0; i < len; i++)
    {
        ll sum=0,B=1,sigma;
        if(!sparsity_map[i*channel_in])  // Skip whole row if this row is marked empty.
        {
            continue;
        }
        for (i64 co = 0; co < channel_in; ++co) 
        {
            i64 g = matIdx(i, co, channel_in); // i*channel_in+co
            qx[co]=convert(val[0][q_offset+g]);
            sum+=qx[co];
        }
        for (i64 co = 0; co < channel_in; ++co) 
        {
            if(!sparsity_map[i*channel_in+co]) // skip zero element
                continue;
            a[co]=qx[co]*real_cn_in-sum;  //TODO here has to change
            B+=a[co]*a[co];
        }
        assert(B!=0);
        sigma=round(sqrt(B));
        assert(sigma!=0);
        assert(sigma*sigma+sigma+1-B<(1ll<<32) && sigma*sigma+sigma+1-B>0);
        assert(B-sigma*sigma+sigma<(1ll<<32) && B-sigma*sigma+sigma>0);
        Fr delta1= Fr(sigma*sigma+sigma+1-B)*Fr(B-sigma*sigma+sigma);
        for (i64 co = 0; co < channel_in; ++co) 
        {
            i64 g = matIdx(i, co, channel_in);
            if(!sparsity_map[g])
            {
                continue;
            }
            ll qy = round(pow(2,e1)*c1*qw[co]*a[co]/sigma+pow(2,e2)*c2*qb[co]);
            
            i64 y_off=orgsize+10+g;
            i64 d2_off=orgsize+10+block_len+len*4+matIdx(i, co, real_cn_in);
            val[0][y_off]=qy;
            ll term1,term2;
            term1=(2*qy+1)*(1ll<<(m-1))*sigma+1-(1ll<<(e1+m))*c1*qw[co]*a[co]-(1ll<<(e2+m))*c2*qb[co]*sigma;
            term2=(1ll<<(e1+m))*c1*qw[co]*a[co]+(1ll<<(e2+m))*c2*qb[co]*sigma-(2*qy-1)*(1ll<<(m-1))*sigma+1;
            assert(term1>0&&term2>0);
            val[0][d2_off]=Fr(term1)*Fr(term2);
            positive_check+=1;  //add one d2
        }
        i64 sum_off=orgsize+10+block_len+i;
        val[0][sum_off]=sum;
        i64 b_off=orgsize+10+block_len+len+i;
        val[0][b_off]=B;
        i64 sig_off=orgsize+10+block_len+len*2+i;
        val[0][sig_off]=sigma;
        i64 d1_off=orgsize+10+block_len+len*3+i;
        val[0][d1_off]=delta1;
        positive_check+=1;  //add one d1
    }
    Fr SUM=0;
    for (i64 i = 0; i < len; i++)
    {
        if(!sparsity_map[i*channel_in])
        {
            continue;
        }
        i64 sum_off=orgsize+10+block_len+i;
        i64 sig_off=orgsize+10+block_len+len*2+i;
        i64 b_off=orgsize+10+block_len+len+i;
        for (i64 co = 0; co < channel_in; ++co) 
        {
            int g = matIdx(i, co, channel_in);
            if(!sparsity_map[g])
            {
                continue;
            }
            circuit.uni_gates.emplace_back(g, q_offset+g, 0, real_cn_in); // verify a with x and sum 
            circuit.uni_gates.emplace_back(g, sum_off, 0, -1);
            if(i==2)
            {
                SUM+=val[0][q_offset+g];
            }
            circuit.uni_gates.emplace_back(block_len+len*2+i, q_offset+g, 0, 1); //+xi
        }
        circuit.bin_gates.emplace_back(block_len+i, sig_off,sig_off, 1,0);//sigma^2
        circuit.uni_gates.emplace_back(block_len+i, sig_off, 0, 1);  //sigma
        circuit.uni_gates.emplace_back(block_len+i, orgsize, 0, 1); //+1
        circuit.uni_gates.emplace_back(block_len+i, b_off, 0, -1); //-B


        circuit.bin_gates.emplace_back(block_len+len+i, sig_off,sig_off, -1,0);//-sigma^2
        circuit.uni_gates.emplace_back(block_len+len+i, sig_off, 0, 1);  //sigma
        circuit.uni_gates.emplace_back(block_len+len+i, b_off, 0, 1); //+B
        circuit.uni_gates.emplace_back(block_len+len*2+i, sum_off, 0, -1);  //-SUM
    }
    
    calcNormalLayer(circuit, layer_id);
    for (i64 i = 0; i < len; i++)
    {
        assert(val[layer_id][block_len+len*2+i].isZero());
    }

    layer_id++;
}

void neuralNetwork::ln_checker_layer2(layer &circuit, i64 &layer_id, int ln_id, int ey,int cy,bool* sparsity_map)
{
    int cw= layer_norm_w_c[ln_id]; //scale w  //TODO we need to fix all layer_norm value's read
    int ew= layer_norm_w_e[ln_id];
    int cb= layer_norm_b_c[ln_id];  //scale b
    int eb= layer_norm_b_e[ln_id];
    int qw_off= layer_norm_w_q_start[ln_id];  // place of w vector
    int qb_off= layer_norm_b_q_start[ln_id];  // place of b vector
    int c1,e1,c2,e2;
    c1=layer_norm_c1[ln_id];
    e1=layer_norm_e1[ln_id];
    c2=layer_norm_c2[ln_id];
    e2=layer_norm_e2[ln_id];
    int m=multi_max::max(-e1,-e2,1);
    i64 block_len = len* channel_in;
    i64 output_size=2*block_len+2*len;
    initLayer(circuit, output_size, layerType::LAYER_NORM_2);  //TODO: the output dim of the layer
    circuit.need_phase2=true;
    circuit.zero_start_id=2*block_len;
    i64 orgsize=ln_aux_start;
    //ll qx[1024],a[1024];
    //ll qw[1024],qb[1024];
    for (i64 i = 0; i < len; i++)
    {
        i64 sum_off=orgsize+10+block_len+i;
        i64 sig_off=orgsize+10+block_len+len*2+i;
        i64 b_off=orgsize+10+block_len+len+i;

        // (2*qy+1)*(1ll<<(m-1))*sigma+1  -(1ll<<(e1+m))*c1*qw[co]*a[co]  -(1ll<<(e2+m))*c2*qb[co]*sigma;

        ll SUM=0,cir=0;
        for (i64 co = 0; co < channel_in; ++co) 
        {
            i64 g = matIdx(i, co, channel_in); // i*channel_in+co
            i64 y_off=orgsize+10+g;
            if(!sparsity_map[g])
            {
                continue;
            }
            circuit.bin_gates.emplace_back(g, g, qw_off+co, -(1ll<<(e1+m))*c1 ,2 ); // -qw[co]*a[co]
            circuit.bin_gates.emplace_back(g, sig_off, qb_off+co, -(1ll<<(e2+m))*c2 ,0); //-(1ll<<(e2+m))*c2*qb[co]*sigma;
            circuit.bin_gates.emplace_back(g, sig_off,y_off ,1<<m ,0 );
            circuit.uni_gates.emplace_back(g, orgsize,0 ,1 );
            circuit.uni_gates.emplace_back(g, sig_off,0 ,(1<<(m-1)) );

            circuit.bin_gates.emplace_back(g+block_len, g, qw_off+co, (1ll<<(e1+m))*c1 ,2 ); // qw[co]*a[co]
            circuit.bin_gates.emplace_back(g+block_len, sig_off, qb_off+co, (1ll<<(e2+m))*c2 ,0); //(1ll<<(e2+m))*c2*qb[co]*sigma;
            circuit.bin_gates.emplace_back(g+block_len, sig_off,y_off ,-(1<<m) ,0 );
            circuit.uni_gates.emplace_back(g+block_len, orgsize,0 ,1 );
            circuit.uni_gates.emplace_back(g+block_len, sig_off,0 ,(1<<(m-1)) );

            circuit.bin_gates.emplace_back(2*block_len+len+i, g, g ,1,1 );
            cir=convert(val[layer_id-1][g]);
            SUM+=cir*cir;
        }
        if(!sparsity_map[i*channel_in])
        {
            continue;
        }
        ll P=convert(val[0][b_off]);
        circuit.bin_gates.emplace_back(2*block_len+i, block_len+i,block_len+len+i, 1,1);
        i64 d1_off=orgsize+10+block_len+len*3+i;
        circuit.uni_gates.emplace_back(2*block_len+i, d1_off, 0, -1); 
        circuit.uni_gates.emplace_back(2*block_len+len+i, orgsize, 0, 1); 
        circuit.uni_gates.emplace_back(2*block_len+len+i, b_off, 0, -1);  //b=sigma^2+1
        /*circuit.bin_gates.emplace_back(block_len+i, sig_off,sig_off, 1,0);//sigma^2
        circuit.uni_gates.emplace_back(block_len+i, sig_off, 0, 1);  //sigma
        circuit.uni_gates.emplace_back(block_len+i, orgsize, 0, 1); //+1
        circuit.uni_gates.emplace_back(block_len+i, b_off, 0, -1); //-B


        circuit.bin_gates.emplace_back(block_len+len+i, sig_off,sig_off, -1,0);//-sigma^2
        circuit.uni_gates.emplace_back(block_len+len+i, sig_off, 0, 1);  //sigma
        circuit.uni_gates.emplace_back(block_len+len+i, b_off, 0, 1); //+B

        circuit.uni_gates.emplace_back(block_len+len*2+i, sum_off, 0, -1);  //-SUM
        */
    }
    
    calcNormalLayer(circuit, layer_id);
    for (i64 i = 0; i < block_len; i++)
    {
        assert(!val[layer_id][i].isNegative());  //only need to check the sparse items of these
        assert(!val[layer_id][block_len+i].isNegative());  
        if(i<block_len)
        {
            if(sparsity_map[i]==false)
                assert(val[layer_id][i].isZero());
            if(sparsity_map[i]==false)
                assert(val[layer_id][i+block_len].isZero());
        }
    }
    for (i64 i = 0; i < len; i++)
    {
        i64 b_off=orgsize+10+block_len+len+i;
        assert(val[layer_id][2*block_len+len+i].isZero());
        assert(val[layer_id][2*block_len+i].isZero());
    }

    layer_id++;
}

void neuralNetwork::ln_checker_layer3(layer &circuit, i64 &layer_id, int ln_id, int ey,int cy,int real_cn_in,bool* sparsity_map)
{
    i64 block_len = len* channel_in;
    i64 output_size=len*real_cn_in;
    initLayer(circuit, output_size, layerType::LAYER_NORM_3);  //TODO: the output dim of the layer
    circuit.need_phase2=true;
    circuit.zero_start_id=0;
    i64 orgsize=ln_aux_start;  // use previous layer's org size
    
    for (i64 i = 0; i < len; i++)
    {
        for (i64 co = 0; co < real_cn_in; ++co) 
        {
            i64 g=matIdx(i,co,channel_in);  // i*channel_in+co
            i64 p=matIdx(i, co, real_cn_in);  // i*real_cn_in+co
            i64 d2_off=orgsize+10+block_len+len*4+p;
            circuit.uni_gates.emplace_back(p, d2_off, 0, -1);  
            circuit.bin_gates.emplace_back(p, g, g+block_len,1,1);
        }
        
    }
    calcNormalLayer(circuit, layer_id);
    for(int g=0;g<output_size;g++)
    {
        assert(val[layer_id][g].isZero());   // assert will fail without UNDEF NDEBUG on cmake
    }
    layer_id++;
    q_offset=ln_aux_start+10;  // get the rounded result for next computation
} 

void neuralNetwork::relu_checker_layer(layer &circuit, i64 &layer_id, int real_cn_out)
{
    i64 block_len   = len * channel_out;      // padded length * padded hidden
    i64 output_size = len * real_cn_out;      // real tokens * real channels

    // Layout in val[0]:
    // [ ... existing ... | const(10) | y(output_size) | abs(output_size) | d(output_size) ]
    i64 orgsize = val[0].size();
    val[0].resize(orgsize + 10 + block_len*2 + output_size);
    for (i64 i = orgsize; i < val[0].size(); ++i) {
        val[0][i] = 0;
    }

    total_relu_in_size += 10 + block_len*2 + output_size;
    positive_check += len * real_cn_out * 2;

    i64 y_off   = orgsize + 10;
    i64 abs_off = y_off + block_len;
    i64 d_off   = abs_off + block_len;

    // The layer outputs:
    //   first  output_size entries: y >= 0
    //   second output_size entries: d >= 0
    //   third  output_size entries: 2y - q - abs = 0
    //   fourth output_size entries: q + abs - 2d = 0
    initLayer(circuit, output_size * 4, layerType::RELU_CHECK);
    circuit.need_phase2 = false;
    circuit.zero_start_id = output_size * 2;

    for (i64 g = 0; g < block_len; ++g)
    {
        if (g % channel_out >= real_cn_out) continue;

        // Compact index over only the real outputs
        i64 gp = (g / channel_out) * real_cn_out + (g % channel_out);

        // q lives at q_offset from the previous roundLayer output
        Fr qf = val[0][q_offset + g];
        long long qx = (long long)convert(qf);

        long long yv = std::max(0LL, qx);
        long long av = std::llabs(qx);
        long long dv = yv - qx;

        val[0][y_off   + gp] = Fr(yv);
        val[0][abs_off + gp] = Fr(av);
        val[0][d_off   + gp] = Fr(dv);

        // 1) y >= 0  -> just copy y as layer output
        circuit.uni_gates.emplace_back(gp, y_off + gp, 0, 1);

        // 2) d >= 0  -> just copy d as layer output
        circuit.uni_gates.emplace_back(output_size + gp, d_off + gp, 0, 1);

        // 3) 2y - q - abs = 0
        circuit.uni_gates.emplace_back(output_size * 2 + gp, y_off + gp,   0,  2);
        circuit.uni_gates.emplace_back(output_size * 2 + gp, q_offset + g, 0, -1);
        circuit.uni_gates.emplace_back(output_size * 2 + gp, abs_off + gp, 0, -1);

        // 4) q + abs - 2d = 0
        circuit.uni_gates.emplace_back(output_size * 3 + gp, q_offset + g, 0,  1);
        circuit.uni_gates.emplace_back(output_size * 3 + gp, abs_off + gp, 0,  1);
        circuit.uni_gates.emplace_back(output_size * 3 + gp, d_off + gp,   0, -2);
    }

    calcNormalLayer(circuit, layer_id);

    // Sanity checks
    for (int g = 0; g < output_size; ++g) {
        assert(!val[layer_id][g].isNegative());
        assert(!val[layer_id][output_size + g].isNegative());
        assert(val[layer_id][output_size * 2 + g].isZero());
        assert(val[layer_id][output_size * 3 + g].isZero());
    }

    // Next FC should consume y
    q_offset = y_off;
    layer_id++;
}

void neuralNetwork::gelu_checker_layer1(layer &circuit, i64 &layer_id, int real_cn_out, int ea,int ca,int eb,int cb,int ec,int cc,int ed,int cd,int ex,int cx,int ey,int cy)
{
    int m1=multi_max::max(0,-ec,-eb-ex,-ea-2*ex,ex-ey);
    int m2=multi_max::max(0,-ed,-ex);
    ll C1 = cx*(1ll<<(m1));
    ll C2 = (1ll<<(ea +2*ex+m1))*cx*ca*cx*cx;
    ll C3 = (1ll<<(eb+ex+m1))*cx*cb*cx;
    ll    C4 = cx*cc * (1ll<<(ec +m1));
    ll    C5 = (1ll<<(ey-ex+m1))*cy;
    
    ll    C6 = cd * (1ll<<(ed +m2));
    ll    C7 = cx * (1ll<<(ex +m2));

    i64 block_len = len* channel_out;
    i64 output_size=6*block_len;
    initLayer(circuit, output_size, layerType::GELU_1);  //TODO: the output dim of the layer
    circuit.need_phase2=true;
    circuit.zero_start_id=block_len*2;
    i64 orgsize=val[0].size() ;
    val[0].resize(orgsize +10 + block_len*3 + len*real_cn_out*3 );// Const; y; abs; t; d1; d2; d3
    for(int i=orgsize;i<val[0].size();i++)
        val[0][i]=0;
    gelu_aux_start=orgsize; 
    positive_check+=len*real_cn_out*3; //add positive check
    total_relu_in_size += 10+ len*real_cn_out*3 + block_len*3;
    for (i64 g = 0; g < block_len; ++g) 
    {
        if(g%channel_out>=real_cn_out)
            continue;
        ll qx=convert(val[0][g+q_offset]);
        
        i64 y_off=orgsize+10;
        i64 abs_off=orgsize+10+block_len;
        i64 t_off=orgsize+10+block_len*2;
        i64 d1_off=orgsize+10+block_len*3;  
        i64 d2_off=orgsize+10+block_len*3+ len*real_cn_out;
        i64 d3_off=orgsize+10+block_len*3+ len*real_cn_out*2;
        
        ll abs,t;
        if(qx<0)
            val[0][abs_off+g]=abs=-qx;
        else
            val[0][abs_off+g]=abs=qx;
        if(C6>=C7*abs)
        {
            val[0][t_off+g]=t=1;
        }
        else
        {
            val[0][t_off+g]=t=0;
        }
        i64 gp=g/channel_out* real_cn_out+g%channel_out;
        val[0][d1_off+gp]=abs+1;
        val[0][d2_off+gp]=t+(1-2*t)*(C7*abs-C6);
        assert(!val[0][d1_off+gp].isNegative());
        assert(!val[0][d2_off+gp].isNegative());
        double inner=(double)ca*cx*cx*qx*qx*pow(2,ea+2*ex)-cb*cx*abs*pow(2,eb+ex)+cc*pow(2,ec);
        double middle=(double)qx+abs-abs*t*inner;
        double final=(double)cx*pow(2,ex-1-ey)*middle/cy;
        ll y=round(final);
        val[0][y_off+g]=(ll)y;
        ll term1=(2*y + 1)*C5 + 1 - C1 * qx - C1 * abs + C2 *t*abs*abs*abs-C3 *t* abs*abs+C4 *t* abs;
        ll term2= C1 * qx + C1 * abs - C2 *t*abs*abs*abs+C3 *t* abs*abs-C4 *t* abs-(2*y-1)*C5+1;
        assert(term1>0);
        assert(term2>0);
        val[0][d3_off+gp]= Fr(term1)*Fr(term2);
    }
    val[0][orgsize]=1; 
    for (i64 g = 0; g < block_len; ++g) 
    {
        if(g%channel_out>=real_cn_out)
            continue;
        i64 gp=g/channel_out* real_cn_out+g%channel_out;
        i64 abs_off=orgsize+10+block_len+g;
        
        i64 t_off=orgsize+10+block_len*2+g;

        i64 d1_off=orgsize+10+block_len*3+gp;  
        i64 d2_off=orgsize+10+block_len*3+ len*real_cn_out+gp;

        i64 c1=orgsize;

        i64 q_off=g+q_offset;

        i64 abs_mult_t_off=g;
        i64 q_square_off=g+block_len;
        i64 dt1_off=g+block_len*2;
        i64 dt2_off=g+block_len*3;
        i64 t2_off=g+block_len*4;
        i64 abs_check_off=g+block_len*5;
        circuit.bin_gates.emplace_back(abs_mult_t_off, abs_off, t_off, 1, 0);  // sc, layer
        
        circuit.bin_gates.emplace_back(q_square_off, q_off, q_off, 1, 0);  // sc, layer

        circuit.uni_gates.emplace_back(dt1_off,abs_off,0,1);  //g,u,lu,sc
        circuit.uni_gates.emplace_back(dt1_off,c1,0,1);  //g,u,lu,sc
        circuit.uni_gates.emplace_back(dt1_off,d1_off,0,-1);  //g,u,lu,sc

        circuit.uni_gates.emplace_back(dt2_off,c1,0,-C6); 
        circuit.uni_gates.emplace_back(dt2_off,t_off,0,2ll*C6+1); 
        circuit.uni_gates.emplace_back(dt2_off,abs_off,0,C7); 
        circuit.bin_gates.emplace_back(dt2_off, abs_off, t_off, -2ll*C7, 0);
        circuit.uni_gates.emplace_back(dt2_off,d2_off,0,-1);  //g,u,lu,sc

        circuit.bin_gates.emplace_back(t2_off, t_off, t_off, 1, 0);  // sc, layer
        circuit.uni_gates.emplace_back(t2_off, t_off, 0,-1);  //g,u,lu,sc

        circuit.bin_gates.emplace_back(abs_check_off, q_off, q_off, 1, 0);  // sc, layer
        circuit.bin_gates.emplace_back(abs_check_off, abs_off, abs_off, -1, 0);  // sc, layer
    }
    calcNormalLayer(circuit, layer_id);
    

    for(int g=0;g<block_len;g++)
    {
        //TODO round for python, work wierd
        assert(!val[layer_id][g].isNegative());   // assert will fail without UNDEF NDEBUG on cmake
        assert(!val[layer_id][g+block_len].isNegative());
        if(g%channel_out>=real_cn_out)
        {
            assert(val[layer_id][g].isZero());   
            assert(val[layer_id][g+block_len].isZero());
        }
        assert(val[layer_id][g+block_len*2].isZero());
        assert(val[layer_id][g+block_len*3].isZero());
        assert(val[layer_id][g+block_len*4].isZero());
        assert(val[layer_id][g+block_len*5].isZero());
    }
    layer_id++;
} 

void neuralNetwork::gelu_checker_layer2(layer &circuit, i64 &layer_id, int real_cn_out,int ea,int ca,int eb,int cb,int ec,int cc,int ed,int cd,int ex,int cx,int ey,int cy)
{
    int m1=multi_max::max(0,-ec,-eb-ex,-ea-2*ex,ex-ey);
    int m2=multi_max::max(0,-ed,-ex);
    ll C1 = cx*(1ll<<(m1));
    ll C2 = (1ll<<(ea +2*ex+m1))*cx*ca*cx*cx;
    ll C3 = (1ll<<(eb+ex+m1))*cx*cb*cx;
    ll    C4 = cx*cc * (1ll<<(ec +m1));
    ll    C5 = (1ll<<(ey-ex+m1))*cy;
    
    ll    C6 = cd * (1ll<<(ed +m2));
    ll    C7 = cx * (1ll<<(ex +m2));
    i64 block_len = len* channel_out;
    i64 output_size=2*block_len;
    initLayer(circuit, output_size, layerType::GELU_2);  //TODO: the output dim of the layer
    circuit.need_phase2=true;
    i64 orgsize=gelu_aux_start;  // use previous layer's org size

    for (i64 g = 0; g < block_len; ++g) 
    {
        if(g%channel_out>=real_cn_out)
            continue;
        i64 y_off=orgsize+10+g;
        i64 abs_off=orgsize+10+block_len+g;
        
        i64 t_off=orgsize+10+block_len*2+g;
        i64 c1=orgsize;

        i64 q_off=g+q_offset;

        i64 term1_off=g;
        i64 term2_off=g+block_len;

        i64 abs_mult_t_off=g;
        i64 q_square_off=g+block_len;

        circuit.uni_gates.emplace_back(term1_off, c1, 0, C5+1);  // C5+1
        circuit.uni_gates.emplace_back(term1_off, y_off, 0, C5*2);  // 2*C5*y
        circuit.uni_gates.emplace_back(term1_off, abs_off, 0, -C1);  
        circuit.uni_gates.emplace_back(term1_off, q_off, 0, -C1);  
        circuit.uni_gates.emplace_back(term1_off, abs_mult_t_off, layer_id-1, C4);  
        circuit.bin_gates.emplace_back(term1_off, abs_mult_t_off,q_square_off,C2,1);
        circuit.bin_gates.emplace_back(term1_off, q_square_off, t_off,-C3,2);


        circuit.uni_gates.emplace_back(term2_off, c1, 0, C5+1);  // C5+1
        circuit.uni_gates.emplace_back(term2_off, y_off, 0, -C5*2);  // 2*C5*y
        circuit.uni_gates.emplace_back(term2_off, abs_off, 0, C1);  
        circuit.uni_gates.emplace_back(term2_off, q_off, 0, C1);  
        circuit.uni_gates.emplace_back(term2_off, abs_mult_t_off, layer_id-1, -C4);  
        circuit.bin_gates.emplace_back(term2_off, abs_mult_t_off,q_square_off,-C2,1);
        circuit.bin_gates.emplace_back(term2_off, q_square_off, t_off,C3,2);
    }
    calcNormalLayer(circuit, layer_id);
    for(int g=0;g<block_len;g++)
    {
        
        if(g%channel_out>=real_cn_out)
        {
            assert(val[layer_id][g].isZero());   
            assert(val[layer_id][g+block_len].isZero());
        }
        assert(!val[layer_id][g].isNegative());   // assert will fail without UNDEF NDEBUG on cmake
        assert(!val[layer_id][g+block_len].isNegative());
    }
    layer_id++;
} 


void neuralNetwork::gelu_checker_layer3(layer &circuit, i64 &layer_id,int real_cn_out, int ea,int ca,int eb,int cb,int ec,int cc,int ed,int cd,int ex,int cx,int ey,int cy)
{

    i64 block_len = len* channel_out;
    i64 output_size=len* real_cn_out;
    initLayer(circuit, output_size, layerType::GELU_3);  //TODO: the output dim of the layer
    circuit.need_phase2=true;
    circuit.zero_start_id=0;
    i64 orgsize=gelu_aux_start;  // use previous layer's org size

    for (i64 g = 0; g < block_len; ++g) 
    {
        if(g%channel_out>=real_cn_out)
            continue;
        i64 gp=g/channel_out* real_cn_out+g%channel_out;

        i64 d3_off=orgsize+10+block_len*3+2*real_cn_out*len+gp;

        circuit.uni_gates.emplace_back(gp, d3_off, 0, -1);  
        circuit.bin_gates.emplace_back(gp, g, g+block_len,1,1);
    }
    calcNormalLayer(circuit, layer_id);
    for(int g=0;g<output_size;g++)
        assert(val[layer_id][g].isZero());   
    layer_id++;
    q_offset=gelu_aux_start+10;  
} 


// we place the computation of after fcon layer here
void neuralNetwork::roundLayer(layer &circuit, i64 &layer_id, float scale,bool* sparsity_map) 
{
    i64 block_len = len* channel_out;
    int c,m;
    pair<int,int> pm=search(scale);
    m=pm.first;
    c=pm.second;
    float virtual_scale=c*pow(2,m);
    i64 size = block_len; 
    initLayer(circuit, size, layerType::RELU);  //TODO: the output dim of the layer
    circuit.need_phase2=true;
    circuit.zero_start_id=0;
    i64 orgsize=val[0].size() ;
    val[0].resize(orgsize + 20 + block_len*2);// Const; Q; delta
    
    q_offset=orgsize + 20 ;  //TODO set the input offset of the next matrix
    total_relu_in_size += 20+ block_len*2; //TODO: need to update here, for all aux vars added
    val[0][orgsize]=1; 
    int M=max(-m,0);
    for(i64 g = 0; g < block_len; ++g) 
    {
        i64 qq=g+orgsize+20;
        double fm;
        double fz;
        ll p=convert(val[layer_id-1][g]);
        ll q=round(p*c*pow(2,m));
        val[0][qq]=q;  // compute non-linear round
        i64 s=qq+block_len;
        
        val[0][s]=Fr(p*c*(1ll<<(m+M+1))+(1<<M)-q*(1<<(M+1)))*Fr(q*(1<<(M+1))+(1<<M)-c*(1ll<<(m+M+1))*p);
        assert(!val[0][s].isNegative());
    }
    for (i64 g = 0; g < block_len; ++g) 
    {
        i64 p=g;
        i64 q=g+orgsize+20;
        i64 c1=orgsize;
        i64 s=q+block_len;
        if(sparsity_map)
        {
            if(!sparsity_map[g])
                continue;
        }
        circuit.bin_gates.emplace_back(g, p, p, -(1ll<<(2*m+2*M+2))*c*c , 1); //  , 
        circuit.bin_gates.emplace_back(g, p, q, (1ll<<(m+2*M+3))*c ,2);  //
        circuit.bin_gates.emplace_back(g, q, q, -(1<<(2*M+2)) ,0);
        circuit.uni_gates.emplace_back(g, c1, 0, (1ll<<(2*M)) );  // this public input is one
        circuit.uni_gates.emplace_back(g, s, 0, -1);
    }   
    calcNormalLayer(circuit, layer_id);
    for(int i=0;i<block_len;i++)
    {
        int p=i;
        int q=i+orgsize+20;
        assert(val[layer_id][i].isZero());   
    }
    layer_id++;
}


void neuralNetwork::multi_head_matrix_QK(layer &circuit, i64 &layer_id)
{
    const int HEAD = headnum;
    const int HSIZE = headdim;
    i64 output_size=HEAD*len*(len+1)/2;
    initLayer(circuit, output_size, layerType::MHA_QK);
    circuit.need_phase2 = true;
    for(int head=0;head<HEAD;head++)
    {
        int T=0;
        for(int i=0;i<len;i++)
        for(int j=0;j<=i;j++)
        {
            i64 targ_gate=head*len*(len+1)/2+T;
            for(int k=0;k<HSIZE;k++)
            {
                // (i, head*64+k)
                // (j, HEAD*HSIZE+head*64+k)
                int col_i=head*HSIZE+k;
                int col_j=HEAD*HSIZE+head*HSIZE+k;
                int idi=i*channel_out+col_i;
                int idj=j*channel_out+col_j;
                i64 gate_i=q_offset+idi;
                i64 gate_j=q_offset+idj;
                circuit.bin_gates.emplace_back(targ_gate, gate_i,gate_j, 1,0 );
            }
            ++T;
        }
    }
    calcNormalLayer(circuit, layer_id);
    layer_id++;
}

void neuralNetwork::compute_e_table()
{
    double St=pow(2,-9),Se=pow(2,-20);
    for(int i=0;i<655360;i++)
    {
        int t=round(exp(-St*i)/Se);
        table[i]=max(t,1);  //TODO: avoid sum_Ei=0, occasionally happens
    }
}

void neuralNetwork::softmax_layer_1(layer &circuit, i64 &layer_id,float SQ,float SK,float Sv, float Sy)
{
    const int HEAD = headnum;
    const int HSIZE = headdim;
    i64 orgsize=val[0].size();
    val[0].resize(orgsize+10+HEAD*2*len+4*HEAD*len*(len+1)/2+HEAD*len*HSIZE+len*channel_in);//sumE,pmax,delta1,delta2,t,E,delta3,Y
    for(int i=orgsize;i<val[0].size();i++)
        val[0][i].clear();
    val[0][orgsize]=1;
    softmax_aux_start=orgsize;
    positive_check+=2*HEAD*len*(len+1)/2+HEAD*len*HSIZE; //add positive check for delta1,delta2,delta3
    exp_check+=HEAD*len*(len+1)/2;  //exp check for (t,E) pair
    total_relu_in_size += 10+2*HEAD*len+4*HEAD*len*(len+1)/2+HEAD*len*HSIZE+len*channel_in;
    i64 output_size=3*HEAD*len*(len+1)/2+HEAD*len+HEAD*len*HSIZE;
    initLayer(circuit, output_size, layerType::SOFTMAX_1);
    circuit.need_phase2 = true;
    circuit.zero_start_id=2*HEAD*len*(len+1)/2+HEAD*len*HSIZE;
    int e1,c1;
    const float St=pow(2,-9);
    const float Se=pow(2,-16);
    pair<int,int> pm=search(SQ*SK/St);
    e1=pm.first;
    c1=pm.second;
    int eprime=max(-e1-1,0);
    for(int head=0;head<HEAD;head++)
    {
        for(int i=1;i<=len;i++)
        {
            i64 sum_E_offset=orgsize+10+head*len+i-1;
            i64 pmax_offset=orgsize+10+HEAD*len+head*len+i-1;
            val[0][sum_E_offset]=0;
            //for(int j=0;j<i;j++)
            //{
            //    int offset=head*len*(len+1)/2+T;
            //    val[0][sum_E_offset]+=val[layer_id-1][offset];
            //}
            ll mx=0;
            for(int j=0;j<i;j++)
            {
                i64 offset=head*len*(len+1)/2+i*(i-1)/2+j;
                ll S=convert(val[layer_id-1][offset]);
                mx=max(mx,S);
            }
            val[0][pmax_offset]=Fr(mx);
            for(int j=0;j<i;j++) // i is length, j is id
            {
                i64 offset=head*len*(len+1)/2+i*(i-1)/2+j; //global offset
                i64 dt1_off=orgsize+10+HEAD*2*len+offset;
                i64 dt2_off=orgsize+10+HEAD*2*len+HEAD*len*(len+1)/2+offset;
                i64 t_off=orgsize+10+HEAD*2*len+2*HEAD*len*(len+1)/2+offset;
                i64 E_off=orgsize+10+HEAD*2*len+3*HEAD*len*(len+1)/2+offset;
                val[0][dt1_off]=val[0][pmax_offset]-val[layer_id-1][offset]; //pmax-pj
                ll pj_=convert(val[0][dt1_off]);
                
                ll tj=round(c1*pow(2,e1+eprime+1)*pj_/pow(2,eprime+1));
                val[0][t_off]=tj;
                assert(tj>=0 && tj<655360);  //TODO change to 65536
                val[0][E_off]=table[tj];
                val[0][dt2_off]=Fr(c1*(1<<(e1+eprime+1))*pj_+(1<<eprime)-tj*(1<<(eprime+1)))*Fr(-c1*(1<<(e1+eprime+1))*pj_+(1<<eprime)+tj*(1<<(eprime+1)));
                val[0][sum_E_offset]+=table[tj];
                ++T;
            }
        }
    }
    for(int head=0;head<HEAD;head++)
    {
        for(int i=0;i<len;i++)
        {
            for(int j=0;j<HSIZE;j++)
            {
                i64 out_ij=head*len*HSIZE+i*HSIZE+j;
                for(int k=0;k<=i;k++)
                {
                    i64 Vkj_offset=q_offset+channel_out*k+2*HEAD*HSIZE+head*HSIZE+j; //Vkj, offset   
                    i64 E_off=orgsize+10+HEAD*2*len+3*HEAD*len*(len+1)/2+head*len*(len+1)/2+(i*(i+1))/2+k;
                    circuit.bin_gates.emplace_back(out_ij, Vkj_offset,E_off,1,0);
                }
            }
        }
        
        for(int i=1;i<=len;i++)
        {
            i64 pmax_offset=orgsize+10+HEAD*len+head*len+i-1;
            i64 sum_E_offset=orgsize+10+head*len+i-1;
            for(int j=0;j<i;j++)
            {
                i64 offset=head*len*(len+1)/2+(i*(i-1))/2+j; //global offset
                i64 dt1_off=orgsize+10+HEAD*2*len+offset;
                i64 check_seg1_offset=2*HEAD*len*(len+1)/2+HEAD*len*HSIZE+offset;
                circuit.uni_gates.emplace_back(check_seg1_offset, dt1_off,0,-1); //-(pm-pi)
                circuit.uni_gates.emplace_back(check_seg1_offset, offset,layer_id-1,-1); //-pi
                circuit.uni_gates.emplace_back(check_seg1_offset, pmax_offset,0,1); //+pmax
                i64 term1_offset=HEAD*len*HSIZE+offset;
                i64 term2_offset=HEAD*len*(len+1)/2+HEAD*len*HSIZE+offset;
                i64 t_off=orgsize+10+HEAD*2*len+2*HEAD*len*(len+1)/2+offset;
                circuit.uni_gates.emplace_back(term1_offset, dt1_off,0,c1*(1<<(e1+1+eprime))); //c1*2^(e1+1+e')
                circuit.uni_gates.emplace_back(term1_offset, orgsize,0,1<<eprime); //+2^e'
                circuit.uni_gates.emplace_back(term1_offset, t_off,0,-(1<<(eprime+1))); //-2^(e'+1)*ti

                circuit.uni_gates.emplace_back(term2_offset, dt1_off,0,-c1*(1<<(e1+1+eprime))); //c1*2^(e1+1+e')
                circuit.uni_gates.emplace_back(term2_offset, orgsize,0,1<<eprime); //+2^e'
                circuit.uni_gates.emplace_back(term2_offset, t_off,0,(1<<(eprime+1))); //-2^(e'+1)*ti
            }
            i64 sum_e_check=3*HEAD*len*(len+1)/2+HEAD*len*HSIZE+head*len+i-1;
            circuit.uni_gates.emplace_back(sum_e_check, sum_E_offset,0,-1); //-2^(e'+1)*ti
            for(int j=0;j<i;j++) // i is length, j is id
            {
                i64 offset=head*len*(len+1)/2+(i*(i-1))/2+j; //global offset
                i64 E_off=orgsize+10+HEAD*2*len+3*HEAD*len*(len+1)/2+offset;
                circuit.uni_gates.emplace_back(sum_e_check, E_off,0,1);
            }
        }
    }
    
    calcNormalLayer(circuit, layer_id);
    for(int i=circuit.zero_start_id;i<val[layer_id].size();i++)
    {
        assert(val[layer_id][i].isZero());
    }
    for(int i=HEAD*len*HSIZE;i<circuit.zero_start_id;i++)
    {
        assert(!val[layer_id][i].isNegative());
    }
    layer_id++;
}

void neuralNetwork::softmax_layer_2(layer &circuit, i64 &layer_id,float SQ,float SK,float Sv, float Sy)
{
    const int HEAD = headnum;
    const int HSIZE = headdim;
    i64 orgsize=softmax_aux_start;
    
    i64 output_size=2*HEAD*len*HSIZE+HEAD*len*(len+1)/2; //delta3_term1, delta3_term2, delta2_check 
    initLayer(circuit, output_size, layerType::SOFTMAX_2);
    circuit.need_phase2 = true;
    circuit.zero_start_id=2*HEAD*len*HSIZE;
    int e1,c1;
    pair<int,int> pm=search(Sv/Sy);
    e1=pm.first;
    c1=pm.second;
    int eprime=max(-e1,1);
    ll f=0,f2=0;
    for(int head=0;head<HEAD;head++)
    {
        for(int i=0;i<len;i++)
        {
            i64 sum_E_offset=orgsize+10+head*len+i;
            ll sumE=convert(val[0][sum_E_offset]);
            assert(sumE!=0);
            for(int j=0;j<HSIZE;j++)
            {
                i64 out_ij=head*len*HSIZE+i*HSIZE+j; //on layer_id-1
                i64 term1_off=out_ij;
                i64 term2_off=out_ij+HEAD*len*HSIZE;
                i64 s_ij=orgsize+10+HEAD*2*len+4*HEAD*len*(len+1)/2+HEAD*len*HSIZE+i*channel_in+head*HSIZE+j;
                i64 d3_off=orgsize+10+HEAD*2*len+4*HEAD*len*(len+1)/2+head*len*HSIZE+i*HSIZE+j;
                ll Qij=convert(val[layer_id-1][out_ij]);
                ll S=(ll)round(Qij*c1*pow(2,e1)/sumE);
                val[0][s_ij]=S;
                ll d1=(sumE*(1<<(eprime-1))+Qij*c1*(1<<(eprime+e1))-S*sumE*(1<<eprime));
                ll d2=(sumE*(1<<(eprime-1))-Qij*c1*(1<<(eprime+e1))+S*sumE*(1<<eprime));
                val[0][d3_off]=Fr(sumE*(1<<(eprime-1))+Qij*c1*(1<<(eprime+e1))-S*sumE*(1<<eprime))*Fr((sumE*(1<<(eprime-1))-Qij*c1*(1<<(eprime+e1))+S*sumE*(1<<eprime)));
                f2=min(f2,S);
                circuit.uni_gates.emplace_back(term1_off, out_ij,layer_id-1,c1*(1ll<<(e1+eprime)));
                circuit.uni_gates.emplace_back(term1_off, sum_E_offset,0,1ll<<(eprime-1));

                circuit.bin_gates.emplace_back(term1_off, s_ij, sum_E_offset,-(1<<eprime),0);
                circuit.uni_gates.emplace_back(term2_off, out_ij,layer_id-1,-c1*(1ll<<(e1+eprime)));
                circuit.uni_gates.emplace_back(term2_off, sum_E_offset,0,1ll<<(eprime-1));
                circuit.bin_gates.emplace_back(term2_off, s_ij, sum_E_offset,(1<<eprime),0);
            }
        }
    }
    for(int head=0;head<HEAD;head++)
    {        
        for(int i=1;i<=len;i++)
        {
            for(int j=0;j<i;j++)
            {
                i64 offset=head*len*(len+1)/2+i*(i-1)/2+j; //global offset
                i64 dt2_off=orgsize+10+HEAD*2*len+HEAD*len*(len+1)/2+offset;
                i64 term1_offset=HEAD*len*HSIZE+offset;
                i64 term2_offset=HEAD*len*(len+1)/2+HEAD*len*HSIZE+offset;
                i64 now_off=2*HEAD*len*HSIZE+offset;
                circuit.bin_gates.emplace_back(now_off, term1_offset, term2_offset,1,1);
                circuit.uni_gates.emplace_back(now_off,dt2_off,0,-1);
            }
        }
    }
    
    
    calcNormalLayer(circuit, layer_id);
    for(int i=circuit.zero_start_id;i<val[layer_id].size();i++)
    {
        assert(val[layer_id][i].isZero());
    }
    for(int i=0;i<circuit.zero_start_id;i++)
    {
        assert(!val[layer_id][i].isNegative());
    }
    layer_id++;
    
}
void neuralNetwork::softmax_layer_3(layer &circuit, i64 &layer_id,float SQ,float SK,float Sv, float Sy)
{
    const int HEAD = headnum;
    const int HSIZE = headdim;
    i64 orgsize=softmax_aux_start;
    
    i64 output_size=HEAD*len*HSIZE; //delta3_check
    initLayer(circuit, output_size, layerType::SOFTMAX_3);
    circuit.need_phase2 = true;
    circuit.zero_start_id=0;
    for(int head=0;head<HEAD;head++)
    {
        for(int i=0;i<len;i++)
        {
            i64 sum_E_offset=orgsize+10+head*len+i;
            ll sumE=convert(val[0][sum_E_offset]);
            assert(sumE!=0);
            for(int j=0;j<HSIZE;j++)
            {
                i64 now=head*len*HSIZE+i*HSIZE+j; //on layer_id-1
                i64 term1_off=now;
                i64 term2_off=now+HEAD*len*HSIZE;
                i64 d3_off=orgsize+10+HEAD*2*len+4*HEAD*len*(len+1)/2+head*len*HSIZE+i*HSIZE+j;
                
                
                circuit.bin_gates.emplace_back(now,term1_off, term2_off, 1,1);
                circuit.uni_gates.emplace_back(now,d3_off ,0,-1);
            }
        }
    }

    calcNormalLayer(circuit, layer_id);
    for(int i=circuit.zero_start_id;i<val[layer_id].size();i++)
    {
        assert(val[layer_id][i].isZero());
    }
    layer_id++;
    q_offset=orgsize+10+HEAD*2*len+4*HEAD*len*(len+1)/2+HEAD*len*HSIZE; 
}

void neuralNetwork::fullyConnLayer(layer &circuit, i64 &layer_id, i64 first_fc_id,  i64 x_offset, int x_layer) 
{
    i64 size = channel_out*len;
    initLayer(circuit, size, layerType::FCONN);
    circuit.need_phase2 = true;
    val[layer_id].resize(circuit.size);
    for (i64 i = 0; i < len; i++)
    {
        for (i64 co = 0; co < channel_out; ++co) 
        {
            i64 g = matIdx(i, co, channel_out);
            val[layer_id][g]=0;
            //circuit.uni_gates.emplace_back(g, first_bias_id + co, 0, 1);  // our protocol doesn't support adding bias for simplicity
            for (i64 ci = 0; ci < channel_in; ++ci) 
            {
                i64 u = x_offset+matIdx(i, ci, channel_in);
                i64 v = first_fc_id + matIdx(co, ci, channel_in);  // the matrix is distributed as (i,ci)*(co,ci)
                val[layer_id][g]+=val[x_layer][u]*val[0][v];
            }
        }
    }
    layer_id++;
}


void neuralNetwork::refreshFCParam(const fconKernel &fc) {
    channel_in = fc.channel_in;
    channel_out = fc.channel_out;
}

i64 neuralNetwork::getFFTLen() const {
    return 1L << getFFTBitLen();
}

i8 neuralNetwork::getFFTBitLen() const {
    return 0;
}









void neuralNetwork::calcSizeAfterPool(const poolKernel &p) {
}

void neuralNetwork::calcInputLayer(layer &circuit) 
{
    val[0].resize(circuit.size);

    assert(val[0].size() == total_in_size);
    auto val_0 = val[0].begin();

    double num, mx = -10000, mn = 10000;
    vector<double> input_dat;
    const int hidden = attn_dim;
    const int padded_hidden = 1 << ceilPow2BitLength(hidden);
    for (i64 i=0;i<len;i++)
    {
        for(i64 j=0;j<hidden;j++)
        {
            in >> num; 
            input_dat.push_back(num);
            mx = max(mx, num);
            mn = min(mn, num);
        }
    }
    pair<int,int> pm=search(0.01);  // 2^input_e * input_c ≈ 0.01
    input_e=pm.first;
    input_c=pm.second;
    
    double sc=input_c*pow(2,input_e);  // 0.009765625
    int k=0;
    for (i64 i=0;i<len;i++)
    {
        for(i64 j=0;j<hidden;j++)
        {
            ll s=input_dat[k++]/sc;
            val[0][i*padded_hidden+j] = F(s);
        }
        for(i64 j=hidden;j<padded_hidden;j++)
            val[0][i*padded_hidden+j] =0;
    }

    val_0=val[0].begin()+len*padded_hidden;
    for (; val_0 < val[0].begin() + circuit.size; ++val_0) 
        val_0 -> clear();
}



void neuralNetwork::readBias(i64 first_bias_id) {
    auto val_0 = val[0].begin() + first_bias_id;

    double num, mx = -10000, mn = 10000;
    vector<double> input_dat;
    for (i64 co = 0; co < channel_out; ++co) 
    {
        in >> num;
        input_dat.push_back(num);
        mx = max(mx, num);
        mn = min(mn, num);
    }

    for (double i : input_dat)  
        *val_0++ = F((i64) (i * exp2(w_bit + x_bit)));

}

void neuralNetwork::readFconWeight(i64 first_fc_id,int real_r,int real_c,int id, bool DEBUG_disable_rand) 
{
    double num, mx = -10000, mn = 10000;
    auto val_0 = val[0].begin() + first_fc_id;
    mat_values[id]=new int[channel_out * channel_in]; // TODO(jm): memory leak
    for (i64 co = 0; co < channel_out; ++co)
        for (i64 ci = 0; ci < channel_in; ++ci) 
        {
            if(co<real_c && ci<real_r)
            {
                mat_values[id][co*channel_in+ci]=DEBUG_disable_rand ? 8 : rand()%1024;
                val_0[co*channel_in+ci]=mat_values[id][co*channel_in+ci];
            }
            else
            {
                mat_values[id][co*channel_in+ci]=0;
                val_0[co*channel_in+ci]=0;
            }
        }
}

void neuralNetwork::prepareDecmpBit(i64 layer_id, i64 idx, i64 dcmp_id, i64 bit_shift) {
    auto data = abs(val[layer_id].at(idx).getInt64());
    val[0].at(dcmp_id) = (data >> bit_shift) & 1;
}

void neuralNetwork::prepareFieldBit(const F &data, i64 dcmp_id, i64 bit_shift) {
    auto tmp = abs(data.getInt64());
    val[0].at(dcmp_id) = (tmp >> bit_shift) & 1;
}

void neuralNetwork::prepareSignBit(i64 layer_id, i64 idx, i64 dcmp_id) {
    val[0].at(dcmp_id) = val[layer_id].at(idx).isNegative() ? F_ONE : F_ZERO;
}

void neuralNetwork::prepareMax(i64 layer_id, i64 idx, i64 max_id) {
    auto data = val[layer_id].at(idx).isNegative() ? F_ZERO : val[layer_id].at(idx);
    if (data > val[0].at(max_id)) val[0].at(max_id) = data;
}

void neuralNetwork::calcNormalLayer(const layer &circuit, i64 layer_id,bool output) 
{
    val[layer_id].resize(circuit.size);
    for (auto &x: val[layer_id]) 
        x.clear();
    for (size_t idx = 0; idx < circuit.uni_gates.size(); ++idx) {
        const auto &gate = circuit.uni_gates[idx];
        val[layer_id].at(gate.g) = val[layer_id].at(gate.g) + val[gate.lu].at(gate.u) * gate.sc;
    }

    for (size_t idx = 0; idx < circuit.bin_gates.size(); ++idx) {
        const auto &gate = circuit.bin_gates[idx];
        uint16_t bin_lu = gate.getLayerIdU(layer_id), bin_lv = gate.getLayerIdV(layer_id);
        val[layer_id].at(gate.g) = val[layer_id].at(gate.g) + val[bin_lu].at(gate.u) * val[bin_lv][gate.v] * gate.sc;
    }
}

void neuralNetwork::checkNormalLayer(const layer &circuit, i64 layer_id,const vector<vector<F> > & val) 
{
    vector<F> valp;

    valp.resize(val[layer_id].size());
    
    for (auto &x: valp) 
        x.clear();
    for (auto &gate: circuit.uni_gates) 
    {
        assert(gate.g>=0 && gate.g<valp.size());
        assert(gate.u>=0 && gate.u<val[gate.lu].size());
        valp.at(gate.g) += val[gate.lu].at(gate.u) * gate.sc;
    }
    for (auto &gate: circuit.bin_gates) 
    {
        uint16_t bin_lu = gate.getLayerIdU(layer_id), bin_lv = gate.getLayerIdV(layer_id);
        assert(gate.g>=0 && gate.g<valp.size());
        valp.at(gate.g)+=  val[bin_lu].at(gate.u) * val[bin_lv][gate.v] * gate.sc;
    }
    for(int i=0;i<circuit.size;i++)
        assert(valp[i]==val[layer_id][i]);
}

void neuralNetwork::calcDotProdLayer(const layer &circuit, i64 layer_id) {
    val[layer_id].resize(circuit.size);
    for (int i = 0; i < circuit.size; ++i) val[layer_id][i].clear();

    char fft_bit = circuit.fft_bit_length;
    u32 fft_len = 1 << fft_bit;
    i64 l = layer_id - 1;
    for (auto &gate: circuit.bin_gates)
        for (int s = 0; s < fft_len; ++s)
            val[layer_id][gate.g << fft_bit | s] = val[layer_id][gate.g << fft_bit | s] +
                    val[l][gate.u << fft_bit | s] * val[l][gate.v << fft_bit | s];
}


int neuralNetwork::getNextBit(int layer_id) {
    F mx = F_ZERO, mn = F_ZERO;
    for (const auto &x: val[layer_id]) {
        if (!x.isNegative()) mx = max(mx, x);
        else mn = max(mn, -x);
    }
    i64 x = (mx + mn).getInt64();
    double real_scale = x / exp2(x_bit + w_bit);
    int res = (int) log2( ((1 << (Q - 1)) - 1) / real_scale );
    return res;
}

void neuralNetwork::printLayerValues(prover &pr) {
    for (i64 i = 0; i < SIZE; ++i) 
    {
        for (i64 j = 0; j < std::min(200u, pr.C.circuit[i].size); ++j)
            if (!pr.val[i][j].isZero()) cerr << pr.val[i][j] << ' ';
        cerr << endl;
        for (i64 j = pr.C.circuit[i].zero_start_id; j < pr.C.circuit[i].size; ++j)
            if (pr.val[i].at(j) != F_ZERO) 
            {
                exit(EXIT_FAILURE);
            }
    }
}

void neuralNetwork::printInfer(prover &pr) {
    // output the inference result with the size of (pic_parallel x n_class)
    if (out.is_open()) 
    {
        int n_class = full_conn.back().channel_out;
        for (int p = 0; p < pic_parallel; ++p) {
            int k = -1;
            F v;
            for (int c = 0; c < n_class; ++c) {
                auto tmp = val[SIZE - 1].at(matIdx(p, c, n_class));
                if (!tmp.isNegative() && (k == -1 || v < tmp)) {
                    k = c;
                    v = tmp;
                }
            }
            out << k << endl;
        }
    }
    out.close();
}
