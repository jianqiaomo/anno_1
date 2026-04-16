//
// Created by 69029 on 3/9/2021.
//

#ifndef ZKCNN_CONVVERIFIER_HPP
#define ZKCNN_CONVVERIFIER_HPP

#include "prover.hpp"
#include "hyrax.hpp"
using namespace hyrax;
using std::unique_ptr;
class verifier 
{
public:
    prover *p;
    const layeredCircuit &C;

    verifier(prover *pr, const layeredCircuit &cir);

    void prove(int commit_thread=4, bool DEBUG_PRINT=false, const std::string& model_name="gpt2-small", int squeeze_merge=1);
    void range_prove(double range_prover_time_) { range_prover_time = range_prover_time_; };
    void setCommitDisabled(bool disabled) { commit_disabled = disabled; }

    timer total_timer, total_slow_timer;
    double verifier_time;
    double prover_time;
    double range_prover_time = 0;
    double matrix_time;
    Commit_return comm;
    
private:
    vector<vector<F>> r_u, r_v;
    vector<F> final_claim_u0, final_claim_v0;
    bool verifyGKR(bool DEBUG_PRINT, const std::string& model_name, int squeeze_merge);
    bool verifyLasso();
    bool openCommit();
    

    vector<F> beta_g;
    void betaInitPhase1(i64 depth, const F &alpha, const F &beta, const vector<F>::const_iterator &r_0, const vector<F>::const_iterator &r_1, const F &relu_rou);
    void betaInitPhase2(i64 depth);

    F uni_value[2];
    F bin_value[3];
    void predicatePhase1(i64 layer_id);
    void predicatePhase2(i64 layer_id);

    F getFinalValue(const F &claim_u0, const F &claim_u1, const F &claim_v0, const F &claim_v1);

    F eval_in;
    bool commit_disabled = false;
    bool debug_print_enabled = false;
    int squeeze_merge_mode = 1;
    std::string debug_model_name = "gpt2-small";
};


#endif //ZKCNN_CONVVERIFIER_HPP
