#include "prover.hpp"
#include <iostream>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <algorithm>
#include <utils.hpp>

static vector<F> beta_gs, beta_u;
using namespace mcl::bn;
using std::unique_ptr;

void dumpVMultPatternForB(
    i64 sumcheck_id,
    int b,
    const std::string& rows,
    const char *log_file,
    int phase = 1
) {
    auto mode = (std::ios::out | std::ios::app);
    std::ofstream out(log_file, mode);
    if (!out.is_open()) {
        std::cerr << "failed to open " << log_file << '\n';
        return;
    }

    out << std::left
        << std::setw(14) << "sumcheck_id"
        << std::setw(6) << "b"
        << std::setw(12) << (phase == 1 ? "u" : "v")
        << std::setw(16) << "layer_id(dep)"
        << std::setw(16) << (phase == 1 ? "ori_id_u[u]|u" : "ori_id_v[v]|v") << '\n';
    out << rows;
    out << '\n';
}

struct MultArrayLogRow {
    u64 gate_u;
    u64 gate_g;
    ll gate_sc;
    int layer_id_v;
    long long ori_id_v_or_v;
    long long gate_v;
};

std::string llToString(ll value) {
    if (value == 0) return "0";
    bool neg = value < 0;
    if (neg) value = -value;
    std::string s;
    while (value > 0) {
        s.push_back(char('0' + (value % 10)));
        value /= 10;
    }
    if (neg) s.push_back('-');
    std::reverse(s.begin(), s.end());
    return s;
}

void dumpMultArrayPatternForIdx(
    i64 sumcheck_id,
    int idx,
    const std::vector<MultArrayLogRow>& rows,
    const char *log_file,
    int phase = 1
) {
    auto mode = (std::ios::out | std::ios::app);
    std::ofstream out(log_file, mode);
    if (!out.is_open()) {
        std::cerr << "failed to open " << log_file << '\n';
        return;
    }

    out << std::left
        << std::setw(14) << "sumcheck_id"
        << std::setw(7) << (phase == 1 ? "lu/idx" : "lv/idx")
        << std::setw(12) << "gate.u"
        << std::setw(12) << "gate.g"
        << std::setw(16) << "gate.sc"
        << std::setw(16) << "VLayerId"
        << std::setw(18) << (phase == 1 ? "ori_id_v|v" : "u0|u1")
        << std::setw(12) << "gate.v"
        << '\n';
    for (const auto& row : rows) {
        out << std::left
            << std::setw(14) << int(sumcheck_id)
            << std::setw(7) << idx
            << std::setw(12) << row.gate_u
            << std::setw(12) << row.gate_g
            << std::setw(16) << llToString(row.gate_sc)
            << std::setw(16) << row.layer_id_v
            << std::setw(18) << row.ori_id_v_or_v
            << std::setw(12) << row.gate_v
            << '\n';
    }
    out << '\n';
}

struct LassoMultLogRow {
    u64 wire;
    u64 local_wire;
};

struct LassoMultLogBlock {
    i64 sumcheck_id;
    std::vector<LassoMultLogRow> u_rows;
    std::vector<LassoMultLogRow> v_rows;
};

void dumpLassoMultPattern(
    const std::vector<LassoMultLogBlock>& blocks,
    const char *log_file
) {
    auto mode = (std::ios::out | std::ios::app);
    std::ofstream out(log_file, mode);
    if (!out.is_open()) {
        std::cerr << "failed to open " << log_file << '\n';
        return;
    }

    for (const auto& block : blocks) {
        out << std::left
            << std::setw(14) << "sumcheck_id"
            << std::setw(16) << "u"
            << std::setw(16) << "hu" << '\n';
        for (const auto& row : block.u_rows) {
            out << std::left
                << std::setw(14) << block.sumcheck_id
                << std::setw(16) << row.wire
                << std::setw(16) << row.local_wire << '\n';
        }
        out << '\n';

        out << std::left
            << std::setw(14) << "sumcheck_id"
            << std::setw(16) << "v"
            << std::setw(16) << "hv" << '\n';
        for (const auto& row : block.v_rows) {
            out << std::left
                << std::setw(14) << block.sumcheck_id
                << std::setw(16) << row.wire
                << std::setw(16) << row.local_wire << '\n';
        }
        out << '\n';
    }
    cout << "Finished dumping lasso mult pattern to " << log_file << '\n';
}

linear_poly interpolate(const F &zero_v, const F &one_v) 
{
    return {one_v - zero_v, zero_v};
}

F prover::getCirValue(uint16_t layer_id, const vector<u64> &ori, u64 u) {
    return !layer_id ? val[0][ori[u]] : val[layer_id][u];
}

void prover::init() // not prover::prover()
{
    proof_size = 0;  // nn.create already use proof_size
    r_u.resize(C.size + 1);
    r_v.resize(C.size + 1);
    const int SIZE=28; // TODO(jm): this assume max(max_bl_u, max_bl_v) <= 28. (assuming 2^28 big enough to store per layer wires)
    const size_t table_size = size_t(1) << SIZE;
    V_mult[0].resize(table_size);
    V_mult[1].resize(table_size);
    mult_array[0].resize(table_size);
    mult_array[1].resize(table_size);
    tmp_V_mult[0].resize(table_size);
    tmp_V_mult[1].resize(table_size);
    tmp_mult_array[0].resize(table_size);
    tmp_mult_array[1].resize(table_size);
    lasso_mult_v.resize(table_size);
    for(size_t i = 0; i < table_size; i++)
        lasso_mult_v[i]=0;
}

/**
 * This is to initialize all process.
 *
 * @param the random point to be evaluated at the output layer
 */
void prover::sumcheckInitAll(const vector<F>::const_iterator &r_0_from_v) 
{
    sumcheck_id = C.size;
    i8 last_bl = C.circuit[sumcheck_id - 1].bit_length;
    r_u[sumcheck_id].resize(last_bl);
    prove_timer.start();
    for (int i = 0; i < last_bl; ++i) 
        r_u[sumcheck_id][i] = r_0_from_v[i];
    prove_timer.stop();
}

/**
 * This is to initialize before the process of a single layer.
 *
 * @param the random combination coefficiants for multiple reduction points
 */
void prover::sumcheckInit(const F &alpha_0, const F &beta_0) 
{
    prove_timer.start();
    auto &cur = C.circuit[sumcheck_id];
    alpha = alpha_0;
    beta = beta_0;
    r_0 = r_u[sumcheck_id].begin();
    r_1 = r_v[sumcheck_id].begin();
    --sumcheck_id;
    prove_timer.stop();
}
static ThreadSafeQueue<int> workerq,endq;

void sc_phase1_uni_worker(std::deque<uniGate> &beg, std::vector<linear_poly> (&mult_array)[2],vector<F>& beta_g, vector<F>& beta_u,int*&L,int*&R) //F*& uni_value, layer &cur_layer,
{
    int idx;
    while (true)
    {
        bool ret=workerq.TryPop(idx);
        if(ret==false)
            return;
        int l=L[idx],r=R[idx];

        for (size_t i = l; i <r ; ++i) 
        {
            auto &gate = beg[i];
            bool idx = gate.lu != 0;
            mult_array[idx][gate.u] = mult_array[idx][gate.u] + beta_g[gate.g] * gate.sc;
        }
        endq.Push(idx);
    }
}

void sc_phase1_bin_worker(layer& cur, std::deque<binGate> &beg, std::vector<linear_poly> (&mult_array)[2],F& V_u0,F&V_u1,vector<vector<F> >& val,vector<F>& beta_g, vector<F>& beta_u,int*&L,int*&R,i64 sumcheck_id) //F*& uni_value, layer &cur_layer,
{
    int idx;
    while (true)
    {
        bool ret=workerq.TryPop(idx);
        if(ret==false)
            return;
        int l=L[idx],r=R[idx];

        for (size_t i = l; i <r ; ++i) 
        {
            auto &gate = beg[i];
            bool idx = gate.getLayerIdU(sumcheck_id) != 0;
            auto val_lv =  !gate.getLayerIdV(sumcheck_id) ? val[0][cur.ori_id_v[gate.v]] : val[gate.getLayerIdV(sumcheck_id)][gate.v];
            mult_array[idx][gate.u] = mult_array[idx][gate.u] + val_lv * beta_g[gate.g] * gate.sc;  // Ahg for phase 1
        }
        endq.Push(idx);
    }
}

void prover::sumcheckInitPhase1(const F &relu_rou_0, bool DEBUG_PRINT, int SqueezeMerge, const std::string& model_name) 
{
    //fprintf(stderr, "sumcheck level %d, phase1 init start\n", sumcheck_id);
    auto &cur = C.circuit[sumcheck_id];
    total[0] = ~cur.bit_length_u[0] ? 1ULL << cur.bit_length_u[0] : 0;
    total_size[0] = cur.size_u[0];
    total[1] = ~cur.bit_length_u[1] ? 1ULL << cur.bit_length_u[1] : 0;
    total_size[1] = cur.size_u[1];

    r_u[sumcheck_id].resize(cur.max_bl_u);
    timer useless_t;
    useless_t.start();
    beta_g.resize(1ULL << cur.bit_length);
    relu_rou = relu_rou_0;
    add_term.clear();

    int scidx = int(sumcheck_id);
    // if (sumcheck_id >= 10) {
    //     scidx = int(sumcheck_id / 10) * 10;
    // }
    // else {
    //     scidx = int(sumcheck_id);
    // }
    std::ostringstream v_mult_log_file_ss, mult_array_log_file_ss;
    v_mult_log_file_ss << "../output/" << model_name << "/SqueezeMerge_" << SqueezeMerge << "/" << model_name << "_initP1_V_mult/" << model_name << "_initP1_V_mult_layers_" << scidx << ".log";
    mult_array_log_file_ss << "../output/" << model_name << "/SqueezeMerge_" << SqueezeMerge << "/" << model_name << "_initP1_mult_array/" << model_name << "_initP1_mult_array_layers_" << scidx << ".log";
    std::string v_mult_log_file = v_mult_log_file_ss.str();
    std::string mult_array_log_file = mult_array_log_file_ss.str();
    std::vector<MultArrayLogRow> mult_array_rows[2];
    
    for (int b = 0; b < 2; ++b)
        for (u32 u = 0; u < total[b]; ++u)
            mult_array[b][u].clear();
    
    for (int b = 0; b < 2; ++b)
    {
        auto dep = !b ? 0 : sumcheck_id - 1;
        std::ostringstream v_mult_rows;
        for (u32 u = 0; u < total[b]; ++u)
        {
            if (u >= cur.size_u[b])
                V_mult[b][u].clear();
            else
            {
                V_mult[b][u] = getCirValue(dep, cur.ori_id_u, u); // put to (ax + b)linear_poly's coeffcient b
                // if (DEBUG_PRINT) {
                //     v_mult_rows << std::left
                //                 << std::setw(14) << int(sumcheck_id)
                //                 << std::setw(6) << b
                //                 << std::setw(12) << u
                //                 << std::setw(16) << int(dep)
                //                 << std::setw(16) << (dep == 0 ? cur.ori_id_u[u] : u) << '\n';
                // }
            }
        }
        // if (DEBUG_PRINT)
        //     dumpVMultPatternForB(sumcheck_id, b, v_mult_rows.str(), v_mult_log_file.c_str());
    }
    useless_t.stop();
    throw_time[sumcheck_id].push_back(useless_t.elapse_sec());
    prove_timer.start();
    
    initBetaTable(beta_g, cur.bit_length, r_0, r_1, alpha, beta);

    // // debug
    // bool all_BetaTable_zero = true;
    // for (const auto &bg : beta_g) {
    //     if (!bg.isZero()) {
    //         all_BetaTable_zero = false;
    //         break;
    //     }
    // }
    // if (all_BetaTable_zero)
    //     std::cout << "sumcheck id " << int(sumcheck_id) << ", beta_g is zero." << std::endl;
    // else
    //     std::cout << "sumcheck id " << int(sumcheck_id) << ", beta_g is not zero." << std::endl;
    
    if(cur.uni_interval.size()>=2 && DEBUG_PRINT == false)
    {
        const int thd=32;
        int *L=new int [cur.uni_interval.size()],*R=new int [cur.uni_interval.size()];  // TODO(jm): memory leak
        for (u64 j = 0; j <cur.uni_interval.size(); ++j) 
        {
                workerq.Push(j);
                L[j]=cur.uni_interval[j].first;
                R[j]=cur.uni_interval[j].second;
        }
        std::vector<std::thread> workers;
        workers.reserve(thd);
        for(int i=0;i<thd;i++)
        {
            workers.emplace_back(
                sc_phase1_uni_worker,
                std::ref(cur.uni_gates),
                std::ref(mult_array),
                std::ref(beta_g),
                std::ref(beta_u),
                std::ref(L),
                std::ref(R)
            );
        }
        while(!workerq.Empty())
            this_thread::sleep_for (std::chrono::microseconds(1));
        while(endq.Size()!=cur.uni_interval.size())
            this_thread::sleep_for (std::chrono::microseconds(1));
        for (auto &worker : workers)
            worker.join();
        endq.Clear();
    }
    else for (auto &gate: cur.uni_gates) 
        {
            bool idx = gate.lu != 0; // if lu=0 then idx=0 else idx=1
            mult_array[idx][gate.u] = mult_array[idx][gate.u] + beta_g[gate.g] * gate.sc;
            if (DEBUG_PRINT) {
                mult_array_rows[idx].push_back({
                    gate.u,
                    gate.g,
                    gate.sc,
                    -1,
                    -1,
                    -1
                });
            }
        }
    if(cur.bin_interval.size()>=2 && DEBUG_PRINT == false)
    {
        const int thd=32;
        int *L=new int [cur.bin_interval.size()],*R=new int [cur.bin_interval.size()];  // TODO(jm): memory leak
        for (u64 j = 0; j <cur.bin_interval.size(); ++j) 
        {
                workerq.Push(j);
                L[j]=cur.bin_interval[j].first;
                R[j]=cur.bin_interval[j].second;
        }
        std::vector<std::thread> workers;
        workers.reserve(thd);
        for(int i=0;i<thd;i++)
        {
            workers.emplace_back(
                sc_phase1_bin_worker,
                std::ref(cur),
                std::ref(cur.bin_gates),
                std::ref(mult_array),
                std::ref(V_u0),
                std::ref(V_u1),
                std::ref(val),
                std::ref(beta_g),
                std::ref(beta_u),
                std::ref(L),
                std::ref(R),
                sumcheck_id
            );
        }
        while(!workerq.Empty())
            this_thread::sleep_for (std::chrono::microseconds(1));
        while(endq.Size()!=cur.bin_interval.size())
            this_thread::sleep_for (std::chrono::microseconds(1));
        for (auto &worker : workers)
            worker.join();
        endq.Clear();
    }
    else  for (auto &gate: cur.bin_gates) 
        {
            bool idx = gate.getLayerIdU(sumcheck_id) != 0; // if layer_id_u=0 then idx=0 else idx=1
            auto val_lv = getCirValue(gate.getLayerIdV(sumcheck_id), cur.ori_id_v, gate.v);
            mult_array[idx][gate.u] = mult_array[idx][gate.u] + val_lv * beta_g[gate.g] * gate.sc;  // Ahg for phase 1
            if (DEBUG_PRINT) {
                long long ori_id_v_or_v = static_cast<long long>((gate.getLayerIdV(sumcheck_id) == 0) ? cur.ori_id_v[gate.v] : gate.v);
                mult_array_rows[idx].push_back({
                    gate.u,
                    gate.g,
                    gate.sc,
                    gate.getLayerIdV(sumcheck_id),
                    ori_id_v_or_v,
                    static_cast<long long>(gate.v)
                });
            }
        }
    if (DEBUG_PRINT) {
        for (int idx = 0; idx < 2; ++idx) {
            std::stable_sort(mult_array_rows[idx].begin(), mult_array_rows[idx].end(), [](const MultArrayLogRow& a, const MultArrayLogRow& b) {
                if (a.gate_u != b.gate_u) return a.gate_u < b.gate_u;
                if (a.gate_g != b.gate_g) return a.gate_g < b.gate_g;
                return a.gate_v < b.gate_v;
            });
            dumpMultArrayPatternForIdx(sumcheck_id, idx, mult_array_rows[idx], mult_array_log_file.c_str());
            cout << "sumcheck id " << int(sumcheck_id) << ", phase 1, idx " << idx << ", mult_array log dumped to " << mult_array_log_file << std::endl;
        }
    }
    round = 0;
    prove_timer.stop();
    //fprintf(stderr, "sumcheck level %d, phase1 init finished\n", sumcheck_id);
}


void sc_phase2_uni_worker( std::deque<uniGate> &beg, F& sum_value,F& V_u0,F&V_u1,vector<F>& beta_g, vector<F>& beta_u,int*&L,int*&R) //F*& uni_value, layer &cur_layer,
{
    int idx;
    while (true)
    {
        bool ret=workerq.TryPop(idx);
        if(ret==false)
            return;
        int l=L[idx],r=R[idx];
        F ss;
        ss.clear();
        for (size_t i = l; i <r ; ++i) 
        {
            auto &gate = beg[i];
            auto V_u = !gate.lu ? V_u0 : V_u1;                  //V_u0 is claim 0, V_u1 is claim 1
            ss +=beta_g[gate.g] * beta_u[gate.u] * V_u * gate.sc;
        }
        sum_value+=ss;
        endq.Push(idx);
    }
}

void sc_phase2_bin_worker( std::deque<binGate> &beg, std::vector<linear_poly> (&mult_array)[2],F& V_u0,F&V_u1,vector<F>& beta_g, vector<F>& beta_u,int*&L,int*&R,i64 sumcheck_id) //F*& uni_value, layer &cur_layer,
{
    int idx;
    while (true)
    {
        bool ret=workerq.TryPop(idx);
        if(ret==false)
            return;
        int l=L[idx],r=R[idx];

        for (size_t i = l; i <r ; ++i) 
        {
            auto &gate = beg[i];
            bool idx = gate.getLayerIdV(sumcheck_id);
            auto V_u = !gate.getLayerIdU(sumcheck_id) ? V_u0 : V_u1;
            mult_array[idx][gate.v] =mult_array[idx][gate.v]+beta_g[gate.g] * beta_u[gate.u] * V_u * gate.sc;
        }
        endq.Push(idx);
    }
}
void prover::sumcheckInitPhase2(bool DEBUG_PRINT, int SqueezeMerge, const std::string& model_name) 
{
    //fprintf(stderr, "sumcheck level %d, phase2 init start\n", sumcheck_id);
    auto &cur = C.circuit[sumcheck_id];
    total[0] = ~cur.bit_length_v[0] ? 1ULL << cur.bit_length_v[0] : 0;
    total_size[0] = cur.size_v[0];
    total[1] = ~cur.bit_length_v[1] ? 1ULL << cur.bit_length_v[1] : 0;
    total_size[1] = cur.size_v[1];
    i8 fft_bl = cur.fft_bit_length;
    i8 cnt_bl = cur.max_bl_v;

    timer useless_time;
    useless_time.start();
    r_v[sumcheck_id].resize(cur.max_bl_v);

    int scidx = int(sumcheck_id);
    std::ostringstream v_mult_log_file_ss, mult_array_log_file_ss;
    v_mult_log_file_ss << "../output/" << model_name << "/SqueezeMerge_" << SqueezeMerge << "/" << model_name << "_initP2_V_mult/" << model_name << "_initP2_V_mult_layers_" << scidx << ".log";
    mult_array_log_file_ss << "../output/" << model_name << "/SqueezeMerge_" << SqueezeMerge << "/" << model_name << "_initP2_mult_array/" << model_name << "_initP2_mult_array_layers_" << scidx << ".log";
    std::string v_mult_log_file = v_mult_log_file_ss.str();
    std::string mult_array_log_file = mult_array_log_file_ss.str();
    std::vector<MultArrayLogRow> mult_array_rows[2];

    beta_u.resize(1ULL << cur.max_bl_u);

    

    add_term.clear();
    for (int b = 0; b < 2; ++b) 
    {
        for (u32 v = 0; v < total[b]; ++v)
            mult_array[b][v].clear();
    }
    useless_time.stop();
    throw_time[sumcheck_id].push_back(useless_time.elapse_sec());
    prove_timer.start();
    initBetaTable(beta_u, cur.max_bl_u, r_u[sumcheck_id].begin(), F_ONE,32); //  beta_u is U in the code
    for (int b = 0; b < 2; ++b) 
    {
        auto dep = !b ? 0 : sumcheck_id - 1;
        std::ostringstream v_mult_rows;
        for (u32 v = 0; v < total[b]; ++v) 
        {
            V_mult[b][v] = v >= cur.size_v[b] ? F_ZERO : getCirValue(dep, cur.ori_id_v, v);
            // if (DEBUG_PRINT && v < cur.size_v[b]) {
            //     v_mult_rows << std::left
            //                 << std::setw(14) << int(sumcheck_id)
            //                 << std::setw(6) << b
            //                 << std::setw(12) << v
            //                 << std::setw(16) << int(dep)
            //                 << std::setw(16) << (dep == 0 ? cur.ori_id_v[v] : v) << '\n';
            // }
        }
        // if (DEBUG_PRINT)
        //     dumpVMultPatternForB(sumcheck_id, b, v_mult_rows.str(), v_mult_log_file.c_str(), 2);
    }
    
    if(cur.uni_interval.size()>=2 && DEBUG_PRINT == false)
    {
        const int thd=32;
        F sum[40];
        
        int *L=new int [cur.uni_interval.size()],*R=new int [cur.uni_interval.size()];
        for (u64 j = 0; j <cur.uni_interval.size(); ++j) 
        {
                workerq.Push(j);
                L[j]=cur.uni_interval[j].first;
                R[j]=cur.uni_interval[j].second;
        }
            std::vector<std::thread> workers;
            workers.reserve(thd);
            for(int i=0;i<thd;i++)
            {
                sum[i].clear(); 
                workers.emplace_back(
                    sc_phase2_uni_worker,
                    std::ref(cur.uni_gates),
                    std::ref(sum[i]),
                    std::ref(V_u0),
                    std::ref(V_u1),
                    std::ref(beta_g),
                    std::ref(beta_u),
                    std::ref(L),
                    std::ref(R)
                );
            }
            while(!workerq.Empty())
                this_thread::sleep_for (std::chrono::microseconds(1));
            while(endq.Size()!=cur.uni_interval.size())
                this_thread::sleep_for (std::chrono::microseconds(1));
            for (auto &worker : workers)
                worker.join();
            endq.Clear();
            for(int i=0;i<thd;i++)
                add_term+=sum[i];
    }
    else for (auto &gate: cur.uni_gates) 
    {
        auto V_u = !gate.lu ? V_u0 : V_u1;                  //V_u0 is claim 0, V_u1 is claim 1
        add_term = add_term + beta_g[gate.g] * beta_u[gate.u] * V_u * gate.sc;
        if (DEBUG_PRINT) {
            bool idx = gate.lu != 0; // if lu=0 then idx=0 else idx=1
            mult_array_rows[idx].push_back({
                gate.u,
                gate.g,
                gate.sc,
                -1,
                (gate.lu == 0) ? 0 : 1,
                -1
            });
        }
    }
    if(cur.bin_interval.size()>=2 && DEBUG_PRINT == false)
    {
        const int thd=32;
        int *L=new int [cur.bin_interval.size()],*R=new int [cur.bin_interval.size()];
        for (u64 j = 0; j <cur.bin_interval.size(); ++j) 
        {
                workerq.Push(j);
                L[j]=cur.bin_interval[j].first;
                R[j]=cur.bin_interval[j].second;
        }
            std::vector<std::thread> workers;
            workers.reserve(thd);
            for(int i=0;i<thd;i++)
            {
                workers.emplace_back(
                    sc_phase2_bin_worker,
                    std::ref(cur.bin_gates),
                    std::ref(mult_array),
                    std::ref(V_u0),
                    std::ref(V_u1),
                    std::ref(beta_g),
                    std::ref(beta_u),
                    std::ref(L),
                    std::ref(R),
                    sumcheck_id
                );
            }
            while(!workerq.Empty())
                this_thread::sleep_for (std::chrono::microseconds(1));
            while(endq.Size()!=cur.bin_interval.size())
                this_thread::sleep_for (std::chrono::microseconds(1));
            for (auto &worker : workers)
                worker.join();
            endq.Clear();
    }
    else for (auto &gate: cur.bin_gates) 
    {
        bool idx = gate.getLayerIdV(sumcheck_id);
        auto V_u = !gate.getLayerIdU(sumcheck_id) ? V_u0 : V_u1;
        mult_array[idx][gate.v] = mult_array[idx][gate.v] + beta_g[gate.g] * beta_u[gate.u] * V_u * gate.sc;
        if (DEBUG_PRINT) {
            mult_array_rows[idx].push_back({
                gate.u,
                gate.g,
                gate.sc,
                idx,
                gate.getLayerIdU(sumcheck_id) == 0 ? 0 : 1,
                static_cast<long long>(gate.v)
            });
        }
    }
    if (DEBUG_PRINT) {
        for (int idx = 0; idx < 2; ++idx) {
            std::stable_sort(mult_array_rows[idx].begin(), mult_array_rows[idx].end(), [](const MultArrayLogRow& a, const MultArrayLogRow& b) {
                if (a.gate_v != b.gate_v) return a.gate_v < b.gate_v;
                if (a.gate_g != b.gate_g) return a.gate_g < b.gate_g;
                return a.gate_u < b.gate_u;
            });
            dumpMultArrayPatternForIdx(sumcheck_id, idx, mult_array_rows[idx], mult_array_log_file.c_str(), 2);
            cout << "sumcheck id " << int(sumcheck_id) << ", phase 2, idx " << idx << ", mult_array log dumped to " << mult_array_log_file << std::endl;
        }
    }
    round = 0;
    prove_timer.stop();
}

void prover::sumcheckLassoInit(const vector<F> &s_u, const vector<F> &s_v,const vector<vector<F>>& r_uu, const vector<vector<F>>& r_vv,
                               bool DEBUG_PRINT, int SqueezeMerge, const std::string& model_name) 
{
    // DEBUG_PRINT = true;
    sumcheck_id = 0;
    total[1] = (1ULL << C.circuit[sumcheck_id].bit_length);
    total_size[1] = C.circuit[sumcheck_id].size;

    r_u[0].resize(C.circuit[0].bit_length);
    timer ggg;
    ggg.start();

    std::vector<LassoMultLogBlock> lasso_blocks;
    std::string lasso_mult_log_file = "../output/" + model_name + "/SqueezeMerge_" + std::to_string(SqueezeMerge) + "/" + model_name + "_lasso_mult_array.log";

    prove_timer.start(); // add
    i8 max_bl = 0;
    for (i64 i = sumcheck_id + 1; i < C.size; ++i)
        max_bl = max(max_bl, max(C.circuit[i].bit_length_u[0], C.circuit[i].bit_length_v[0]));
    beta_g.resize(1ULL << max_bl);
    for (i64 i = sumcheck_id + 1; i < C.size; ++i) 
    {
        LassoMultLogBlock block{i, {}, {}};
        i8 bit_length_i = C.circuit[i].bit_length_u[0];
        u32 size_i = C.circuit[i].size_u[0];
        //timer a,b;
        if (~bit_length_i) // run if bit_length_i != -1, otherwise skip since no wire in this layer
        {
            r_u[i].resize(C.circuit[i].max_bl_u);
            for(int j=0;j<C.circuit[i].max_bl_u;j++)
                r_u[i][j]=r_uu[i][j];
            initBetaTable(beta_g, bit_length_i, r_u[i].begin(), s_u[i - 1],32);
            for (u32 hu = 0; hu < size_i; ++hu) 
            {
                u64 u = C.circuit[i].ori_id_u[hu];
                lasso_mult_v[u] += beta_g[hu];
                if (DEBUG_PRINT)
                    block.u_rows.push_back({u, hu});
            }
        }
        bit_length_i = C.circuit[i].bit_length_v[0];
        size_i = C.circuit[i].size_v[0];
        if (~bit_length_i) 
        {
            r_v[i].resize(C.circuit[i].max_bl_v);
            for(int j=0;j<C.circuit[i].max_bl_v;j++)
                r_v[i][j]=r_vv[i][j];
            initBetaTable( beta_g, bit_length_i, r_v[i].begin(), s_v[i - 1],32);
            for (u32 hv = 0; hv < size_i; ++hv) 
            {
                u64 v = C.circuit[i].ori_id_v[hv];
                lasso_mult_v[v] += beta_g[hv];
                if (DEBUG_PRINT)
                    block.v_rows.push_back({v, hv});
            }
        }
        if (DEBUG_PRINT && (!block.u_rows.empty() || !block.v_rows.empty()))
            lasso_blocks.push_back(std::move(block));
    }
    if (DEBUG_PRINT) {
        for (auto& block : lasso_blocks) {
            std::stable_sort(block.u_rows.begin(), block.u_rows.end(), [](const LassoMultLogRow& a, const LassoMultLogRow& b) {
                if (a.wire != b.wire) return a.wire < b.wire;
                return a.local_wire < b.local_wire;
            });
            std::stable_sort(block.v_rows.begin(), block.v_rows.end(), [](const LassoMultLogRow& a, const LassoMultLogRow& b) {
                if (a.wire != b.wire) return a.wire < b.wire;
                return a.local_wire < b.local_wire;
            });
        }
        std::stable_sort(lasso_blocks.begin(), lasso_blocks.end(), [](const LassoMultLogBlock& a, const LassoMultLogBlock& b) {
            return a.sumcheck_id < b.sumcheck_id;
        });
        std::ofstream clear_file(lasso_mult_log_file, std::ios::out | std::ios::trunc);
        clear_file.close();
        dumpLassoMultPattern(lasso_blocks, lasso_mult_log_file.c_str());
    }
    round = 0;
    prove_timer.stop();
}

quadratic_poly prover::sumcheckUpdate1(const F &previous_random) {
    return sumcheckUpdate(previous_random, r_u[sumcheck_id]);
}

quadratic_poly prover::sumcheckUpdate2(const F &previous_random) {
    return sumcheckUpdate(previous_random, r_v[sumcheck_id]);
}

quadratic_poly prover::sumcheckUpdate(const F &previous_random, vector<F> &r_arr) 
{
    prove_timer.start();

    if (round) r_arr.at(round - 1) = previous_random;
    ++round;
    quadratic_poly ret;

    add_term = add_term * (F_ONE - previous_random);
    for (int b = 0; b < 2; ++b)
        ret = ret + sumcheckUpdateEach(previous_random, b);
    ret = ret + quadratic_poly(F_ZERO, -add_term, add_term);

    prove_timer.stop();
    proof_size += F_BYTE_SIZE * 3;
    return ret;
}



void sumcheckUpdate_worker(quadratic_poly &sum,vector<linear_poly> &tmp_v,vector<linear_poly> &tmp_mult,vector<linear_poly> &tmp_v_2,vector<linear_poly> &tmp_mult_2,int*&L,int*&R,Fr previous_random,int total_size)
{
    int idx;
    while (true)
    {
        bool ret=workerq.TryPop(idx);
        if(ret==false)
            return;
        int l=L[idx],r=R[idx];
        for(int i=l;i<r;i++)
        {
            u32 g0 = i << 1, g1 = i << 1 | 1;
            if (g0 >= total_size) 
                break;
            tmp_v_2[i] = interpolate(tmp_v[g0].eval(previous_random), tmp_v[g1].eval(previous_random));
            tmp_mult_2[i] = interpolate(tmp_mult[g0].eval(previous_random), tmp_mult[g1].eval(previous_random));
            sum = sum + tmp_mult_2[i] * tmp_v_2[i];
        }
        endq.Push(idx);
    }
}
quadratic_poly prover::sumcheckUpdateEach(const F &previous_random, bool idx) 
{
    auto &tmp_mult = mult_array[idx];
    auto &tmp_v = V_mult[idx];
    auto &tmp_mult_2 = tmp_mult_array[idx];
    auto &tmp_v_2 = tmp_V_mult[idx];

    if (total[idx] == 1) 
    {
        tmp_v[0] = tmp_v[0].eval(previous_random);
        tmp_mult[0] = tmp_mult[0].eval(previous_random);
        add_term = add_term + tmp_v[0].b * tmp_mult[0].b;
    }

    quadratic_poly ret;
    ret.clear();
    if(total[idx]<(1<<15))
    {
        for (u32 i = 0; i < (total[idx] >> 1); ++i) 
        {
            u32 g0 = i << 1, g1 = i << 1 | 1;
            if (g0 >= total_size[idx]) 
            {
                tmp_v[i].clear();
                tmp_mult[i].clear();
                continue;
            }
            tmp_v[i] = interpolate(tmp_v[g0].eval(previous_random), tmp_v[g1].eval(previous_random));
            tmp_mult[i] = interpolate(tmp_mult[g0].eval(previous_random), tmp_mult[g1].eval(previous_random));
            ret = ret + tmp_mult[i] * tmp_v[i];
        }
    }
    else
    {
        timer tt_f1,tt_f2;
        tt_f1.start();
        const int k=10;
        int total_work=(total[idx] >> 1);
        int *L=new int [size_t(1) << k],*R=new int [size_t(1) << k];
        const int thd=32;
        for (u64 j = 0; j < (1<<k); ++j) 
        {
            workerq.Push(j);
            L[j]=(total_work>>k)*j;
            R[j]=(total_work>>k)*(1+j);
        }
        quadratic_poly qp[thd];
        std::vector<std::thread> workers;
        workers.reserve(thd);
        for(int j=0;j<thd;j++)
        {
            workers.emplace_back(
                sumcheckUpdate_worker,
                std::ref(qp[j]),
                std::ref(tmp_v),
                std::ref(tmp_mult),
                std::ref(tmp_v_2),
                std::ref(tmp_mult_2),
                std::ref(L),
                std::ref(R),
                previous_random,
                total_size[idx]
            );
        }
        while(endq.Size()!=(1<<k))
            this_thread::sleep_for(std::chrono::microseconds(1));
        for (auto &worker : workers)
            worker.join();
        endq.Clear();
        for(int j=0;j<thd;j++)
            ret=ret+qp[j];
        tt_f1.stop();
        tt_f2.start();
        for(int i=0;i<total_work;i++)
        {
            if((i<<1)<total_size[idx])
            {
                tmp_mult[i]=tmp_mult_2[i];
                tmp_v[i]=tmp_v_2[i];
            }
            else
            {
                tmp_mult[i].clear();
                tmp_v[i].clear();
            }
        }
        tt_f2.stop();
    }
    
    total[idx] >>= 1;
    total_size[idx] = (total_size[idx] + 1) >> 1;

    return ret;
}


/**
 * This is to evaluate a multi-linear extension at a random point.
 *
 * @param the value of the array & random point & the size of the array & the size of the random point
 * @return sum of `values`, or 0.0 if `values` is empty.
 */
F prover::Vres(const vector<F>::const_iterator &r, u32 output_size, i64 r_size,int layer_id,int start) 
{
    prove_timer.start();

    vector<F> output(output_size);
    for (u32 i = 0; i < output_size; ++i)
        output[i] = val[layer_id][i+start];
    u32 whole = 1ULL << r_size;
    for (i64 i = 0; i < r_size; ++i) {
        for (u32 j = 0; j < (whole >> 1); ++j) {
            if (j > 0)
                output[j].clear();
            if ((j << 1) < output_size)
                output[j] = output[j << 1] * (F_ONE - r[i]);
            if ((j << 1 | 1) < output_size)
                output[j] = output[j] + output[j << 1 | 1] * (r[i]);
        }
        whole >>= 1;
    }
    F res = output[0];

    prove_timer.stop();
    proof_size += F_BYTE_SIZE;
    return res;
}

void prover::sumcheckFinalize1(const F &previous_random, F &claim_0, F &claim_1) {
    prove_timer.start();
    r_u[sumcheck_id].at(round - 1) = previous_random;
    V_u0 = claim_0 = total[0] ? V_mult[0][0].eval(previous_random) : (~C.circuit[sumcheck_id].bit_length_u[0]) ? V_mult[0][0].b : F_ZERO;
    V_u1 = claim_1 = total[1] ? V_mult[1][0].eval(previous_random) : (~C.circuit[sumcheck_id].bit_length_u[1]) ? V_mult[1][0].b : F_ZERO;
    prove_timer.stop();

    mult_array[0].clear();
    mult_array[1].clear();
    V_mult[0].clear();
    V_mult[1].clear();
    proof_size += F_BYTE_SIZE * 2;
}

void prover::sumcheckFinalize2(const F &previous_random, F &claim_0, F &claim_1) {
    prove_timer.start();
    r_v[sumcheck_id].at(round - 1) = previous_random;
    claim_0 = total[0] ? V_mult[0][0].eval(previous_random) : (~C.circuit[sumcheck_id].bit_length_v[0]) ? V_mult[0][0].b : F_ZERO;
    claim_1 = total[1] ? V_mult[1][0].eval(previous_random) : (~C.circuit[sumcheck_id].bit_length_v[1]) ? V_mult[1][0].b : F_ZERO;
    prove_timer.stop();

    mult_array[0].clear();
    mult_array[1].clear();
    V_mult[0].clear();
    V_mult[1].clear();
    proof_size += F_BYTE_SIZE * 2;
}

void prover::sumcheck_lasso_Finalize(const F &previous_random, F &claim_1) 
{
    prove_timer.start();
    r_u[sumcheck_id].at(round - 1) = previous_random;
    claim_1 = total[1] ? V_mult[1][0].eval(previous_random) : V_mult[1][0].b;
    prove_timer.stop();
    proof_size += F_BYTE_SIZE;
}



void prover::commitInput(const vector<G1> &gens,int thr, bool disable) 
{
    if (disable) {
        std::cout << "commitInput is skipped" << std::endl;
        cc.l = 0;
        cc.w = nullptr;
        cc.ww = nullptr;
        cc.comm = nullptr;
        cc.g = gens.empty() ? nullptr : const_cast<G1*>(gens.data());
        cc.G.clear();
        return;
    }

    const size_t original_size = val[0].size();
    if (C.circuit[0].size != (1ULL << C.circuit[0].bit_length)) 
    {
        val[0].resize(1ULL << C.circuit[0].bit_length);
        for (size_t i = C.circuit[0].size; i < val[0].size(); ++i)
            val[0][i].clear();
    }
    
    int l=ceil(log2(val[0].size()));  // 28
    const size_t padded_size = size_t(1) << l;
    ll* vi=new ll[padded_size];  // at least pr.val[0].size()
    memset(vi,0,sizeof(ll)*padded_size);  // set vi to 0
    ll mx=-1e9,mn=1e9;
    for(size_t i = 0; i < val[0].size(); i++)
    {
        vi[i]=convert(val[0][i]);  // usually inputs/weights, safely convert Fr 254 to 64-bit
        mx=max(mx,vi[i]);
        mn=min(mn,vi[i]);
        
    }

    
    
    Fr* dat=new Fr[padded_size];
    memset(dat,0,sizeof(Fr)*padded_size);
    memcpy(dat,val[0].data(),sizeof(Fr)*val[0].size());
    G1* ret=prover_commit(vi,(G1*)gens.data(),l,thr);
    cc.comm=ret;
    cc.G=gens.back();
    cc.g=(G1*)gens.data();
    cc.l=l;
    cc.w=dat;  // TODO(jm): memory leak
    cc.ww=vi;  // TODO(jm): memory leak
    val[0].resize(original_size);
}

// convert Fr to 64-bit int (taking the lower 64 bits)
__int128 convert(Fr x)	
{	
    int sign=0;	
    Fr abs;	
    if(x.isNegative())	
    {	
        sign=1;	
        abs=-x;	
    }	
    else	
        abs=x;	

    uint8_t bf[16]={0};	 //64 bit
    int size=abs.getLittleEndian(bf,16);	
    ll V=0;	
    for(int j=size-1;j>=0;j--)	
        V=V*256+bf[j];	
    if(sign)	
        V=-V;	
    return V;	
}
