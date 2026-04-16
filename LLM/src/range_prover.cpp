#include "range_prover.hpp"
#include <iostream>
#include <utils.hpp>

static vector<F> beta_gs, beta_u;
using namespace mcl::bn;
using std::unique_ptr;

int lg2(int x) 
{
    int res = 0;
    while (x >>= 1) ++res;
    return res;
}
Fr poly_eval(Fr x0,Fr x1,Fr x2,Fr x3,Fr u)  //lagrange poly for degree 3
{
    Fr y=1/Fr(6)*((-x0)*(u-1)*(u-2)*(u-3)+3*x1*u*(u-2)*(u-3)-3*x2*u*(u-1)*(u-3)+x3*u*(u-1)*(u-2));
    return y;
}

range_prover::SC_Return range_prover::sumcheck_deg1(int l, Fr* f, Fr S) // sum_i f_i=S
{
    // P send V sum S
    Fr *ran=new Fr[l];
    for(int i=l;i>=1;i--) // round i
    {
        Fr sum0=0,sum1=0;
        for(int j=0;j<(1<<i);j++)
        {
            if((j&1)==0)
                sum0+=f[j];
            else
            {
                sum1+=f[j];
            }
        }
        // assert(sum0+sum1==S);
        //send poly: sum0,sum1
        Fr new_chlg;
        new_chlg.setByCSPRNG();
        ran[l-i]=new_chlg;
        //V send new_chlg
        S=new_chlg*(sum1-sum0)+sum0; // update target sum
        Fr* new_f=new Fr[size_t(1) << (i-1)];
        Fr sum0p=0,sum1p=0;
        for(int j=0;j<(1<<(i-1));j++)
            new_f[j]=(1-new_chlg)*f[j*2]+new_chlg*f[j*2+1];
        f=new_f;
    }
    SC_Return s;
    s.random=ran;
    s.claim_f=f[0];
    return s;
}

range_prover::SC_Return range_prover::sumcheck_deg3(int l, Fr* r, Fr* f, Fr* g, Fr S) // sum_i eq(r,i) f_i g_i=S
{
    Fr* lag=range_proof_get_eq(r,l);
    // P send V sum S
    Fr *ran=new Fr[l];
    Fr *S0=new Fr[size_t(1) << l],*S1=new Fr[size_t(1) << l],*S2=new Fr[size_t(1) << l],*S3=new Fr[size_t(1) << l];
    for(int i=l;i>=1;i--) // round i
    {
        memset(S0,0,sizeof(Fr)*(1<<i));
        memset(S1,0,sizeof(Fr)*(1<<i));
        memset(S2,0,sizeof(Fr)*(1<<i));
        memset(S3,0,sizeof(Fr)*(1<<i));

        Fr sum0=0,sum1=0,sum2=0,sum3=0;

        #pragma omp parallel for
        for(int j=0;j<(1<<i);j+=2)
        {
            S0[j>>1]=lag[j]*f[j]*g[j];
            S1[j>>1]=lag[j+1]*f[j+1]*g[j+1];
            S2[j>>1]=(lag[j+1]+lag[j+1]-lag[j])*(f[j+1]+f[j+1]-f[j])*(g[j+1]+g[j+1]-g[j]);
            S3[j>>1]=(lag[j+1]+lag[j+1]+lag[j+1]-lag[j]-lag[j])*(f[j+1]+f[j+1]+f[j+1]-f[j]-f[j])*(g[j+1]+g[j+1]+g[j+1]-g[j]-g[j]);  
        }

        if(i<8)
        {
            for(int j=0;j<(1<<(i-1));j++)
            {
                sum0+=S0[j];
                sum1+=S1[j];
                sum2+=S2[j];
                sum3+=S3[j];
            }
        }
        else
        {
            Fr s0[8],s1[8],s2[8],s3[8];
            memset(s0,0,sizeof(Fr)*8);
            memset(s1,0,sizeof(Fr)*8);
            memset(s2,0,sizeof(Fr)*8);
            memset(s3,0,sizeof(Fr)*8);
            #pragma omp parallel for
            for(int k=0;k<(1<<3);k++)
            {
                for(int j=0;j<(1<<(i-1-3));j++)
                {
                    s0[k]+=S0[(k<<(i-1-3))+j];
                    s1[k]+=S1[(k<<(i-1-3))+j];
                    s2[k]+=S2[(k<<(i-1-3))+j];
                    s3[k]+=S3[(k<<(i-1-3))+j];
                }
            }
            sum0=s0[0]+s0[1]+s0[2]+s0[3]+s0[4]+s0[5]+s0[6]+s0[7];
            sum1=s1[0]+s1[1]+s1[2]+s1[3]+s1[4]+s1[5]+s1[6]+s1[7];
            sum2=s2[0]+s2[1]+s2[2]+s2[3]+s2[4]+s2[5]+s2[6]+s2[7];
            sum3=s3[0]+s3[1]+s3[2]+s3[3]+s3[4]+s3[5]+s3[6]+s3[7];
        }
        assert(sum0+sum1==S);
        //send poly: sum0,sum1,sum2,sum3
        Fr new_chlg;
        new_chlg.setByCSPRNG();
        ran[l-i]=new_chlg;
        //V send new_chlg
        S=poly_eval(sum0,sum1,sum2,sum3,new_chlg); // update target sum
        Fr* new_lag=new Fr[size_t(1) << (i-1)];
        Fr* new_f=new Fr[size_t(1) << (i-1)];
        Fr* new_g=new Fr[size_t(1) << (i-1)];
        Fr sum0p=0,sum1p=0;
        #pragma omp parallel for 
        for(int j=0;j<(1<<(i-1));j++)
        {
            new_lag[j]=lag[j*2]+new_chlg*(lag[j*2+1]-lag[j*2]);
            new_f[j]=f[j*2]+new_chlg*(f[j*2+1]-f[j*2]);
            new_g[j]=g[j*2]+new_chlg*(g[j*2+1]-g[j*2]);
        }
        f=new_f;
        g=new_g;
        lag=new_lag;
    }
    SC_Return s;
    s.random=ran;
    s.claim_f=f[0];
    s.claim_g=g[0];
    return s;
}

void Constraint::prepare() {
    static std::mt19937_64 rng(std::random_device{}());
    int l = query_size;
    assert(l <= (1 << MAXL));
    for (int i = 0; i < l; i++) {
        inputs[i] = rng() & ((1 << range_size) - 1); // 2^range_size -1
    }
}

void range_prover::init() {
    GG=range_proof_gen_gi(g,1<<(MAXL-MAXL/2));
}
void range_prover::push_back(NonlinearOpType op_type, const std::vector<std::pair<int, int>>& constraint_params) {
    OP op;
    op.op_type = op_type;
    for (const auto& [query_size, range_size] : constraint_params) {
        Constraint c;
        c.query_size = query_size;
        c.range_size = range_size;
        c.inputs = inputs;
        op.constraints.push_back(c);
    }
    ops.push_back(op);
}
void range_prover::range_prove(ll * x,int range,int m,int thread_num) {
    
    // int log = 9; // range = 16 or 32
    int log = 7; 
    assert(range%log==0);
    cout << "range_prover::range_prove, range=" << range << ", log=" << log << ", m=" << m << endl;
    int l=range/log;  // preparing a lookup table for a Lasso-based range check.
    ll** data=new ll*[l]; //data size=l*m; TODO(jm): memory leak
    for(int i=0;i<l;i++)
        data[i]=new ll[m];
    for(int i=0;i<l;i++)
    {
        for(int j=0;j<m;j++)
        {
            data[i][j]=(x[j]>>(log*i))&((1<<log)-1);
        }
    }
    ll* t=new ll[size_t(1) << log];
    for(int i=0;i<(1<<log);i++)
    {
        t[i]=i;
    }
    for(int i=0;i<l;i++)
    {
        logup(data[i],t,m,1<<log,thread_num,i);
    }
}


void range_prover::logup(ll * f,ll *t,int m,int n,int thread, int idx)
{
    ll * c=new ll[n];
    memset(c,0,sizeof(ll)*n);
    for(int i=0;i<m;i++)
        c[f[i]]++;
    
    Fr r;
    r.setByCSPRNG();

    auto jmdbg_timer0 = std::chrono::high_resolution_clock::now();
    Fr* F=new Fr[m];
    for(int i=0;i<m;i++)
        F[i]=r+Fr(f[i]);
    auto jmdbg_timer1 = std::chrono::high_resolution_clock::now();
    auto jmdbg_ms1 = std::chrono::duration_cast<std::chrono::microseconds>(jmdbg_timer1 - jmdbg_timer0).count();
    std::cout << "(jmTimer)range_prover::logup, " << idx << ", F[i], time=" << jmdbg_ms1 << " us" << std::endl;

    auto jmdbg_timer2 = std::chrono::high_resolution_clock::now();
    Fr* Hp=new Fr[n];
    for(int i=0;i<n;i++)
        Hp[i]=r+Fr(t[i]);
    auto jmdbg_timer3 = std::chrono::high_resolution_clock::now();
    auto jmdbg_ms2 = std::chrono::duration_cast<std::chrono::microseconds>(jmdbg_timer3 - jmdbg_timer2).count();
    std::cout << "(jmTimer)range_prover::logup, " << idx << ", Hp[i], time=" << jmdbg_ms2 << " us" << std::endl;

    auto jmdbg_timer4 = std::chrono::high_resolution_clock::now();
    Fr* G=new Fr[m];
    invVec(G,F,m);  // 1/(r+fi)
    auto jmdbg_timer5 = std::chrono::high_resolution_clock::now();
    auto jmdbg_ms3 = std::chrono::duration_cast<std::chrono::microseconds>(jmdbg_timer5 - jmdbg_timer4).count();
    std::cout << "(jmTimer)range_prover::logup, " << idx << ", invVec_G, time=" << jmdbg_ms3 << " us" << std::endl;

    auto jmdbg_timer6 = std::chrono::high_resolution_clock::now();
    Fr* H=new Fr[n];
    invVec(H,Hp,n);  // 1/(r+ti)
    auto jmdbg_timer7 = std::chrono::high_resolution_clock::now();
    auto jmdbg_ms4 = std::chrono::duration_cast<std::chrono::microseconds>(jmdbg_timer7 - jmdbg_timer6).count();
    std::cout << "(jmTimer)range_prover::logup, " << idx << ", invVec_H, time=" << jmdbg_ms4 << " us" << std::endl;
    
    Fr s1=0,s2=0;
    for(int i=0;i<m;i++)
        s1+=G[i];
    for(int i=0;i<n;i++)
    {
        H[i]=H[i]*Fr(c[i]);
        s2+=H[i];
    }


    auto jmdbg_timer8 = std::chrono::high_resolution_clock::now();
    G1* f_comm = range_proof_prover_commit(f, g, lg2(m), thread);
    auto jmdbg_timer9 = std::chrono::high_resolution_clock::now();
    auto jmdbg_ms5 = std::chrono::duration_cast<std::chrono::microseconds>(jmdbg_timer9 - jmdbg_timer8).count();
    std::cout << "(jmTimer)range_prover::logup, " << idx << ", commit f, time=" << jmdbg_ms5 << " us" << std::endl;

    auto jmdbg_timer10 = std::chrono::high_resolution_clock::now();
    G1* t_comm = range_proof_prover_commit(t, g, lg2(n), thread);
    auto jmdbg_timer11 = std::chrono::high_resolution_clock::now();
    auto jmdbg_ms6 = std::chrono::duration_cast<std::chrono::microseconds>(jmdbg_timer11 - jmdbg_timer10).count();
    std::cout << "(jmTimer)range_prover::logup, " << idx << ", commit t, time=" << jmdbg_ms6 << " us" << std::endl;

    auto jmdbg_timer12 = std::chrono::high_resolution_clock::now();
    G1* c_comm = range_proof_prover_commit(c, g, lg2(n), thread);
    auto jmdbg_timer13 = std::chrono::high_resolution_clock::now();
    auto jmdbg_ms7 = std::chrono::duration_cast<std::chrono::microseconds>(jmdbg_timer13 - jmdbg_timer12).count();
    std::cout << "(jmTimer)range_prover::logup, " << idx << ", commit c, time=" << jmdbg_ms7 << " us" << std::endl;

    auto jmdbg_timer14 = std::chrono::high_resolution_clock::now();
    Fr *diff=new Fr[n];
    for(int i=0;i<n;i++)
    {
        diff[i]=1/(r+t[i]);
    }
    auto jmdbg_timer15 = std::chrono::high_resolution_clock::now();
    auto jmdbg_ms8 = std::chrono::duration_cast<std::chrono::microseconds>(jmdbg_timer15 - jmdbg_timer14).count();
    std::cout << "(jmTimer)range_prover::logup, " << idx << ", compute diff, time=" << jmdbg_ms8 << " us" << std::endl;

    auto jmdbg_timer16 = std::chrono::high_resolution_clock::now();
    G1* g_comm=range_proof_prover_commit_fr(f,diff,n,g,lg2(m),thread);
    auto jmdbg_timer17 = std::chrono::high_resolution_clock::now();
    auto jmdbg_ms9 = std::chrono::duration_cast<std::chrono::microseconds>(jmdbg_timer17 - jmdbg_timer16).count();
    std::cout << "(jmTimer)range_prover::logup, " << idx << ", commit diff, time=" << jmdbg_ms9 << " us" << std::endl;

    auto jmdbg_timer18 = std::chrono::high_resolution_clock::now();
    G1* h_comm=range_proof_prover_commit_fr_general(H,g,lg2(n),thread);
    auto jmdbg_timer19 = std::chrono::high_resolution_clock::now();
    auto jmdbg_ms10 = std::chrono::duration_cast<std::chrono::microseconds>(jmdbg_timer19 - jmdbg_timer18).count();
    std::cout << "(jmTimer)range_prover::logup, " << idx << ", commit H, time=" << jmdbg_ms10 << " us" << std::endl;
    
    Fr sum=0;
    for(int i=0;i<m;i++)
        sum+=G[i];
    Fr* rp1=new Fr[lg2(n)];
    for(int i=0;i<lg2(n);i++)
        rp1[i].setByCSPRNG(); //verifier challenge

    auto jmdbg_timer20 = std::chrono::high_resolution_clock::now();
    Fr c_eva=range_proof_prover_evaluate(c,rp1,lg2(n)); // c_eva = Σ_{k=0}^{2^μ-1} eq(rp1, k) * c[k]
    auto jmdbg_timer21 = std::chrono::high_resolution_clock::now();
    auto jmdbg_ms20 = std::chrono::duration_cast<std::chrono::microseconds>(jmdbg_timer21 - jmdbg_timer20).count();
    std::cout << "(jmTimer)range_prover::logup, " << idx << ", c_eva time=" << jmdbg_ms20 << " us" << std::endl;

    
    auto jmdbg_timer22 = std::chrono::high_resolution_clock::now();
    range_prover::SC_Return ret1=sumcheck_deg3(lg2(n),rp1,H,Hp,c_eva); // sumcheck, reciprocal well-formed   Hi*(r+ti)=ci
    auto jmdbg_timer23 = std::chrono::high_resolution_clock::now();
    auto jmdbg_ms22 = std::chrono::duration_cast<std::chrono::microseconds>(jmdbg_timer23 - jmdbg_timer22).count();
    std::cout << "(jmTimer)range_prover::logup, " << idx << ", sumcheck3 H Hp ret1 time=" << jmdbg_ms22 << " us" << std::endl;

    Fr* rp2=new Fr[lg2(m)];
    for(int i=0;i<lg2(m);i++)
        rp2[i].setByCSPRNG(); //verifier challenge

    auto jmdbg_timer24 = std::chrono::high_resolution_clock::now();
    range_prover::SC_Return ret2=sumcheck_deg3(lg2(m),rp2,G,F,1); // sumcheck, reciprocal well-formed  Gi*(r+fi)=1
    auto jmdbg_timer25 = std::chrono::high_resolution_clock::now();
    auto jmdbg_ms24 = std::chrono::duration_cast<std::chrono::microseconds>(jmdbg_timer25 - jmdbg_timer24).count();
    std::cout << "(jmTimer)range_prover::logup, " << idx << ", sumcheck3 G F ret2 time=" << jmdbg_ms24 << " us" << std::endl;


    auto jmdbg_timer26 = std::chrono::high_resolution_clock::now();
    range_prover::SC_Return ret3=sumcheck_deg1(lg2(m),G,sum); // sumcheck, reciprocal sum
    auto jmdbg_timer27 = std::chrono::high_resolution_clock::now();
    auto jmdbg_ms26 = std::chrono::duration_cast<std::chrono::microseconds>(jmdbg_timer27 - jmdbg_timer26).count();
    std::cout << "(jmTimer)range_prover::logup, " << idx << ", sumcheck1 G ret3 time=" << jmdbg_ms26 << " us" << std::endl;
    
    auto jmdbg_timer28 = std::chrono::high_resolution_clock::now();
    range_prover::SC_Return ret4=sumcheck_deg1(lg2(n),H,sum); // sumcheck, reciprocal sum
    auto jmdbg_timer29 = std::chrono::high_resolution_clock::now();
    auto jmdbg_ms28 = std::chrono::duration_cast<std::chrono::microseconds>(jmdbg_timer29 - jmdbg_timer28).count();
    std::cout << "(jmTimer)range_prover::logup, " << idx << ", sumcheck1 H ret4 time=" << jmdbg_ms28 << " us" << std::endl;

    

    
    auto jmdbg_timer30 = std::chrono::high_resolution_clock::now();
    range_proof_open(c,rp1,c_eva,GG,g,c_comm,lg2(n));
    auto jmdbg_timer31 = std::chrono::high_resolution_clock::now();
    auto jmdbg_ms30 = std::chrono::duration_cast<std::chrono::microseconds>(jmdbg_timer31 - jmdbg_timer30).count();
    std::cout << "(jmTimer)range_prover::logup, " << idx << ", open c time=" << jmdbg_ms30 << " us" << std::endl;

    auto jmdbg_timer32 = std::chrono::high_resolution_clock::now();
    range_proof_open(H,ret1.random,ret1.claim_f,GG,g,h_comm,lg2(n));
    auto jmdbg_timer33 = std::chrono::high_resolution_clock::now();
    auto jmdbg_ms32 = std::chrono::duration_cast<std::chrono::microseconds>(jmdbg_timer33 - jmdbg_timer32).count();
    std::cout << "(jmTimer)range_prover::logup, " << idx << ", open H (ret1) time=" << jmdbg_ms32 << " us" << std::endl;

    auto jmdbg_timer34 = std::chrono::high_resolution_clock::now();
    range_proof_open(t,ret1.random,ret1.claim_g-r,GG,g,t_comm,lg2(n));
    auto jmdbg_timer35 = std::chrono::high_resolution_clock::now();
    auto jmdbg_ms34 = std::chrono::duration_cast<std::chrono::microseconds>(jmdbg_timer35 - jmdbg_timer34).count();
    std::cout << "(jmTimer)range_prover::logup, " << idx << ", open t time=" << jmdbg_ms34 << " us" << std::endl;

    auto jmdbg_timer36 = std::chrono::high_resolution_clock::now();
    range_proof_open(G,ret2.random,ret2.claim_f,GG,g,g_comm,lg2(m));
    auto jmdbg_timer37 = std::chrono::high_resolution_clock::now();
    auto jmdbg_ms36 = std::chrono::duration_cast<std::chrono::microseconds>(jmdbg_timer37 - jmdbg_timer36).count();
    std::cout << "(jmTimer)range_prover::logup, " << idx << ", open G (ret2) time=" << jmdbg_ms36 << " us" << std::endl;

    auto jmdbg_timer38 = std::chrono::high_resolution_clock::now();
    range_proof_open(f,ret2.random,ret2.claim_g-r,GG,g,f_comm,lg2(m));
    auto jmdbg_timer39 = std::chrono::high_resolution_clock::now();
    auto jmdbg_ms38 = std::chrono::duration_cast<std::chrono::microseconds>(jmdbg_timer39 - jmdbg_timer38).count();
    std::cout << "(jmTimer)range_prover::logup, " << idx << ", open f time=" << jmdbg_ms38 << " us" << std::endl;

    auto jmdbg_timer40 = std::chrono::high_resolution_clock::now();
    range_proof_open(G,ret3.random,ret3.claim_f,GG,g,g_comm,lg2(m));
    auto jmdbg_timer41 = std::chrono::high_resolution_clock::now();
    auto jmdbg_ms40 = std::chrono::duration_cast<std::chrono::microseconds>(jmdbg_timer41 - jmdbg_timer40).count();
    std::cout << "(jmTimer)range_prover::logup, " << idx << ", open G (ret3) time=" << jmdbg_ms40 << " us" << std::endl;

    auto jmdbg_timer42 = std::chrono::high_resolution_clock::now();
    range_proof_open(H,ret4.random,ret4.claim_f,GG,g,h_comm,lg2(n));
    auto jmdbg_timer43 = std::chrono::high_resolution_clock::now();
    auto jmdbg_ms42 = std::chrono::duration_cast<std::chrono::microseconds>(jmdbg_timer43 - jmdbg_timer42).count();
    std::cout << "(jmTimer)range_prover::logup, " << idx << ", open H (ret4) time=" << jmdbg_ms42 << " us" << std::endl;
    
    delete[] G;
    delete[] F;
}

double range_prover::prove() {
    double prover_time = 0;
    for (auto& op : ops) {
        for (auto& constraint : op.constraints) {
            cout << "start prepare inputs" << endl;
            constraint.prepare();
            cout << "start range prove"<<" query_size "<< constraint.query_size << endl;
            prove_timer.start();
            range_prove(constraint.inputs, constraint.range_size, constraint.query_size, 32);
            cout << "end range prove" << endl;
            prove_timer.stop();
            prover_time += prove_timer.elapse_sec();
            cout << "time: " << prove_timer.elapse_sec() << endl;
        }
    }
    return prover_time;
}

int next_power_of_2(int n) {
    int power = 1;
    while (power < n) {
        power *= 2;
    }
    return power;
}

void range_prover::build() {
    
    int query_size = 0;
    std::vector<std::pair<int, int>> params;
    
    if (!merged){
        // layernorm
        query_size += (3*seq_len + 2*seq_len*AttnDim);

        // attetion
        query_size += (3*HeadNum*seq_len*(seq_len+1)/2 + seq_len*AttnDim);

        // layernorms
        query_size += (3*seq_len + 2*seq_len*AttnDim);

        // linear (GELU)
        query_size += (2*seq_len*LinearDim);

        query_size = next_power_of_2(query_size*LayerNum);
        push_back(NonlinearOpType::NonLinear, {{query_size, 49}}); // query_size, range_size
    }
    else {
        // layernorm
        query_size += (seq_len + seq_len*AttnDim);

        // attention
        query_size += (HeadNum*seq_len*(seq_len+1)/2 + seq_len*AttnDim);

        // layernorms
        query_size += (seq_len + seq_len*AttnDim);

        // linear (GELU)
        query_size += (seq_len*LinearDim);

        query_size = next_power_of_2(query_size*LayerNum);
        push_back(NonlinearOpType::NonLinear, {{query_size, 49}}); // query_size, range_size
        
    }
    
}
