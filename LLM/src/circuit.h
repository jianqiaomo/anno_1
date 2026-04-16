#pragma once

#include <vector>
#include <deque>
#include <unordered_map>
#include <utility>

#include <unordered_set>
#include <iostream>
#include "global_var.hpp"
#include "hyrax.hpp"
using std::cerr;
using std::endl;
using std::vector;

struct uniGate {
    u64 g, u;
    uint16_t lu;
    ll sc;
    template <typename G, typename U, typename LU>
    uniGate(G _g, U _u, LU _lu, ll _sc) :
        g(static_cast<u64>(_g)), u(static_cast<u64>(_u)), lu(static_cast<uint16_t>(_lu)), sc(_sc) {
    }
};

struct binGate 
{
    u64 g, u, v;
    uint16_t  l;
    ll sc;
    template <typename G, typename U, typename V, typename L>
    binGate(G _g, U _u, V _v, ll _sc, L _l):
        g(static_cast<u64>(_g)), u(static_cast<u64>(_u)), v(static_cast<u64>(_v)), sc(_sc), l(static_cast<uint16_t>(_l)) 
        {
    }
    [[nodiscard]] uint16_t getLayerIdU(uint16_t layer_id) const { return !l ? 0 : layer_id - 1; }
    [[nodiscard]] uint16_t getLayerIdV(uint16_t layer_id) const { return !(l & 1) ? 0 : layer_id - 1; }
};

enum class layerType {
//    0     1     2      3       4     5        6           7          8         9        10      11         12            13             14       15     16     17     18      19       20        21          22           23
    INPUT, FFT, IFFT, ADD_BIAS, RELU, Sqr, OPT_AVG_POOL, MAX_POOL, AVG_POOL, DOT_PROD, PADDING, FCONN,  LAYER_NORM_1, LAYER_NORM_2 ,LAYER_NORM_3,GELU_1,GELU_2,GELU_3,MHA_QK,SOFTMAX_1,SOFTMAX_2,SOFTMAX_3,PLACE_HOLDER,RELU_CHECK,
};

class layer {
public:
    layerType ty;
    std::vector<pair<int,int> > uni_interval,bin_interval;
	u32 size{}, size_u[2]{}, size_v[2]{};
	i8 bit_length_u[2]{}, bit_length_v[2]{}, bit_length{};
    i8 max_bl_u{}, max_bl_v{};

    bool need_phase2;

    // bit decomp related
    u32 zero_start_id;

    std::deque<uniGate> uni_gates;
	std::deque<binGate> bin_gates;

	vector<u64> ori_id_u, ori_id_v;
    i8 fft_bit_length;

    // iFFT or avg pooling.
    //F scale;

	layer() 
    {
        bit_length_u[0] = bit_length_v[0] = -1;
        size_u[0] = size_v[0] = 0;
        bit_length_u[1] = bit_length_v[1] = -1;
        size_u[1] = size_v[1] = 0;
        need_phase2 = false;
        zero_start_id = 0;
        fft_bit_length = -1;
        //scale = F_ONE;
	}

	void updateSize() {
	    max_bl_u = std::max(bit_length_u[0], bit_length_u[1]);
	    max_bl_v = 0;
	    if (!need_phase2) return;

        max_bl_v = std::max(bit_length_v[0], bit_length_v[1]);
	}
};

class layeredCircuit {
public:
	vector<layer> circuit;
    i64 size;
    void init(u8 q_bit_size, i64 _layer_sz);
	void initSubset();
};
