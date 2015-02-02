#pragma GCC diagnostic ignored "-Wunused-result"

#ifndef _COMMON_H_
#define _COMMON_H_

#define flag { printf("\nLINE: %d\n", __LINE__); fflush(stdout); }

#include <vector>
#include <cmath>
#include <unordered_map>
#include <pmmintrin.h>

#include "pairhash.h"

struct Problem
{
    Problem(uint32_t const nr_instance, uint32_t const nr_feature, uint64_t const range_sum)
          :nr_instance(nr_instance), nr_feature(nr_feature), range_sum(range_sum),  
          norm(2.0f/static_cast<float>(nr_feature)),
          J(static_cast<uint64_t>(nr_instance) * nr_feature, range_sum), 
          Y(nr_instance){}
    uint32_t nr_instance, nr_feature;
    uint64_t range_sum;
    float norm;
    std::vector<uint64_t> J; 
    std::vector<float> Y; 
};

Problem read_tr_problem_fast(std::string const tr_path, std::unordered_map<std::pair<uint32_t, uint32_t>, uint64_t, pairhash> &fviMap);

Problem read_tr_problem(std::string const tr_path, std::unordered_map<std::pair<uint32_t, uint32_t>, uint64_t, pairhash> &fviMap);

Problem read_va_problem(std::string const va_path, std::unordered_map<std::pair<uint32_t, uint32_t>, uint64_t, pairhash> &fviMap, uint64_t const range_sum, uint32_t const nr_feature);

uint32_t const NODE_SIZE = 2;

struct Model
{
    Model(uint64_t const range_sum, uint32_t const nr_factor) 
        : range_sum(range_sum), nr_factor(nr_factor), w0(0.0f), w0_sg(1.0f),
        W(range_sum * NODE_SIZE),
        V(range_sum * nr_factor * NODE_SIZE),
        L(nr_factor * NODE_SIZE) {}
    uint64_t range_sum;
    uint32_t nr_factor;
    float w0, w0_sg;
    std::vector<float> W;
    std::vector<float> V;
    std::vector<float> L;
};

FILE *open_c_file(std::string const &path, std::string const &mode);

std::vector<std::string> 
argv_to_args(int const argc, char const * const * const argv);

inline float wTx(Problem const &prob, Model &model, uint32_t const i, 
    float const kappa=0, float const eta0=0, float const eta1 = 0,
    float const eta2=0, float const eta3=0, float const l0=0, float const l1=0,
    float const l2=0, float const l3=0, bool const do_update=false)
{
    uint32_t const nr_feature = prob.nr_feature;
    uint64_t const range_sum = prob.range_sum; 
    uint32_t const nr_factor = model.nr_factor;

    uint32_t const valign = nr_factor * NODE_SIZE; // +valign: next feature value

    //initialize pointers
    uint64_t const * const J = &prob.J[i * nr_feature]; 
    float * const W = model.W.data(); //first order parameter
    float * const V = model.V.data(); //second order parameter
    float * const L = model.L.data(); //Lambda

    float predict = 0.0;

    //compute sum_vx[f] = \sum_{j = 1}^n v_{j, f}x_j and its square
    float sum_vx[nr_factor], sum_vx_square[nr_factor];
    for(uint32_t d = 0; d < nr_factor; ++d)
    {
        sum_vx[d] = 0.0;
        for(uint32_t f = 0; f < nr_feature; ++f)
        {
            uint64_t const j = J[f];
            if(j >= range_sum)
                continue;
            sum_vx[d] += *(V + j * valign + d);
        }
        sum_vx_square[d] = sum_vx[d] * sum_vx[d];
    }

    if(do_update)
    {
        //update w0
        float const gradient = kappa + l0 * model.w0;
        model.w0 -= static_cast<float>(eta0 * gradient / sqrt(model.w0_sg));
        model.w0_sg += (gradient * gradient);

        //update first order and diagonal parameters
        for(uint32_t f = 0; f < nr_feature; ++f)
        {
            uint64_t const j = J[f];
            if(j >= range_sum) //ignore if not present in training set
                continue;
            float * const w = W + j * NODE_SIZE;
            float * const shg = w + 1;
            float const gradient = kappa + l1 * *w;
            *w -= static_cast<float>(eta1 * gradient / sqrt(*shg) ); 
            *shg += (gradient * gradient); //update history gradient sum
        }

        //update second order parameters
        for(uint32_t d = 0; d < nr_factor; ++d)
        {
            for(uint32_t f = 0; f < nr_feature; ++f)
            {
                uint64_t const j = J[f];
                if(j >= range_sum) 
                    continue;
                float * const v = V + j * valign + d;  
                float * const shg = v + nr_factor;
                float const gradient = kappa * sum_vx[d] * *(L+d*NODE_SIZE) + l2 * *v; 
                *v -= static_cast<float> (eta2 * gradient / sqrt(*shg)); 
                *shg += (gradient * gradient); //update sum history gradient
            }
        }

        //update lambda parameters
        for(uint32_t d = 0; d < nr_factor; ++d)
        {
            float * const l = L + d * NODE_SIZE; 
            float * const shg = l + 1;
            float const gradient = kappa * sum_vx_square[d] / 2 + l3 * *l;
            *l -= static_cast<float>(eta3 * gradient / sqrt(*shg) );
            *shg += (gradient * gradient);
        }
    }
    else
    {
        predict += model.w0;

        //compute first order value
        for(uint32_t f = 0; f < nr_feature; ++f)
        {
            uint64_t const j = J[f];
            if(j >= range_sum)
                continue;
            predict += *(W + j * NODE_SIZE);
        }
        
        //compute second order value
        float second_order = 0.0;
        for(uint32_t d = 0; d < nr_factor; ++d)
        {
            second_order += *(L + d * NODE_SIZE) * sum_vx_square[d];
        }
        predict += (second_order / 2);
    }

    if(do_update)
        return 0;
    return predict;
}

inline float wTx_sse(Problem const &prob, Model &model, uint32_t const i, 
    float const kappa=0, float const eta0=0, float const eta1 = 0,
    float const eta2=0, float const eta3=0, float const l0=0, float const l1=0,
    float const l2=0, float const l3=0, bool const do_update=false)
{
    uint32_t const nr_feature = prob.nr_feature;
    uint64_t const range_sum = prob.range_sum; 
    uint32_t const nr_factor = model.nr_factor;

    uint32_t const valign = nr_factor * NODE_SIZE; // +valign: next feature value

    //initialize pointers
    uint64_t const * const J = &prob.J[i * nr_feature]; 
    float * const W = model.W.data(); //first order parameter
    float * const V = model.V.data(); //second order parameter
    float * const L = model.L.data(); //Lambda

    float predict = 0.0;

    //compute sum_vx[f] = \sum_{j = 1}^n v_{j, f}x_j and its square
    std::vector<float> sum_vx(nr_factor, 0.0), sum_vx_square(nr_factor, 0.0);
    for(uint32_t d = 0; d < nr_factor; ++d)
    {
        sum_vx[d] = 0.0;
        for(uint32_t f = 0; f < nr_feature; ++f)
        {
            uint64_t const j = J[f];
            if(j >= range_sum)
                continue;
            sum_vx[d] += *(V + j * valign + d);
        }
        sum_vx_square[d] = sum_vx[d] * sum_vx[d];
    }
    float * const S = sum_vx.data();

    if(do_update)
    {
        //update w0
        float const gradient = kappa + l0 * model.w0;
        model.w0 -= static_cast<float>(eta0 * gradient / sqrt(model.w0_sg));
        model.w0_sg += (gradient * gradient);

        //update first order and diagonal parameters
        for(uint32_t f = 0; f < nr_feature; ++f)
        {
            uint64_t const j = J[f];
            if(j >= range_sum) //ignore if not present in training set
                continue;
            float * const w = W + j * NODE_SIZE;
            float * const shg = w + 1;
            float const gradient = kappa + l1 * *w;
            *w -= static_cast<float>(eta1 * gradient / sqrt(*shg) ); 
            *shg += (gradient * gradient); //update history gradient sum
        }

        //update second order parameters
        __m128 const xMMkappa = _mm_set1_ps(kappa * prob.norm);
        __m128 const xMMeta = _mm_set1_ps(eta2);
        __m128 const xMMlambda = _mm_set1_ps(l2);
        for(uint32_t f = 0; f < nr_feature; ++f)
        {
            uint64_t const j = J[f];
            if(j >= range_sum)
                continue;

            float * const v = V + j * valign;
            float * const vg = v + nr_factor;
            
            for(uint32_t d = 0; d < nr_factor; d+=4)
            {
                __m128 xMMv = _mm_load_ps(v + d);
                __m128 xMMs = _mm_load_ps(S + d); 
                __m128 xMMvg = _mm_load_ps(vg + d);
                __m128 xMMl = _mm_set_ps(*(L + (d+3)*NODE_SIZE),
                                         *(L + (d+2)*NODE_SIZE),
                                         *(L + (d+1)*NODE_SIZE),
                                         *(L + (d+0)*NODE_SIZE));
                __m128 xMMg = _mm_add_ps(
                              _mm_mul_ps(xMMkappa, _mm_mul_ps(xMMs, xMMl)),
                              _mm_mul_ps(xMMlambda, xMMv));
                xMMv = _mm_sub_ps(xMMv, _mm_mul_ps(xMMeta,
                       _mm_mul_ps(_mm_rsqrt_ps(xMMvg), xMMg)));
                xMMvg = _mm_add_ps(xMMvg, _mm_mul_ps(xMMg, xMMg));

                _mm_store_ps(v+d, xMMv);
                _mm_store_ps(vg+d, xMMvg);    
            }
        }

        //update lambda parameters
        for(uint32_t d = 0; d < nr_factor; ++d)
        {
            float * const l = L + d * NODE_SIZE; 
            float * const shg = l + 1;
            float const gradient = kappa * sum_vx_square[d] / 2 + l3 * *l;
            *l -= static_cast<float>(eta3 * gradient / sqrt(*shg) );
            *shg += (gradient * gradient);
        }
    }
    else
    {
        predict += model.w0;

        //compute first order value
        for(uint32_t f = 0; f < nr_feature; ++f)
        {
            uint64_t const j = J[f];
            if(j >= range_sum)
                continue;
            predict += *(W + j * NODE_SIZE);
        }
        
        //compute second order value
        float second_order = 0.0;
        for(uint32_t d = 0; d < nr_factor; ++d)
        {
            second_order += *(L + d * NODE_SIZE) * sum_vx_square[d];
        }
        second_order *= prob.norm;
        predict += (second_order / 2);
    }

    if(do_update)
        return 0;
    return predict;
}


float predict(Problem const &prob, Model &model, 
    std::string const &output_path = std::string(""));
#endif // _COMMON_H_
