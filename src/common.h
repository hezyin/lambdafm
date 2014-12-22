#pragma GCC diagnostic ignored "-Wunused-result"

#ifndef _COMMON_H_
#define _COMMON_H_

#define flag { printf("\nLINE: %d\n", __LINE__); fflush(stdout); }

#include <vector>
#include <cmath>

struct Problem
{
    Problem(uint32_t const nr_instance, uint32_t const nr_feature, uint64_t const range_sum)
          :nr_instance(nr_instance), nr_feature(nr_feature), range_sum(range_sum),  
          J(static_cast<uint64_t>(nr_instance) * nr_feature, range_sum), 
          Y(nr_instance){}
    uint32_t nr_instance, nr_feature;
    uint64_t range_sum;
    std::vector<uint64_t> J; 
    std::vector<float> Y; 
};

Problem read_tr_problem(std::string const tr_path, std::map<std::pair<uint32_t, uint32_t>, uint64_t> &fviMap);

Problem read_va_problem(std::string const va_path, std::map<std::pair<uint32_t, uint32_t>, uint64_t> &fviMap, uint64_t const range_sum, uint32_t const nr_feature);

uint32_t const NODE_SIZE = 2;

struct Model
{
    Model(uint64_t const range_sum, uint32_t const nr_factor) 
        : range_sum(range_sum), nr_factor(nr_factor),
        W(range_sum * NODE_SIZE),
        V(range_sum * nr_factor * NODE_SIZE),
        L(nr_factor * NODE_SIZE),
        D(range_sum * NODE_SIZE) {} // W is initialized by init_model() 
    uint64_t range_sum;
    uint32_t nr_factor;
    std::vector<float> W;
    std::vector<float> V;
    std::vector<float> L;
    std::vector<float> D;
};

FILE *open_c_file(std::string const &path, std::string const &mode);

std::vector<std::string> 
argv_to_args(int const argc, char const * const * const argv);

inline float wTx(Problem const &prob, Model &model, uint32_t const i, 
    float const kappa=0, float const eta=0, float const lambda=0, 
    bool const do_update=false)
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
    float * const D = model.D.data();

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
        //update first order and diagonal parameters
        for(uint32_t f = 0; f < nr_feature; ++f)
        {
            uint64_t const j = J[f];
            if(j >= range_sum) //ignore if not present in training set
                continue;
            float * const w = W + j * NODE_SIZE;
            float * const wshg = w + 1;
            float const wgradient = kappa + lambda * *w;
            *w -= static_cast<float>(eta * wgradient / sqrt(*wshg) ); 
            *wshg += (wgradient * wgradient); //update history gradient sum

            float * const d = D + j * NODE_SIZE;
            float * const dshg = d + 1;
            float const dgradient = kappa + lambda * *d;
            *d -= static_cast<float>(eta * dgradient / sqrt(*dshg) );
            *dshg += (dgradient * dgradient);
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
                float const gradient = kappa * (sum_vx[d] - *v) + lambda * *v; 
                *v -= static_cast<float> (eta * gradient / sqrt(*shg)); 
                *shg += (gradient * gradient); //update sum history gradient
            }
        }

        //update lambda parameters
        for(uint32_t d = 0; d < nr_factor; ++d)
        {
            float * const l = L + d * NODE_SIZE; 
            float * const shg = l + 1;
            float const gradient = kappa * sum_vx_square[d] / 2 + lambda * *l;
            *l -= static_cast<float>(eta * gradient / sqrt(*shg) );
            *shg += (gradient * gradient);
        }
    }
    else
    {
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
        for(uint32_t f = 0; f < nr_feature; ++f)
        {
            uint64_t const j = J[f];
            if(j >= range_sum)
                continue;
            second_order -= *(D + j * NODE_SIZE);
        }
        predict += (second_order / 2);
    }

    if(do_update)
        return 0;
    return predict;
}

float predict(Problem const &prob, Model &model, 
    std::string const &output_path = std::string(""));
#endif // _COMMON_H_
