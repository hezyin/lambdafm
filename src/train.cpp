#include <iostream>
#include <algorithm>
#include <stdexcept>
#include <omp.h>
#include <stdio.h>
#include <cstring>
#include <unordered_map>
#include <map>

#include "common.h"
#include "timer.h"
#include "pairhash.h"

namespace {

int const kMaxLineSize = 1000000;

// original eta 0.1f lambda 0.00002f
struct Option
{
    Option() 
        : eta0(0.004f), eta1(0.01f), eta2(0.001f), eta3(0.001f), 
          l0(0.00002f), l1(0.00002f), l2(0.00002f), l3(0.00002f),
          iter(15), nr_factor(4), nr_threads(1), do_prediction(true), 
          model_existed(false), save_model(false) {}
    std::string Tr_path, Va_path, model_path, log_path;
    float eta0, eta1, eta2, eta3, l0, l1, l2, l3;
    uint32_t iter, nr_factor, nr_threads;
    bool do_prediction;
    bool model_existed;
    bool save_model;
};

std::string train_help()
{
    return std::string(
"usage: fm [<options>] <validation_path> <train_path>\n"
"\n"
"<validation_path>.out will be automatically generated at the end of training\n"
"\n"
"options:\n"
"-l <lambda>: set the regularization penalty\n"
"-k <factor>: set the number of latent factors, which must be a multiple of 4\n"
"-t <iteration>: set the number of iterations\n"
"-e0 <eta0>: set the learning rate for zero order parameters\n"
"-e1 <eta1>: set the learning rate for first order parameters\n"
"-e2 <eta2>: set the learning rate for second order parameters\n"
"-e3 <eta3>: set the learning rate for diagonal lambda parameters\n"
"-s <nr_threads>: set the number of threads\n"
"-q: if it is set, then there is no output file\n"
"-m <model>: use existed model instead of training\n"
"-g: <log_path> set the log file path\n"
"-v: save model after training\n");
}

Option parse_option(std::vector<std::string> const &args)
{
    uint32_t const argc = static_cast<uint32_t>(args.size());

    if(argc == 0)
        throw std::invalid_argument(train_help());

    Option opt; 

    uint32_t i = 0;
    for(; i < argc; ++i)
    {
        if(args[i].compare("-t") == 0)
        {
            if(i == argc-1)
                throw std::invalid_argument("invalid command\n");
            opt.iter = std::stoi(args[++i]);
        }
        else if(args[i].compare("-k") == 0)
        {
            if(i == argc-1)
                throw std::invalid_argument("invalid command\n");
            opt.nr_factor = std::stoi(args[++i]);
            /*if(opt.nr_factor%4 != 0)
                throw std::invalid_argument("k should be a multiple of 4\n");*/
        }
        else if(args[i].compare("-e0") == 0)
        {
            if(i == argc-1)
                throw std::invalid_argument("invalid command\n");
            opt.eta0 = std::stof(args[++i]);
        }
        else if(args[i].compare("-e1") == 0)
        {
            if(i == argc-1)
                throw std::invalid_argument("invalid command\n");
            opt.eta1 = std::stof(args[++i]);
        }
        else if(args[i].compare("-e2") == 0)
        {
            if(i == argc-1)
                throw std::invalid_argument("invalid command\n");
            opt.eta2 = std::stof(args[++i]);
        }
        else if(args[i].compare("-e3") == 0)
        {
            if(i == argc-1)
                throw std::invalid_argument("invalid command\n");
            opt.eta3 = std::stof(args[++i]);
        }
        else if(args[i].compare("-l0") == 0)
        {
            if(i == argc-1)
                throw std::invalid_argument("invalid command\n");
            opt.l0 = std::stof(args[++i]);
        }
        else if(args[i].compare("-l1") == 0)
        {
            if(i == argc-1)
                throw std::invalid_argument("invalid command\n");
            opt.l1 = std::stof(args[++i]);
        }
        else if(args[i].compare("-l2") == 0)
        {
            if(i == argc-1)
                throw std::invalid_argument("invalid command\n");
            opt.l2 = std::stof(args[++i]);
        }
        else if(args[i].compare("-l3") == 0)
        {
            if(i == argc-1)
                throw std::invalid_argument("invalid command\n");
            opt.l3 = std::stof(args[++i]);
        }
        else if(args[i].compare("-s") == 0)
        {
            if(i == argc-1)
                throw std::invalid_argument("invalid command\n");
            opt.nr_threads = std::stoi(args[++i]);
        }
        else if(args[i].compare("-q") == 0)
        {
            opt.do_prediction = false;
        }
        else if(args[i].compare("-m") == 0)
        {
            opt.model_existed = true;
            if(i == argc-1)
                throw std::invalid_argument("invalid command\n");
            opt.model_path = args[++i];
        }
        else if(args[i].compare("-g") == 0)
        {
            if(i == argc-1)
                throw std::invalid_argument("invalid command\n");
            opt.log_path = args[++i];
        }
        else if(args[i].compare("-v") == 0)
        {
            opt.save_model = true;
        }
        else
        {
            break;
        }
    }

    if(i >= argc-1)
        throw std::invalid_argument("training or test set not specified\n");

    opt.Va_path = args[i++];
    opt.Tr_path = args[i++];

    return opt;
}

void init_model(Model &model)
{
    uint64_t const range_sum = model.range_sum;
    uint32_t const nr_factor = model.nr_factor;
    float const coef = 
        static_cast<float>(0.5/sqrt(static_cast<double>(nr_factor)));

    model.w0 = static_cast<float>(0.01 * drand48());

    float * w = model.W.data();
    for(uint64_t i = 0; i < range_sum; ++i)
    {
        *(w++) = static_cast<float>(0.01 * drand48());
        *(w++) = 1;
    }
  
    float * v = model.V.data();
    for(uint64_t i = 0; i < range_sum; ++i)
    {
        for(uint32_t d = 0; d < nr_factor; ++d, ++v)
            *v = coef*static_cast<float>(drand48());
        for(uint32_t d = 0; d < nr_factor; ++d, ++v)
            *v = 1;
    }

    float * l = model.L.data();
    for(uint32_t d = 0; d < nr_factor; ++d)
    {
        *(l++) = static_cast<float>(0.5 * drand48());
        *(l++) = 1;
    }
}

/*

void save_model(Model &model, std::unordered_map<std::pair<uint32_t, uint32_t>, uint64_t, pairhash> &fviMap, std::string output_path)
{
    uint64_t range_sum = model.range_sum;
    uint32_t nr_factor = model.nr_factor;
    uint32_t align = model.nr_factor * kW_NODE_SIZE;

    float * const fW = model.W.data();
    float * const sW = model.W.data() + range_sum * kW_NODE_SIZE;

    FILE* modelfile = fopen(output_path.c_str(), "w+");
    fprintf(modelfile, "range_sum %llu\n", range_sum);
    fprintf(modelfile, "nr_factor %d\n", nr_factor);

    std::map<std::pair<uint32_t, uint32_t>, uint64_t>::iterator it;

    for(it = fviMap.begin(); it != fviMap.end(); ++it)
    {
        fprintf(modelfile, "%d %d %llu\n", (it->first).first, (it->first).second, it->second);
    }

    for(uint64_t i = 0; i < range_sum; ++i)
    {
        fprintf(modelfile, "%llu %.5f\n", i+1, *(fW + i * kW_NODE_SIZE));
    }

    for(uint64_t i = 0; i < range_sum; ++i)
    {
        fprintf(modelfile, "# Item %llu\n", i+1);
        for(uint32_t k = 0; k < nr_factor; k++)
        {
            fprintf(modelfile, "%.5f ", *(sW + i * align + k));
        }
        fprintf(modelfile, "\n\n");
    }
    fclose(modelfile);
}

*/

/*
void read_model(std::string const &path, Model& model)
{
    if(path.empty())
        return;

    FILE* f = fopen(path.c_str(), "r");
    char line[kMaxLineSize];
    int model_parameter;

    fgets(line, kMaxLineSize, f);
    strtok(line, " \t");
    model_parameter = std::stoi(strtok(NULL, " \t"));
    model.feature = model_parameter;

    fgets(line, kMaxLineSize, f);
    strtok(line, " \t");
    model_parameter = std::stoi(strtok(NULL, " \t"));
    model.feature_range = model_parameter;
    
    fgets(line, kMaxLineSize, f);
    strtok(line, " \t");
    model_parameter = std::stoi(strtok(NULL, " \t"));
    model.factor = model_parameter;

    model.W.resize(model.feature * model.feature_range * model.factor * kW_NODE_SIZE);
    init_model(model);

    float* w = model.W.data();

    for(uint32_t i = 0; i < model.feature; ++i)
    {
        fgets(line, kMaxLineSize, f); // white line
        for(uint32_t j = 0; j < model.feature_range; ++j)
        {
            fgets(line, kMaxLineSize, f); // content line
            float p = std::stof(strtok(line, " "));
            for(uint32_t k = 0; k < model.factor - 1; ++k, ++w)
            {
                *w = p;
                p = std::stof(strtok(NULL, " "));
            }
            *w = p;
            w += model.factor + 1;
        }
        fgets(line, kMaxLineSize, f); // white line
    }
}

*/

bool first_element_comparator(std::pair<float, uint32_t> const &p1, std::pair<float, uint32_t> const &p2)
{
    return p1.first < p2.first;
}

float auc(Problem const &prob, std::string predict_path)
{
    std::map<float, float> fp_tp;

    uint32_t n = prob.nr_instance;
    std::vector<std::pair<float, uint32_t>> pbs(n);

    FILE* f = fopen(predict_path.c_str(), "r");
    char line[kMaxLineSize];

    //initialize pbs
    for(uint32_t i = 0; i < n; i++)
    {
        pbs[i] = std::make_pair(std::stof(fgets(line, kMaxLineSize, f)), i);
    }
    fclose(f);

    std::sort(pbs.begin(), pbs.end(), first_element_comparator);

    //initialize parameters
    float fpr, tpr;
    uint32_t tp, tn, fp, fn;
    tp = tn = fp = fn = 0;
    for(uint32_t i = 0; i < n; i++)
    {
        if(prob.Y[pbs[i].second] > 0) { tp++; } else { fp++; }
    }
    fpr = (float) fp / (float) (fp + tn);
    tpr = (float) tp / (float) (tp + fn);
    fp_tp[fpr] = tpr;
    
    //compute tpr and fpr all other points 
    for(uint32_t i = 1; i < n; i++)
    {
        //use i as threshold point, for all points before i, prediction value is -1
        if(prob.Y[pbs[i].second] > 0)
        {
            tp--; fn++;
        }
        else
        {
            fp--; tn++;
        }
        fpr = (float) fp / (float) (fp + tn);
        tpr = (float) tp / (float) (tp + fn);
        fp_tp[fpr] = tpr;
    }
    std::map<float, float>::iterator iter;
    /*for(iter = fp_tp.begin(); iter != fp_tp.end(); iter++)
    {
        printf("(%f %f)\n", iter->first, iter->second);
    }*/

    float area = 0.0, x, y;
    for(iter = fp_tp.begin(); iter != fp_tp.end(); iter++)
    {
        if(iter != fp_tp.begin())
            area += (iter->second + y) * (iter->first - x) / 2;
        x = iter->first;
        y = iter->second;
    }
    return area; 
}

void train(Problem const &Tr, Problem const &Va, Model &model, Option const &opt)
{
    std::vector<uint32_t> order(Tr.Y.size());
    for(uint32_t i = 0; i < Tr.Y.size(); ++i)
        order[i] = i;

    Timer timer;
    FILE * logfile = nullptr;
    if(!opt.log_path.empty())
    {
        logfile = fopen(opt.log_path.c_str(), "w+");
        fprintf(logfile, "iter     time    tr_loss    va_loss\n");
        fflush(logfile);
    }
    
    printf("iter     time    tr_loss    va_loss\n");
    for(uint32_t iter = 0; iter < opt.iter; ++iter)
    {
        timer.tic();

        double Tr_loss = 0;
        std::random_shuffle(order.begin(), order.end()); // stochastic gradient descent
        #pragma omp parallel for schedule(static)
        for(uint32_t i_ = 0; i_ < order.size(); ++i_)
        {
            uint32_t const i = order[i_];

            float const y = Tr.Y[i];

            float const t = wTx_sse(Tr, model, i);
            
            float const expnyt = static_cast<float>(exp(-y*t));

            Tr_loss += log(1+expnyt);

            float const kappa = -y*expnyt/(1+expnyt);

            wTx_sse(Tr, model, i, kappa, opt.eta0, opt.eta1, opt.eta2, opt.eta3, opt.l0, opt.l1, opt.l2, opt.l3, true);
        }

        //Tr_loss /= static_cast<double>(Tr.Y.size());

        Tr_loss = predict(Tr, model);

        double const Va_loss = predict(Va, model);

        if(!opt.log_path.empty())
        {
            fprintf(logfile, "%4d %8.1f %10.5f %10.5f\n", iter+1, timer.toc(), Tr_loss, Va_loss);
            fflush(logfile);
        }

        printf("%4d %8.1f %10.5f %10.5f\n", 
              iter+1, timer.toc(), Tr_loss, Va_loss);
        fflush(stdout);
    }
}

} //unnamed namespace

int main(int const argc, char const * const * const argv)
{
    Option opt;
    try
    {
        opt = parse_option(argv_to_args(argc, argv));
    }
    catch(std::invalid_argument const &e)
    {
        std::cout << e.what();
        return EXIT_FAILURE;
    }
    
    std::unordered_map<std::pair<uint32_t, uint32_t>, uint64_t, pairhash> fviMap; // key: pair of feature number and its value, value: corresponding index in weight vector 

    std::cout << "reading tr dataset...\n" << std::flush;
    Problem const Tr = read_tr_problem(opt.Tr_path, fviMap);
    std::cout << "reading va dataset...\n" << std::flush;
    Problem const Va = read_va_problem(opt.Va_path, fviMap, Tr.range_sum, Tr.nr_feature);

    /*     
    for(uint32_t i = 0; i < 10; i++)
    {
        std::cout << Tr.Y[i] << " ";
        for(uint32_t j = 0; j < Tr.nr_feature; j++)
        {
            std::cout << Tr.J[i * Tr.nr_feature + j] << " ";
        }
        std::cout << "\n";
    }
    std::cout << "\n";
    for(uint32_t i = 0; i < 10; i++)
    {
        std::cout << Va.Y[i] << " ";
        for(uint32_t j = 0; j < Va.nr_feature; j++)
        {
            std::cout << Va.J[i * Va.nr_feature + j] << " ";
        }
        std::cout << "\n";
    }*/

    Model model(Tr.range_sum, opt.nr_factor); 

    if(opt.model_existed)
    {
        std::cout << "reading existed model..." << std::flush;
        //read_model(opt.model_path, model);
        std::cout << "done\n" << std::flush;
    }
    else
    {
        std::cout << "initializing model..." << std::flush;
        init_model(model);
        std::cout << "done\n" << std::flush;

        /*
        uint32_t const align = model.nr_factor * kW_NODE_SIZE;
        for(uint64_t i = 0; i < model.range_sum; i++)
        {
            for(uint32_t j = 0; j < align; j++)
            {
                std::cout << model.W[i * align + j] << " ";
            }
            std::cout << "\n";
        }*/


	      omp_set_num_threads(static_cast<int>(opt.nr_threads));

        train(Tr, Va, model, opt);

        /*
        std::cout << "saving model..." << std::flush;
        save_model(model, fviMap, "model.txt");
        std::cout << "done\n" << std::flush;*/

	      omp_set_num_threads(1);
    }
    if(opt.do_prediction)
    {
        double Va_loss = predict(Va, model, opt.Va_path+".out");
        printf("%f\n", Va_loss);
        double auc_score = auc(Va, opt.Va_path+".out");
        printf("%f\n", auc_score);
    }

    return EXIT_SUCCESS;
}
