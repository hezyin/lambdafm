#include <stdexcept>
#include <cstring>
#include <omp.h>
#include <iostream>
#include <map>

#include "common.h"

namespace {

int const kMaxLineSize = 1000000;

uint32_t get_nr_line(std::string const path)
{
    FILE *f = open_c_file(path.c_str(), "r");
    char line[kMaxLineSize];

    uint32_t nr_line = 0;
    while(fgets(line, kMaxLineSize, f) != nullptr)
        ++nr_line;

    fclose(f);

    return nr_line;
}

} //unamed namespace


Problem read_tr_problem(std::string const tr_path, std::map<std::pair<uint32_t, uint32_t>, uint64_t> &fviMap)
{
    if(tr_path.empty())
        return Problem(0, 0, 0);

    FILE *f;
    char *record, *y_char;
    char line[kMaxLineSize], feature[100], value[100];
    uint64_t range_sum, w_ind;
    uint32_t nr_line, nr_feature, cf, cv;
    std::pair<uint32_t, uint32_t> temp_pair;
    std::map<std::pair<uint32_t, uint32_t>, uint64_t>::iterator it;

    f = open_c_file(tr_path.c_str(), "r");
    w_ind = nr_line = nr_feature = 0;
    while(fgets(line, kMaxLineSize, f) != nullptr)
    {
        nr_line++;
        size_t ln = strlen(line) - 1;
        if(line[ln] == '\n')
            line[ln] = '\0';

        strtok(line, " \t");
        record = strtok(nullptr, " \t");
        do {
            sscanf(record, "%[^:]:%s", feature, value);
            record = strtok(nullptr, " \t");
            cf = static_cast<uint32_t>(std::stol(feature));
            if(cf > nr_feature)
                nr_feature = cf;
            try
            {
                cv = static_cast<uint32_t>(std::stol(value));
            }
            catch(std::invalid_argument const &e)
            {
                continue;
            }
            temp_pair = std::make_pair(cf, cv);
            it = fviMap.find(temp_pair);
            if (it == fviMap.end())
            {
                fviMap[temp_pair] = w_ind;
                w_ind++;
            }
        } while(record != nullptr);
    }
    fclose(f);

    range_sum = fviMap.size();
    std::cout << "nr_line: " << nr_line << "\n";
    std::cout << "nr_feature: " << nr_feature << "\n";
    std::cout << "range_sum: " << range_sum << "\n";
    Problem prob(nr_line, nr_feature, range_sum);

    f = open_c_file(tr_path.c_str(), "r");
    for(uint32_t i = 0; fgets(line, kMaxLineSize, f) != nullptr; ++i)
    {
        size_t ln = strlen(line) - 1;
        if(line[ln] == '\n')
            line[ln] = '\0';

        y_char = strtok(line, " \t");
        prob.Y[i] = (atoi(y_char)>0)? 1.0f : -1.0f;
        record = strtok(nullptr, " \t");
        do {
            sscanf(record, "%[^:]:%s", feature, value);
            record = strtok(nullptr, " \t");
            cf = static_cast<uint32_t>(std::stol(feature));
            try
            {
                cv = static_cast<uint32_t>(std::stol(value));
            }
            catch(std::invalid_argument const &e)
            {
                continue;
            }
            temp_pair = std::make_pair(cf, cv);
            it = fviMap.find(temp_pair);
            if(it == fviMap.end())
            {
                fviMap[temp_pair] = w_ind;
                prob.J[i * nr_feature + cf - 1] = w_ind;
                w_ind++;
            }
            else
            {
                prob.J[i * nr_feature + cf - 1]  = it->second;
            }   
        } while(record != nullptr);
    }
    fclose(f);
    return prob;
}

Problem read_va_problem(std::string const va_path, std::map<std::pair<uint32_t, uint32_t>, uint64_t> &fviMap, uint64_t const range_sum, uint32_t const nr_feature)
{
    if(va_path.empty())
        return Problem(0, 0, 0);

    uint32_t nr_line = get_nr_line(va_path);
    std::cout << "nr_line: " << nr_line << "\n";
    Problem prob(nr_line, nr_feature, range_sum);

    FILE *f;
    char *record, *y_char;
    char line[kMaxLineSize], feature[100], value[100];
    uint32_t cf, cv;
    std::pair<uint32_t, uint32_t> temp_pair;
    std::map<std::pair<uint32_t, uint32_t>, uint64_t>::iterator it;

    f = open_c_file(va_path.c_str(), "r");
    for(uint32_t i = 0; fgets(line, kMaxLineSize, f) != nullptr; ++i)
    {
        size_t ln = strlen(line) - 1;
        if(line[ln] == '\n')
            line[ln] = '\0';

        y_char = strtok(line, " \t");
        prob.Y[i] = (atoi(y_char)>0)? 1.0f : -1.0f;
        record = strtok(nullptr, " \t");
        do {
            sscanf(record, "%[^:]:%s", feature, value);
            record = strtok(nullptr, " \t");
            cf = static_cast<uint32_t>(std::stol(feature));
            try
            {
                cv = static_cast<uint32_t>(std::stol(value));
            }
            catch(std::invalid_argument const &e)
            {
                continue;
            }
            temp_pair = std::make_pair(cf, cv);
            it = fviMap.find(temp_pair);
            if(it != fviMap.end())
            {
                prob.J[i * nr_feature + cf - 1]  = it->second;
            }   
        } while(record != nullptr);
    }
    fclose(f);
    return prob;
}

FILE *open_c_file(std::string const &path, std::string const &mode)
{
    FILE *f = fopen(path.c_str(), mode.c_str());
    if(!f)
        throw std::runtime_error(std::string("cannot open ")+path);
    return f;
}

std::vector<std::string> 
argv_to_args(int const argc, char const * const * const argv)
{
    std::vector<std::string> args;
    for(int i = 1; i < argc; ++i)
        args.emplace_back(argv[i]);
    return args;
}

float predict(Problem const &prob, Model &model, 
    std::string const &output_path)
{
    FILE *f = nullptr;
    if(!output_path.empty())
        f = open_c_file(output_path, "w");

    double loss = 0;
    for(uint32_t i = 0; i < prob.Y.size(); ++i)
    {
        float const y = prob.Y[i];

        float const t = wTx(prob, model, i);
        
        float const prob = 1/(1+static_cast<float>(exp(-t)));

        float const expnyt = static_cast<float>(exp(-y*t));

        loss += log(1+expnyt);

        if(f)
            fprintf(f, "%lf\n", prob);
    }

    if(f)
        fclose(f);

    return static_cast<float>(loss/static_cast<double>(prob.Y.size()));
}
