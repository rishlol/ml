#pragma once
#include <fstream>
#include <string>
#include "xtensor/containers/xarray.hpp"

typedef xt::xarray<double> data_array;

class Dataset {
private:
    bool good;
    data_array features;
    data_array labels;
public:
    Dataset(std::string);
    Dataset(std::string, bool);
    Dataset(const Dataset &) = default;

    inline data_array & get_features() { return features; }
    inline data_array & get_labels() { return labels; }

    inline bool isGood() { return good; }
};