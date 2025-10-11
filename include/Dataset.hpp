#pragma once
#include <fstream>
#include <string>
#include "xtensor/containers/xarray.hpp"

class Dataset {
private:
    bool good;
    xt::xarray<double> features;
    xt::xarray<double> labels;
public:
    Dataset(std::string);
    Dataset(std::string, bool);
    Dataset(const Dataset &) = default;

    inline xt::xarray<double> & get_features() { return features; }
    inline xt::xarray<double> & get_labels() { return labels; }

    inline bool isGood() { return good; }
};