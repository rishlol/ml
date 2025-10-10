#pragma once
#include <fstream>
#include <string>
#include "xtensor/containers/xarray.hpp"

class Dataset {
private:
    bool good;
    xt::xarray<double> csv;
public:
    Dataset(std::string, bool);
    Dataset(std::string);

    inline xt::xarray<double> & get_csv() { return csv; }
    inline bool isGood() { return good; }
};