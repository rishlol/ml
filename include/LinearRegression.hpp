#pragma once
#include "Model.hpp"
#include "Dataset.hpp"
#include "utils/ML_Utils.hpp"
#include "xtensor/containers/xarray.hpp"

typedef xt::xarray<double> reg_array;

class LinearRegression : public Model {
public:
    LinearRegression(Dataset &, bool, size_t);
    LinearRegression(Dataset &, bool);
    LinearRegression(Dataset &, size_t);
    LinearRegression(Dataset &);

    ~LinearRegression() {
        delete_feat_bias();
        delete_y_label();
    }

    inline double getYMean() const { return y_norm.mean; }
    inline double getYSTD() const { return y_norm.std; }

    static double MSE(const reg_array &, const reg_array &);
    static double SSE(const reg_array &, const reg_array &);
    void train(size_t, double);
    reg_array output_raw(reg_array);
    reg_array output(reg_array);
    reg_array operator()(reg_array);
};