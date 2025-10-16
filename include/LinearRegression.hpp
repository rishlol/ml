#pragma once
#include "Model.hpp"
#include "utils/Dataset.hpp"
#include "xtensor/containers/xarray.hpp"

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

    static double MSE(const model_arr &, const model_arr &);
    static double SSE(const model_arr &, const model_arr &);
    void train(size_t, double);
    model_arr output_raw(model_arr);
    model_arr output(model_arr);
    model_arr operator()(model_arr);
};