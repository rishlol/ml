#pragma once
#include "Model.hpp"
#include "utils/Dataset.hpp"
#include "xtensor/containers/xarray.hpp"

class Perceptron : public Model {
public:
    Perceptron(Dataset &, size_t);

    ~Perceptron() {
        delete_feat_bias();
        delete_y_label();
    }

    static double P_Loss(const model_arr &, const model_arr &);
    void train(size_t, double);
    model_arr output(model_arr);
    model_arr operator()(model_arr);
};