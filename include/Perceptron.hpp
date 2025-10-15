#pragma once
#include "Model.hpp"
#include "Dataset.hpp"
#include "xtensor/containers/xarray.hpp"

typedef xt::xarray<double> perc_arr;

class Perceptron : public Model {
public:
    Perceptron(Dataset &, size_t);

    ~Perceptron() {
        delete_feat_bias();
        delete_y_label();
    }

    static double P_Loss(const perc_arr &, const perc_arr &);
    void train(size_t, double);
    perc_arr output(perc_arr);
    perc_arr operator()(perc_arr);
};