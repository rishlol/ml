#pragma once
#include "Model.hpp"
#include "utils/Dataset.hpp"
#include "xtensor/containers/xarray.hpp"

class SupportVectorMachine : public Model {
public:
    SupportVectorMachine(Dataset &, size_t);

    ~SupportVectorMachine() {
        delete_feat_bias();
        delete_y_label();
    }

    static double Hinge(const model_arr &, const model_arr &);
    void train(size_t, double);
    model_arr output(model_arr);
    model_arr operator()(model_arr);
};