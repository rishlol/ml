#pragma once
#include "Model.hpp"
#include "Dataset.hpp"
#include "utils/ML_Utils.hpp"
#include "xtensor/containers/xarray.hpp"

typedef xt::xarray<double> svm_array;

class SupportVectorMachine : public Model {
public:
    SupportVectorMachine(Dataset &, size_t);

    ~SupportVectorMachine() {
        delete_feat_bias();
        delete_y_label();
    }

    static double Hinge(const svm_array &, const svm_array &);
    void train(size_t, double);
    svm_array output(svm_array);
    svm_array operator()(svm_array);
};