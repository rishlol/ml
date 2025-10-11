#pragma once
#include "Dataset.hpp"
#include "utils/ML_Utils.hpp"
#include <tuple>
#include <unordered_map>
#include "xtensor/containers/xarray.hpp"

typedef xt::xarray<double> svm_array;

class SupportVectorMachine {
private:
    svm_array *y_label = nullptr;
    svm_array *feat_bias = nullptr;         // (n, d + 1)
    svm_array weights;                      // (d + 1, n)
    std::tuple<size_t, size_t> fb_shape;

    std::unordered_map<size_t, ZScaleNormalizer> feat_norms;

    inline void delete_feat_bias() {
        delete feat_bias;
        feat_bias = nullptr;
    }
    inline void delete_y_label() {
        delete y_label;
        y_label = nullptr;
    }
public:
    SupportVectorMachine();

    ~SupportVectorMachine() {
        delete_feat_bias();
        delete_y_label();
    }

    inline svm_array & getLabels() { return *y_label; }
    inline svm_array & getFeatures() { return *feat_bias; }
    inline svm_array & getWeights() { return weights; }
    inline std::tuple<size_t, size_t> getShape() const { return fb_shape; }
};