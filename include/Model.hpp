#pragma once
#include "Dataset.hpp"
#include "utils/ML_Utils.hpp"
#include <unordered_map>
#include "xtensor/containers/xarray.hpp"
#include "xtensor/views/xview.hpp"

typedef xt::xarray<double> model_arr;
typedef ML::ZScaleNormalizer ZScaleNormalizer;

class Model {
protected:
    model_arr *y_label = nullptr;
    model_arr *feat_bias = nullptr;         // (n, d + 1)
    model_arr weights;                      // (d + 1, 1)
    std::tuple<size_t, size_t> fb_shape;

    bool normalizeLabels = false;
    ZScaleNormalizer y_norm;
    std::unordered_map<size_t, ZScaleNormalizer> feat_norms;

    Model(Dataset &d, size_t start_norm) {
        y_label = new model_arr(d.get_labels());

        // Create feature vector with bias column (first column)
        feat_bias = new model_arr(ML::generate_feat_bias(d.get_features()));

        // Store feature matrix shape and normalize
        fb_shape = std::make_tuple(feat_bias->shape().at(0), feat_bias->shape().at(1));
        for(size_t c = start_norm + 1; c < std::get<1>(fb_shape); c += 1) {
            feat_norms.insert({ c, ZScaleNormalizer(
                xt::mean(xt::col(*feat_bias, c))(),
                xt::stddev(xt::col(*feat_bias, c))()
            )});
            ZScaleNormalizer c_norm = feat_norms.at(c);
            xt::col(*feat_bias, c) = (xt::col(*feat_bias, c) - c_norm.mean) / c_norm.std;
        }

        // Initialize weights
        weights = xt::zeros<double>({ std::get<1>(fb_shape), (size_t)1 });
    }
    Model(Dataset &d, bool norm_lab, size_t start_norm) {
        // Get labels and normalize if needed
        normalizeLabels = norm_lab;
        y_label = new model_arr(d.get_labels());
        if(normalizeLabels) {
            y_norm.mean = xt::mean(*y_label)();
            y_norm.std = xt::stddev(*y_label)();
            *y_label = (*y_label - y_norm.mean) / y_norm.std;
        }

        // Create feature vector with bias column (first column)
        feat_bias = new model_arr(ML::generate_feat_bias(d.get_features()));

        // Store feature matrix shape and normalize
        fb_shape = std::make_tuple(feat_bias->shape().at(0), feat_bias->shape().at(1));
        for(size_t c = start_norm + 1; c < std::get<1>(fb_shape); c += 1) {
            feat_norms.insert({ c, ZScaleNormalizer(
                xt::mean(xt::col(*feat_bias, c))(),
                xt::stddev(xt::col(*feat_bias, c))()
            )});
            ZScaleNormalizer c_norm = feat_norms.at(c);
            xt::col(*feat_bias, c) = (xt::col(*feat_bias, c) - c_norm.mean) / c_norm.std;
        }
        
        // Initialize weights
        weights = xt::zeros<double>({ (size_t)std::get<1>(fb_shape), (size_t)1 });
    }

    inline void delete_feat_bias() {
        delete feat_bias;
        feat_bias = nullptr;
    }
    inline void delete_y_label() {
        delete y_label;
        y_label = nullptr;
    }

    inline model_arr & getLabels() { return *y_label; }
    inline model_arr & getFeatures() { return *feat_bias; }
    inline model_arr & getWeights() { return weights; }
    inline std::tuple<size_t, size_t> getShape() const { return fb_shape; }
    inline int getNumFeatures() const { return std::get<1>(fb_shape) - 1; }
};