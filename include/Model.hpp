#pragma once
#include "utils/Dataset.hpp"
#include <unordered_map>
#include "xtensor/containers/xarray.hpp"
#include "xtensor/views/xview.hpp"
#include "xtensor/core/xoperation.hpp"

typedef xt::xarray<double> model_arr;

namespace ML {

struct ZScaleNormalizer {
    double mean;
    double std;

    /**
     * @brief Initialize ZScaleNormalizer struct.
     * 
     * @param m Mean of normalizer.
     * @param s Standard deviation of normalizer.
     */
    ZScaleNormalizer(double m, double s) {
        mean = m;
        std = s;
    }
    ZScaleNormalizer() = default;
};

template<typename T>
inline bool xarray_same_shape(xt::xarray<T> a1, xt::xarray<T> a2) {
    bool differentShape = a1.shape().size() != a2.shape().size();
    if(differentShape) { std::cerr << "Different number of dimensions!\n"; return false; }
    for(int i = 0; i < a1.shape().size(); i += 1) {
        if(a1.shape().at(i) != a2.shape().at(i)) { std::cerr << "Different number of elements in "
                                                                << i << " dimension!\n"; return false; }
    }
    return true;
}

/**
 * @brief Adds bias column to feature array.
 * 
 * Takes xarray as input and adds a bias column to the beginning.
 * Bias column filled with ones.
 * 
 * @param features Feature xarray.
 * @return New xarray with bias column before features.
 */
inline model_arr generate_feat_bias(model_arr &features) {
    model_arr fb = xt::ones<double>({ (size_t)features.shape().at(0), (size_t)features.shape().at(1) + 1 });
    for(size_t c = 0; c < features.shape().at(1); c += 1)
        xt::col(fb, c + 1) = xt::col(features, c);
    return std::move(fb);
}

/**
 * @brief Calculates R^2 value.
 * 
 * Takes model outputs and labels and calculates R^2.
 * This can be used to evaluate model performance.
 * A R^2 value close to 1 indicates good performance.
 * 
 * @param y_lab xarray of expected/desired model output.
 * @param y xarray of model outputs.
 * @return R^2 value (double).
 */
inline double R_Squared(const model_arr &y_lab, const model_arr &y) {
    // Make sure input shapes are the same
    if(!xarray_same_shape(y_lab, y)) {
        std::cerr << "Cannot calculate loss! y_label and y_train have different dimensions!\n";
        return std::numeric_limits<double>::quiet_NaN();
    }

    // R^2
    double ss_tot = xt::sum(xt::square(y_lab - xt::mean(y_lab)()))();
    if(ss_tot == 0.0)
        return 1;
    double ss_res = xt::sum(xt::square(y_lab - y))();
    return 1 - (ss_res / ss_tot);
}

/**
 * @brief Calculates accuracy value for classification models.
 * 
 * Takes model outputs and labels and calculates accuracy (TP + TN) / (TP + FP + TN + FN).
 * This can be used to evaluate model performance for classification models that output { -1, 1 }.
 * A accuracy value close to 1 indicates good performance.
 * 
 * @param y_lab xarray of expected/desired model output.
 * @param y xarray of model outputs.
 * @return accuracy value (double).
 */
inline double accuracy(const model_arr &y_lab, const model_arr &y) {
    model_arr correct = xt::equal(y_lab, y);
    return xt::mean(correct)();
}

}

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

    /**
     * @brief Create Model from Dataset.
     * 
     * Prepares Model object for training.
     * Dataset `d` already stores feature and labels.
     * Constructor takes features and labels from Dataset `d`.
     * Also normalizes labels, adds bias column to feature matrix, initializes weights, stores label normalization information.
     * The first `start_norm - 1` columns of the Dataset feature matrix will not be normalized.
     * 
     * @param d Dataset object.
     * @param start_norm size_t: column index from which normalization will be applied.
     */
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

    /**
     * @brief Create Model from Dataset.
     * 
     * Prepares Model object for training.
     * Dataset `d` already stores feature and labels.
     * Constructor takes features and labels from Dataset `d`.
     * Also normalizes labels, adds bias column to feature matrix, initializes weights, stores label normalization information.
     * The first `start_norm - 1` columns of the Dataset feature matrix will not be normalized.
     * 
     * @param d Dataset object.
     * @param norm_lab bool: determines whether labels will be normalized.
     * @param start_norm size_t: column index from which normalization will be applied.
     */
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