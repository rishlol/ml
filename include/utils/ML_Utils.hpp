#pragma once
#include "xtensor/containers/xarray.hpp"
#include "xtensor/views/xview.hpp"

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
inline xt::xarray<double> generate_feat_bias(xt::xarray<double> &features) {
    xt::xarray<double> fb = xt::ones<double>({ (size_t)features.shape().at(0), (size_t)features.shape().at(1) + 1 });
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
inline double R_Squared(const xt::xarray<double> &y_lab, const xt::xarray<double> &y) {
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

}