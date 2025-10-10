#pragma once
#include <tuple>
#include <vector>
#include "xtensor/containers/xarray.hpp"

class LinearRegression {
private:
    xt::xarray<double> y_label;
    xt::xarray<double> feat_bias;           // (n, d + 1)
    xt::xarray<double> weights;             // (d + 1, 1)
    std::tuple<size_t, size_t> fb_shape;
protected:
    template<typename T>
    static bool xarray_same_shape(xt::xarray<T> a1, xt::xarray<T> a2) {
        bool differentShape = a1.shape().size() != a2.shape().size();
        if(differentShape) { std::cerr << "Different number of dimensions!\n"; return false; }
        for(int i = 0; i < a1.shape().size(); i += 1) {
            if(a1.shape().at(i) != a2.shape().at(i)) { std::cerr << "Different number of elements in "
                                                                 << i << " dimension!\n"; return false; }
        }
        return true;
    }
public:
    LinearRegression(const xt::xarray<double> &);

    inline xt::xarray<double> & getLabels() { return y_label; }
    inline xt::xarray<double> & getFeatures() { return feat_bias; }
    inline xt::xarray<double> & getWeights() { return weights; }
    inline std::tuple<size_t, size_t> getShape() const { return fb_shape; }
    inline int getNumFeatures() const { return std::get<1>(fb_shape) - 1; }

    template<typename E> // Can take xarray or view
    inline void z_scale_normalize(E &&m) { m = (m - xt::mean(m)()) / xt::stddev(m)(); }
    static double MSE_loss(const xt::xarray<double> &, const xt::xarray<double> &);
    void train(size_t, double);
    double eval(xt::xarray<double>);
};