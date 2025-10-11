#pragma once
#include "Dataset.hpp"
#include <tuple>
#include <vector>
#include <unordered_map>
#include "xtensor/containers/xarray.hpp"

typedef xt::xarray<double> reg_array;
struct ZScaleNormalizer {
    double mean;
    double std;
};

class LinearRegression {
private:
    reg_array y_label;
    reg_array feat_bias;           // (n, d + 1)
    reg_array weights;             // (d + 1, 1)
    std::tuple<size_t, size_t> fb_shape;

    bool normalizeLabels = false;
    ZScaleNormalizer y_norm;
    std::unordered_map<size_t, ZScaleNormalizer> feat_norms;
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
    LinearRegression(Dataset &, bool);
    LinearRegression(Dataset &);

    inline reg_array & getLabels() { return y_label; }
    inline reg_array & getFeatures() { return feat_bias; }
    inline reg_array & getWeights() { return weights; }
    inline std::tuple<size_t, size_t> getShape() const { return fb_shape; }
    inline int getNumFeatures() const { return std::get<1>(fb_shape) - 1; }
    inline double getYMean() const { return y_norm.mean; }
    inline double getYSTD() const { return y_norm.std; }

    static reg_array generate_feat_bias(reg_array &);
    static double MSE_loss(const reg_array &, const reg_array &);
    void train(size_t, double);
    reg_array eval(reg_array &);
    reg_array output(reg_array &);

    reg_array operator()(reg_array &);
};