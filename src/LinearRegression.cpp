#include "LinearRegression.hpp"
#include "xtensor/containers/xarray.hpp"
#include "xtensor/views/xview.hpp"
#include "xtensor-blas/xlinalg.hpp"

LinearRegression::LinearRegression(const xt::xarray<double> &csv) {
    // Create label array (y_label) from last column in data and normalize
    y_label = xt::col(csv, csv.shape().at(1) - 1);
    z_scale_normalize(y_label);

    // Ensure 2d shape
    y_label.reshape({ (size_t)y_label.shape().at(0), (size_t)1 });

    // Create feature vector from csv with bias column (d + 1), ignore label column
    feat_bias = xt::ones<double>({ (size_t)csv.shape().at(0), (size_t)csv.shape().at(1) });
    for(size_t c = 0; c < csv.shape().at(1) - 1; c += 1) {
        xt::col(feat_bias, c + 1) = xt::col(csv, c);
    }

    // Store feature matrix shape and normalize
    fb_shape = std::make_tuple(feat_bias.shape().at(0), feat_bias.shape().at(1));
    for(size_t c = 2; c < std::get<1>(fb_shape); c += 1) {
        z_scale_normalize(xt::col(feat_bias, c));
    }

    // Initialize weights
    weights = xt::ones<double>({ (size_t)std::get<1>(fb_shape), (size_t)1 });
}

double LinearRegression::MSE_loss(const xt::xarray<double> &y_label, const xt::xarray<double> &y_train) {
    // Make sure input shapes are the same
    if(!xarray_same_shape(y_label, y_train)) {
        std::cerr << "Cannot calculate loss! y_label and y_train have different dimensions!\n";
        return -1;
    }

    // MSE
    xt::xarray<double> sq_diff = xt::square(y_label - y_train);
    double m = xt::mean(sq_diff)();
    return m;
}

void LinearRegression::train(size_t epochs, double lr) {
    for(size_t i = 0; i < epochs; i += 1) {
        // forward pass
        xt::xarray<double> y_train = xt::linalg::dot(feat_bias, weights);
        double loss = MSE_loss(y_label, y_train);
        std::cout << "Epoch: " << i + 1 << " Loss: " << loss << std::endl;

        // Calculate gradient and update weights: 2 * x * (f(x) - y)
        xt::xarray<double> grad = (2.0 / (double)std::get<0>(fb_shape)) * xt::linalg::dot(xt::transpose(feat_bias), (y_train - y_label));
        weights -= lr * grad;
    }
}