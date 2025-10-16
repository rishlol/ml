#include "LinearRegression.hpp"
#include "utils/Dataset.hpp"
#include <limits>
#include "xtensor/containers/xarray.hpp"
#include "xtensor-blas/xlinalg.hpp"
#include "xtensor/generators/xbuilder.hpp"

LinearRegression::LinearRegression(Dataset &d, bool norm_lab, size_t start_norm) : Model(d, norm_lab, start_norm) {}

LinearRegression::LinearRegression(Dataset &d) : LinearRegression(d, false, 0) {}
LinearRegression::LinearRegression(Dataset &d, bool norm_lab) : LinearRegression(d, norm_lab, 0) {}
LinearRegression::LinearRegression(Dataset &d, size_t start_norm) : LinearRegression(d, false, start_norm) {}

/**
 * @brief Calculates MSE.
 * 
 * Takes model outputs and labels and calculates mean squared error.
 * 
 * @param y_lab xarray of expected/desired model output.
 * @param y xarray of model outputs.
 * @return MSE value (double).
 */
double LinearRegression::MSE(const model_arr &y_lab, const model_arr &y) {
    // Make sure input shapes are the same
    if(!ML::xarray_same_shape(y_lab, y)) {
        std::cerr << "Cannot calculate loss! y_label and y_train have different dimensions!\n";
        return -1;
    }

    // MSE
    model_arr sq_diff = xt::square(y_lab - y);
    double m = xt::mean(sq_diff)();
    return m;
}

/**
 * @brief Calculates SSE.
 * 
 * Takes model outputs and labels and calculates sum squared error.
 * 
 * @param y_lab xarray of expected/desired model output.
 * @param y xarray of model outputs.
 * @return SSE value (double).
 */
double LinearRegression::SSE(const model_arr &y_lab, const model_arr &y) {
    // Make sure input shapes are the same
    if(!ML::xarray_same_shape(y_lab, y)) {
        std::cerr << "Cannot calculate loss! y_label and y_train have different dimensions!\n";
        return -1;
    }

    // SSE
    model_arr sq_diff = xt::square(y_lab - y);
    double s = xt::sum(sq_diff)();
    return s;
}

/**
 * @brief Trains LinearRegression using feat_bias features, y_label, and weights
 * 
 * Trains weights based on the feature and label matrices using the given epoch and learning rate values.
 * 
 * @param epochs Number of time dataset will be fed into model during training
 * @param lr Step size for updating weights.
 */
void LinearRegression::train(size_t epochs, double lr) {
    model_arr feat_bias_T = xt::transpose(*feat_bias);
    for(size_t i = 0; i < epochs; i += 1) {
        // forward pass
        model_arr y_train = xt::linalg::dot(*feat_bias, weights);
        double loss = MSE(*y_label, y_train);
        std::cout << "Epoch: " << i + 1 << " Loss: " << loss << std::endl;

        // Calculate gradient and update weights: (2 / N) * x * (f(x) - y)
        model_arr grad = (2.0 / (double)std::get<0>(fb_shape)) * xt::linalg::dot(feat_bias_T, (y_train - *y_label));
        weights -= lr * grad;
    }
    delete_feat_bias();
    delete_y_label();
}

/**
 * @brief Inference without normalization.
 * 
 * Output is not normalized.
 * If labels were normalized during training, denormalize.
 * 
 * @param input_feat Feature matrix with bias column.
 * @return Model outputs without any normalization.
 */
model_arr LinearRegression::output_raw(model_arr input_feat) {
    model_arr y = (*this)(input_feat);
    if(normalizeLabels)
        y = (y * y_norm.std) + y_norm.mean;
    return std::move(y);
}

/**
 * @brief Inference in a normalized space.
 * 
 * Output is normalized if labels were normalized during training.
 * 
 * @param input_feat Feature matrix with bias column.
 * @return Model outputs.
 */
model_arr LinearRegression::output(model_arr input_feat) {
    for(size_t c = 3; c < input_feat.shape().at(1); c += 1) {
        ZScaleNormalizer c_norm = feat_norms.at(c);
        xt::col(input_feat, c) = (xt::col(input_feat, c) - c_norm.mean) / c_norm.std;
    }
    model_arr y = xt::linalg::dot(input_feat, weights);
    return std::move(y);
}

model_arr LinearRegression::operator()(model_arr input_feat) {
    return output(input_feat);
}