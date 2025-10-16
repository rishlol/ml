#include "Perceptron.hpp"
#include "utils/Dataset.hpp"
#include "xtensor/containers/xarray.hpp"
#include "xtensor/generators/xbuilder.hpp"
#include "xtensor-blas/xlinalg.hpp"

Perceptron::Perceptron(Dataset &d, size_t start_norm) : Model(d, start_norm) {
    weights = xt::ones<double>({ std::get<1>(fb_shape), (size_t)1 });
}

double Perceptron::P_Loss(const model_arr &y_lab, const model_arr &y) {
    if(!ML::xarray_same_shape(y_lab, y)) {
        std::cerr << "Not same shape!\n";
        return -1;
    }

    model_arr zeros = xt::zeros_like(y_lab);
    double h = xt::mean(xt::maximum(zeros, -1.0 * (y_lab * y)))();;
    return h;
}

/**
 * @brief Trains Perceptron using feat_bias features, y_label, and weights
 * 
 * Trains weights based on the feature and label matrices using the given epoch and learning rate values.
 * 
 * @param epochs Number of time dataset will be fed into model during training
 * @param lr Step size for updating weights.
 */
void Perceptron::train(size_t epochs, double lr) {
    model_arr feat_bias_T = xt::transpose(*feat_bias);
    for(size_t i = 0; i < epochs; i += 1) {
        // Forward pass
        model_arr y_train = xt::linalg::dot(*feat_bias, weights);
        double loss = P_Loss(*y_label, y_train);
        std::cout << "Epoch: " << i + 1 << " Loss: " << loss << std::endl;

        // Subgradient descent (-xy)
        model_arr mask = -1.0 * (*y_label * y_train);
        model_arr y_grad = xt::where(mask > 0, *y_label, 0.0);

        model_arr grad = (-1.0 / std::get<0>(fb_shape)) * xt::linalg::dot(feat_bias_T, y_grad);
        weights -= lr * grad;
    }
    delete_feat_bias();
    delete_y_label();
}

model_arr Perceptron::output(model_arr input_feat) {
    model_arr raw = xt::linalg::dot(input_feat, weights);
    model_arr classes = xt::where(raw > 0.0, 1.0, -1.0);
    return std::move(classes);
}

model_arr Perceptron::operator()(model_arr input_feat) {
    return output(input_feat);
}