#include "Perceptron.hpp"
#include "Dataset.hpp"
#include "utils/ML_Utils.hpp"
#include "xtensor/containers/xarray.hpp"
#include "xtensor/generators/xbuilder.hpp"
#include "xtensor-blas/xlinalg.hpp"

Perceptron::Perceptron(Dataset &d, size_t start_norm) : Model(d, start_norm) {
    weights = xt::ones<double>({ std::get<1>(fb_shape), (size_t)1 });
}

double Perceptron::P_Loss(const perc_arr &y_lab, const perc_arr &y) {
    if(!ML::xarray_same_shape(y_lab, y)) {
        std::cerr << "Not same shape!\n";
        return -1;
    }

    perc_arr zeros = xt::zeros_like(y_lab);
    double h = xt::mean(xt::maximum(zeros, -1.0 * (y_lab * y)))();;
    return h;
}

void Perceptron::train(size_t epochs, double lr) {
    perc_arr feat_bias_T = xt::transpose(*feat_bias);
    for(size_t i = 0; i < epochs; i += 1) {
        // Forward pass
        perc_arr y_train = xt::linalg::dot(*feat_bias, weights);
        double loss = P_Loss(*y_label, y_train);
        std::cout << "Epoch: " << i + 1 << " Loss: " << loss << std::endl;

        // Subgradient descent (-xy)
        perc_arr mask = -1.0 * (*y_label * y_train);
        perc_arr y_grad = xt::where(mask > 0, *y_label, 0.0);

        perc_arr grad = (-1.0 / std::get<0>(fb_shape)) * xt::linalg::dot(feat_bias_T, y_grad);
        weights -= lr * grad;
    }
    delete_feat_bias();
    delete_y_label();
}