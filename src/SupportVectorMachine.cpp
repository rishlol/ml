#include "SupportVectorMachine.hpp"
#include <tuple>
#include "xtensor/containers/xarray.hpp"
#include "xtensor/generators/xbuilder.hpp"
#include "xtensor-blas/xlinalg.hpp"
#include "xtensor/core/xoperation.hpp"

SupportVectorMachine::SupportVectorMachine(Dataset &d, size_t start_norm) : Model(d, start_norm) {}

double SupportVectorMachine::Hinge(const model_arr &y_lab, const model_arr &y) {
    if(!ML::xarray_same_shape(y_lab, y)) {
        std::cerr << "Not same shape!\n";
        return -1;
    }
    
    // Hinge
    model_arr zeros = xt::zeros_like(y_lab);
    model_arr ones = xt::ones_like(y_lab);
    double h = xt::mean(xt::maximum(zeros, ones - (y_lab * y)))();;
    return h;
}

/**
 * @brief Trains SupportVectorMachine using feat_bias features, y_label, and weights
 * 
 * Trains weights based on the feature and label matrices using the given epoch and learning rate values.
 * 
 * @param epochs Number of time dataset will be fed into model during training
 * @param lr Step size for updating weights.
 */
void SupportVectorMachine::train(size_t epochs, double lr) {
    model_arr feat_bias_T = xt::transpose(*feat_bias);
    for(size_t i = 0; i < epochs; i += 1) {
        // Forward pass
        model_arr y_train = xt::linalg::dot(*feat_bias, weights);     
        double loss = Hinge(*y_label, y_train);
        std::cout << "Epoch: " << i + 1 << " Loss: " << loss << std::endl;
        
        // Subgradient mask (-xy)
        model_arr mask = 1.0 - (*y_label * y_train);
        model_arr y_grad = xt::where(mask > 0.0, *y_label, 0.0);

        model_arr grad = (-1.0 / std::get<0>(fb_shape)) * xt::linalg::dot(feat_bias_T, y_grad);
        weights -= lr * grad;
    }
    delete_feat_bias();
    delete_y_label();
}

model_arr SupportVectorMachine::output(model_arr input_feat) {
    model_arr raw = xt::linalg::dot(input_feat, weights);
    model_arr classes = xt::where(raw >= 0.0, 1.0, -1.0);
    return std::move(classes);
}

model_arr SupportVectorMachine::operator()(model_arr input_feat) {
    return output(input_feat);
}