#include "SupportVectorMachine.hpp"
#include "utils/ML_Utils.hpp"
#include <tuple>
#include "xtensor/containers/xarray.hpp"
#include "xtensor/generators/xbuilder.hpp"
#include "xtensor-blas/xlinalg.hpp"
#include "xtensor/core/xoperation.hpp"

/**
 * @brief Create SupportVectorMachine from Dataset.
 * 
 * Prepares SupportVectorMachine object for training.
 * Dataset `d` already stores feature and labels.
 * Constructor takes features and labels from Dataset `d`.
 * Also normalizes labels, adds bias column to feature matrix, initializes weights, stores label normalization information.
 * The first `start_norm - 1` columns of the Dataset feature matrix will not be normalized.
 * 
 * @param d Dataset object.
 * @param start_norm size_t: column index from which normalization will be applied.
 */
SupportVectorMachine::SupportVectorMachine(Dataset &d, size_t start_norm) : Model(d, start_norm) {}

double SupportVectorMachine::Hinge(const svm_array &y_lab, const svm_array &y) {
    if(!ML::xarray_same_shape(y_lab, y)) {
        std::cerr << "Not same shape!\n";
        return -1;
    }
    
    // Hinge
    svm_array zeros = xt::zeros_like(y_lab);
    svm_array ones = xt::ones_like(y_lab);
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
    svm_array feat_bias_T = xt::transpose(*feat_bias);
    for(size_t i = 0; i < epochs; i += 1) {
        // Forward pass
        svm_array y_train = xt::linalg::dot(*feat_bias, weights);     
        double loss = Hinge(*y_label, y_train);
        std::cout << "Epoch: " << i + 1 << " Loss: " << loss << std::endl;
        
        // Subgradient mask (-xy)
        svm_array mask = 1.0 - (*y_label * y_train);
        svm_array y_grad = xt::where(mask > 0.0, *y_label, 0.0);

        svm_array grad = (-1.0 / std::get<0>(fb_shape)) * xt::linalg::dot(feat_bias_T, y_grad);
        weights -= lr * grad;
    }
    delete_feat_bias();
    delete_y_label();
}

svm_array SupportVectorMachine::output(svm_array input_feat) {
    svm_array raw = xt::linalg::dot(input_feat, weights);
    svm_array classes = xt::where(raw >= 0.0, 1.0, -1.0);
    return std::move(classes);
}

svm_array SupportVectorMachine::operator()(svm_array input_feat) {
    return output(input_feat);
}