#include "LinearRegression.hpp"
#include "Dataset.hpp"
#include <limits>
#include "xtensor/containers/xarray.hpp"
#include "xtensor/views/xview.hpp"
#include "xtensor-blas/xlinalg.hpp"
#include "xtensor/generators/xbuilder.hpp"

/**
 * @brief Create LinearRegression from Dataset.
 * 
 * Prepares LinearRegression object for training.
 * Dataset d already stores feature and labels.
 * Constructor takes Dataset object and stores features and labels.
 * Also normalizes labels, adds bias column to feature matrix, initializes weights, stores label normalization information.
 * 
 * @param d Dataset object.
 * @param norm_lab Boolean flag that determines whether labels will be normalized.
 */
LinearRegression::LinearRegression(Dataset &d, bool norm_lab) {
    // Get labels and normalize if needed
    normalizeLabels = norm_lab;
    y_label = new reg_array(d.get_labels());
    if(normalizeLabels) {
        y_norm.mean = xt::mean(*y_label)();
        y_norm.std = xt::stddev(*y_label)();
        *y_label = (*y_label - y_norm.mean) / y_norm.std;
    }

    // Create feature vector with bias column (first column)
    feat_bias = new reg_array(generate_feat_bias(d.get_features()));

    // Store feature matrix shape and normalize
    fb_shape = std::make_tuple(feat_bias->shape().at(0), feat_bias->shape().at(1));
    for(size_t c = 3; c < std::get<1>(fb_shape); c += 1) {
        feat_norms.insert({ c, ZScaleNormalizer(xt::mean(xt::col(*feat_bias, c))(), xt::stddev(xt::col(*feat_bias, c))()) });
        ZScaleNormalizer c_norm = feat_norms.at(c);
        xt::col(*feat_bias, c) = (xt::col(*feat_bias, c) - c_norm.mean) / c_norm.std;
    }
    
    // Initialize weights
    weights = xt::zeros<double>({ (size_t)std::get<1>(fb_shape), (size_t)1 });
}

LinearRegression::LinearRegression(Dataset &d) : LinearRegression(d, false) {}

/**
 * @brief Adds bias column to feature array.
 * 
 * Takes xarray as input and adds a bias column to the beginning.
 * Bias column filled with ones.
 * 
 * @param features Feature xarray.
 * @return New xarray with bias column before features.
 */
reg_array LinearRegression::generate_feat_bias(reg_array &features) {
    reg_array fb = xt::ones<double>({ (size_t)features.shape().at(0), (size_t)features.shape().at(1) + 1 });
    for(size_t c = 0; c < features.shape().at(1); c += 1)
        xt::col(fb, c + 1) = xt::col(features, c);
    return std::move(fb);
}

/**
 * @brief Calculates MSE.
 * 
 * Takes model outputs and labels and calculates mean squared error.
 * 
 * @param y_lab xarray of expected/desired model output.
 * @param y xarray of model outputs.
 * @return MSE value (double).
 */
double LinearRegression::MSE(const reg_array &y_lab, const reg_array &y) {
    // Make sure input shapes are the same
    if(!xarray_same_shape(y_lab, y)) {
        std::cerr << "Cannot calculate loss! y_label and y_train have different dimensions!\n";
        return -1;
    }

    // MSE
    reg_array sq_diff = xt::square(y_lab - y);
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
double LinearRegression::SSE(const reg_array &y_lab, const reg_array &y) {
    // Make sure input shapes are the same
    if(!xarray_same_shape(y_lab, y)) {
        std::cerr << "Cannot calculate loss! y_label and y_train have different dimensions!\n";
        return -1;
    }

    // SSE
    reg_array sq_diff = xt::square(y_lab - y);
    double s = xt::sum(sq_diff)();
    return s;
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
double LinearRegression::R_Squared(const reg_array &y_lab, const reg_array &y) {
    // Make sure input shapes are the same
    if(!xarray_same_shape(y_lab, y)) {
        std::cerr << "Cannot calculate loss! y_label and y_train have different dimensions!\n";
        return std::numeric_limits<double>::quiet_NaN();
    }

    // R^2
    // reg_array avg_label = xt::full_like(y_label, xt::mean(y_label)());
    double ss_tot = xt::sum(xt::square(y_lab - xt::mean(y_lab)()))();
    if(ss_tot == 0.0)
        return 1;
    double ss_res = xt::sum(xt::square(y_lab - y))();
    return 1 - (ss_res / ss_tot);
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
    for(size_t i = 0; i < epochs; i += 1) {
        // forward pass
        reg_array y_train = xt::linalg::dot(*feat_bias, weights);
        double loss = MSE(*y_label, y_train);
        std::cout << "Epoch: " << i + 1 << " Loss: " << loss << std::endl;

        // Calculate gradient and update weights: (2 / N) * x * (f(x) - y)
        reg_array grad = (2.0 / (double)std::get<0>(fb_shape)) * xt::linalg::dot(xt::transpose(*feat_bias), (y_train - *y_label));
        weights -= lr * grad;
    }
    delete feat_bias;
    delete y_label;
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
reg_array LinearRegression::output_raw(reg_array input_feat) {
    reg_array y = (*this)(input_feat);
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
reg_array LinearRegression::output(reg_array input_feat) {
    return std::move((*this)(input_feat));
}

reg_array LinearRegression::operator()(reg_array input_feat) {
    for(size_t c = 3; c < input_feat.shape().at(1); c += 1) {
        ZScaleNormalizer c_norm = feat_norms.at(c);
        xt::col(input_feat, c) = (xt::col(input_feat, c) - c_norm.mean) / c_norm.std;
    }
    reg_array y = xt::linalg::dot(input_feat, weights);
    return std::move(y);
}