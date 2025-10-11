#include "LinearRegression.hpp"
#include "Dataset.hpp"
#include "xtensor/containers/xarray.hpp"
#include "xtensor/views/xview.hpp"
#include "xtensor-blas/xlinalg.hpp"

/**
 * @brief Create LinearRegression from Dataset.
 * 
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
    y_label = d.get_labels();
    if(normalizeLabels) {
        y_norm.mean = xt::mean(y_label)();
        y_norm.std = xt::stddev(y_label)();
        y_label = (y_label - y_norm.mean) / y_norm.std;
    }

    // Create feature vector with bias column (first column)
    feat_bias = generate_feat_bias(d.get_features());

    // Store feature matrix shape and normalize
    fb_shape = std::make_tuple(feat_bias.shape().at(0), feat_bias.shape().at(1));
    for(size_t c = 3; c < std::get<1>(fb_shape); c += 1) {
        ZScaleNormalizer c_norm{ xt::mean(xt::col(feat_bias, c))(), xt::stddev(xt::col(feat_bias, c))() };
        feat_norms.insert({ c, c_norm });
        xt::col(feat_bias, c) = (xt::col(feat_bias, c) - c_norm.mean) / c_norm.std;
    }
    
    // Initialize weights
    weights = xt::ones<double>({ (size_t)std::get<1>(fb_shape), (size_t)1 });
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
 * @brief Calculates MSE loss.
 * 
 * Takes model outputs and labels and calculates MSE loss.
 * 
 * @param y_label xarray of expected/desired model output.
 * @param y_train xarray of model outputs.
 * @return MSE loss value (double).
 */
double LinearRegression::MSE_loss(const reg_array &y_label, const reg_array &y_train) {
    // Make sure input shapes are the same
    if(!xarray_same_shape(y_label, y_train)) {
        std::cerr << "Cannot calculate loss! y_label and y_train have different dimensions!\n";
        return -1;
    }

    // MSE
    reg_array sq_diff = xt::square(y_label - y_train);
    double m = xt::mean(sq_diff)();
    return m;
}

void LinearRegression::train(size_t epochs, double lr) {
    for(size_t i = 0; i < epochs; i += 1) {
        // forward pass
        reg_array y_train = xt::linalg::dot(feat_bias, weights);
        double loss = MSE_loss(y_label, y_train);
        std::cout << "Epoch: " << i + 1 << " Loss: " << loss << std::endl;

        // Calculate gradient and update weights: 2 * x * (f(x) - y)
        reg_array grad = (2.0 / (double)std::get<0>(fb_shape)) * xt::linalg::dot(xt::transpose(feat_bias), (y_train - y_label));
        weights -= lr * grad;
    }
}

reg_array LinearRegression::eval(reg_array &input_feat) {
    for(size_t c = 3; c < input_feat.shape().at(1); c += 1) {
        ZScaleNormalizer c_norm = feat_norms.at(c);
        xt::col(input_feat, c) = (xt::col(input_feat, c) - c_norm.mean) / c_norm.std;
    }
    reg_array y = (*this)(input_feat);
    if(normalizeLabels)
    y = y * y_norm.std + y_norm.mean;
    return std::move(y);
}

reg_array LinearRegression::output(reg_array &input_feat) {
    return std::move((*this)(input_feat));
}

reg_array LinearRegression::operator()(reg_array &input_feat) {
    for(size_t c = 3; c < input_feat.shape().at(1); c += 1) {
        ZScaleNormalizer c_norm = feat_norms.at(c);
        xt::col(input_feat, c) = (xt::col(input_feat, c) - c_norm.mean) / c_norm.std;
    }
    reg_array y = xt::linalg::dot(input_feat, weights);
    return std::move(y);
}