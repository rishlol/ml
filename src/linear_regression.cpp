#include <iostream>
#include <fstream>
#include <tuple>
#include <ctime>
#include "boost/program_options.hpp"
#include "xtensor/containers/xarray.hpp"
#include "xtensor/io/xcsv.hpp"
#include "xtensor/io/xio.hpp"
#include "xtensor/core/xexpression.hpp"
#include "xtensor-blas/xlinalg.hpp"

namespace po = boost::program_options;

double MSE_loss(const xt::xarray<double> &, const xt::xarray<double> &);

int main(int argc, char **argv) {
    // Get CLI options
    po::options_description desc("Allowed options");
    desc.add_options()
        ("help,h", "<csv filename> -[N]")
        ("input-file,I", po::value<std::string>(), "input csv file")
        ("no-header,N", po::value<bool>()->default_value(false), "flag if csv has no header")
    ;
    po::positional_options_description p;
    p.add("input-file", -1);
    
    po::variables_map vm;
    po::store(po::command_line_parser(argc, argv).options(desc).positional(p).run(), vm);
    po::notify(vm);
    
    if(vm.count("help")) {
        std::cout << desc << std::endl;
    }
    
    bool no_header = vm["no-header"].as<bool>();
    std::string input = vm["input-file"].as<std::string>();

    // Load csv filestream
    std::ifstream f(input);
    if(f.fail()) {
        std::cerr << "Could not open file!\n";
        return 1;
    }
    
    // Get csv header
    std::string csv_header;
    if(!no_header)
        std::getline(f, csv_header);
    
    // Load csv into xarray
    xt::xarray<double> csv = xt::load_csv<double>(f);
    
    // Create label array (y_hat) and normalize
    xt::xarray<double> y_label = xt::col(csv, -1);
    y_label = (y_label - xt::mean(y_label)()) / xt::stddev(y_label)();
    y_label = y_label.reshape({ (ssize_t)y_label.shape().at(0), (ssize_t)1 });

    // Create feature vector with column of 1s for bias (built into weight vector)
    // Remove last column of csv since it stores labels
    xt::xarray<double> bias = xt::ones<double>({(int)csv.shape().at(0), 1});
    xt::xarray<double> feat_bias = xt::hstack(xt::xtuple(
        bias,
        xt::view( csv, xt::all(), xt::range(0, (int)csv.shape().at(1) - 1) )
    ));
    std::tuple fb_shape = std::make_tuple(feat_bias.shape().at(0), feat_bias.shape().at(1));

    // Normalize features
    for(int c = 2; c < std::get<1>(fb_shape); c += 1) {
        xt::col(feat_bias, c) = (xt::col(feat_bias, c) - xt::mean(xt::col(feat_bias, c))()) / xt::stddev(xt::col(feat_bias, c))();
    }
    
    // Define hyperparameters
    int epochs = 5000;
    double lr = 7.5e-4;
    
    // Weight vector
    xt::xarray<double> weights = xt::ones<double>({ (ssize_t)std::get<1>(fb_shape), (ssize_t)1 });

    // Train
    for(ssize_t i = 0; i < epochs; i += 1) {
        // forward pass
        xt::xarray<double> y_train = xt::linalg::dot(feat_bias, weights);
        double loss = MSE_loss(y_label, y_train);
        std::cout << "Epoch: " << i + 1 << " Loss: " << loss << std::endl;

        // Calculate gradient and update weights: 2 * x * (f(x) - y)
        xt::xarray<double> grad = (2.0 / (double)std::get<0>(fb_shape)) * xt::linalg::dot(xt::transpose(feat_bias), (y_train - y_label));
        weights -= lr * grad;
    }

    return 0;
}

double MSE_loss(const xt::xarray<double> &y_label, const xt::xarray<double> &y_train) {
    // Make sure input shapes are the same
    int dim_train = 0, dim_label = 0;
    for(const auto &x : y_train.shape())
        dim_train += 1;
    for(const auto &x : y_label.shape())
        dim_label += 1;
    if(dim_train != dim_label || y_train.shape().at(0) != y_label.shape().at(0) || y_train.shape().at(1) != y_label.shape().at(1)) {
        if(dim_train != dim_label)
            std::cerr << "MSE: Train and Label input shapes not same shape! Train: " << dim_train << " Label: " << dim_label << std::endl;
        if(y_train.shape().at(0) != y_label.shape().at(0))
            std::cerr << "MSE: Train and Label input first dimension not same!\n";
        if(y_train.shape().at(1) != y_label.shape().at(1))
            std::cerr << "MSE: Train and Label input second dimension not same!\n";
        return -1;
    }

    // MSE
    xt::xarray<double> sq_diff = xt::square(y_label - y_train);
    double m = xt::mean(sq_diff)();
    return m;
}
