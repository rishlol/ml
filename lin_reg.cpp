#include "Dataset.hpp"
#include "LinearRegression.hpp"
#include "utils/ML_CLIOptions.hpp"
#include <iostream>
#include "xtensor/containers/xarray.hpp"
#include "xtensor/views/xview.hpp"
#include "xtensor/generators/xbuilder.hpp"

void validation(LinearRegression &, std::string, bool);

int main(int argc, char **argv) {
    ML_CLIOptions cli;
    cli.parse_args(argc, argv);
    
    if(cli.vm.count("help")) {
        std::cout << cli.desc << std::endl;
        return 0;
    }
    
    bool no_header = cli.vm["no-header"].as<bool>();
    std::string input = cli.vm["input-file"].as<std::string>();

    // Load dataset
    Dataset data(input, no_header);
    if(!data.isGood()) {
        std::cerr << "Could not read input CSV!\n";
        return -1;
    }
    
    // Create regression
    LinearRegression lin_reg(data, true);
    
    // Train
    size_t epochs = cli.vm["epochs"].as<int>();
    double lr = cli.vm["lr"].as<double>();
    std::cout << "Training with epochs=" << epochs << " lr=" << lr << std::endl;
    lin_reg.train(epochs, lr);

    // Validation
    if(cli.vm.count("test-file")) {
        std::string val_file = cli.vm["test-file"].as<std::string>();
        validation(lin_reg, val_file, no_header);
    }

    return 0;
}

void validation(LinearRegression &lin_reg, std::string val_file, bool no_header) {
    Dataset val_data(val_file, no_header);
    if(!val_data.isGood()) {
        std::cerr << "Could not read test CSV!\n";
        return;
    }

    xt::xarray<double> f1 = LinearRegression::generate_feat_bias(val_data.get_features());
    xt::xarray<double> f2 = LinearRegression::generate_feat_bias(val_data.get_features());
    xt::xarray<double> res_norm = lin_reg(f1);                                                                  // Model output (raw)    
    xt::xarray<double> labels_norm = (val_data.get_labels() - lin_reg.getYMean()) / lin_reg.getYSTD();          // Labels (normalized)
    xt::xarray<double> res = lin_reg.output_raw(f2);                                                            // Model output (normalized)
    xt::xarray<double> labels = val_data.get_labels();                                                          // Labels (raw)
    std::cout << "MSE Loss (normalized): " << LinearRegression::MSE(labels_norm, res_norm) << std::endl
              << "MSE Loss (raw):        " << LinearRegression::MSE(labels, res) << std::endl
              << "R^2 (normalized):      " << LinearRegression::R_Squared(labels_norm, res_norm) << std::endl
              << "R^2 (raw):             " << LinearRegression::R_Squared(labels, res) << std::endl;
}
