#include "Dataset.hpp"
#include "LinearRegression.hpp"
#include <iostream>
#include "boost/program_options.hpp"
#include "xtensor/containers/xarray.hpp"
#include "xtensor/io/xio.hpp"
#include "xtensor/views/xview.hpp"
#include "xtensor/generators/xbuilder.hpp"

namespace po = boost::program_options;

int main(int argc, char **argv) {
    // Get CLI options
    po::options_description desc("Allowed options");
    desc.add_options()
        ("help,h", "Help:")
        ("input-file,I", po::value<std::string>(), "Input CSV file")
        ("test-file,T", po::value<std::string>(), "Validation CSV file")
        ("no-header,N", po::value<bool>()->default_value(false), "Flag if CSV file has no header")
        ("epochs,e", po::value<int>()->default_value(20), "Number of epochs for training")
        ("lr", po::value<double>()->default_value(1e-3), "Learning rate for training")
    ;
    po::positional_options_description p;
    p.add("input-file", 1);
    p.add("test-file", 1);
    
    po::variables_map vm;
    po::store(po::command_line_parser(argc, argv).options(desc).positional(p).run(), vm);
    po::notify(vm);
    
    if(vm.count("help")) {
        std::cout << desc << std::endl;
        return 0;
    }
    
    bool no_header = vm["no-header"].as<bool>();
    std::string input = vm["input-file"].as<std::string>();

    // Load dataset
    Dataset data(input, no_header);
    if(!data.isGood()) {
        std::cerr << "Could not read input CSV!\n";
        return -1;
    }
    
    // Create regression
    LinearRegression lin_reg(data, true);
    
    // Train
    size_t epochs = vm["epochs"].as<int>();
    double lr = vm["lr"].as<double>();
    std::cout << "Training with epochs=" << epochs << " lr=" << lr << std::endl;
    lin_reg.train(epochs, lr);

    // Validation
    if(vm.count("test-file")) {
        std::string val_file = vm["test-file"].as<std::string>();
        Dataset val_data(val_file, no_header);
        if(!val_data.isGood()) {
            std::cerr << "Could not read test CSV!\n";
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

    return 0;
}
