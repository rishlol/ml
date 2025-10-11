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
        ("help,h", "<Train CSV> <Test CSV>")
        ("input-file,I", po::value<std::string>(), "input csv file")
        ("test-file,T", po::value<std::string>(), "validation csv file")
        ("no-header,N", po::value<bool>()->default_value(false), "flag if csv has no header")
    ;
    po::positional_options_description p;
    p.add("input-file", 1);
    p.add("test-file", 1);
    
    po::variables_map vm;
    po::store(po::command_line_parser(argc, argv).options(desc).positional(p).run(), vm);
    po::notify(vm);
    
    if(vm.count("help")) {
        std::cout << desc << std::endl;
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
    size_t epochs = 5000;
    double lr = 7.5e-4;
    std::cout << "Training with epochs=" << epochs << " lr=" << lr << std::endl;
    lin_reg.train(epochs, lr);

    // Validation
    if(vm.count("test-file")) {
        std::string val_file = vm["test-file"].as<std::string>();
        Dataset val_data(val_file, no_header);
        if(!val_data.isGood()) {
            std::cerr << "Could not read test CSV!\n";
        }

        xt::xarray<double> f = LinearRegression::generate_feat_bias(val_data.get_features());
        xt::xarray<double> res_norm = lin_reg(f);                                                                   // Model output (raw)    
        xt::xarray<double> norm_labels = (val_data.get_labels() - lin_reg.getYMean()) / lin_reg.getYSTD();          // Labels (normalized)
        xt::xarray<double> res = lin_reg.eval(f);                                                                   // Model output (normalized)
        xt::xarray<double> labels = val_data.get_labels();                                                          // Labels (raw)
        std::cout << "MSE Loss (normalized): " << LinearRegression::MSE_loss(norm_labels, res_norm) << std::endl
                  << "MSE Loss (raw):        " << LinearRegression::MSE_loss(labels, res) << std::endl;
    }

    return 0;
}
