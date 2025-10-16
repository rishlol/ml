#include "utils/Dataset.hpp"
#include "SupportVectorMachine.hpp"
#include "utils/ML_CLIOptions.hpp"
#include <iostream>
#include "xtensor/containers/xarray.hpp"

void validation(SupportVectorMachine &, std::string, bool);

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
    Dataset data(input);
    if(!data.isGood()) {
        std::cerr << "Could not open CSV!\n";
        return -1;
    }

    // Create SVM
    SupportVectorMachine svm(data, 28);

    // Train
    size_t epochs = cli.vm["epochs"].as<size_t>();
    double lr = cli.vm["lr"].as<double>();
    std::cout << "Training with epochs=" << epochs << " lr=" << lr << std::endl;
    svm.train(epochs, lr);

    if(cli.vm.count("test-file"))
        validation(svm, cli.vm["test-file"].as<std::string>(), no_header);

    return 0;
}

void validation(SupportVectorMachine &svm, std::string val_file, bool no_header) {
    Dataset val_data(val_file);
    if(!val_data.isGood()) {
        std::cerr << "Could not open validation dataset!\n";
        return;
    }

    xt::xarray<double> f = ML::generate_feat_bias(val_data.get_features());
    xt::xarray<double> labels = val_data.get_labels();
    xt::xarray<double> outputs = svm(f);
    std::cout << "Mean Hinge Loss: " << SupportVectorMachine::Hinge(labels, outputs) << std::endl;
    std::cout << "Accuracy       : " << ML::accuracy(labels, outputs) << std::endl;
}