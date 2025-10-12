#include "Dataset.hpp"
#include "SupportVectorMachine.hpp"
#include "utils/ML_CLIOptions.hpp"
#include <iostream>

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

    return 0;
}