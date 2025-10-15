#include "Perceptron.hpp"
#include "utils/ML_CLIOptions.hpp"

int main(int argc, char **argv) {
    ML_CLIOptions cli;
    cli.parse_args(argc, argv);

    Dataset data(cli.vm["input-file"].as<std::string>());

    Perceptron p(data, 28);

    size_t epochs = cli.vm["epochs"].as<size_t>();
    double lr = cli.vm["lr"].as<double>();
    std::cout << "Training with epochs=" << epochs << " lr=" << lr << std::endl;
    p.train(epochs, lr);

    return 0;
}