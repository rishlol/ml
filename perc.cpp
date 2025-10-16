#include "Perceptron.hpp"
#include "utils/ML_CLIOptions.hpp"

void validation(Perceptron &, std::string, bool);

int main(int argc, char **argv) {
    ML_CLIOptions cli;
    cli.parse_args(argc, argv);

    std::string input_file = cli.vm["input-file"].as<std::string>();
    bool no_header = cli.vm["no-header"].as<bool>();

    // Load dataset
    Dataset data(input_file, no_header);
    if(!data.isGood()) {
        std::cerr << "Could not load training dataset!\n";
        return -1;
    }

    Perceptron p(data, 28);

    // Train
    size_t epochs = cli.vm["epochs"].as<size_t>();
    double lr = cli.vm["lr"].as<double>();
    std::cout << "Training with epochs=" << epochs << " lr=" << lr << std::endl;
    p.train(epochs, lr);

    if(cli.vm.count("test-file")) {
        validation(p, cli.vm["test-file"].as<std::string>(), no_header);
    }

    return 0;
}

void validation(Perceptron &p, std::string test_file, bool no_header) {
    Dataset val(test_file, no_header);
    xt::xarray<double> y_labels = val.get_labels();
    xt::xarray<double> input_feat = ML::generate_feat_bias(val.get_features());
    xt::xarray<double> y = p(input_feat);
    std::cout << ML::accuracy(y_labels, y) << std::endl;
}