#include "Dataset.hpp"
#include "LinearRegression.hpp"
#include <iostream>
#include "boost/program_options.hpp"
#include "xtensor/containers/xarray.hpp"

namespace po = boost::program_options;

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

    // Load dataset
    Dataset data(input, no_header);
    if(!data.isGood()) {
        std::cerr << "Could not read input csv!\n";
        return -1;
    }
    xt::xarray<double> csv = data.get_csv();
    
    // Create regression
    LinearRegression lin_reg(csv);
    
    // Train
    size_t epochs = 5000;
    double lr = 7.5e-4;
    lin_reg.train(epochs, lr);

    return 0;
}
