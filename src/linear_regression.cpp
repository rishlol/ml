#include <iostream>
#include <fstream>
#include <tuple>
#include <ctime>
#include "boost/program_options.hpp"
#include "xtensor/containers/xarray.hpp"
#include "xtensor/io/xcsv.hpp"
#include "xtensor/generators/xrandom.hpp"
#include "xtensor/io/xio.hpp"
#include "xtensor/core/xexpression.hpp"

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
    std::cout << input << std::endl;

    // Load csv filestream
    std::ifstream f(input);
    if (f.fail()) {
        std::cerr << "Could not open file!\n";
        return 1;
    }
    
    // Get csv header
    std::string csv_header;
    if(!no_header)
        std::getline(f, csv_header);
    
    // Load csv into xarray
    xt::xarray<double> csv = xt::load_csv<double>(f);
    
    // Create label array (y_hat)
    xt::xarray<double> y_label = xt::col(csv, -1);

    // Create feature vector with column of 1s for bias (built into weight vector)
    xt::xarray<double> bias = xt::ones<double>({(int)csv.shape()[0], 1});
    xt::xarray<double> features_wbias = xt::hstack(xt::xtuple(bias, xt::view(csv, xt::all(), xt::range(0, (int)csv.shape()[1] - 1))));
    std::tuple s = std::make_tuple(features_wbias.shape()[0], features_wbias.shape()[1]);
    
    // Train
    int epochs = 20;
    double lr = 1e-5;
    xt::xarray<double> weights = xt::ones<double>({std::get<1>(s), std::get<0>(s)});
    for(ssize_t i = 0; i < epochs; i += 1) {
        
    }

    return 0;
}
