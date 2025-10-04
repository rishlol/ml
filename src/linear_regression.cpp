#include <iostream>
#include <fstream>
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

    std::ifstream f(input);
    if (f.fail()) {
        std::cerr << "Could not open file!";
        return 1;
    }
    std::string csv_header;
    if(!no_header)
        std::getline(f, csv_header);
    auto expr = xt::load_csv<double>(f);
    std::cout << expr << std::endl;

    return 0;
}
