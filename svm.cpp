#include "Dataset.hpp"
#include <iostream>
#include "boost/program_options.hpp"

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

    return 0;
}