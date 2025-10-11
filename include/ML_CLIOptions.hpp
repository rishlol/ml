#pragma once
#include "boost/program_options.hpp"

namespace po = boost::program_options;

class ML_CLIOptions {
public:
    po::positional_options_description p;
    po::options_description desc;
    po::variables_map vm;

    /**
     * @brief Construct default parser
     * 
     * Takes input and test CSV files as positional arguments.
     * No header option if CSV files do not have header line.
     * Can specify epochs and learning rate.
     * Default epochs, learning rate are 20, 1e-3 respectively.
     * 
     * @return void
     */
    ML_CLIOptions() : desc("Allowed options:") {
        desc.add_options()
            ("help,h", "Help:")
            ("input-file,I", po::value<std::string>()->default_value(""), "Input CSV file")
            ("test-file,T", po::value<std::string>()->default_value(""), "Validation CSV file")
            ("no-header,N", po::value<bool>()->default_value(false), "Flag if CSV file has no header")
            ("epochs,e", po::value<int>()->default_value(20), "Number of epochs for training")
            ("lr", po::value<double>()->default_value(1e-3), "Learning rate for training")
        ;
        
        p.add("input-file", 1);
        p.add("test-file", 1);
    }

    /**
     * @brief Parse arguments into variable_map vm
     * 
     * Parse arguments given argc, argv and loads all arguments into vm.
     * Access each argument a with vm[a].as<type>().
     * 
     * @param argc Number of CLI arguments (int)
     * @param argv CLI arguments
     * @return void
     */
    void parse_args(int argc, char **argv) {
        po::store(po::command_line_parser(argc, argv).options(desc).positional(p).run(), vm);
        po::notify(vm);
    }
};