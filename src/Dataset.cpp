#include "Dataset.hpp"
#include <fstream>
#include <string>
#include "xtensor/io/xcsv.hpp"

Dataset::Dataset(std::string input, bool no_header) {
    good = true;

    // Load csv filestream
    std::ifstream f(input);
    if(f.fail()) {
        std::cerr << "Could not open file!\n";
        good = false;
        return;
    }

    // Get csv header
    std::string csv_header;
    if(!no_header)
        std::getline(f, csv_header);

    csv = xt::load_csv<double>(f);
}

Dataset::Dataset(std::string input) : Dataset(input, false) {}