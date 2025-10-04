#include <iostream>
#include <fstream>
#include "boost/program_options.hpp"
#include "xtensor/containers/xarray.hpp"
#include "xtensor/io/xcsv.hpp"
#include "xtensor/generators/xrandom.hpp"
#include "xtensor/io/xio.hpp"
#include "xtensor/core/xexpression.hpp"

int main(int argc, char **argv) {
    std::cout << argv[1] << std::endl;

    std::ifstream f(argv[1]);
    if (f.fail()) {
        std::cerr << "Could not open file!";
        return 1;
    }
    std::string csv_header;
    std::getline(f, csv_header);
    auto expr = xt::load_csv<double>(f);
    std::cout << expr;

    return 0;
}