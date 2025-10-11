#include "Dataset.hpp"
#include "SupportVectorMachine.hpp"
#include "utils/ML_CLIOptions.hpp"
#include <iostream>

int main(int argc, char **argv) {
    ML_CLIOptions cli;
    cli.parse_args(argc, argv);

    return 0;
}