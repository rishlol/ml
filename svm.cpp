#include "Dataset.hpp"
#include "ML_CLIOptions.hpp"
#include "SupportVectorMachine.hpp"
#include <iostream>

int main(int argc, char **argv) {
    ML_CLIOptions cli;
    cli.parse_args(argc, argv);

    return 0;
}