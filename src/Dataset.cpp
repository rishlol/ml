#include "Dataset.hpp"
#include <fstream>
#include <string>
#include "xtensor/io/xcsv.hpp"
#include "xtensor/views/xview.hpp"

/**
 * @brief Creates dataset out of CSV and extracts features and labels.
 * 
 * This function takes a string containing CSV file path.
 * The csv can contain a header line or no header line.
 * First (n - 1) columns are stored as features.
 * Last columns is stored as label.
 * 
 * @param input CSV file path.
 * @param no_header Whether CSV has header or not. False by default.
*/
Dataset::Dataset(std::string input) : Dataset(input, false) {}
Dataset::Dataset(std::string input, bool no_header) {
    good = true;

    // Load csv filestream
    std::ifstream f(input);
    if(f.fail()) {
        std::cerr << "Could not open file!\n";
        good = false;
        return;
    }

    // Get csv header and store csv
    std::string csv_header;
    if(!no_header)
        std::getline(f, csv_header);
    xt::xarray<double> csv = xt::load_csv<double>(f);

    // Extract last columns (labels)
    labels = xt::col(csv, csv.shape().at(1) - 1);
    if(labels.shape().size() == 1)
        labels.reshape({ (size_t)labels.shape().at(0), (size_t)1 });

    // Extract data (features)
    features = xt::view(csv, xt::all(), xt::range((size_t)0, (size_t)csv.shape().at(1) - 1));
}