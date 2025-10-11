#pragma once

struct ZScaleNormalizer {
    double mean;
    double std;

    /**
     * @brief Initialize ZScaleNormalizer struct.
     * 
     * @param m Mean of normalizer.
     * @param s Standard deviation of normalizer.
     */
    ZScaleNormalizer(double m, double s) {
        mean = m;
        std = s;
    }
    ZScaleNormalizer() = default;
};