// Forward - DCT II
#include "DCT.hpp"


// Define qTable
Matrix qTable = {
    {16, 11, 10, 16, 24, 40, 51, 61},
    {12, 12, 14, 19, 26, 58, 60, 55},
    {14, 13, 16, 24, 40, 57, 69, 56},
    {14, 17, 22, 29, 51, 87, 80, 62},
    {18, 22, 37, 56, 68, 109, 103, 77},
    {24, 35, 55, 64, 81, 104, 113, 92},
    {49, 64, 78, 87, 103, 121, 120, 101},
    {72, 92, 95, 98, 112, 100, 103, 99}
};


Matrix forwardDCT(const Matrix& block) {
    int m = block.size();
    int n = block[0].size();
    Matrix OUT(m, vector<float>(n, 0.0));

    for (int p = 0; p < m; ++p) {
        for (int q = 0; q < n; ++q) {
            float ap = (p == 0) ? 1 / sqrt(m) : sqrt(2.0 / m);
            float aq = (q == 0) ? 1 / sqrt(n) : sqrt(2.0 / n);
            float sum = 0.0;

            for (int i = 0; i < m; ++i) {
                for (int j = 0; j < n; ++j) {
                    sum += block[i][j] *
                        std::cos(M_PI * (2 * i + 1) * p / (2 * m)) *
                        std::cos(M_PI * (2 * j + 1) * q / (2 * n));
                }
            }
            OUT[p][q] = ap * aq * sum;
        }
    }
    return OUT;
}

// Quantize DCT coefficients
void quantize(Matrix& block, const Matrix& qTable) {
    int m = block.size();
    int n = block[0].size();
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            block[i][j] = round(block[i][j] / qTable[i][j]);
        }
    }
}

// Dequantize DCT coefficients
void dequantize(Matrix& block, const Matrix& qTable) {
    int m = block.size();
    int n = block[0].size();
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            block[i][j] *= qTable[i][j];
        }
    }
}

Matrix backwardDCT(const Matrix& block) {
    int m = block.size();
    int n = block[0].size();
    Matrix OUT(m, vector<float>(n, 0.0));

    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            double sum = 0.0;

            for (int p = 0; p < m; ++p) {
                for (int q = 0; q < n; ++q) {
                    double ap = (p == 0) ? 1 / sqrt(m) : sqrt(2.0 / m);
                    double aq = (q == 0) ? 1 / sqrt(n) : sqrt(2.0 / n);
                    sum += ap * aq * block[p][q] *
                        cos(M_PI * (2 * i + 1) * p / (2 * m)) *
                        cos(M_PI * (2 * j + 1) * q / (2 * n));
                }
            }
            OUT[i][j] = sum;
        }
    }
    return OUT;
}