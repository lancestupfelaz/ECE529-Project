// Forward - DCT II
#include "DCT.hpp"

Matrix forwardDCT(const Matrix& block) {
    int m = block.size();
    int n = block[0].size();
    Matrix OUT(m, vector<double>(n, 0.0));

    for (int p = 0; p < m; ++p) {
        for (int q = 0; q < n; ++q) {
            double ap = (p == 0) ? 1 / sqrt(m) : sqrt(2.0 / m);
            double aq = (q == 0) ? 1 / sqrt(n) : sqrt(2.0 / n);
            double sum = 0.0;

            for (int i = 0; i < m; ++i) {
                for (int j = 0; j < n; ++j) {
                    sum += block[i][j] *
                        cos(M_PI * (2 * i + 1) * p / (2 * m)) *
                        cos(M_PI * (2 * j + 1) * q / (2 * n));
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
    Matrix OUT(m, vector<double>(n, 0.0));

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