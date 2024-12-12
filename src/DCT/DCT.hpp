#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <algorithm>
#include <cstdint>

using namespace std;
using Matrix = vector<vector<float>>;
#ifndef QTABLE_H
#define QTABLE_H

// Declare qTable as extern
extern Matrix qTable;

#endif // QTABLE_H

#define M_PI 3.14159

// Forward - DCT II
Matrix forwardDCT(const Matrix& block);

// Quantize DCT coefficients
void quantize(Matrix& block, const Matrix& qTable);

// Dequantize DCT coefficients
void dequantize(Matrix& block, const Matrix& qTable);

// Inverse DCT function - DCT III
Matrix backwardDCT(const Matrix& block);

