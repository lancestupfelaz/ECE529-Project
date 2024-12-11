#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <algorithm>


using namespace std;

// 2D vector
using Matrix = vector<vector<double>>;



// Quantization table
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

/*
This assumes some greyscale image input
for each 8x8 block in the image, perform the forward DCT(DCT II)
followed by quantization, and then do dequantization, and
the backward DCT II (DCT III)
Then populate another matrix 8x8 at a time with the output of the 
8x8 matrix of the DCT III or backwardDCT function
potentially save the output
Then calculate the SSIM, CR, and MSE
    */
#define M_PI 3.14159

// Forward - DCT II
Matrix forwardDCT(const Matrix& block);

// Quantize DCT coefficients
void quantize(Matrix& block, const Matrix& qTable);

// Dequantize DCT coefficients
void dequantize(Matrix& block, const Matrix& qTable);

// Inverse DCT function - DCT III
Matrix backwardDCT(const Matrix& block);

