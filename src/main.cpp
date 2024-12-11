#include <stdio.h>
#include <iostream>
#include "MSE.hpp"
#include "WaveletTransform.hpp"
#include "DCT.hpp"
#include <chrono>
//#include <opencv2/opencv.hpp>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "ThirdPartyLibraries/stb_image.h"
#include "ThirdPartyLibraries/stb_image_write.h"



// Global variable for the data path
std::string dataPath = "..\\..\\Data\\women_512x512.png";

void printImageStats(char* fn);
void DWT2IDWT();
void DCT2IDCT();
std::vector<std::vector<float>> imageToVector(unsigned char* input_image, int width, int height, int channels);
void saveGrayscaleImage(const std::vector<std::vector<float>>& grayscale, const std::string& filename);
using Matrix = std::vector<std::vector<float>>;
int main() {

    // Start measuring time
    auto startDWT = std::chrono::high_resolution_clock::now();
    DWT2IDWT();
    ////printImageStats("paris-1213603.jpg");
    //WT::testConvolve();
    //WT::testDownsample();
    ////WT::testCustomDWT();
    //WT::testCustomIDWT();
    // 
    // 
        // Stop measuring time
    auto stopDWT = std::chrono::high_resolution_clock::now();
    // Calculate the duration in milliseconds
    auto durationDWT = std::chrono::duration_cast<std::chrono::milliseconds>(stopDWT - startDWT);
    // Print the duration

    std::cout << "DWT2IDWT function took " << durationDWT.count() << " milliseconds to execute." << std::endl;
    ///////////////////////////////////////////////////////////////////////////////////////////////////////
    // Start measuring time
    auto start = std::chrono::high_resolution_clock::now();
    DCT2IDCT();
    // Stop measuring time
    auto stop = std::chrono::high_resolution_clock::now();
    // Calculate the duration in milliseconds
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    // Print the duration
    std::cout << "DCT2IDCT function took " << duration.count() << " milliseconds to execute." << std::endl;

    return 0;
}

void DCT2IDCT()

{
    int width, height, channels;
    unsigned char* data = stbi_load(dataPath.c_str(), &width, &height, &channels, 0);
    std::vector<std::vector<float>> img = imageToVector(data, width, height, channels);


    // Output matrix for the processed image
    Matrix imgOut(height, vector<float>(width, 0.0f));
    Matrix coeffs(height, vector<float>(width, 0.0f));

    // Iterate over each 8x8 block
    for (int m = 0; m < height; m += 8) {
        for (int n = 0; n < width; n += 8) {
            // Extract 8x8 block
            Matrix block(8, vector<float>(8, 0.0f));
            for (int i = 0; i < 8 && m + i < height; ++i) {
                for (int j = 0; j < 8 && n + j < width; ++j) {
                    block[i][j] = img[m + i][n + j];
                }
            }

            // Process block
            Matrix processedBlock = forwardDCT(block);
            quantize(processedBlock,qTable);

            // Copy back the processed block
            for (int i = 0; i < 8 && m + i < height; ++i) {
                for (int j = 0; j < 8 && n + j < width; ++j) {
                    coeffs[m + i][n + j] = processedBlock[i][j];
                }
            }
        }
    }

    // Iterate over each 8x8 block
    for (int m = 0; m < height; m += 8) {
        for (int n = 0; n < width; n += 8) {
            // Extract 8x8 block
            Matrix block(8, vector<float>(8, 0.0f));
            for (int i = 0; i < 8 && m + i < height; ++i) {
                for (int j = 0; j < 8 && n + j < width; ++j) {
                    block[i][j] = coeffs[m + i][n + j];
                }
            }

            // Process block
            Matrix processedBlock = backwardDCT(block);
            dequantize(processedBlock,qTable);

            // Copy back the processed block
            for (int i = 0; i < 8 && m + i < height; ++i) {
                for (int j = 0; j < 8 && n + j < width; ++j) {
                    imgOut[m + i][n + j] = processedBlock[i][j];
                }
            }
            
        }
    }

saveGrayscaleImage(imgOut, "outputImgDCT.png");
// compare img and imgOut SSIM, MSE CR

}

void DWT2IDWT()
{
    std::string dataPath = "..\\..\\Data\\women_512x512.png";//DATA_DIR;
    int width, height, channels;

    system("cd");

    unsigned char* data = stbi_load(dataPath.c_str(), &width, &height, &channels, 0);
    
    std::vector<std::vector<float>> img = imageToVector(data, width, height, channels);

    std::vector<std::vector<std::vector<std::vector<float>>>> coefficents = WT::customDWT(img, 2);

    std::vector<std::vector<float>> reconstructedImage = WT::customIDWT(coefficents);
    saveGrayscaleImage(reconstructedImage, "outputImg.png");

}

void saveGrayscaleImage(const std::vector<std::vector<float>>& grayscale, const std::string& filename) {

    int height = grayscale.size();
    int width = grayscale[0].size();

    std::vector<unsigned char> imageData(height * width);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {

            float value = grayscale[y][x];

            imageData[y * width + x] = static_cast<unsigned char>(value);
        }
    }

    // Write the grayscale image to a file (PNG format)
    if (!stbi_write_png(filename.c_str(), width, height, 1, imageData.data(), width)) {
        throw std::runtime_error("Failed to save the image: " + filename);
    }

    std::cout << "Image saved successfully to " << filename << std::endl;
}


std::vector<std::vector<float>> imageToVector(unsigned char* input_image, int width, int height, int channels)
{
    std::vector<std::vector<float>> output = WT::formOutputMatrix(height, width);

    for (int row = 0; row < height; row++)
    {
        for (int column = 0; column < width; column++)
        {

            int index = (row * width + column) * channels;
            float gray = 0.0f;
            if (channels >= 3) { // At least RGB
                gray = 0.2126f * input_image[index]     // Red
                    + 0.7152f * input_image[index + 1] // Green
                    + 0.0722f * input_image[index + 2]; // Blue
            }
            else { // If grayscale or alpha-only
                gray = input_image[index];
            }


            output[row][column] = gray;
        }
    }
    return output;
}

void printImageStats(char* fn) 
{
    int width, height, channels;

    std::string dataPath = "";//DATA_DIR;
    dataPath.append(fn);

    unsigned char* data = stbi_load(dataPath.c_str(), &width, &height, &channels, 0);

    double mse = MSE::error(data, data, width * height * channels);
    printf("Height: %d, Width: %d, channels: %d, self MSE: %lf", height, width, channels, mse);

    stbi_write_png("paris-1213603-RED.jpeg", width, height, channels, data, width * channels);

    stbi_image_free(data);

}

// for wavelet compression we need to 
// 1. decompose the image in terms of the choosen wavelet.
// 2. threshold the detail coefficents
// 3. reconstruct the image using the coefficents.

// so first we need to find the wavelet we want to use.

