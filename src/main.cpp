#include <stdio.h>
#include <iostream>
#include "MSE.hpp"
#include "WaveletTransform.hpp"


#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "ThirdPartyLibraries/stb_image.h"
#include "ThirdPartyLibraries/stb_image_write.h"


void printImageStats(char* fn);
void DWT2IDWT();
std::vector<std::vector<float>> imageToVector(unsigned char* input_image, int width, int height, int channels);
void saveGrayscaleImage(const std::vector<std::vector<float>>& grayscale, const std::string& filename);

int main() {

    DWT2IDWT();
    ////printImageStats("paris-1213603.jpg");
    //WT::testConvolve();
    //WT::testDownsample();
    ////WT::testCustomDWT();
    //WT::testCustomIDWT();



    return 0;
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

    if (grayscale.empty() || grayscale[0].empty()) {
        throw std::invalid_argument("Grayscale image data is empty.");
    }

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

