#include <stdio.h>
#include <iostream>
#include "MSE.hpp"
#include "WaveletGenerator.hpp"


#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "ThirdPartyLibraries/stb_image.h"
#include "ThirdPartyLibraries/stb_image_write.h"


void printImageStats(char* fn);

int main() {
    //printImageStats("paris-1213603.jpg");
    WaveletGenerator::testConvolve();
    WaveletGenerator::testDownsample();
    WaveletGenerator::testCustomDWT();

    /*  constexpr unsigned int waveletLength = 201U;
    std::array<float, waveletLength> wavelet = WaveletGenerator::generateMoreletWavelet<waveletLength>(1);
    for (size_t i = 0; i < waveletLength; i++)
    {
        printf("%lf\n", wavelet[i]);
    }*/

    return 0;
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

