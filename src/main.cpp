#include <stdio.h>
#include <iostream>
#include "MSE.hpp"
#include "WaveletTransform.hpp"


#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "ThirdPartyLibraries/stb_image.h"
#include "ThirdPartyLibraries/stb_image_write.h"


void printImageStats(char* fn);

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
    std::string dataPath = "";//DATA_DIR;


    unsigned char* data = stbi_load(dataPath.c_str(), &width, &height, &channels, 0);

    std::vector<std::vector<std::vector<std::vector<float>>>> coefficents = WT::customDWT(input, 2);


    std::vector<std::vector<float>> reconstructedImage = WT::customIDWT(coefficents);


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

