#include <stdio.h>
#include <iostream>
#include "MSE.hpp"



#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "ThirdPartyLibraries/stb_image.h"
#include "ThirdPartyLibraries/stb_image_write.h"




int main() {
    int width, height, channels;

    std::string fn = DATA_DIR;
    fn.append("\\paris-1213603.jpg");

    unsigned char* data = stbi_load(fn.c_str(), &width, &height, &channels, 0);

    if (data) {
        // Save the image as PNG
        stbi_write_png("output.png", width, height, channels, data, width * channels);
        stbi_image_free(data);
    }
    else {
        std::cerr << "Failed to load image" << std::endl;
    }
    return 0;
}