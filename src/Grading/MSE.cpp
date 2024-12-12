#include "MSE.hpp"
#include "stdio.h"
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/hal/interface.h>
#include <opencv2/core/utils/logger.hpp>
//#include <opencv2/quality/qualityssim.hpp>

// Lets make sure this algorithm works
using Matrix = std::vector<std::vector<float>>;
using namespace cv;

namespace MSE
{
	void testMSE()
	{
		float x[3] = { 0.0, 1.0,  2.0 };
		float y[3] = { 0.0, 2.0,  2.0 };

		float e = error(x, y, 3);
		printf("error: %lf\n", e);

	}


// Huffman encoding function
void huffmanEncoding(const Matrix& input, std::vector<unsigned char>& output) {
    // Convert Matrix to Mat
    Mat matInput(input.size(), input[0].size(), CV_32F);
    for (int i = 0; i < input.size(); ++i) {
        for (int j = 0; j < input[i].size(); ++j) {
            matInput.at<float>(i, j) = input[i][j];
        }
    }

    // Convert Mat to CV_8U type and then to a vector of uchar
    Mat matInput8U;
    matInput.convertTo(matInput8U, CV_8U);
    imencode(".png", matInput8U, output, std::vector<int>{IMWRITE_PNG_COMPRESSION, 9});
}

// Metrics calculation function
float calculateMSE(const Matrix& img1, const Matrix& img2) {
    float mse = 0.0;
    int height = img1.size();
    int width = img1[0].size();
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            mse += pow(img1[i][j] - img2[i][j], 2);
        }
    }
    return mse / (height * width);
}
/*
float calculateSSIM(const Matrix& img1, const Matrix& img2) {
    Mat mat1(img1.size(), img1[0].size(), CV_32F);
    Mat mat2(img2.size(), img2[0].size(), CV_32F);
    for (int i = 0; i < img1.size(); ++i) {
        for (int j = 0; j < img1[i].size(); ++j) {
            mat1.at<float>(i, j) = img1[i][j];
            mat2.at<float>(i, j) = img2[i][j];
        }
    }
    return cv::quality::QualitySSIM::compute(mat1, mat2, noArray()).val[0];
}
*/


}