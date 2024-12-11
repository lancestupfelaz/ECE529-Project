#ifndef WAVELETGENERATOR_HPP
#define WAVELETGENERATOR_HPP
#include <array>
#include <stdio.h>
#include <vector>



namespace WT
{
	////////////////////////////////////////// IDWT ////////////////////////////////////////////////////

	std::vector<std::vector<float>> customIDWT(std::vector<std::vector<std::vector<std::vector<float>>>>& coefficents);


	std::vector<std::vector<float>> interpolateColumnsAndLowPass(const std::vector<std::vector<float>>& input);
	std::vector<std::vector<float>> interpolateColumnsAndHighPass(const std::vector<std::vector<float>>& input);

	std::vector<std::vector<float>> interpolateRowsAndHighPass(const std::vector<std::vector<float>>& input);
	std::vector<std::vector<float>> interpolateRowsAndLowPass(const std::vector<std::vector<float>>& input);

	std::vector<std::vector<float>> interpolateRowsAndConv(std::vector<float> filter, std::vector<std::vector<float>> input);
	std::vector<std::vector<float>> interpolateColumnsAndConv(std::vector<float> filter, std::vector<std::vector<float>> input);
	std::vector<float> upsample(const std::vector<float>& input, const int rate);

	////////////////////////////////////////// DWT ////////////////////////////////////////////////////

	std::vector<std::vector<std::vector<std::vector<float>>>> customDWT(std::vector<std::vector<float>> image, unsigned int levels);

	std::vector<std::vector<float>> lowpassRowsAndDecimate(std::vector<std::vector<float>>& input);
	std::vector<std::vector<float>> lowpassColumnsAndDecimate(std::vector<std::vector<float>>& input);
	std::vector<std::vector<float>> highpassColumnsAndDecimate(std::vector<std::vector<float>>& input);
	std::vector<std::vector<float>> highpassRowsAndDecimate(std::vector<std::vector<float>>& input);


	std::vector<std::vector<float>> convColumnsAndDecimate(const std::vector<float>& filter, const std::vector<std::vector<float>>& input);
	std::vector<std::vector<float>> convRowsAndDecimate(const std::vector<float>& filter, const std::vector<std::vector<float>>& input);


	std::vector<float> convolve(const std::vector<float>& a, const std::vector<float>& b);
	std::vector<float> downsample(const std::vector<float>& input, const int rate);

	std::vector<std::vector<float>> formOutputMatrix(const unsigned int rows, const unsigned int columns);

	std::vector<float> getRow(const std::vector<std::vector<float>>& input, const unsigned int rowIdx);
	std::vector<float> getColumn(const std::vector<std::vector<float>>& input, const unsigned int columnIdx);
	void setRow(const std::vector<float> input, std::vector<std::vector<float>>& output, const unsigned int rowIdx);
	void setColumn(const std::vector<float> input, std::vector<std::vector<float>>& output, const unsigned int columnIdx);

	void testCustomDWT();
	void testCustomIDWT();

	void testDownsample();
	void testConvolve();

	void printMatrix(std::vector<std::vector<float>> input);

	// operator overloads

	std::vector<float> operator+(const std::vector<float>& a, const std::vector<float>& b);
	std::vector<std::vector<float>> operator+(const std::vector<std::vector<float>>& a, const std::vector<std::vector<float>>& b);


}


#endif