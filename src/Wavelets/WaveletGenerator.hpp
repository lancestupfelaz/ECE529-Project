#ifndef WAVELETGENERATOR_HPP
#define WAVELETGENERATOR_HPP
#include <array>
#include <stdio.h>
#include <vector>

// algorithm:
// https://en.wikipedia.org/wiki/Morlet_wavelet


namespace WaveletGenerator
{
	template<unsigned int length>
	std::array<float, length> generateMoreletWavelet(float k);


	void customDWT(std::vector<std::vector<float>> image, unsigned int levels);

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
	void testDownsample();
	void testConvolve();

	void printMatrix(std::vector<std::vector<float>> input);
}


#endif