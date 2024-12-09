#include "WaveletGenerator.hpp"
#include <stdexcept>
#include <cmath>
#include <iostream>
#include <iomanip>

// Lets make sure this algorithm works

namespace WaveletGenerator
{

	void customDWT(std::vector<std::vector<float>> image, unsigned int levels)
	{


		std::vector<std::vector<float>> L = lowpassRowsAndDecimate(image);

		std::vector<std::vector<float>> LL = lowpassColumnsAndDecimate(L);
		std::vector<std::vector<float>> LH = highpassColumnsAndDecimate(L);



		printMatrix(LL);
		printMatrix(LH);


	}

	std::vector<std::vector<float>> highpassRowsAndDecimate(std::vector<std::vector<float>>& input)
	{
		std::vector<float> filter = { 0.7071f, -0.7071f };
		return convRowsAndDecimate(filter, input);
	}
	std::vector<std::vector<float>> highpassColumnsAndDecimate(std::vector<std::vector<float>>& input)
	{
		std::vector<float> filter = { 0.7071f, -0.7071f };
		return convColumnsAndDecimate(filter, input);
	}


	std::vector<std::vector<float>> lowpassRowsAndDecimate(std::vector<std::vector<float>>& input)
	{
		std::vector<float> filter = { 0.7071f, 0.7071f };

		return convRowsAndDecimate(filter, input);
	}

	std::vector<std::vector<float>> lowpassColumnsAndDecimate(std::vector<std::vector<float>>& input)
	{
		std::vector<float> filter = { 0.7071f, 0.7071f };

		return convColumnsAndDecimate(filter, input);
	}

	std::vector<std::vector<float>> convRowsAndDecimate(const std::vector<float>& filter, const std::vector<std::vector<float>>& input)
	{
		const unsigned int numColumns = input.size();
		const unsigned int numRows = input[0].size();

		std::vector<std::vector<float>> output = formOutputMatrix(numRows, numColumns / 2);

		for (unsigned int rowIdx = 0; rowIdx < numRows; rowIdx++)
		{

			std::vector<float> row = getRow(input, rowIdx);
			std::vector<float> convRow = convolve(row, filter);
			std::vector<float> downSampledRow = downsample(convRow, 2);

			setRow(downSampledRow, output, rowIdx);
		}

		return output;
	}

	// input is rows by columns
	std::vector<std::vector<float>> convColumnsAndDecimate(const std::vector<float>& filter, const std::vector<std::vector<float>>& input)
	{
		const unsigned int numColumns = input[0].size();
		const unsigned int numRows = input.size();

		std::vector<std::vector<float>> output = formOutputMatrix(numRows / 2, numColumns);

		for (unsigned int columnIdx = 0; columnIdx < numColumns; columnIdx++)
		{

			std::vector<float> column = getColumn(input, columnIdx);
			std::vector<float> convColumn = convolve(column, filter);
			std::vector<float> downSampledColumn = downsample(convColumn, 2);

			setColumn(downSampledColumn, output, columnIdx);
		}

		return output;
	}

	std::vector<std::vector<float>> formOutputMatrix(const unsigned int rows, const unsigned int columns)
	{
		std::vector<std::vector<float>> output;
		output.resize(rows);

		for (int rowIdx = 0; rowIdx < rows; rowIdx++)
		{
			output[rowIdx].resize(columns);
		}

		return output;
	}


	std::vector<float> getColumn(const std::vector<std::vector<float>>& input, const unsigned int columnIdx)
	{
		const unsigned int numRows = input.size();
		std::vector<float> column(numRows);

		for (unsigned int rowIdx = 0; rowIdx < numRows; rowIdx++)
		{
			column[rowIdx] = input[rowIdx][columnIdx];
		}

		return column;
	}
	std::vector<float> getRow(const std::vector<std::vector<float>>& input, const unsigned int rowIdx)
	{
		return input[rowIdx];
	}
	void setColumn(const std::vector<float> input, std::vector<std::vector<float>>& output, const unsigned int columnIdx) 
	{
		for (size_t rowIdx = 0; rowIdx < input.size(); rowIdx++)
		{
			output[rowIdx][columnIdx] = input[rowIdx];
		}
	}
	void setRow(const std::vector<float> input, std::vector<std::vector<float>>& output, const unsigned int rowIdx)
	{
		output[rowIdx] = input;
	}

	std::vector<float> downsample(const std::vector<float>& input, const int rate)
	{
		std::vector<float> output;
		output.resize(input.size() / 2);

		for (int i = 0; i / 2 < output.size(); i += 2)
		{
			output[i / 2] = input[i];
		}

		return output;
	}

	std::vector<float> convolve(const std::vector<float>& a, const std::vector<float>& b)
	{
		// https://en.wikipedia.org/wiki/Convolution

		int n = a.size(); // Size of array a
		int m = b.size(); // Size of array b

		// Result size will be n + m - 1
		std::vector<float> result(n + m - 1, 0.0f);

		// Perform convolution
		for (int i = 0; i < n; ++i) {
			for (int j = 0; j < m; ++j) {
				result[i + j] += a[i] * b[j];
			}
		}
		
		result.erase(result.begin());
		
		return result;


	}

	void printMatrix(std::vector<std::vector<float>> input)
	{
		// Determine the width for formatting
		int width = 8; // Adjust as needed for larger or smaller values
		int precision = 2; // Number of decimal places

		for (const auto& row : input) {
			for (const auto& value : row) {
				std::cout << std::setw(width) << std::fixed << std::setprecision(precision) << value;
			}
			std::cout << std::endl; // Move to the next line after printing a row
		}
		
	}

	void testCustomDWT()
	{
		std::vector<std::vector<float>> input = {
			{1.0f, 2.0f, 3.0f, 4.0f},
			{5.0f, 6.0f, 7.0f, 8.0f},
			{9.0f,10.0f,11.0f,12.0f},
			{13.0f,14.0f,15.0f,16.0f}
		};


		customDWT(input, 1);


	}
	void testConvolve()
	{
		// remove the first value from the convolution so we end up with an output the same size of the input.
		// equation { 1, 2, 3, 2, 1} conv { 1, 2 } =  {4, 7, 8, 5, 2}

		std::vector<float> a = { 1, 2, 3, 2, 1 };
		std::vector<float> b = { 1, 2 };


		std::vector<float> c = convolve(a, b);

		


		std::vector<float> expectedValue = { 4, 7, 8, 5, 2 };
		if (expectedValue == c)
		{
			std::cout << "Passed" << std::endl;
		}
		else
		{
			std::cout << "Failed";
			for (size_t i = 0; i < c.size(); i++)
			{
				std::cout << c[i] << ",";
			}
		}
	}
	void testDownsample()
	{
		std::vector<float> a = { 1, 2, 3, 2, 1, 5 };
		std::vector<float> output;
		output.reserve(3);
		output = downsample(a, 2);

		// expect {1, 3, 1}
		std::vector<float> expect = { 1, 3, 1 };
		if (expect == output)
		{
			std::cout << "Passed" << std::endl;
		}
		else
		{
			std::cout << "failed";
		}


	}

}