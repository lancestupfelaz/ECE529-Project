#include "WaveletTransform.hpp"
#include <stdexcept>
#include <cmath>
#include <iostream>
#include <iomanip>

// Lets make sure this algorithm works

namespace WT
{

	////////////////////////////////////////// IDWT ////////////////////////////////////////////////////
	
	std::vector<std::vector<float>> customIDWT(std::vector<std::vector<std::vector<std::vector<float>>>>& coefficents)
	{
		// start at the highest level
		const unsigned int levels = coefficents.size() - 1;
		
		std::vector<std::vector<float>> S;

		for (int level = levels; level >= 0; level--)
		{
			std::vector<std::vector<float>> H = coefficents[level][0];
			std::vector<std::vector<float>> V = coefficents[level][1];
			std::vector<std::vector<float>> D = coefficents[level][2];
			std::vector<std::vector<float>> A;

			if (level == levels)
			{
				A = coefficents[level][3];
			}
			else 
			{
				A = S;
			}
			A = interpolateColumnsAndLowPass(A);
			H = interpolateColumnsAndHighPass(H);

			std::vector<std::vector<float>> L = interpolateRowsAndLowPass(A + H);

		
			D = interpolateColumnsAndHighPass(D);
			V = interpolateColumnsAndLowPass(V);

			H = interpolateRowsAndHighPass(D + V);

			S = H + L;

		}

		return S;
	
	}


	std::vector<std::vector<float>> interpolateColumnsAndHighPass(const std::vector<std::vector<float>>& input)
	{
		std::vector<float> highpass = { -0.7071, 0.7071 };
		return interpolateColumnsAndConv(highpass, input);
	}

	std::vector<std::vector<float>> interpolateColumnsAndLowPass(const std::vector<std::vector<float>>& input)
	{
		std::vector<float> lowpass = { 0.7071, 0.7071 };
		return interpolateColumnsAndConv(lowpass, input);

	}

	std::vector<std::vector<float>> interpolateRowsAndHighPass(const std::vector<std::vector<float>>& input)
	{
		std::vector<float> highpass = { -0.7071, 0.7071 };
		return interpolateRowsAndConv(highpass, input);
	}

	std::vector<std::vector<float>> interpolateRowsAndLowPass(const std::vector<std::vector<float>>& input)
	{
		std::vector<float> lowpass = { 0.7071, 0.7071 };
		return interpolateRowsAndConv(lowpass, input);
	}

	std::vector<std::vector<float>> interpolateRowsAndConv(std::vector<float> filter, std::vector<std::vector<float>> input)
	{
		const unsigned int numColumns = input[0].size();
		const unsigned int numRows = input.size();

		std::vector<std::vector<float>> output = formOutputMatrix(numRows, numColumns*2);

		for (unsigned int rowIdx = 0; rowIdx < numRows; rowIdx++)
		{

			std::vector<float> row = getRow(input, rowIdx);
			// pad column symmetricly
			row.insert(row.begin(), row[1]);
			std::vector<float> upSampledRow = upsample(row, 2);
			std::vector<float> convRow = convolve(upSampledRow, filter);
			convRow.erase(convRow.begin());
			convRow.erase(convRow.begin() + convRow.size() - 1);

			setRow(convRow, output, rowIdx);
		}

		return output;
	}

	std::vector<std::vector<float>> interpolateColumnsAndConv(std::vector<float> filter, std::vector<std::vector<float>> input)
	{
		const unsigned int numColumns = input[0].size();
		const unsigned int numRows = input.size();

		std::vector<std::vector<float>> output = formOutputMatrix(numRows * 2, numColumns);

		for (unsigned int columnIdx = 0; columnIdx < numColumns; columnIdx++)
		{

			std::vector<float> column = getColumn(input, columnIdx);
			// pad column symmetricly
			column.insert(column.begin(), column[1]);
			std::vector<float> upSampledColumn = upsample(column, 2);
			std::vector<float> convColumn = convolve(upSampledColumn, filter);
			convColumn.erase(convColumn.begin());
			convColumn.erase(convColumn.begin() + convColumn.size() - 1);

			setColumn(convColumn, output, columnIdx);
		}

		return output;
	}

	std::vector<float> upsample(const std::vector<float>& input, const int rate)
	{
		std::vector<float> output;
		output.resize(input.size() * rate);

		for (int i = 0; i < output.size(); i++)
		{
			output[i] = (i % rate == 0) ? input[i / rate] : 0.0f;
		}

		return output;
	}


	void testCustomIDWT()
	{
		std::vector<std::vector<float>> input = {
			{ 1.0f,  2.0f,  3.0f,  4.0f,  5.0f,  6.0f,  7.0f,  8.0f},
			{ 9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f},
			{17.0f, 18.0f, 19.0f, 20.0f, 21.0f, 22.0f, 23.0f, 24.0f},
			{25.0f, 26.0f, 27.0f, 28.0f, 29.0f, 30.0f, 31.0f, 32.0f},
			{33.0f, 34.0f, 35.0f, 36.0f, 37.0f, 38.0f, 39.0f, 40.0f},
			{41.0f, 42.0f, 43.0f, 44.0f, 45.0f, 46.0f, 47.0f, 48.0f},
			{49.0f, 50.0f, 51.0f, 52.0f, 53.0f, 54.0f, 55.0f, 56.0f},
			{57.0f, 58.0f, 59.0f, 60.0f, 61.0f, 62.0f, 63.0f, 64.0f}
		};

		std::vector<std::vector<std::vector<std::vector<float>>>> coefficents = customDWT(input, 2);

		std::vector<std::vector<float>> reconstructedImage = customIDWT(coefficents);

		std::cout << "reconstructed Image" << std::endl;
		printMatrix(reconstructedImage);

	}



	////////////////////////////////////////// DWT ////////////////////////////////////////////////////

	std::vector<std::vector<std::vector<std::vector<float>>>> customDWT(std::vector<std::vector<float>> image, unsigned int levels)
	{
		std::vector<std::vector<std::vector<std::vector<float>>>> coefficentsByLevel;
		coefficentsByLevel.resize(levels);
		

		for (unsigned int level = 0; level < levels; level++)
		{
			std::vector<std::vector<float>> L = lowpassRowsAndDecimate(image);

			std::vector<std::vector<float>> LL = lowpassColumnsAndDecimate(L);  // Approximate coefficents
			std::vector<std::vector<float>> LH = highpassColumnsAndDecimate(L); // Horizontal coefficents

			std::vector<std::vector<float>> H = highpassRowsAndDecimate(image);

			std::vector<std::vector<float>> HH = highpassColumnsAndDecimate(H); // Veritcal coefficents
			std::vector<std::vector<float>> HL = lowpassColumnsAndDecimate(H);  // Diagonal coefficents


			// final level has four coefficent matrices
			unsigned int numCoefficents = (level == levels - 1) ? 4 : 3;
			coefficentsByLevel[level].resize(numCoefficents);

			coefficentsByLevel[level][0] = LH;
			coefficentsByLevel[level][1] = HL;
			coefficentsByLevel[level][2] = HH;
			
			if (level == levels - 1)
			{
				coefficentsByLevel[level][3] = LL;
			}
			else
			{
				image = LL;
			}
			
			/*
			std::cout << "=== Level " << level + 1 << " ===";

			std::cout << "\nH\n";
			printMatrix(LH);
			
			std::cout << "\nV\n";
			printMatrix(HH);

			std::cout << "\nD\n";
			printMatrix(HL);

			if (level == levels - 1)
			{
				std::cout << "\nA\n";
				printMatrix(LL);
			}
			*/

		}

		return coefficentsByLevel;
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
		const unsigned int numColumns = input[0].size();
		const unsigned int numRows = input.size();

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


		customDWT(input, 2);


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


	std::vector<float> operator+(const std::vector<float>& a, const std::vector<float>& b) 
	{

		if (a.size() != b.size()) {
			throw std::invalid_argument("Vectors must be of the same size for addition.");
		}

		std::vector<float> output(a.size());
		for (size_t i = 0; i < a.size(); ++i) {
			output[i] = a[i] + b[i];
		}

		return output;
	}

	std::vector<std::vector<float>> operator+(const std::vector<std::vector<float>>& a, const std::vector<std::vector<float>>& b)
	{

		if (a.size() != b.size()) {
			throw std::invalid_argument("Outer sizes of the matrices must be the same for addition.");
		}

		std::vector<std::vector<float>> output;
		output.resize(a.size());

		for (size_t i = 0; i < a.size(); i++)
		{
			output[i] = a[i] + b[i];
		}

		return output;
	}
}