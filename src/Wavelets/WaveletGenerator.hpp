#ifndef WAVELETGENERATOR_HPP
#define WAVELETGENERATOR_HPP
#include <array>
#include <stdio.h>
#include <vector>
#include <fftw3.h>

// algorithm:
// https://en.wikipedia.org/wiki/Morlet_wavelet


namespace WaveletGenerator
{
	template<unsigned int length>
	std::array<float, length> generateMoreletWavelet(float k);
	fftwf_complex* convolve(float a[], float b[], int a_size, int b_size);
	void testConvolve();

	void testWavelet();

}


#endif