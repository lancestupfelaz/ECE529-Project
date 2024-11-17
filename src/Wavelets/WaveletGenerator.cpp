#include "WaveletGenerator.hpp"
#include <stdexcept>
#include <cmath>

// Lets make sure this algorithm works

namespace WaveletGenerator
{
#define PI 3.14f

	template<unsigned int length>
	std::array<float, length> generateMoreletWavelet(float k)
	{


		//static_assertstatic_assert

		std::array<float, length> output = {};

		// lets evaluate the wavelet function at each of the samples of the vector
		for (size_t i = 0; i < length; i++)
		{
			// lets make the vector start at 0 and go to length: 0,1,2,...length
			// lets normalize the vector scale between [0,1]
			// lets multiply by 2pi then subtract pi to get a distrabution between [-pi,pi]

			float t = static_cast<float>(i) / static_cast<float>(length) * 2.0f * PI - PI;


			// wavelet function = exp(-abs(t/k)).*cos(2*pi*t)
			// where t is the vector size of the length.
			// where k is the stepness factor of the complex envelope

			output[i] = std::exp(-std::abs(t / k)) * std::cos(2 * PI * t);

		}




		return output;
	}


	// https://en.wikipedia.org/wiki/Convolution

	fftwf_complex* convolve(float a[], float b[], int a_size, int b_size)
	{
		int largest_signal_length = std::max(a_size, b_size);
		int length_pow_2 = 2 << static_cast<int>(std::ceil(std::log2f(largest_signal_length)));

		float* a_in = new float[length_pow_2];
		float* b_in = new float[length_pow_2];

		fftwf_complex* out_a = new fftwf_complex[length_pow_2];
		fftwf_complex* out_b = new fftwf_complex[length_pow_2];

		fftwf_complex* in_c = new fftwf_complex[length_pow_2];
		fftwf_complex* out_c = new fftwf_complex[length_pow_2];


		fftwf_plan plan_a;
		fftwf_plan plan_b;

		// we need to zero pad signals so they are the same length

		for (size_t i = 0; i < length_pow_2; i++)
		{
			a_in[i] = (a_size >= i) ? a[i] : 0.0;
			b_in[i] = (a_size >= i) ? b[i] : 0.0;
		}


		plan_a = fftwf_plan_dft_r2c_1d(length_pow_2, a, out_a, FFTW_ESTIMATE);
		plan_b = fftwf_plan_dft_r2c_1d(length_pow_2, b, out_b, FFTW_ESTIMATE);


		fftwf_execute(plan_a);
		fftwf_execute(plan_b);

		delete[] a_in;
		delete[] b_in;

		fftwf_destroy_plan(plan_a);
		fftwf_destroy_plan(plan_b);

		// multiply plan_a and b
		for (size_t i = 0; i < length_pow_2; i++)
		{
			//(a + ib) (c + id) = (ac - bd) + i(ad + bc).
			float real = out_a[i][0] * out_b[i][0] - out_a[i][1] * out_b[i][1];
			float imag = out_a[i][0] * out_b[i][1] + out_b[i][0] * out_a[i][1];
			in_c[i][0] = real; // real
			in_c[i][1] = imag; // complex

		}

		fftwf_plan plan_inverse = fftwf_plan_dft_1d(length_pow_2, in_c, out_c, FFTW_BACKWARD, FFTW_ESTIMATE);

		fftwf_execute(plan_inverse);
		fftwf_destroy_plan(plan_inverse);


		fftw_cleanup();

		return out_c;
	}

	void testConvolve()
	{
		float a[] = { 1, 2, 3, 2, 1 };
		float b[] = { 1, 2 };

		fftwf_complex * output = convolve(a, b, 5, 2);

		for (size_t i = 0; i < 5+2-1; i++)
		{
			printf("%lf+1i*%lf\n", output[i][0], output[i][1]);
		}

	}


	void testWavelet()
	{
		constexpr unsigned int waveletLength = 201U;
		constexpr float k = 1.0f;
		std::array<float, waveletLength> wavelet = generateMoreletWavelet<waveletLength>(k);

	}

}