#ifndef MSE_HPP
#define MSE_HPP
// This class will find the mean squared error between two images.

// algorithm:
// https://en.wikipedia.org/wiki/Mean_squared_error


namespace MSE
{
	template<typename T>
	double error(T* v1, T* v2, const unsigned int length)
	{
		T sum = 0;
		for (unsigned int i = 0; i < length; i++)
		{
			int diff = static_cast<int>(v1[i]) - static_cast<int>(v2[i]);
			diff *= diff;
			sum += diff;
		}
		sum /= length;

		return sum;
	}
	void testMSE();

}


#endif