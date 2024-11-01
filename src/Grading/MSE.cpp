#include "MSE.hpp"
#include "stdio.h"

// Lets make sure this algorithm works

namespace MSE
{
	void testMSE()
	{
		float x[3] = { 0.0, 1.0,  2.0 };
		float y[3] = { 0.0, 2.0,  2.0 };

		float e = error(x, y, 3);
		printf("error: %lf\n", e);

	}

}