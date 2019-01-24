#ifndef _MATHEMATIX_HPP_
#define _MATHEMATIX_HPP_

#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <assert.h>
#include "Matrix.hpp"

using namespace std;

class Mathematix
{
public:
	static void multiplyMatrix(Matrix *a, Matrix *b, Matrix *c);
};




#endif // !_MATHEMATIX_HPP_
