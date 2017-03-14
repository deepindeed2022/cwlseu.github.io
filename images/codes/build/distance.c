#include <math.h>
#include <stdio.h>
#include "gd.h"
/**
  This is a comment
  */

int main(int argc, char const *argv[])
{
	double a = get_double("Enter a number:", 1.0, 100.0);
	printf("%lf\n", pow(2,a));
	return 0;
}