#ifndef DISTANCE_H_
#define DISTANCE_H_
#include <stdio.h>
/**
 * update this file
 */
double get_double(char* prompt, double min, double max)
{
	double input;
	do{
		printf("%s\n", prompt);
		scanf("%lf", &input);
		if(input < min) printf("Must be at least %lf\n",min);
		if(input > max) printf("Must be < %lf\n", max);
	}while(input < min || input > max);
	return input;
}
#endif