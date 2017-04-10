#include <stdio.h>
int main(int argc, char const *argv[])
{
	for(auto i = 0; i < 10; ++i)
	{
#if defined(__GNUC__) || defined(__GNUG__)
	printf("the number of trailing 0-bits in %d is %d \n", i,  __builtin_ctz(i));
	printf("%d have %d 1-bits\n", i, __builtin_popcount(i));
	printf("%d parity value: %d\n", i, __builtin_parity(i));
	printf("%d swap32 %d\n", i, __builtin_bswap32(i));
#endif
	}
#if defined(__GNUC__) || defined(__GNUG__)
	printf("test __builtin___clear_cache\n");
	char* a;
	for(int i = 0; i < 10; ++i)
	{
		__builtin___clear_cache(a, a + 4096);
		a = new char[4096];
		delete[] a;
	}
#endif
	return 0;
}