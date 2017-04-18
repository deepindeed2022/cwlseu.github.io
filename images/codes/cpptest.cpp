#include <iostream>
using namespace std;
int main(int argc, char const *argv[])
{
	int i = 100;
	std::cout << (i++) << std::endl;
	std::cout << i << std::endl;
	return 0;
}