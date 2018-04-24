#include <iostream>

void func(float a) {
	std::cout << "float func:" << a << std::endl;
}

void func(int a) {
	std::cout << "int func:" << a << std::endl;

}
template <class T>
void func(T a) {
	std::cout << "template func:" << a << std::endl;
}


int main(int argc, char const *argv[])
{
	int ia = 1;
	func(ia);
	func<int>(ia);

	float fb = 2;
	func(fb);
	func<float>(fb);
	double db = 3;
	func(db);
	func<double>(db);
	return 0;
}