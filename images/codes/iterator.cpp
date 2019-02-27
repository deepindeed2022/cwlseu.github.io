#include <iostream>
#include <cstdlib>
// 定义迭代器类型
struct Iterator_tag {};
struct Forward_Iter_tag : public Iterator_tag {};
struct Bidirectional_Iter_tag: public Iterator_tag {};
struct Random_Iter_tag: public Iterator_tag {};

template <typename T>
class Iterator
{
public:
    typedef Iterator_tag iterator_category;  //注意这里
    Iterator(T* inT): pointer(inT) {}
    virtual ~Iterator() {}
    T* get() const 
    {
        return pointer;
    }
    operator T* () const {  return pointer; } 
    T operator * () const { return *pointer; }
    friend T* operator + (const Iterator& it, const int i) 
    {
        return it.doPlus(i);
    }
    friend T* operator + (const int i, const Iterator& it) 
    {
        return it.doPlus(i);
    }
    friend T* operator - (const Iterator& it, const int i) 
    {
        return it.doPlus(-i);
    }
    friend T* operator - (const int i, const Iterator& it) 
    {
        return it.doPlus(-i);
    }
private:
    T* pointer;

    T* doPlus(const int i) const
    {
        return pointer + i;
    }
};


template<typename Iter>
struct Iterator_traits
{
    typedef typename Iter::iterator_category iterator_category;
};

//针对内置指针的偏特化
template<typename Iter>
struct Iterator_traits<Iter*>
{
    typedef Random_Iter_tag iterator_category;
};

template<typename T>
class Forward_Iter: public Iterator<T>
{

public:
    typedef Forward_Iter_tag iterator_category;
    Forward_Iter(T* inT):Iterator<T>(inT) {}
};

template<typename T>
class Bidirectional_Iter: public Iterator<T>
{
public:
    typedef Bidirectional_Iter_tag iterator_category;
    Bidirectional_Iter(T* inT):Iterator<T>(inT) {}
};

template<typename T>
class Random_Iter: public Iterator<T>
{

public:
    typedef Random_Iter_tag iterator_category;
    Random_Iter(T* inT):Iterator<T>(inT) {}
};

template<typename Iter, typename Dist>
void doAdvance(Iter& iter, Dist dist, Forward_Iter_tag)
{
    if(dist >=0 )
        while(dist--) iter = iter + 1;
    else 
    {
        std::cerr << "error!" << std::endl;
        abort();
    }
}

template<typename Iter, typename Dist>
void doAdvance(Iter& iter, Dist dist, Bidirectional_Iter_tag)
{
    if(dist >= 0)
        while(dist--) iter = iter+1;
    else 
        while(dist++) iter = iter - 1;
}
template<typename Iter, typename Dist>
void doAdvance(Iter& iter, Dist dist, Random_Iter_tag)
{
    iter = iter + dist;
}
template<typename Iter, typename Dist>
void doAdvance(Iter& iter, Dist dist, Iterator_tag)
{
    iter = iter + dist;
}

template<typename Iter, typename Dist>
void advance_test(Iter& iter, Dist dist)
{
    doAdvance(iter, dist, 
        typename Iterator_traits<Iter>::iterator_category());
}

int main(int argc, char const *argv[])
{
    int a[1024];
    for(int i = 0;i < 1024; ++i) {
        a[i] = i;
    }
    Forward_Iter<int> forward_iter(a);
    forward_iter = forward_iter + 8;
    std::cout << *forward_iter << std::endl;
    Bidirectional_Iter<int> bd_iter(a);

    bd_iter = bd_iter + 8;
    advance_test(bd_iter, 8);
    std::cout << *bd_iter << std::endl;
    advance_test(bd_iter, -4);
    std::cout<<*bd_iter<<std::endl;

    int* b = a + 9;
    advance_test(b, 10);
    std::cout << *b << std::endl;
    return 0;
}