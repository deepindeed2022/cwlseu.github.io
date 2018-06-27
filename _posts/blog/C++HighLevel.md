

## 调试
```c
#define stub  fprintf(stderr, "error param in %s:%s:%d\n",  __FUNCTION__, __FILE__, __LINE__);
```

## mutable关键字用来解决常函数中不能修改对象的数据成员的问题

##
这是因为结构体内存分配有自己的对齐规则，结构体内存对齐默认的规则如下：
1、 分配内存的顺序是按照声明的顺序。
2、 每个变量相对于起始位置的偏移量必须是该变量类型大小的整数倍，不是整数倍空出内存，直到偏移量是整数倍为止。
3、 最后整个结构体的大小必须是里面变量类型最大值的整数倍。

内存对其 https://www.cnblogs.com/suntp/p/MemAlignment.html

## 使用trait的陷阱
有的时候
```cpp
template <typename T>
struct image_crop_trait;

template <typename>
struct image_crop_trait<float>{

};
```

```cpp
template <>
struct image_crop_trait{};
struct image_crop_trait<float>{

};
```