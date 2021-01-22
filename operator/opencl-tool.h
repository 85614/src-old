#pragma once
#include <string>
using namespace std;

#define MY_DEBUG(x) (void)(cout << #x " is " << (x) << endl);

// 返回值为右值，为const没有用，而且还让编译器不能进行返回值优化
inline string my_GetFullName(const char *name)
{
  int status = -1;
  char *fullName = abi::__cxa_demangle(name, NULL, NULL, &status);
  const char *const demangledName = (status == 0) ? fullName : name;
  string ret_val(demangledName);
  free(fullName);
  return ret_val;
}
template <typename _Ty>
inline const string &my_GetFullName()
{
  static string name = my_GetFullName(typeid(_Ty).name());
  return name;
}

inline int my_get_load_type(size_t N)
{
  using namespace mshadow;
  if (N % 8 == 0)
  {
    return kFloat64;
  }
  else if (N % 4 == 0)
  {
    return kFloat32;
  }
  else if (N % 2 == 0)
  {
    return kFloat16;
  }
  else
  {
    return kUint8;
  }
}

template <typename... _Args>
string my_strcat(const string &first, _Args &&... args)
{ // 字符串连接
  return first + my_strcat(std::forward<_Args>(args)...);
}
template <>
inline string my_strcat(const string &first)
{
  return first;
}

template <typename... _Args>
const string &make_kernel_name(const char *basic_kernel_name)
{
  // 获取添加类型信息的kernel名，得到静态string变量
  cout << "Test make_kernel_name\n";
  static string name = basic_kenel_name;
  // static string name = my_strcat(basic_kernel_name, ("_" + my_GetFullName<_Args>())...);
  // 应该不会有指针类型，数组类型什么的吧
  return name;
}