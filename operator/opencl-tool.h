#pragma once
#include <string>
using namespace std;

#define MY_DEBUG(x)                      \
  {                                      \
    cout << #x << " is " << (x) << endl; \
  }

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



template <typename First, typename... _Args>
const string &make_kernel_name(const char *basic_name)
{
  // 给basic_name拼接类型信息
  static string name = make_kernel_name<_Args...>(basic_name) + "_" + my_GetFullName<DType>();
  // 递归实现虽然会有不必要的静态变量，但是消耗的内存并不大
  return name;
}
template<>
const string &make_kernel_name<>(const char *basic_name)
{
  static string name(basic_name);
  cout << "Test make_kernel_name\n";
  return name;
}