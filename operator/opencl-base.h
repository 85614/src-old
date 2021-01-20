#pragma once
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <time.h>
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif
#include "opencl-func.h"
#include "opencl-tool.h"
using namespace std;

// 获取设备信息，初始化context和queue
inline bool
my_ClDeviceInitializer(cl_context &context, cl_device_id &device, cl_command_queue &queue)
{
    //首先获得系统上所有的OpenCL platform，调用两次clGetPlatformIDs函数，第一次获取可用的平台数量，第二次获取一个可用的平台ID,不用动即可。
    cl_int err;
    cl_uint num;
    err = clGetPlatformIDs(0, 0, &num);
    if (err != CL_SUCCESS)
    {
        cout << err << "Unable to get platforms\n";
        return 1;
    }
    vector<cl_platform_id> platforms(num);
    err = clGetPlatformIDs(num, &platforms[0], &num);
    if (err != CL_SUCCESS)
    {
        cout << "Unable to get platform ID\n";
        return 1;
    }
    /*
    调用clCreateContextFromType创建一个上下文（context），即OpenCL的Platform上共享和使用资源的环境，包括kernel、device、memory objects、command queue等。
    使用中一般一个Platform对应一个Context。不用动即可。
    */
    cl_context_properties prop[] = {CL_CONTEXT_PLATFORM,
                                    reinterpret_cast<cl_context_properties>(platforms[0]), 0};
    context = clCreateContextFromType(prop, CL_DEVICE_TYPE_DEFAULT, NULL,
                                      NULL, NULL);
    if (context == 0)
    {
        cout << "Can't create OpenCL context\n";
        return 1;
    }

    size_t cb;
    clGetContextInfo(context, CL_CONTEXT_DEVICES, 0, NULL, &cb);
    vector<cl_device_id> devices(cb / sizeof(cl_device_id));
    clGetContextInfo(context, CL_CONTEXT_DEVICES, cb, &devices[0], 0);
    clGetDeviceInfo(devices[0], CL_DEVICE_NAME, 0, NULL, &cb);
    string devname;
    devname.resize(cb);
    clGetDeviceInfo(devices[0], CL_DEVICE_NAME, cb, &devname[0], 0);
    device = devices[0];
    cout << "Device:" << devname.c_str() << "\n";

    /*
    Create a command queue(调用clCreateCommandQueue函数）一个设备device对应一个command queue。
    上下文conetxt将命令发送到设备对应的command queue，设备就可以按顺序执行命令队列里的命令。
    */
    queue = clCreateCommandQueue(context, devices[0], CL_QUEUE_PROFILING_ENABLE, 0);
    if (queue == 0)
    {
        cout << "Can't create command queue\n";
        clReleaseContext(context);
        return 1;
    }
    return 0;
}

class ClSystem
{
public:
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    bool is_good = false;
    static ClSystem &singleton()
    {
        static ClSystem instance;
        return instance;
    }
    static ClSystem *construct()
    {
        ClSystem &instance = singleton();
        if (instance.is_good)
            return &instance;
        return nullptr;
    }
    static void destruct(ClSystem *clsys)
    {
        // do nothing
    }

private:
    ClSystem()
    {
        // cl_int err;
        // cl_uint num;
        if (my_ClDeviceInitializer(context, device, queue))
            is_good = false;
        else
            is_good = true;
    }

public:
    ~ClSystem()
    {
        if (is_good)
        {
            clReleaseContext(context);
            clReleaseCommandQueue(queue);
        }
    }
};

class ProgramManager
{
public:
    cl_program program;
    bool is_good = false;

    ProgramManager(cl_context &context, cl_device_id &device, /*cl_command_queue &queue, */ const std::string &src)
    {
        cl_int err;
        const char *source = src.c_str();
        // MY_DEBUG(source);
        program = clCreateProgramWithSource(context, 1, &source, 0, 0);
        err = clBuildProgram(program, 0, 0, 0, 0, 0);
        if (err != CL_SUCCESS)
        {
            cout << "Can't load or build program\n";
            if (err == CL_BUILD_PROGRAM_FAILURE)
            {
                size_t log_size;
                clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
                char *log = (char *)malloc(log_size);
                clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
                fprintf(stderr, "%s\n", log);
                free(log);
            }
            // clReleaseContext(context);
            clReleaseProgram(program);
            // clReleaseCommandQueue(queue);
            is_good = false;
            return;
        }
        is_good = true;
    }
    ~ProgramManager()
    {
        if (is_good)
            clReleaseProgram(program);
    }
};
class KernelManager
{
public:
    cl_kernel kernel;
    bool is_good = false;

    KernelManager(cl_program &program, const char *kernal_name)
    {
        cl_int err;
        kernel = clCreateKernel(program, kernal_name, &err); //引号中名称换为改写后的kernel名称
        MY_DEBUG(err);
        if (kernel == 0)
        {
            cout << "Can't load kernel\n";
            // clReleaseContext(context);
            // clReleaseProgram(program);
            // clReleaseCommandQueue(queue);
            return;
        }
        is_good = true;
    }

    ~KernelManager()
    {
        if (is_good)
            clReleaseKernel(kernel);
    }
};
inline void setArgs(cl_kernel kernel, int index) {}
template <typename _First, typename... _Args>
void setArgs(cl_kernel kernel, int index, _First &first, _Args &... args)
{
    clSetKernelArg(kernel, index, sizeof(first), &first);
    setArgs(kernel, index + 1, args...);
}

template <typename... _Args>
void setArgs(cl_kernel kernel, _Args &&... args)
{
    setArgs(kernel, 0, args...);
}
class MemManager
{
public:
    vector<cl_mem> mems;
    bool is_good;
    void addMem(cl_context context, // The context where the memory will be allocated
                cl_mem_flags flags,
                size_t size, // The size in bytes
                void *host_ptr)
    {
        if (!is_good)
            return;
        cl_int errcode_ret;
        cl_mem mem = clCreateBuffer(context, flags, size, host_ptr, &errcode_ret);
        mems.push_back(mem);
        if (mem == 0 || errcode_ret != CL_SUCCESS) // 这里是不是和CL_SUCCESS比较没有去确定
        {
            is_good = false;
        }
        is_good = true;
    }
    void clear()
    {
        for (cl_mem &mem : mems)
            if (mem)
            {
                clReleaseMemObject(mem);
                mem = 0;
            }
        mems.clear();
    }
    void reset()
    {
        clear();
        is_good = true;
    }
    ~MemManager()
    {
        reset();
    }
};

