#pragma once
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <time.h>
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif
#include "opencl-func.h"
#include "opencl-tool.h"
using namespace std;

enum
{
    NK_SUCCESS = 0,
    NK_FAIL
};

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

inline bool __make_kernel(cl_kernel &kernel, cl_program &program, const char *kernal_name)
{
    cl_int err;
    kernel = clCreateKernel(program, kernal_name, &err); //引号中名称换为改写后的kernel名称
    if (kernel == 0)
    {
        cout << "Can't load kernel\n";
        // clReleaseContext(context);
        // clReleaseProgram(program);
        // clReleaseCommandQueue(queue);
        return NK_FAIL;
    }
    return NK_SUCCESS;
}

inline bool __make_program(cl_program &program, cl_context &context, cl_device_id &device, /*cl_command_queue &queue, */ const std::string &src)
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
        return NK_FAIL;
    }
    return NK_SUCCESS;
}

class Manager
{
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    unordered_map<const string *, cl_program> program_record; // 程序的记录
    unordered_map<const string *, cl_kernel> kernel_record;   // kernel的记录
    bool init = false;
    Manager();

public:
    Manager(const Manager &) = delete; // 禁止复制
    ~Manager();
    static Manager &instance(); // 可以单例也可以不
    // operator bool() { return init; } // 判断状态，NK_SUCCESS是0，就很难受，不用这个了
    bool inited() { return init; }

    cl_context get_context() { return context; }
    cl_device_id get_device() { return device; }
    cl_command_queue get_queue() { return queue; }

    // 用这个接口，kernel不能实现自动释放，只能由Manager来释放
    int make_kernel(cl_kernel &kernel, const string &kernel_name, const string &program_src);

private:
    int make_kernel_program(cl_program &program, const string &program_src);
};

inline Manager &Manager::instance()
{
    static Manager manager; // 单例
    return manager;
}
inline Manager::Manager()
{
    init = NK_SUCCESS == my_ClDeviceInitializer(context, device, queue);
}
inline int Manager::make_kernel(cl_kernel &kernel, const string &kernel_name, const string &program_src)
{
    {
        // 尝试从记录里获得
        auto it = kernel_record.find(&program_src);
        if (it != kernel_record.end())
        {
            cout << &kernel_name << " get kernel from record\n";
            kernel = (*it).second;
            return NK_SUCCESS;
        }
    }
    cl_program program;
    if (NK_SUCCESS != make_kernel_program(program, program_src))
        return NK_FAIL;
    if (NK_SUCCESS != __make_kernel(kernel, program, kernel_name.c_str()))
        return NK_FAIL;
    kernel_record.insert(make_pair(&kernel_name, kernel));
    return NK_SUCCESS;
};

inline int Manager::make_kernel_program(cl_program &program, const string &program_src)
{

    {
        // 尝试从记录里获得
        auto it = program_record.find(&program_src);
        if (it != program_record.end())
        {
            cout << &program_src << " get program from record\n";
            program = (*it).second;
            return NK_SUCCESS;
        }
    }
    if (NK_SUCCESS != __make_program(program, context, device, /*cl_command_queue &queue, */ program_src))
        return NK_FAIL;
    program_record.insert(make_pair(&program_src, program));
    return NK_SUCCESS;
}

inline Manager::~Manager()
{
    if (!init)
        return;
    for (auto kernel_pair : kernel_record)
        clReleaseKernel(kernel_pair.second);
    for (auto program_pair : program_record)
        clReleaseProgram(program_pair.second);
    clReleaseContext(context);
    clReleaseCommandQueue(queue);
}