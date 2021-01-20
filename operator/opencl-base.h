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
    // 运行环境
public:
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    bool is_good = false; // 状态
    static ClSystem *singleton()
    {
        // 单例
        static ClSystem instance;
        return &instance;
    }
    static ClSystem *construct()
    {
        ClSystem &instance = *singleton();
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
        MY_DEBUG(queue)
        if (is_good)
        {
            clReleaseContext(context);
            clReleaseCommandQueue(queue);
        }
    }
};

class ProgramManager
{
    // 管理 program
public:
    cl_program program;
    bool is_good = false; // 状态

    static ProgramManager *make_kernel_program(const string &program_src)
    {
        static unordered_map<const string *, shared_ptr<ProgramManager>> record;
        {
            // 尝试获得过去的记录
            auto it = record.find(&program_src);
            if (it != record.end())
            {
                ProgramManager *programM = (*it).second.get();
                return programM->is_good ? programM : nullptr;
            }
        }
        auto clsys = ClSystem::singleton();
        if (!clsys)
            return nullptr;
        ProgramManager *programM = new ProgramManager(clsys->context, clsys->device, program_src);
        if (programM->is_good)
        {
            record.insert(std::make_pair(&program_src, (programM)));
            return programM;
        }
        else
        {
            delete programM;
            return nullptr;
        }
    }
    ProgramManager(const ProgramManager &_Right) = delete; // 禁止复制
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
    // 管理 kernel
public:
    cl_kernel kernel;
    bool is_good = false; // 状态
    static KernelManager *make_kernel(const string &kernel_name, const string &program_src)
    {
        static unordered_map<const string *, shared_ptr<KernelManager>> record;
        {
            auto it = record.find(&kernel_name);
            if (it != record.end())
            {
                KernelManager *kernelM = (*it).second.get();
                return kernelM && kernelM->is_good ? kernelM : nullptr;
            }
        }
        ProgramManager *programM = ProgramManager::make_kernel_program(program_src);
        if (!programM || !programM->is_good)
            return nullptr;
        KernelManager *kernelM = new KernelManager(programM->program, kernel_name.c_str());
        if (kernelM->is_good)
        {
            record.insert(std::make_pair(&kernel_name, (kernelM)));
            return kernelM;
        }
        else
        {
            delete kernelM;
            return nullptr;
        }
    }
    KernelManager(cl_program &program, const char *kernal_name)
    {
        cl_int err;
        kernel = clCreateKernel(program, kernal_name, &err); //引号中名称换为改写后的kernel名称

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

inline void __setArgs(cl_kernel kernel, int index) {}
template <typename _First, typename... _Args>
void __setArgs(cl_kernel kernel, int index, _First &first, _Args &... args)
{
    clSetKernelArg(kernel, index, sizeof(first), &first);
    __setArgs(kernel, index + 1, args...);
}
// 方便地设置参数
template <typename... _Args>
void setArgs(cl_kernel kernel, _Args &&... args)
{
    __setArgs(kernel, 0, args...);
}

class MemManager
{
    // 管理几个内存
public:
    vector<cl_mem> mems;            // 或许可以用map实现，用字符串索引
    bool is_good;                   // 状态
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
        // 清空
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
        // 重置
        clear();
        is_good = true;
    }
    ~MemManager()
    {
        reset();
    }
};
