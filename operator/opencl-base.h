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
        {
            return;
        }
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
        cl_program program = clCreateProgramWithSource(context, 1, &source, 0, 0);
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
        cl_kernel kernel = clCreateKernel(program, kernal_name, 0); //引号中名称换为改写后的kernel名称
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
void setArgs(cl_kernel kernel, int index){}
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

template <typename DType>
void my_ClKernelLauncher(Tensor<cpu, 1, DType> bias, Tensor<cpu, 2, DType> data,
                         Tensor<cpu, 2, DType> out, Stream<cpu> *s,
                         string src)
{
#ifdef NK_TIMING_OPTION
    clock_t time_start = clock();
#endif

    cl_int err;
    cl_uint num;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    if (my_ClDeviceInitializer(context, device, queue))
    {
        return;
    }

    // kernel编写和编译
    const char *source = src.c_str();

    cl_program program = clCreateProgramWithSource(context, 1, &source, 0, 0);
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
        clReleaseContext(context);
        clReleaseProgram(program);
        clReleaseCommandQueue(queue);
        return;
    }

    //一个 OpenCL kernel 中可能有很多函数，这里获得函数的进入点。
    cl_kernel tempkernel = clCreateKernel(program, "add_bias_kernel", 0); //引号中名称换为改写后的kernel名称
    if (tempkernel == 0)
    {
        cout << "Can't load kernel\n";
        clReleaseContext(context);
        clReleaseProgram(program);
        clReleaseCommandQueue(queue);
        return;
    }
    size_t N = out.shape_[0] * out.shape_[1];
    size_t bias_N = bias.shape_[0];
    int lead_dim = data.size(0);
    int bias_length = bias.shape_[0];
    std::cout << "----------------------:" << sizeof(DType) << "  " << sizeof(float) << std::endl;

    /*
    Create device buffers(调用clCreateBuffer函数）
　　　　Buffer中保存的是数据对象，就是设备执行程序需要的数据保存在其中。
　　　　Buffer由上下文conetxt创建，这样上下文管理的多个设备就会共享Buffer中的数据。
       所转的kernel有几个参数需要创建几个Buffer，另外再加上需要创建结果存储的Buffer。
       结果存放在创建的cl_res,此处注意其中数据类型也要相应修改，即sizeof(cl_int)，例如：若为float则为sizeof(cl_float)
    */
    cl_mem cl_bias = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(DType) * bias_N, bias.dptr_, NULL);
    cl_mem cl_mat = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(DType) * N, out.dptr_, NULL);

    if (cl_bias == 0 || cl_mat == 0)
    {
        cout << "Can't create OpenCL buffer\n";
        clReleaseCommandQueue(queue);
        clReleaseProgram(program);
        clReleaseContext(context);
        clReleaseKernel(tempkernel);
        clReleaseMemObject(cl_bias);
        clReleaseMemObject(cl_mat);
        return;
    }

    //要执行kernel，只需要先设定kernel的参数，这里有四个参数。
    clSetKernelArg(tempkernel, 0, sizeof(cl_mem), &cl_mat);
    clSetKernelArg(tempkernel, 1, sizeof(cl_mem), &cl_bias);
    clSetKernelArg(tempkernel, 2, sizeof(int), &lead_dim);
    clSetKernelArg(tempkernel, 3, sizeof(int), &bias_length);

    // kernel执行与时间统计
    const int nthreads_addbias = 256;
    size_t work_size = lead_dim * nthreads_addbias;
#ifdef NK_TIMING_OPTION
    int loop = 1000;
    cl_event *timing_event = new cl_event[loop];
    cl_int err_code, kerneltimer;
    for (int j = 0; j < loop; j++)
    {

        err = clEnqueueNDRangeKernel(queue, tempkernel, 1, 0, &work_size, 0, 0, 0, &timing_event[j]);
        clFinish(queue);
    }
    clock_t time_end = clock();

    cl_ulong starttime, endtime;
    unsigned long elapsed = 0;
    for (int j = 0; j < loop; j++)
    {
        err_code = clGetEventProfilingInfo(timing_event[j], CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &starttime, NULL);
        kerneltimer = clGetEventProfilingInfo(timing_event[j], CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &endtime, NULL);
        elapsed += (unsigned long)(endtime - starttime);
    }
#else
    cout << "test nullptr\n";
    err = clEnqueueNDRangeKernel(queue, tempkernel, 1, nullptr, &work_size, nullptr, 0, nullptr, nullptr);
    clFinish(queue);
#endif

    //执行结果在OpenCL设备内存中，所以要取回结果到cpu中
    if (err == CL_SUCCESS)
    {
        // 从GPU取回结果
        err = clEnqueueReadBuffer(queue, cl_mat, CL_TRUE, 0, sizeof(DType) * N, out.dptr_, 0, 0, 0);
        cout << out.dptr_[0] << "  " << out.dptr_[1] << endl;
        if (err != CL_SUCCESS)
        {
            cout << "Can't run kernel or read back data\n";
        }
    }

#ifdef NK_TIMING_OPTION
    for (int j = 0; j < loop; j++)
    {
        clReleaseEvent(timing_event[j]);
    }
    delete[] timing_event;
    cout << "opencl time use:" << 1000 * (time_end - time_start) / (double)CLOCKS_PER_SEC << "us" << endl;
    cout << "opencl kernel time use:" << elapsed / 1000.0 / 1000 << "us" << endl;
#endif

    clReleaseMemObject(cl_bias);
    clReleaseMemObject(cl_mat);
    clReleaseKernel(tempkernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return;
}
