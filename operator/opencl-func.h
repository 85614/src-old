
#pragma once
/*
 * 这个头文件把opencl一些函数的声明列出来，方便编辑器提示
 * 
 */
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif
cl_int  clEnqueueNDRangeKernel (cl_command_queue command_queue, 
                            cl_kernel kernel, 
                            cl_uint  work_dim,    // Choose if we are using 1D, 2D or 3D work-items and work-groups
                            const size_t *global_work_offset,
                            const size_t *global_work_size,   // The total number of work-items (must have work_dim dimensions)
                            const size_t *local_work_size,     // The number of work-items per work-group (must have work_dim dimensions)
                            cl_uint num_events_in_wait_list, 
                            const cl_event *event_wait_list, 
                            cl_event *event);

cl_int clSetKernelArg(cl_kernel kernel, // Which kernel

                      cl_uint arg_index, // Which argument

                      size_t arg_size, // Size of the next argument (not of the value pointed by it!)

                      const void *arg_value); // Value

// Returns the cl_mem object referencing the memory allocated on the device

cl_mem clCreateBuffer(cl_context context, // The context where the memory will be allocated

                      cl_mem_flags flags,

                      size_t size, // The size in bytes

                      void *host_ptr,

                      cl_int *errcode_ret);

// CL_MEM_READ_WRITE

// CL_MEM_WRITE_ONLY

// CL_MEM_READ_ONLY

// CL_MEM_USE_HOST_PTR

// CL_MEM_ALLOC_HOST_PTR

// CL_MEM_COPY_HOST_PTR – 从 host_ptr处拷贝数据

cl_int clEnqueueReadBuffer(cl_command_queue command_queue,
                           cl_mem buffer,         // from which buffer
                           cl_bool blocking_read, // whether is a blocking or non-blocking read
                           size_t offset,         // offset from the beginning
                           size_t cb,             // size to be read (in bytes)
                           void *ptr,             // pointer to the host memory
                           cl_uint num_events_in_wait_list,
                           const cl_event *event_wait_list,
                           cl_event *event);
