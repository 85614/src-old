/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * Copyright (c) 2015 by Contributors
 * \file fully_connect_op-inl.h
 * \brief fully connect operator and symbol
*/
#ifndef MXNET_OPERATOR_NN_FULLY_CONNECTED_INL_H_
#define MXNET_OPERATOR_NN_FULLY_CONNECTED_INL_H_
#define MY_DEBUG(x) {cout << #x << " is " << (x) << endl;}

#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <map>
#include <vector>
#include <string>
#include <utility>
#include <limits>
#include <algorithm>
#include "../operator_common.h"
#include "../elemwise_op_common.h"
#include "../linalg.h"
#include "../../common/utils.h"
#include "../tensor/broadcast_reduce_op.h"


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
using namespace std;
namespace mxnet {
namespace op {

// Declare enumeration of input order to make code more intuitive.
// These enums are only visible within this header
namespace fullc {
enum FullyConnectedOpInputs {kData, kWeight, kBias};
enum FullyConnectedOpResource {kTempSpace};
enum FullyConnectedOpOutputs {kOut};
enum FullyConnectedGradGradOutputs { kOyGrad, kXGradGrad, kWGradGrad, kBGradGrad };
enum GradGradInputs { kOxGrad, kOwGrad, };
enum GradGradInputsBias { kObGrad = 2, kOyBias, };
enum GradGradInputsNoBias { kOy = 2, };
}  // namespace fullc

namespace quantized_fullc {
enum QuantizedFCInputMinMax {kDataMin, kDataMax, kWeightMin, kWeightMax, kBiasMin, kBiasMax};
enum QuantizedFCOutputs {kOut, kOutMin, kOutMax};
}  // quantized_fullc


struct FullyConnectedParam : public dmlc::Parameter<FullyConnectedParam> {
  int num_hidden;
  bool no_bias;
  bool flatten;

  DMLC_DECLARE_PARAMETER(FullyConnectedParam) {
    // TODO(bing) add support for boolean
    DMLC_DECLARE_FIELD(num_hidden).set_lower_bound(1)
    .describe("Number of hidden nodes of the output.");
    DMLC_DECLARE_FIELD(no_bias).set_default(false)
    .describe("Whether to disable bias parameter.");
    DMLC_DECLARE_FIELD(flatten).set_default(true)
    .describe("Whether to collapse all but the first axis of the input data tensor.");
  }
  bool operator==(const FullyConnectedParam& other) const {
    return this->num_hidden == other.num_hidden &&
           this->no_bias == other.no_bias &&
           this->flatten == other.flatten;
  }
};

/**
 * Flatten additional dimensions after the first
 * @tparam xpu
 * @tparam DType
 * @param tblob
 * @param ctx
 * @return 2 Dimensional Tensor with upper shapes collapsed
 */
template<typename xpu, typename DType>
Tensor<xpu, 2, DType> FlattenAs2DTail(const TBlob& tblob, const OpContext& ctx) {
  const TShape& shape = tblob.shape_;
  Stream<xpu> *stream = ctx.get_stream<xpu>();
  return tblob.get_with_shape<xpu, 2, DType>(
      Shape2(shape[0], shape.ProdShape(1, shape.ndim())), stream);
}

/**
 * Flatten dimensions except last
 * @tparam xpu
 * @tparam DType
 * @param tblob
 * @param ctx
 * @return 2 Dimensional tensor with front shapes collapsed
 */
template<typename xpu, typename DType>
Tensor<xpu, 2, DType> FlattenAs2DHead(const TBlob& tblob, const OpContext& ctx) {
  const TShape& shape = tblob.shape_;
  Stream<xpu> *stream = ctx.get_stream<xpu>();
  return tblob.get_with_shape<xpu, 2, DType>(
      Shape2(shape.ProdShape(0, shape.ndim()-1), shape[shape.ndim()-1]), stream);
}

// template<typename DType>
// void AddBias(Tensor<cpu, 1, DType> bias, Tensor<cpu, 2, DType> data,
//              Tensor<cpu, 2, DType> out, Stream<cpu>*) {
//   using namespace mshadow;
//   using namespace mshadow::expr;
//   out += repmat(bias, data.size(0));
// }


#if defined(__CUDACC__)

namespace {
  constexpr int nthreads_addbias = 256;
  constexpr int nthreads_addbiasgrad_phase1 = 512;
  constexpr int nthreads_addbiasgrad_phase2 = 128;
  constexpr int threads_per_warp = 32;

  inline int ceil_div(int x, int y) {
    return (x + y - 1) / y;
  }
}  // namespace

template <typename DType, typename LType>
__global__ void add_bias_kernel(DType* mat, DType* bias, size_t lead_dim, size_t bias_length) {
  __shared__ LType scratch[nthreads_addbias * 2];
  const index_t N = bias_length * sizeof(DType)/sizeof(LType);
  const index_t base = blockIdx.x * N;
  LType* const mat_aligned = reinterpret_cast<LType*>(mat) + base;
  const LType* const bias_aligned = reinterpret_cast<LType*>(bias);
  LType* const scratch_bias_load = scratch + threadIdx.x;
  DType* const scratch_bias = reinterpret_cast<DType*>(scratch_bias_load);
  LType* const scratch_mat_load = scratch_bias_load + nthreads_addbias;
  DType* const scratch_mat = reinterpret_cast<DType*>(scratch_mat_load);
  for (index_t i = threadIdx.x; i < N; i += blockDim.x) {
    *scratch_bias_load = bias_aligned[i];
    *scratch_mat_load = mat_aligned[i];
#pragma unroll
    for (int j = 0; j < sizeof(LType)/sizeof(DType); ++j) {
      scratch_mat[j] += scratch_bias[j];
    }
    mat_aligned[i] = *scratch_mat_load;
  }
}

template<typename DType>
void AddBias(Tensor<gpu, 1, DType> bias, Tensor<gpu, 2, DType> data,
             Tensor<gpu, 2, DType> out, Stream<gpu>* s) {
    int ltype = mxnet::common::cuda::get_load_type(bias.shape_[0] * sizeof(DType));
    MXNET_LOAD_TYPE_SWITCH(ltype, LType, {
    add_bias_kernel<DType, LType><<<data.size(0),
                                    nthreads_addbias,
                                    0,
                                    Stream<gpu>::GetStream(s)>>>(out.dptr_,
                                                                 bias.dptr_,
                                                                 data.size(0),
                                                                 bias.shape_[0]);
    });
}

#endif  // __CUDACC__


#ifndef MY_OPENCLTEST
#define MY_OPENCLTEST


/*
把改写的OpenCL kernel 封装为Op的具体实现，其中主要内容即为OpenCL编程过程，即编写一个OpenCl代码时的所进行的步骤，和CUDA是无关的，
*/

// 获取设备信息，初始化context和queue
inline bool my_ClDeviceInitializer(cl_context &context, cl_device_id &device, cl_command_queue &queue) {
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

inline int my_get_load_type(size_t N) {
  using namespace mshadow;
  if (N % 8 == 0) {
    return kFloat64;
  } else if (N % 4 == 0) {
    return kFloat32;
  } else if (N % 2 == 0) {
    return kFloat16;
  } else {
    return kUint8;
  }
}
// 返回值为右值，为const没有用，而且还让编译器不能做返回值优化
inline string my_GetFullName(const char* name)
{
    int status = -1;
    char* fullName = abi::__cxa_demangle(name, NULL, NULL, &status);
    const char* const demangledName = (status==0) ? fullName : name;
    string ret_val(demangledName);
    free(fullName);
    return ret_val;
}


template <typename DType>
void my_ClKernelLauncher(Tensor<cpu, 1, DType> bias, Tensor<cpu, 2, DType> data,
             Tensor<cpu, 2, DType> out, Stream<cpu>* s,
             string src
             ) { 
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
        if (err == CL_BUILD_PROGRAM_FAILURE) {
		    size_t log_size;
		    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
		    char *log = (char *) malloc(log_size);
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
    cl_kernel tempkernel = clCreateKernel(program, "add_bias_kernel", 0);//引号中名称换为改写后的kernel名称
    if (tempkernel == 0)
    {
        cout << "Can't load kernel\n";
        clReleaseContext(context);
        clReleaseProgram(program);
        clReleaseCommandQueue(queue);
        return;
    }
    size_t N = out.shape_[0]*out.shape_[1];
    size_t bias_N = bias.shape_[0];
    int lead_dim=data.size(0);
    int bias_length=bias.shape_[0];
    std::cout<<"----------------------:"<<sizeof(DType)<<"  "<<sizeof(float)<<std::endl;

    /*
    Create device buffers(调用clCreateBuffer函数）
　　　　Buffer中保存的是数据对象，就是设备执行程序需要的数据保存在其中。
　　　　Buffer由上下文conetxt创建，这样上下文管理的多个设备就会共享Buffer中的数据。
       所转的kernel有几个参数需要创建几个Buffer，另外再加上需要创建结果存储的Buffer。
       结果存放在创建的cl_res,此处注意其中数据类型也要相应修改，即sizeof(cl_int)，例如：若为float则为sizeof(cl_float)
    */
    cl_mem cl_bias = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(DType) * bias_N, const_cast<DType *>(bias.dptr_), NULL);
    cl_mem cl_mat = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(DType) * N, const_cast<DType *>(out.dptr_), NULL);

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
    size_t work_size = lead_dim*nthreads_addbias;
    #ifdef NK_TIMING_OPTION
        int loop = 1000;
        cl_event *timing_event = new cl_event[loop];
        cl_int err_code, kerneltimer;
        for(int j = 0; j < loop; j++) {
            err = clEnqueueNDRangeKernel(queue, tempkernel, 1, 0, &work_size, 0, 0, 0, &timing_event[j]);
            clFinish(queue);
        }
        clock_t time_end = clock();

        cl_ulong starttime, endtime;
        unsigned long elapsed = 0;
        for(int j = 0; j < loop; j++) {
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
        cout<<out.dptr_[0]<<"  "<<out.dptr_[1]<<endl;
        if (err != CL_SUCCESS)
        {
            cout << "Can't run kernel or read back data\n";
        }
    }

    #ifdef NK_TIMING_OPTION
        for(int j = 0; j < loop; j++) {
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
template<typename DType, typename LType>
std::string make_AddBias() {
    string DType_name = my_GetFullName(typeid(DType).name());
    string LType_name = my_GetFullName(typeid(LType).name());
    MY_DEBUG(LType_name);
    MY_DEBUG(DType_name);
    return "__kernel void add_bias_kernel(__global " + DType_name + "* mat, \
                              __global " + DType_name + "* bias, \
                              int lead_dim, int bias_length) {"
    "typedef "+LType_name+" LType;" // 放在函数内部主要是为了防止重定义
    "typedef " + DType_name + " DType;"
    "const int nthreads_addbias = 256; \
    LType scratch[512]; \
    const size_t N = bias_length * sizeof(DType)/sizeof(LType); \
    const size_t base = get_group_id(0) * N; \
    __global LType* const mat_aligned = (__global LType*)(mat) + base; \
    __global const LType* const bias_aligned = (__global LType*)(bias); \
    LType* const scratch_bias_load = scratch + get_local_id(0); \
    DType* const scratch_bias = (DType*)(scratch_bias_load); \
    LType* const scratch_mat_load = scratch_bias_load + nthreads_addbias; \
    DType* const scratch_mat = (DType*)(scratch_mat_load); \
    for (int i = get_local_id(0); i < N; i += get_local_size(0)) { \
        *scratch_bias_load = bias_aligned[i]; \
        *scratch_mat_load = mat_aligned[i]; \
        for (int j = 0; j < sizeof(LType)/sizeof(DType); ++j) { \
            scratch_mat[j] += scratch_bias[j]; \
        } \
        mat_aligned[i] = *scratch_mat_load; \
    } \
};";
}

// template<typename DType>
// void AddBias(Tensor<cpu, 1, DType> bias, Tensor<cpu, 2, DType> data,
//              Tensor<cpu, 2, DType> out, Stream<cpu>* s) {
//   my_ClKernelLauncher<DType>(bias,data,out,s);
  
// }
template<typename DType>
void AddBias(Tensor<cpu, 1, DType> bias, Tensor<cpu, 2, DType> data,
             Tensor<cpu, 2, DType> out, Stream<cpu>* s) {
    int ltype = my_get_load_type(bias.shape_[0] * sizeof(DType));
    // int ltype = mxnet::common::cuda::get_load_type(bias.shape_[0] * sizeof(DType));
    
    MXNET_LOAD_TYPE_SWITCH(ltype, LType, {
    string src = make_AddBias<DType, LType>();
    my_ClKernelLauncher<DType>(bias,data,out,s, src);
    // add_bias_kernel<DType, LType><<<data.size(0),
    //                                 nthreads_addbias,
    //                                 0,
    //                                 Stream<gpu>::GetStream(s)>>>(out.dptr_,
    //                                                              bias.dptr_,
    //                                                              data.size(0),
    //                                                              bias.shape_[0]);
    });
    
    

}
#endif  //MY_OPENCLTEST



template<typename xpu, typename DType>
void FCForward(const OpContext &ctx, const FullyConnectedParam &param,
               const std::vector<TBlob> &in_data, const std::vector<OpReqType> &req,
               const std::vector<TBlob> &out_data) {
  std::cout<<"++++++++++++++++fcforwardstart"<<std::endl;
  using namespace mshadow;
  using namespace mshadow::expr;
  if (req[fullc::kOut] == kNullOp) return;
  CHECK_EQ(req[fullc::kOut], kWriteTo);
  // TODO(bing): check the BLAS Handle, be careful
  // maybe need blas handle from context
  // TODO(bing): judge shape to remove flatten op
  Stream<xpu> *s = ctx.get_stream<xpu>();
#if defined(__CUDACC__)
  CHECK_EQ(s->blas_handle_ownership_, Stream<xpu>::OwnHandle)
      << "Must init CuBLAS handle in stream";
#endif  // __CUDACC__
  Tensor<xpu, 2, DType> wmat = in_data[fullc::kWeight].get<xpu, 2, DType>(s);
  Tensor<xpu, 2, DType> data, out;
  if (!param.flatten) {
    data = FlattenAs2DHead<xpu, DType>(in_data[fullc::kData], ctx);
    out = FlattenAs2DHead<xpu, DType>(out_data[fullc::kOut], ctx);
  } else {
    data = FlattenAs2DTail<xpu, DType>(in_data[fullc::kData], ctx);
    out = FlattenAs2DTail<xpu, DType>(out_data[fullc::kOut], ctx);
  }

  CHECK_EQ(data.shape_[1], wmat.shape_[1])
    << "Incomplete weight tensor detected: weight.data().shape[1] != prod(data.data().shape[1:])."
       " This is not supported by FCForward. If weight is in row_sparse format,"
       " please make sure all row ids are present.";
  // Legacy approach shown here for comparison:
  //   out = dot(data, wmat.T());
  linalg_gemm(data, wmat, out, false, true, s);
  if (!param.no_bias) {
    Tensor<xpu, 1, DType> bias = in_data[fullc::kBias].get_with_shape<xpu, 1, DType>(
      Shape1(wmat.shape_[0]), s);
    CHECK_EQ(bias.shape_[0], wmat.shape_[0])
      << "Incomplete bias tensor detected: bias.data().shape[1] != weight.data().shape[0]."
         " This is not supported by FCForward. If bias is in row_sparse format, please"
         " make sure all row ids are present.";
    AddBias(bias, data, out, s);
  }
  std::cout<<"++++++++++++++++fcforwardend"<<std::endl;
}

#if defined (__CUDACC__)

template<typename LType, typename DType, typename AType>
__global__ void AddBiasGradKernelPhase1(AType * temp_space, const DType* grad,
                                        const size_t lead_dim, const size_t other_dim) {
  constexpr int num_warps = nthreads_addbiasgrad_phase1 / threads_per_warp;
  const int values_per_read = sizeof(LType) >= sizeof(DType) ? sizeof(LType) / sizeof(DType) : 1;
  const size_t stride = lead_dim / values_per_read;
  __shared__ AType scratch[threads_per_warp * num_warps  * values_per_read];
  LType * my_scratch_load = &(reinterpret_cast<LType *>(scratch)[threadIdx.x]);
  DType * my_values_load = reinterpret_cast<DType *>(my_scratch_load);
  AType * my_values_acc = &(scratch[threadIdx.x * values_per_read]);
  AType acc[values_per_read];  // NOLINT(*)
#pragma unroll
  for (int i = 0; i < values_per_read; ++i) {
    acc[i] = 0;
  }
  const size_t offset = blockIdx.x * threads_per_warp;
  const int my_warp = threadIdx.x / threads_per_warp;
  const int my_id = threadIdx.x % threads_per_warp;
  const LType* aligned_grad = reinterpret_cast<const LType*>(grad);
  const int rows_per_block = (other_dim + gridDim.y - 1) / gridDim.y;
  const size_t start_row = my_warp + rows_per_block * blockIdx.y;
  const size_t end_row = min(other_dim, static_cast<size_t>(rows_per_block * (blockIdx.y + 1)));
  if (offset + my_id < stride) {
    for (size_t i = start_row; i < end_row; i += num_warps) {
      *my_scratch_load = aligned_grad[i * stride + offset + my_id];
#pragma unroll
      for (int j = 0; j < values_per_read; ++j) {
        acc[j] += static_cast<AType>(my_values_load[j]);
      }
    }
  }
  __syncthreads();
#pragma unroll
  for (int i = 0; i < values_per_read; ++i) {
    my_values_acc[i] = acc[i];
  }

  __syncthreads();

  for (int i = num_warps / 2; i > 0; i /= 2) {
    if (my_warp < i) {
      const int shared_offset = values_per_read * i * threads_per_warp;
#pragma unroll
      for (int j = 0; j < values_per_read; ++j) {
        my_values_acc[j] += my_values_acc[j + shared_offset];
      }
    }
    __syncthreads();
  }

  if (threadIdx.x < min(threads_per_warp * values_per_read,
                        static_cast<int>(lead_dim - values_per_read * offset))) {
    const size_t offset_out = values_per_read * offset +
                              blockIdx.y * lead_dim;
    temp_space[offset_out + threadIdx.x] = scratch[threadIdx.x];
  }
}

template <typename DType, typename AType>
__global__ void AddBiasGradKernelPhase2(const AType * temp_space, DType * out,
                                        int lead_dim, int n_blocks, OpReqType req) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < lead_dim) {
    AType acc = 0;
    for (int i = tid; i < lead_dim * n_blocks; i += lead_dim) {
      acc += temp_space[i];
    }
    KERNEL_ASSIGN(out[tid], req, static_cast<DType>(acc));
  }
}

template<typename DType>
void AddBiasGrad(const TBlob& in_grad,
                 Tensor<gpu, 2, DType> grad,
                 OpReqType req,
                 int num_hidden,
                 const OpContext& ctx) {
  if (req == kNullOp) return;
  using AType = typename mxnet_op::AccType<DType>::type;
  mshadow::Stream<gpu> *s = ctx.get_stream<gpu>();
  Tensor<gpu, 1, DType> gbias = in_grad.get<gpu, 1, DType>(s);
  TBlob grad_blob = TBlob(grad);
  TBlob gbias_blob = TBlob(gbias);
  mxnet::TShape x(1, 0);
  mxnet::TShape small;
  if (shape_assign(&gbias_blob.shape_, Shape2(num_hidden, 1))) {
    small = gbias_blob.shape_;
  } else {
    small = ReduceAxesShapeImpl(grad_blob.shape_, dmlc::optional<mxnet::TShape>(x), true, false);
  }
  const int N = small.Size();
  int ltype = mxnet::common::cuda::get_load_type(N * sizeof(DType));
  const int M = grad_blob.shape_.Size() / N;
  MXNET_LOAD_TYPE_SWITCH(ltype, LType, {
    const unsigned int blocks_x = ceil_div(N * sizeof(DType),
                                           threads_per_warp * sizeof(LType));
    const unsigned int preferred_number_of_blocks = 2 *
                                                    MultiprocessorCount(ctx.run_ctx.ctx.dev_id);
    const unsigned int blocks_y = std::max(preferred_number_of_blocks / blocks_x, 1u);
    const dim3 n_blocks = {blocks_x, blocks_y, 1};
    auto scratch_space = ctx.requested[fullc::kTempSpace]
                            .get_space_typed<gpu, 1, AType>(mshadow::Shape1(N * blocks_y), s);
    auto stream = mshadow::Stream<gpu>::GetStream(s);
    AddBiasGradKernelPhase1<LType><<<n_blocks,
                                     nthreads_addbiasgrad_phase1,
                                     0,
                                     stream>>>(scratch_space.dptr_,
                                               grad.dptr_, N, M);
    const int nblocks_phase2 = ceil_div(N, nthreads_addbiasgrad_phase2);
    AddBiasGradKernelPhase2<<<nblocks_phase2,
                              nthreads_addbiasgrad_phase2,
                              0,
                              stream>>>(scratch_space.dptr_,
                                        gbias.dptr_, N,
                                        blocks_y, req);
  });
}
#endif

template<typename DType>
void AddBiasGrad(const TBlob& in_grad,
                 Tensor<cpu, 2, DType> grad,
                 OpReqType req,
                 int num_hidden,
                 const OpContext& ctx) {
  mshadow::Stream<cpu> *s = ctx.get_stream<cpu>();
  Tensor<cpu, 1, DType> gbias = in_grad.get<cpu, 1, DType>(s);
  TBlob grad_blob = TBlob(grad);
  TBlob gbias_blob = TBlob(gbias);
  mxnet::TShape x(1, 0);
  mxnet::TShape small;
  if (shape_assign(&gbias_blob.shape_, Shape2(num_hidden, 1))) {
    small = gbias_blob.shape_;
  } else {
    small = ReduceAxesShapeImpl(grad_blob.shape_, dmlc::optional<mxnet::TShape>(x), true, false);
  }
  ReduceAxesComputeImpl<cpu, mshadow::red::sum, false, false,
                        mshadow_op::identity>(ctx, {grad_blob}, {req},
                                              {in_grad}, small);
}

template<typename xpu, typename DType>
void FCBackward(const OpContext &ctx, const FullyConnectedParam &param,
                const std::vector<TBlob> &out_grad, const std::vector<TBlob> &in_data,
                const std::vector<OpReqType> &req, const std::vector<TBlob> &in_grad) {
  using namespace mshadow;
  using namespace mshadow::expr;
  // TODO(bing): check the BLAS Handle, be careful
  //  maybe need blas handle from context
  Stream<xpu> *stream = ctx.get_stream<xpu>();
  Tensor<xpu, 2, DType> wmat = in_data[fullc::kWeight].get<xpu, 2, DType>(stream);
  Tensor<xpu, 2, DType> x, y_grad, x_grad;
  if (!param.flatten) {
    x = FlattenAs2DHead<xpu, DType>(in_data[fullc::kData], ctx);
    y_grad = FlattenAs2DHead<xpu, DType>(out_grad[fullc::kOut], ctx);
    x_grad = FlattenAs2DHead<xpu, DType>(in_grad[fullc::kData], ctx);
  } else {
    x = FlattenAs2DTail<xpu, DType>(in_data[fullc::kData], ctx);
    y_grad = FlattenAs2DTail<xpu, DType>(out_grad[fullc::kOut], ctx);
    x_grad = FlattenAs2DTail<xpu, DType>(in_grad[fullc::kData], ctx);
  }

#if defined(__CUDACC__)
  CHECK_EQ(stream->blas_handle_ownership_, Stream<xpu>::OwnHandle)
      << "Must init CuBLAS handle in stream";
#endif
  //  backprop
  CHECK_NE(req[fullc::kWeight], kWriteInplace) << "cannot write weight inplace";
  // gradient of weight
  Tensor<xpu, 2, DType> w_grad = in_grad[fullc::kWeight].get<xpu, 2, DType>(stream);
  // Legacy approach shown here for comparison:
  //   out = Assign(w_grad, req[fullc::kWeight], dot(grad.T(), data));
  linalg_gemm(y_grad, x, w_grad, true, false, stream, req[fullc::kWeight]);
  // gradient of bias
  if (!param.no_bias) {
      AddBiasGrad(in_grad[fullc::kBias], y_grad, req[fullc::kBias], param.num_hidden, ctx);
  }
  // gradient of data
  // Legacy approach shown here for comparison:
  //   Assign(x_grad, req[fullc::kData], dot(y_grad, wmat));
  linalg_gemm(y_grad, wmat, x_grad, false, false, stream, req[fullc::kData]);
}

template<typename xpu>
void FullyConnectedCompute(const nnvm::NodeAttrs& attrs,
                           const OpContext& ctx,
                           const std::vector<TBlob>& inputs,
                           const std::vector<OpReqType>& req,
                           const std::vector<TBlob>& outputs) {
  std::cout<<"++++++++++++++++fcstart"<<std::endl;

  const FullyConnectedParam& param = nnvm::get<FullyConnectedParam>(attrs.parsed);
  uint32_t in_expected = param.no_bias ? 2 : 3;
  CHECK_EQ(inputs.size(), in_expected);
  CHECK_EQ(outputs.size(), 1U);
  int dtype = inputs[0].type_flag_;

  switch (dtype) {
  case mshadow::kFloat32:
    FCForward<xpu, float>(ctx, param, inputs, req, outputs);
    break;
  case mshadow::kFloat64:
    FCForward<xpu, double>(ctx, param, inputs, req, outputs);
    break;
  case mshadow::kFloat16:
    LOG(FATAL) << "float16 fully connected layer is currently"
                  "only supported by CuDNN version.";
    break;
  default:
    LOG(FATAL) << "Unsupported type " << dtype;
  }
  std::cout<<"++++++++++++++++fcend"<<std::endl;
}

template<typename xpu>
void FullyConnectedGradCompute(const nnvm::NodeAttrs& attrs,
                               const OpContext& ctx,
                               const std::vector<TBlob>& inputs,
                               const std::vector<OpReqType>& req,
                               const std::vector<TBlob>& outputs) {
  const FullyConnectedParam& param = nnvm::get<FullyConnectedParam>(attrs.parsed);
  uint32_t out_expected = param.no_bias ? 2 : 3;
  CHECK_EQ(inputs.size(), 3U);  // ograd_y, x, w
  CHECK_EQ(outputs.size(), out_expected);
  CHECK_EQ(req.size(), out_expected);

  std::vector<TBlob> out_grad{inputs[0]};
  std::vector<TBlob> in_data(inputs.begin() + 1, inputs.end());
  int dtype = inputs[0].type_flag_;

  switch (dtype) {
  case mshadow::kFloat32:
    FCBackward<xpu, float>(ctx, param, out_grad, in_data, req, outputs);
    break;
  case mshadow::kFloat64:
    FCBackward<xpu, double>(ctx, param, out_grad, in_data, req, outputs);
    break;
  case mshadow::kFloat16:
    LOG(FATAL) << "float16 fully connected layer is currently"
                  "only supported by CuDNN version.";
    break;
  default:
    LOG(FATAL) << "Unsupported type " << dtype;
  }
}



///
// Inputs are:
// o_x_grad : head gradient for x_grad
// o_w_grad : head gradient for w_grad
// o_b_grad : if param.no_bias is false
// o_y : head gradient of y
//
// outputs are:
// o_y_grad : gradient of o_y
// x_grad_grad : o_y *  o_w_grad
// w_grad_grad : o_y.T * o_x_grad
// b_grad_grad: if param.no_bias is false
//
// For implementation details see this PR: https://github.com/apache/incubator-mxnet/pull/14779

/**
 * Second order gradient for Fully Connected
 * x_grad_grad = o_y * o_w_grad
 * w_grad_grad = o_y.T * o_x_grad
 *
 * @tparam xpu
 * @tparam DType
 * @param attrs
 * @param ctx
 * @param inputs
 * @param req
 * @param outputs
 */
template<typename xpu, typename DType>
void FullyConnectedGradGradCompute(const nnvm::NodeAttrs& attrs,
                                   const OpContext& ctx,
                                   const std::vector<TBlob>& inputs,
                                   const std::vector<OpReqType>& req,
                                   const std::vector<TBlob>& outputs) {
  using namespace std;
  using namespace fullc;
  Stream<xpu> *stream = ctx.get_stream<xpu>();
  const FullyConnectedParam& param = nnvm::get<FullyConnectedParam>(attrs.parsed);
  const size_t num_inputs = param.no_bias ? 3U : 4U;
  // outputs are: o_x_grad, o_w_grad, o_y   || o_x_grad, o_w_grad, o_b_grad, o_y
  const size_t num_outputs = 3U;
  CHECK_EQ(inputs.size(), num_inputs);
  CHECK_EQ(outputs.size(), num_outputs);
  CHECK_EQ(req.size(), num_outputs);

  // inputs
  Tensor<xpu, 2, DType> o_x_grad;
  Tensor<xpu, 2, DType> o_w_grad;
  Tensor<xpu, 2, DType> o_y;
  // unused
  // Tensor<xpu, 1, DType> o_b_grad;

  // outputs
  Tensor<xpu, 2, DType> o_y_grad;
  TBlob o_y_grad_blob = outputs[kOyGrad];
  Tensor<xpu, 2, DType> x_grad_grad;
  Tensor<xpu, 2, DType> w_grad_grad;
  Tensor<xpu, 1, DType> b_grad_grad;
  size_t o_y_idx = std::numeric_limits<size_t>::max();
  if (param.no_bias)
    o_y_idx = kOy;
  else
    o_y_idx = kOyBias;
  if (!param.flatten) {
    o_x_grad = FlattenAs2DHead<xpu, DType>(inputs[kOxGrad], ctx);
    o_w_grad = inputs[kOwGrad].get<xpu, 2, DType>(stream);
    o_y = FlattenAs2DHead<xpu, DType>(inputs[o_y_idx], ctx);
    x_grad_grad = FlattenAs2DHead<xpu, DType>(outputs[kXGradGrad], ctx);
    w_grad_grad = FlattenAs2DHead<xpu, DType>(outputs[kWGradGrad], ctx);
  } else {
    o_x_grad = FlattenAs2DTail<xpu, DType>(inputs[kOxGrad], ctx);
    o_w_grad = FlattenAs2DTail<xpu, DType>(inputs[kOwGrad], ctx);
    o_y = inputs[o_y_idx].get<xpu, 2, DType>(stream);
    x_grad_grad = FlattenAs2DTail<xpu, DType>(outputs[kXGradGrad], ctx);
    w_grad_grad = FlattenAs2DTail<xpu, DType>(outputs[kWGradGrad], ctx);
  }
  linalg_gemm(o_y, o_w_grad, x_grad_grad, false, false, stream, req[kXGradGrad]);
  linalg_gemm(o_y, o_x_grad, w_grad_grad, true, false, stream, req[kWGradGrad]);
  // 3rd order not supported
  Fill(stream, o_y_grad_blob, kWriteTo, static_cast<DType>(0));
  /* TODO(larroy) bias is not supported yet as there's no bias input to backward. Bias grad grad is
   * zero.
  if (!param.no_bias) {
    // The second order gradient for b doesn't depend on x or w. Thus we set it to 0.
    b_grad_grad = outputs.at(kBGradGrad).get<xpu, 1, DType>(stream);
    TBlob b_grad_grad_blob = TBlob(b_grad_grad);
    Fill(stream, b_grad_grad_blob, kWriteTo, static_cast<DType>(0));
  }
  */
}


template<typename xpu>
void FullyConnectedGradGradDTypeDispatch(
    const nnvm::NodeAttrs& attrs,
    const OpContext& ctx,
    const std::vector<TBlob>& inputs,
    const std::vector<OpReqType>& req,
    const std::vector<TBlob>& outputs) {
  const int dtype = inputs[0].type_flag_;
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    FullyConnectedGradGradCompute<xpu, DType>(attrs, ctx, inputs, req, outputs);
  });
}


}  // namespace op
}  // namespace mxnet
namespace std {
template<>
struct hash<mxnet::op::FullyConnectedParam> {
  size_t operator()(const mxnet::op::FullyConnectedParam& val) {
    size_t ret = 0;
    ret = dmlc::HashCombine(ret, val.num_hidden);
    ret = dmlc::HashCombine(ret, val.no_bias);
    ret = dmlc::HashCombine(ret, val.flatten);
    return ret;
  }
};
}  // namespace std
#endif  // MXNET_OPERATOR_NN_FULLY_CONNECTED_INL_H_
