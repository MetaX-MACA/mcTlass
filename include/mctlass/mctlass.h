/***************************************************************************************************
 * Copyright (c) 2017 - 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

/*! \file
    \brief Basic include for MCTLASS.
*/

#pragma once

#include "mc_runtime_types.h"
#include "mc_runtime_api.h"
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef MCTLASS_NAMESPACE
#define mctlass MCTLASS_NAMESPACE
#endif

////////////////////////////////////////////////////////////////////////////////////////////////////

#define MCTLASS_UNUSED(expr) do { ; } while (&expr != &expr)

#if !defined(__MACACC_RTC__)

#include <assert.h>

#if defined(_MSC_VER)
  #define MCTLASS_NOT_IMPLEMENTED() assert(0 && __FUNCSIG__)
#else
  #define MCTLASS_NOT_IMPLEMENTED() assert(0 && __PRETTY_FUNCTION__)
#endif

#else

#if defined(_MSC_VER)
  #define MCTLASS_NOT_IMPLEMENTED() assert(0 && __FUNCSIG__)
#else
  #define MCTLASS_NOT_IMPLEMENTED() assert(0 && __PRETTY_FUNCTION__)
#endif

#endif

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace mctlass {

////////////////////////////////////////////////////////////////////////////////////////////////////

#if defined(__MXCC__) || (defined(__clang__) && defined(__MACA__))
  #define MCTLASS_HOST_DEVICE __forceinline__ __device__ __host__
  #define MCTLASS_DEVICE __forceinline__ __device__
#elif defined(__MACACC_RTC__)
  #define MCTLASS_HOST_DEVICE __forceinline__ __device__
  #define MCTLASS_DEVICE __forceinline__ __device__
#else
  #define MCTLASS_HOST_DEVICE inline
  #define MCTLASS_DEVICE inline
#endif

/// Status code returned by MCTLASS operations
enum class Status {
  kSuccess,                    ///< Operation was successful.
  kErrorMisalignedOperand,     ///< operands fail alignment requirements.
  kErrorInvalidDataType,       ///< DataType fails requirement.
  kErrorInvalidLayout,         ///< Layout fails alignment requirement.
  kErrorInvalidProblem,        ///< Specified problem size is not supported by operator.
  kErrorNotSupported,          ///< Operation is not supported on current device.
  kErrorWorkspaceNull,         ///< The given workspace is null when it is required to be non-null.
  kErrorInternal,              ///< An error within MCTLASS occurred.
  kErrorArchMismatch,          ///< MCTLASS runs on a device that it was not compiled for.
  kErrorInsufficientDriver,    ///< MCTLASS runs with a driver that is too old.
  kErrorMemoryAllocation,      ///< Kernel launch failed due to insufficient device memory.
  kInvalid                     ///< Status is unspecified.
};

/// Convert mctlass status to status strings
MCTLASS_HOST_DEVICE
static char const* mctlassGetStatusString(mctlass::Status status) {
  switch (status) {
    case mctlass::Status::kSuccess:
      return "Success";
    case mctlass::Status::kErrorMisalignedOperand:
      return "Error Misaligned Operand";
    case mctlass::Status::kErrorInvalidDataType:
      return "Error Invalid Data Type";
    case mctlass::Status::kErrorInvalidLayout:
      return "Error Invalid Layout";
    case mctlass::Status::kErrorInvalidProblem:
      return "Error Invalid Problem";
    case mctlass::Status::kErrorNotSupported:
      return "Error Not Supported";
    case mctlass::Status::kErrorWorkspaceNull:
      return "Error Workspace Null";
    case mctlass::Status::kErrorInternal:
      return "Error Internal";
    case mctlass::Status::kErrorInsufficientDriver:
      return "Error Insufficient Driver";
    case mctlass::Status::kErrorArchMismatch:
      return "Error Architecture Mismatch";
    case mctlass::Status::kErrorMemoryAllocation:
      return "Error Memory Allocation failed";
    case mctlass::Status::kInvalid: break;
  }

  return "Invalid status";
}

template<class T>
static __inline__ __host__ mcError_t mcFuncSetAttribute(
  T *entry,
  mcFuncAttribute attr,
  int value
)
{
  return mcFuncSetAttribute((const void *)entry,attr,value);
}

////////////////////////////////////////////////////////////////////////////////////////////////////


#ifndef MCTLASS_CONV_UNIT_TEST_RIGOROUS_SIZE_ENABLED
#define MCTLASS_CONV_UNIT_TEST_RIGOROUS_SIZE_ENABLED 0
#endif


// CUDA 10.1 introduces the mma instruction
#if !defined(MCTLASS_ENABLE_TENSOR_CORE_MMA)
#define MCTLASS_ENABLE_TENSOR_CORE_MMA 0
#endif

////////////////////////////////////////////////////////////////////////////////////////////////////

#define MCTLASS_ASSERT(x) assert(x)

////////////////////////////////////////////////////////////////////////////////////////////////////

// MCTLASS_PRAGMA_(UNROLL|NO_UNROLL) optimization directives for the compiler.
#if defined(__MACA_ARCH__)
  #if defined(__MACACC_RTC__) || (defined(__clang__) && defined(__MACA__))
    #define MCTLASS_PRAGMA_UNROLL _Pragma("unroll")
    #define MCTLASS_PRAGMA_NO_UNROLL _Pragma("unroll 1")
  #else
    // We cannot build with this define for mxcc
    //#define MCTLASS_PRAGMA_UNROLL #pragma unroll
    //#define MCTLASS_PRAGMA_NO_UNROLL #pragma unroll 1
    #define MCTLASS_PRAGMA_UNROLL _Pragma("unroll")
    #define MCTLASS_PRAGMA_NO_UNROLL _Pragma("unroll 1")
  #endif

  #define MCTLASS_GEMM_LOOP MCTLASS_PRAGMA_NO_UNROLL

#else

    #define MCTLASS_PRAGMA_UNROLL
    #define MCTLASS_PRAGMA_NO_UNROLL
    #define MCTLASS_GEMM_LOOP

#endif

////////////////////////////////////////////////////////////////////////////////////////////////////

static const int NUM_THREADS_PER_WARP = 32;
static const int NUM_THREADS_PER_HALF_WARP = NUM_THREADS_PER_WARP / 2;
static const int NUM_THREADS_PER_QUAD = 4;
static const int NUM_THREADS_PER_QUAD_PAIR = NUM_THREADS_PER_QUAD * 2;

////////////////////////////////////////////////////////////////////////////////////////////////////

/// Helper function to return true when called by thread 0 of threadblock 0.
MCTLASS_HOST_DEVICE bool thread0() {
  #if defined(__MACA_ARCH__)
    return (!threadIdx.x && !threadIdx.y && !threadIdx.z) && (!blockIdx.x && !blockIdx.y && !blockIdx.z);
  #else
    return false;
  #endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace mctlass

////////////////////////////////////////////////////////////////////////////////////////////////////
