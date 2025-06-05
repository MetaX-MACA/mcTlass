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
    \brief Matrix multiply
*/

#pragma once

#if defined(__MACACC_RTC__)
#include <cuda/std/cassert>
#else
#include <assert.h>
#endif

#include "mctlass/mctlass.h"
#include "mma.h"
#include "mctlass/layout/matrix.h"
#include "mctlass/numeric_types.h"


////////////////////////////////////////////////////////////////////////////////

//#if ((__CUDACC_VER_MAJOR__ > 11) || (__CUDACC_VER_MAJOR__ == 11 && __CUDACC_VER_MINOR__ >= 0))
#if 1
#define MCTLASS_ARCH_MMA_SM80_SUPPORTED 1

#if defined(__MACA_ARCH__)
#define MCTLASS_ARCH_MMA_SM80_ENABLED
#include "__clang_maca_mma_functions.h"
#endif
#endif

////////////////////////////////////////////////////////////////////////////////

namespace mctlass {
namespace arch {

////////////////////////////////////////////////////////////////////////////////
//
// Matrix Multiply 1688 - Float BF16, FP32 accumulation
//
////////////////////////////////////////////////////////////////////////////////

/// Matrix multiply-add operation - F32 = bf16 * bf16 + F32
template <>
struct Mma<
  gemm::GemmShape<16, 8, 8>,
  32,
  bfloat16_t,
  layout::RowMajor,
  bfloat16_t,
  layout::ColumnMajor,
  float,
  layout::RowMajor,
  OpMultiplyAdd> {

  using Shape = gemm::GemmShape<16, 8, 8>;

  using ElementA = bfloat16_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<bfloat16_t, 4>;

  using ElementB = bfloat16_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<bfloat16_t, 2>;

  using ElementC = float;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<float, 4>;

  using Operator = OpMultiplyAdd;
  using ArchTag = arch::Sm80;

  MCTLASS_HOST_DEVICE
  void operator()(FragmentC &d, FragmentA const &a, FragmentB const &b,
                  FragmentC const &c) const {

//#if defined(MCTLASS_ARCH_MMA_SM80_ENABLED)
#if 0 //mctlass not support this data-type and shape now

  uint32_t const *A = reinterpret_cast<uint32_t const *>(&a);
  uint32_t const *B = reinterpret_cast<uint32_t const *>(&b);
  float const *C = reinterpret_cast<float const *>(&c);
  float *D = reinterpret_cast<float *>(&d);
  asm(
      "mma.sync.aligned.m16n8k8.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%8,%9,%10};\n"
      : "=f"(D[0]), "=f"(D[1]), "=f"(D[2]), "=f"(D[3])
      :
        "r"(A[0]), "r"(A[1]),
        "r"(B[0]),
        "f"(C[0]), "f"(C[1]), "f"(C[2]), "f"(C[3])
  );
#else

    MCTLASS_UNUSED(d);
    MCTLASS_UNUSED(a);
    MCTLASS_UNUSED(b);
    MCTLASS_UNUSED(c);
    MCTLASS_NOT_IMPLEMENTED();

#endif
  }
};

////////////////////////////////////////////////////////////////////////////////
//
// Matrix Multiply 1684 - Float TF32
//
////////////////////////////////////////////////////////////////////////////////

/// Matrix multiply-add operation: F32 = tf32 * tf32 + F32
template <>
struct Mma<
  gemm::GemmShape<16, 8, 4>,
  32,
  tfloat32_t,
  layout::RowMajor,
  tfloat32_t,
  layout::ColumnMajor,
  float,
  layout::RowMajor,
  OpMultiplyAdd> {

  using Shape = gemm::GemmShape<16, 8, 4>;

  using ElementA = tfloat32_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<tfloat32_t, 2>;

  using ElementB = tfloat32_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<tfloat32_t, 1>;

  using ElementC = float;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<float, 4>;

  using Operator = OpMultiplyAdd;
  using ArchTag = arch::Sm80;

  MCTLASS_HOST_DEVICE
  void operator()(
    FragmentC &d,
    FragmentA const &a,
    FragmentB const &b,
    FragmentC const &c
  ) const {

    // uint32_t const *A = reinterpret_cast<uint32_t const *>(&a);
    // uint32_t const *B = reinterpret_cast<uint32_t const *>(&b);
    // float const *C = reinterpret_cast<float const *>(&c);
    // float *D = reinterpret_cast<float *>(&d);
    // asm volatile(
    //     "mma.sync.aligned.m16n8k4.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%8,%9,%10};\n"
    //     : "=f"(D[0]), "=f"(D[1]), "=f"(D[2]), "=f"(D[3])
    //     :
    //       "r"(A[0]), "r"(A[1]),
    //       "r"(B[0]),
    //       "f"(C[0]), "f"(C[1]), "f"(C[2]), "f"(C[3])
    // );
#if defined(MCTLASS_ARCH_MMA_SM80_ENABLED)
    float const *A = reinterpret_cast<float const *>(&a);
    float const *B = reinterpret_cast<float const *>(&b);
    float const *C = reinterpret_cast<float const *>(&c);
    float *D = reinterpret_cast<float *>(&d);

    auto temp = __builtin_mxc_mma_16x16x8tf32({A[0], 0}, {B[0], 0}, {C[0], C[1], C[2], C[3]});

    D[0] = temp[0];
    D[1] = temp[1];
    D[2] = temp[2];
    D[3] = temp[3];
#else

    MCTLASS_UNUSED(d);
    MCTLASS_UNUSED(a);
    MCTLASS_UNUSED(b);
    MCTLASS_UNUSED(c);
    MCTLASS_NOT_IMPLEMENTED();
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////
//
// Matrix Multiply 1688 - Float TF32
//
////////////////////////////////////////////////////////////////////////////////

/// Matrix multiply-add operation: F32 = tf32 * tf32 + F32
template <>
struct Mma<gemm::GemmShape<16, 8, 8>, 32, tfloat32_t, layout::RowMajor,
           tfloat32_t, layout::ColumnMajor, float, layout::RowMajor,
           OpMultiplyAdd> {
  using Shape = gemm::GemmShape<16, 8, 8>;

  using ElementA = tfloat32_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<tfloat32_t, 4>;

  using ElementB = tfloat32_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<tfloat32_t, 2>;

  using ElementC = float;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<float, 4>;

  using Operator = OpMultiplyAdd;
  using ArchTag = arch::Sm80;

  MCTLASS_HOST_DEVICE
  void operator()(FragmentC &d, FragmentA const &a, FragmentB const &b,
                  FragmentC const &c) const {

// #if defined(MCTLASS_ARCH_MMA_SM80_ENABLED)

//     uint32_t const *A = reinterpret_cast<uint32_t const *>(&a);
//     uint32_t const *B = reinterpret_cast<uint32_t const *>(&b);
//     float const *C = reinterpret_cast<float const *>(&c);
//     float *D = reinterpret_cast<float *>(&d);
//     asm volatile(
//         "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
//         "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
//         : "=f"(D[0]), "=f"(D[1]), "=f"(D[2]), "=f"(D[3])
//         : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]),
//           "f"(C[0]), "f"(C[1]), "f"(C[2]), "f"(C[3]));
// #endif
#if defined(MCTLASS_ARCH_MMA_SM80_ENABLED)
    float const *fA = reinterpret_cast<float const *>(&a);
    float const *fB = reinterpret_cast<float const *>(&b);
    float const *C = reinterpret_cast<float const *>(&c);
    float *D = reinterpret_cast<float *>(&d);

    auto temp = __builtin_mxc_mma_16x16x8tf32({fA[0], fA[1]}, {fB[0], fB[1]}, {C[0], C[1], C[2], C[3]});

    D[0] = temp[0];
    D[1] = temp[1];
    D[2] = temp[2];
    D[3] = temp[3];
#else

    MCTLASS_UNUSED(d);
    MCTLASS_UNUSED(a);
    MCTLASS_UNUSED(b);
    MCTLASS_UNUSED(c);
    MCTLASS_NOT_IMPLEMENTED();
#endif
  }
};

/// Matrix multiply-add operation: F32 = tf32 * tf32 + F32
template <>
struct Mma<gemm::GemmShape<16, 8, 8>, 32, float, layout::RowMajor,
           float, layout::ColumnMajor, float, layout::RowMajor,
           OpMultiplyAdd> {
  using Shape = gemm::GemmShape<16, 8, 8>;

  using ElementA = float;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<float, 4>;

  using ElementB = float;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<float, 2>;

  using ElementC = float;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<float, 4>;

  using Operator = OpMultiplyAdd;
  using ArchTag = arch::Sm80;

  MCTLASS_HOST_DEVICE
  void operator()(FragmentC &d, FragmentA const &a, FragmentB const &b,
                  FragmentC const &c) const {

// #if defined(MCTLASS_ARCH_MMA_SM80_ENABLED)

//     uint32_t const *A = reinterpret_cast<uint32_t const *>(&a);
//     uint32_t const *B = reinterpret_cast<uint32_t const *>(&b);
//     float const *C = reinterpret_cast<float const *>(&c);
//     float *D = reinterpret_cast<float *>(&d);
//     asm volatile(
//         "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
//         "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
//         : "=f"(D[0]), "=f"(D[1]), "=f"(D[2]), "=f"(D[3])
//         : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]),
//           "f"(C[0]), "f"(C[1]), "f"(C[2]), "f"(C[3]));
// #endif
#if defined(MCTLASS_ARCH_MMA_SM80_ENABLED)
    float const *C = reinterpret_cast<float const *>(&c);
    float *D = reinterpret_cast<float *>(&d);

    auto temp = __builtin_mxc_mma_16x16x4f32(a[0], b[0],
                                            {C[0], C[1], C[2], C[3]});
    auto res = __builtin_mxc_mma_16x16x4f32(a[1], b[1],
                                            {temp[0], temp[1], temp[2], temp[3]});

    D[0] = res[0];
    D[1] = res[1];
    D[2] = res[2];
    D[3] = res[3];
#else

    MCTLASS_UNUSED(d);
    MCTLASS_UNUSED(a);
    MCTLASS_UNUSED(b);
    MCTLASS_UNUSED(c);
    MCTLASS_NOT_IMPLEMENTED();
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////
//
// Matrix Multiply 16816
//
////////////////////////////////////////////////////////////////////////////////

/// Matrix multiply-add operation: F16 = F16 * F16 + F16
template <>
struct Mma<
  gemm::GemmShape<16, 8, 16>,
  32,
  half_t,
  layout::RowMajor,
  half_t,
  layout::ColumnMajor,
  half_t,
  layout::RowMajor,
  OpMultiplyAdd> {

  using Shape = gemm::GemmShape<16, 8, 16>;

  using ElementA = half_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<half_t, 8>;

  using ElementB = half_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<half_t, 4>;

  using ElementC = half_t;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<half_t, 4>;

  using Operator = OpMultiplyAdd;
  using ArchTag = arch::Sm80;

  /// Computes multiply-add
  MCTLASS_HOST_DEVICE
  void operator()(FragmentC &d, FragmentA const &a, FragmentB const &b,
                  FragmentC const &c) const {

#if defined(MCTLASS_ARCH_MMA_SM80_ENABLED)
//
//  uint32_t const *A = reinterpret_cast<uint32_t const *>(&a);
//  uint32_t const *B = reinterpret_cast<uint32_t const *>(&b);
//  uint32_t const *C = reinterpret_cast<uint32_t const *>(&c);
//  uint32_t *D = reinterpret_cast<uint32_t *>(&d);
//  asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%8,%9};\n"
//      : "=r"(D[0]), "=r"(D[1])
//      : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]),
//        "r"(B[0]), "r"(B[1]),
//        "r"(C[0]), "r"(C[1])
//  );
    auto results = __builtin_mxc_mma_16x16x16f16(
      a.to_macahalf4(),
      b.to_macahalf4(),
      {c[0].get().to_macahalf(), c[1].get().to_macahalf(),
        c[2].get().to_macahalf(), c[3].get().to_macahalf()});

    d[0] = mctlass::half_t(results[0]);
    d[1] = mctlass::half_t(results[1]);
    d[2] = mctlass::half_t(results[2]);
    d[3] = mctlass::half_t(results[3]);
#else

   MCTLASS_UNUSED(d);
   MCTLASS_UNUSED(a);
   MCTLASS_UNUSED(b);
   MCTLASS_UNUSED(c);
   MCTLASS_NOT_IMPLEMENTED();

#endif
  }
};

////////////////////////////////////////////////////////////////////////////////

/// Matrix multiply-add operation: F32 = bf16 * bf16 + F32
template <>
struct Mma<
  gemm::GemmShape<16, 8, 16>,
  32,
  bfloat16_t,
  layout::RowMajor,
  bfloat16_t,
  layout::ColumnMajor,
  float,
  layout::RowMajor,
  OpMultiplyAdd> {

  using Shape = gemm::GemmShape<16, 8, 16>;

  using ElementA = bfloat16_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<bfloat16_t, 8>;

  using ElementB = bfloat16_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<bfloat16_t, 4>;

  using ElementC = float;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<float, 4>;

  using Operator = OpMultiplyAdd;
  using ArchTag = arch::Sm80;

  /// Computes multiply-add
  MCTLASS_HOST_DEVICE
  void operator()(
    FragmentC &d,
    FragmentA const &a,
    FragmentB const &b,
    FragmentC const &c
  ) const {

//#if defined(MCTLASS_ARCH_MMA_SM80_ENABLED)
#if 0 //mctlass not support this data-type and shape now

    uint32_t const *A = reinterpret_cast<uint32_t const *>(&a);
    uint32_t const *B = reinterpret_cast<uint32_t const *>(&b);
    float const *C = reinterpret_cast<float const *>(&c);
    float *D = reinterpret_cast<float *>(&d);
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
        : "=f"(D[0]), "=f"(D[1]), "=f"(D[2]), "=f"(D[3])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]),
          "f"(C[0]), "f"(C[1]), "f"(C[2]), "f"(C[3]));
#else

    MCTLASS_UNUSED(d);
    MCTLASS_UNUSED(a);
    MCTLASS_UNUSED(b);
    MCTLASS_UNUSED(c);
    MCTLASS_NOT_IMPLEMENTED();

#endif
  }
};

////////////////////////////////////////////////////////////////////////////////

/// Matrix multiply-add operation: F32 = F16 * F16 + F32
template <>
struct Mma<
  gemm::GemmShape<16, 8, 16>,
  32,
  half_t,
  layout::RowMajor,
  half_t,
  layout::ColumnMajor,
  float,
  layout::RowMajor,
  OpMultiplyAdd> {

  using Shape = gemm::GemmShape<16, 8, 16>;

  using ElementA = half_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<half_t, 8>;

  using ElementB = half_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<half_t, 4>;

  using ElementC = float;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<float, 4>;

  using Operator = OpMultiplyAdd;
  using ArchTag = arch::Sm80;

  /// Computes multiply-add
  MCTLASS_HOST_DEVICE
  void operator()(
    FragmentC &d,
    FragmentA const &a,
    FragmentB const &b,
    FragmentC const &c
  ) const {

#if defined(MCTLASS_ARCH_MMA_SM80_ENABLED)

    // uint32_t const *A = reinterpret_cast<uint32_t const *>(&a);
    // uint32_t const *B = reinterpret_cast<uint32_t const *>(&b);
    // float const *C = reinterpret_cast<float const *>(&c);
    // float *D = reinterpret_cast<float *>(&d);
    // asm volatile(
    //     "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32  {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
    //     "{%10,%11,%12,%13};\n"
    //     : "=f"(D[0]), "=f"(D[1]), "=f"(D[2]), "=f"(D[3])
    //     : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]),
    //       "f"(C[0]), "f"(C[1]), "f"(C[2]), "f"(C[3]));

    auto results = __builtin_mxc_mma_16x16x16f16(
        a.to_macahalf4(),
        b.to_macahalf4(),
        c.to_macafloat4());

    d[0] = results[0];
    d[1] = results[1];
    d[2] = results[2];
    d[3] = results[3];

#else

    MCTLASS_UNUSED(d);
    MCTLASS_UNUSED(a);
    MCTLASS_UNUSED(b);
    MCTLASS_UNUSED(c);
    MCTLASS_NOT_IMPLEMENTED();

#endif
  }
};

////////////////////////////////////////////////////////////////////////////////
//
// Matrix Multiply 16164 - F32
//
////////////////////////////////////////////////////////////////////////////////

/// Matrix multiply-add operation: F32 = F32 * F32 + F32
template <>
struct Mma<
  gemm::GemmShape<16,16,4>,
  32,
  float,
  layout::RowMajor,
  float,
  layout::ColumnMajor,
  float,
  layout::RowMajor,
  OpMultiplyAdd> {

  using Shape = gemm::GemmShape<16,16,4>;

  using ElementA = float;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<float, 1>;

  using ElementB = float;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<float, 1>;

  using ElementC = float;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<float, 4>;

  using Operator = OpMultiplyAdd;

  using ArchTag = arch::Sm80;

  MCTLASS_HOST_DEVICE
  void operator()(FragmentC &d, FragmentA const &a, FragmentB const &b,
                  FragmentC const &c) const {

#if defined(MCTLASS_ARCH_MMA_SM80_ENABLED)

  float const & A = reinterpret_cast<float const &>(a);
  float const & B = reinterpret_cast<float const &>(b);

  float const *C = reinterpret_cast<float const *>(&c);
  float *D = reinterpret_cast<float *>(&d);

  auto temp = __builtin_mxc_mma_16x16x4f32(B, A,
                                           {C[0], C[1], C[2], C[3]});
  D[0] = temp[0];
  D[1] = temp[1];
  D[2] = temp[2];
  D[3] = temp[3];

#endif
  }
};

////////////////////////////////////////////////////////////////////////////////
//
// Matrix Multiply 16168 - F32
//
////////////////////////////////////////////////////////////////////////////////

/// Matrix multiply-add operation: F32 = F32 * F32 + F32
template <>
struct Mma<
  gemm::GemmShape<16,16,8>,
  32,
  float,
  layout::RowMajor,
  float,
  layout::ColumnMajor,
  float,
  layout::RowMajor,
  OpMultiplyAdd> {

  using Shape = gemm::GemmShape<16,16,8>;

  using ElementA = float;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<float, 2>;

  using ElementB = float;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<float, 2>;

  using ElementC = float;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<float, 4>;

  using Operator = OpMultiplyAdd;

  using ArchTag = arch::Sm80;

  MCTLASS_HOST_DEVICE
  void operator()(FragmentC &d, FragmentA const &a, FragmentB const &b,
                  FragmentC const &c) const {

#if defined(MCTLASS_ARCH_MMA_SM80_ENABLED)

  float const *C = reinterpret_cast<float const *>(&c);
  float *D = reinterpret_cast<float *>(&d);

  auto temp = __builtin_mxc_mma_16x16x4f32(b[0], a[0],
                                           {C[0], C[1], C[2], C[3]});
  auto res = __builtin_mxc_mma_16x16x4f32(b[1], a[1],
                                           {temp[0], temp[1], temp[2], temp[3]});
  D[0] = res[0];
  D[1] = res[1];
  D[2] = res[2];
  D[3] = res[3];

#endif
  }
};

/// Matrix multiply-add operation: F32 = F32 * F32 + F32
template <>
struct Mma<
  gemm::GemmShape<16,16,8>,
  32,
  tfloat32_t,
  layout::RowMajor,
  tfloat32_t,
  layout::ColumnMajor,
  float,
  layout::RowMajor,
  OpMultiplyAdd> {

  using Shape = gemm::GemmShape<16,16,8>;

  using ElementA = tfloat32_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<tfloat32_t, 2>;

  using ElementB = tfloat32_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<tfloat32_t, 2>;

  using ElementC = float;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<float, 4>;

  using Operator = OpMultiplyAdd;

  using ArchTag = arch::Sm80;

  MCTLASS_HOST_DEVICE
  void operator()(FragmentC &d, FragmentA const &a, FragmentB const &b,
                  FragmentC const &c) const {

#if defined(MCTLASS_ARCH_MMA_SM80_ENABLED)

  float const *A = reinterpret_cast<float const *>(&a);
  float const *B = reinterpret_cast<float const *>(&b);
  float const *C = reinterpret_cast<float const *>(&c);
  float *D = reinterpret_cast<float *>(&d);

  auto res = __builtin_mxc_mma_16x16x8tf32({B[0], B[1]}, {A[0], A[1]},
                                           {C[0], C[1], C[2], C[3]});
  D[0] = res[0];
  D[1] = res[1];
  D[2] = res[2];
  D[3] = res[3];

#endif
  }
};

////////////////////////////////////////////////////////////////////////////////
//
// Matrix Multiply 884 - F64
//
////////////////////////////////////////////////////////////////////////////////

/// Matrix multiply-add operation: F64 = F64 * F64 + F64
template <>
struct Mma<
  gemm::GemmShape<8,8,4>,
  32,
  double,
  layout::RowMajor,
  double,
  layout::ColumnMajor,
  double,
  layout::RowMajor,
  OpMultiplyAdd> {

  using Shape = gemm::GemmShape<8,8,4>;

  using ElementA = double;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<double, 1>;

  using ElementB = double;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<double, 1>;

  using ElementC = double;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<double, 2>;

  using Operator = OpMultiplyAdd;

  using ArchTag = arch::Sm80;

  MCTLASS_HOST_DEVICE
  void operator()(FragmentC &d, FragmentA const &a, FragmentB const &b,
                  FragmentC const &c) const {

#if defined(MCTLASS_ARCH_MMA_SM80_ENABLED)

  double const & A = reinterpret_cast<double const &>(a);
  double const & B = reinterpret_cast<double const &>(b);

  double const *C = reinterpret_cast<double const *>(&c);
  double *D = reinterpret_cast<double *>(&d);

__syncthreads(); //We need remove this someday
  auto temp = __builtin_mxc_mma_16x16x4f64(A, B,
                                           {C[0], C[1], D[0], D[1]});
__syncthreads();//We need remove this someday
  D[0] = temp[0];
  D[1] = temp[1];

#endif
  }
};

////////////////////////////////////////////////////////////////////////////////
//
// Matrix Multiply 16816 - S8 input, S32 accumulation
//
////////////////////////////////////////////////////////////////////////////////

/// Matrix multiply-add operation: S32 = S8 * S8 + S32
template <>
struct Mma<
  gemm::GemmShape<16,8,16>,
  32,
  int8_t,
  layout::RowMajor,
  int8_t,
  layout::ColumnMajor,
  int,
  layout::RowMajor,
  OpMultiplyAdd> {

  using Shape = gemm::GemmShape<16,8,16>;

  using ElementA = int8_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<int8_t, 8>;

  using ElementB = int8_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<int8_t, 4>;

  using ElementC = int;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<int, 4>;

  using Operator = OpMultiplyAdd;

  using ArchTag = arch::Sm80;

  /// Computes multiply-add
  MCTLASS_HOST_DEVICE
  void operator()(
    FragmentC &d,
    FragmentA const &a,
    FragmentB const &b,
    FragmentC const &c
  ) const {

//#if defined(MCTLASS_ARCH_MMA_SM80_ENABLED)
#if 0 //mctlass not support this data-type and shape now
    uint32_t const *A = reinterpret_cast<uint32_t const *>(&a);
    uint32_t const &B = reinterpret_cast<uint32_t const &>(b);

    int const *C = reinterpret_cast<int const *>(&c);
    int *D = reinterpret_cast<int *>(&d);
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5}, {%6}, "
        "{%7,%8,%9,%10};\n"
        : "=r"(D[0]), "=r"(D[1]), "=r"(D[2]), "=r"(D[3])
        : "r"(A[0]), "r"(A[1]), "r"(B), "r"(C[0]), "r"(C[1]), "r"(C[2]),
          "r"(C[3]));
#else
    MCTLASS_UNUSED(a);
    MCTLASS_UNUSED(b);
    MCTLASS_UNUSED(c);
    MCTLASS_UNUSED(d);
    assert(0);
#endif
  }
};

/// Matrix multiply-add operation: S32 = U8 * S8 + S32
template <>
struct Mma<
  gemm::GemmShape<16,8,16>,
  32,
  uint8_t,
  layout::RowMajor,
  int8_t,
  layout::ColumnMajor,
  int,
  layout::RowMajor,
  OpMultiplyAdd> {

  using Shape = gemm::GemmShape<16,8,16>;

  using ElementA = uint8_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<uint8_t, 8>;

  using ElementB = int8_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<int8_t, 4>;

  using ElementC = int;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<int, 4>;

  using Operator = OpMultiplyAdd;
  using ArchTag = arch::Sm80;

  /// Computes multiply-add
  MCTLASS_HOST_DEVICE
  void operator()(
    FragmentC &d,
    FragmentA const &a,
    FragmentB const &b,
    FragmentC const &c
  ) const {

//#if defined(MCTLASS_ARCH_MMA_SM80_ENABLED)
#if 0 //mctlass not support this data-type and shape now
    uint32_t const *A = reinterpret_cast<uint32_t const *>(&a);
    uint32_t const &B = reinterpret_cast<uint32_t const &>(b);

    int const *C = reinterpret_cast<int const *>(&c);
    int *D = reinterpret_cast<int *>(&d);
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.s32.u8.s8.s32 {%0,%1,%2,%3}, {%4,%5}, {%6}, "
        "{%7,%8,%9,%10};\n"
        : "=r"(D[0]), "=r"(D[1]), "=r"(D[2]), "=r"(D[3])
        : "r"(A[0]), "r"(A[1]), "r"(B), "r"(C[0]), "r"(C[1]), "r"(C[2]),
          "r"(C[3]));
#else
    assert(0);
#endif
  }
};

/// Matrix multiply-add operation: S32 = S8 * U8 + S32
template <>
struct Mma<
  gemm::GemmShape<16,8,16>,
  32,
  int8_t,
  layout::RowMajor,
  uint8_t,
  layout::ColumnMajor,
  int,
  layout::RowMajor,
  OpMultiplyAdd> {

  using Shape = gemm::GemmShape<16,8,16>;

  using ElementA = int8_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<int8_t, 8>;

  using ElementB = uint8_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<uint8_t, 4>;

  using ElementC = int;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<int, 4>;

  using Operator = OpMultiplyAdd;
  using ArchTag = arch::Sm80;

  /// Computes multiply-add
  MCTLASS_HOST_DEVICE
  void operator()(
    FragmentC &d,
    FragmentA const &a,
    FragmentB const &b,
    FragmentC const &c
  ) const {

//#if defined(MCTLASS_ARCH_MMA_SM80_ENABLED)
#if 0 //mctlass not support this data-type and shape now

    uint32_t const *A = reinterpret_cast<uint32_t const *>(&a);
    uint32_t const &B = reinterpret_cast<uint32_t const &>(b);

    int const *C = reinterpret_cast<int const *>(&c);
    int *D = reinterpret_cast<int *>(&d);
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.s32.s8.u8.s32 {%0,%1,%2,%3}, {%4,%5}, {%6}, "
        "{%7,%8,%9,%10};\n"
        : "=r"(D[0]), "=r"(D[1]), "=r"(D[2]), "=r"(D[3])
        : "r"(A[0]), "r"(A[1]), "r"(B), "r"(C[0]), "r"(C[1]), "r"(C[2]),
          "r"(C[3]));
#else
    assert(0);
#endif
  }
};

/// Matrix multiply-add operation: S32 = U8 * U8 + S32
template <>
struct Mma<
  gemm::GemmShape<16,8,16>,
  32,
  uint8_t,
  layout::RowMajor,
  uint8_t,
  layout::ColumnMajor,
  int,
  layout::RowMajor,
  OpMultiplyAdd> {

  using Shape = gemm::GemmShape<16,8,16>;

  using ElementA = uint8_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<uint8_t, 8>;

  using ElementB = uint8_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<uint8_t, 4>;

  using ElementC = int;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<int, 4>;

  using Operator = OpMultiplyAdd;
  using ArchTag = arch::Sm80;

  /// Computes multiply-add
  MCTLASS_HOST_DEVICE
  void operator()(
    FragmentC &d,
    FragmentA const &a,
    FragmentB const &b,
    FragmentC const &c
  ) const {

//#if defined(MCTLASS_ARCH_MMA_SM80_ENABLED)
#if 0 //mctlass not support this data-type and shape now

    uint32_t const *A = reinterpret_cast<uint32_t const *>(&a);
    uint32_t const &B = reinterpret_cast<uint32_t const &>(b);

    int const *C = reinterpret_cast<int const *>(&c);
    int *D = reinterpret_cast<int *>(&d);
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.s32.u8.u8.s32 {%0,%1,%2,%3}, {%4,%5}, {%6}, "
        "{%7,%8,%9,%10};\n"
        : "=r"(D[0]), "=r"(D[1]), "=r"(D[2]), "=r"(D[3])
        : "r"(A[0]), "r"(A[1]), "r"(B), "r"(C[0]), "r"(C[1]), "r"(C[2]),
          "r"(C[3]));

#else
    assert(0);
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////
//
// Matrix Multiply 16816 - S8 input, S32 accumulation - SATURATE
//
////////////////////////////////////////////////////////////////////////////////

/// Matrix multiply-add operation: S32 = S8 * S8 + S32
template <>
struct Mma<
  gemm::GemmShape<16,8,16>,
  32,
  int8_t,
  layout::RowMajor,
  int8_t,
  layout::ColumnMajor,
  int,
  layout::RowMajor,
  OpMultiplyAddSaturate> {

  using Shape = gemm::GemmShape<16,8,16>;

  using ElementA = int8_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<int8_t, 8>;

  using ElementB = int8_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<int8_t, 4>;

  using ElementC = int;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<int, 4>;

  using Operator = OpMultiplyAddSaturate;
  using ArchTag = arch::Sm80;

  /// Computes multiply-add
  MCTLASS_HOST_DEVICE
  void operator()(
    FragmentC &d,
    FragmentA const &a,
    FragmentB const &b,
    FragmentC const &c
  ) const {

#if defined(MCTLASS_ARCH_MMA_SM80_ENABLED)

    uint32_t const *A = reinterpret_cast<uint32_t const *>(&a);
    uint32_t const &B = reinterpret_cast<uint32_t const &>(b);

    int const *C = reinterpret_cast<int const *>(&c);
    int *D = reinterpret_cast<int *>(&d);
    //asm volatile(
    //    "mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32.satfinite {%0,%1,%2,%3}, {%4,%5}, "
    //    "{%6}, {%7,%8,%9,%10};\n"
    //    : "=r"(D[0]), "=r"(D[1]), "=r"(D[2]), "=r"(D[3])
    //    : "r"(A[0]), "r"(A[1]), "r"(B), "r"(C[0]), "r"(C[1]), "r"(C[2]),
    //      "r"(C[3]));
    auto temp = __builtin_mxc_mma_16x16x16i8(A[0], B, {C[0], C[1], C[2], C[3]});
    D[0] = temp[0];
    D[1] = temp[1];
    D[2] = temp[2];
    D[3] = temp[3];
#endif
  }
};

/// Matrix multiply-add operation: S32 = U8 * S8 + S32
template <>
struct Mma<
  gemm::GemmShape<16,8,16>,
  32,
  uint8_t,
  layout::RowMajor,
  int8_t,
  layout::ColumnMajor,
  int,
  layout::RowMajor,
  OpMultiplyAddSaturate> {

  using Shape = gemm::GemmShape<16,8,16>;

  using ElementA = uint8_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<uint8_t, 8>;

  using ElementB = int8_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<int8_t, 4>;

  using ElementC = int;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<int, 4>;

  using Operator = OpMultiplyAddSaturate;
  using ArchTag = arch::Sm80;

  /// Computes multiply-add
  MCTLASS_HOST_DEVICE
  void operator()(
    FragmentC &d,
    FragmentA const &a,
    FragmentB const &b,
    FragmentC const &c
  ) const {

//#if defined(MCTLASS_ARCH_MMA_SM80_ENABLED)
#if 0 //mctlass not support this data-type and shape now

    uint32_t const *A = reinterpret_cast<uint32_t const *>(&a);
    uint32_t const &B = reinterpret_cast<uint32_t const &>(b);

    int const *C = reinterpret_cast<int const *>(&c);
    int *D = reinterpret_cast<int *>(&d);
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.s32.u8.s8.s32.satfinite {%0,%1,%2,%3}, {%4,%5}, "
        "{%6}, {%7,%8,%9,%10};\n"
        : "=r"(D[0]), "=r"(D[1]), "=r"(D[2]), "=r"(D[3])
        : "r"(A[0]), "r"(A[1]), "r"(B), "r"(C[0]), "r"(C[1]), "r"(C[2]),
          "r"(C[3]));
#else
    assert(0);
#endif
  }
};

/// Matrix multiply-add operation: S32 = S8 * U8 + S32
template <>
struct Mma<
  gemm::GemmShape<16,8,16>,
  32,
  int8_t,
  layout::RowMajor,
  uint8_t,
  layout::ColumnMajor,
  int,
  layout::RowMajor,
  OpMultiplyAddSaturate> {

  using Shape = gemm::GemmShape<16,8,16>;

  using ElementA = int8_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<int8_t, 8>;

  using ElementB = uint8_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<uint8_t, 4>;

  using ElementC = int;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<int, 4>;

  using Operator = OpMultiplyAddSaturate;
  using ArchTag = arch::Sm80;

  /// Computes multiply-add
  MCTLASS_HOST_DEVICE
  void operator()(
    FragmentC &d,
    FragmentA const &a,
    FragmentB const &b,
    FragmentC const &c
  ) const {

//#if defined(MCTLASS_ARCH_MMA_SM80_ENABLED)
#if 0 //mctlass not support this data-type and shape now

    uint32_t const *A = reinterpret_cast<uint32_t const *>(&a);
    uint32_t const &B = reinterpret_cast<uint32_t const &>(b);

    int const *C = reinterpret_cast<int const *>(&c);
    int *D = reinterpret_cast<int *>(&d);
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.s32.s8.u8.s32.satfinite {%0,%1,%2,%3}, {%4,%5}, "
        "{%6}, {%7,%8,%9,%10};\n"
        : "=r"(D[0]), "=r"(D[1]), "=r"(D[2]), "=r"(D[3])
        : "r"(A[0]), "r"(A[1]), "r"(B), "r"(C[0]), "r"(C[1]), "r"(C[2]),
          "r"(C[3]));
#else
    assert(0);
#endif
  }
};

/// Matrix multiply-add operation: S32 = U8 * U8 + S32
template <>
struct Mma<
  gemm::GemmShape<16,8,16>,
  32,
  uint8_t,
  layout::RowMajor,
  uint8_t,
  layout::ColumnMajor,
  int,
  layout::RowMajor,
  OpMultiplyAddSaturate> {

  using Shape = gemm::GemmShape<16,8,16>;

  using ElementA = uint8_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<uint8_t, 8>;

  using ElementB = uint8_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<uint8_t, 4>;

  using ElementC = int;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<int, 4>;

  using Operator = OpMultiplyAddSaturate;
  using ArchTag = arch::Sm80;

  /// Computes multiply-add
  MCTLASS_HOST_DEVICE
  void operator()(
    FragmentC &d,
    FragmentA const &a,
    FragmentB const &b,
    FragmentC const &c
  ) const {

//#if defined(MCTLASS_ARCH_MMA_SM80_ENABLED)
#if 0 //mctlass not support this data-type and shape now

    uint32_t const *A = reinterpret_cast<uint32_t const *>(&a);
    uint32_t const &B = reinterpret_cast<uint32_t const &>(b);

    int const *C = reinterpret_cast<int const *>(&c);
    int *D = reinterpret_cast<int *>(&d);
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.s32.u8.u8.s32.satfinite {%0,%1,%2,%3}, {%4,%5}, "
        "{%6}, {%7,%8,%9,%10};\n"
        : "=r"(D[0]), "=r"(D[1]), "=r"(D[2]), "=r"(D[3])
        : "r"(A[0]), "r"(A[1]), "r"(B), "r"(C[0]), "r"(C[1]), "r"(C[2]),
          "r"(C[3]));
#else
    assert(0);
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////
//
// Matrix Multiply 16832 - S8 input, S32 accumulation
//
////////////////////////////////////////////////////////////////////////////////

/// Matrix multiply-add operation: S32 = S8 * S8 + S32
template <>
struct Mma<
  gemm::GemmShape<16,8,32>,
  32,
  int8_t,
  layout::RowMajor,
  int8_t,
  layout::ColumnMajor,
  int,
  layout::RowMajor,
  OpMultiplyAdd> {

  using Shape = gemm::GemmShape<16,8,32>;

  using ElementA = int8_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<int8_t, 16>;

  using ElementB = int8_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<int8_t, 8>;

  using ElementC = int;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<int, 4>;

  using Operator = OpMultiplyAdd;
  using ArchTag = arch::Sm80;

  /// Computes multiply-add
  MCTLASS_HOST_DEVICE
  void operator()(
    FragmentC &d,
    FragmentA const &a,
    FragmentB const &b,
    FragmentC const &c
  ) const {

//#if defined(MCTLASS_ARCH_MMA_SM80_ENABLED)
#if 0 //mctlass not support this data-type and shape now

    uint32_t const *A = reinterpret_cast<uint32_t const *>(&a);
    uint32_t const *B = reinterpret_cast<uint32_t const *>(&b);

    int const *C = reinterpret_cast<int const *>(&c);
    int *D = reinterpret_cast<int *>(&d);
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, "
        "{%8,%9}, {%10,%11,%12,%13};\n"
        : "=r"(D[0]), "=r"(D[1]), "=r"(D[2]), "=r"(D[3])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]),
          "r"(C[0]), "r"(C[1]), "r"(C[2]), "r"(C[3]));
#else
    assert(0);
#endif
  }
};

/// Matrix multiply-add operation: S32 = U8 * S8 + S32
template <>
struct Mma<
  gemm::GemmShape<16,8,32>,
  32,
  uint8_t,
  layout::RowMajor,
  int8_t,
  layout::ColumnMajor,
  int,
  layout::RowMajor,
  OpMultiplyAdd> {

  using Shape = gemm::GemmShape<16,8,32>;

  using ElementA = uint8_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<uint8_t, 16>;

  using ElementB = int8_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<int8_t, 8>;

  using ElementC = int;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<int, 4>;

  using Operator = OpMultiplyAdd;
  using ArchTag = arch::Sm80;

  /// Computes multiply-add
  MCTLASS_HOST_DEVICE
  void operator()(
    FragmentC &d,
    FragmentA const &a,
    FragmentB const &b,
    FragmentC const &c
  ) const {

//#if defined(MCTLASS_ARCH_MMA_SM80_ENABLED)
#if 0 //mctlass not support this data-type and shape now

    uint32_t const *A = reinterpret_cast<uint32_t const *>(&a);
    uint32_t const *B = reinterpret_cast<uint32_t const *>(&b);

    int const *C = reinterpret_cast<int const *>(&c);
    int *D = reinterpret_cast<int *>(&d);
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.u8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, "
        "{%8,%9}, {%10,%11,%12,%13};\n"
        : "=r"(D[0]), "=r"(D[1]), "=r"(D[2]), "=r"(D[3])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]),
          "r"(C[0]), "r"(C[1]), "r"(C[2]), "r"(C[3]));
#else
    assert(0);
#endif
  }
};

/// Matrix multiply-add operation: S32 = S8 * U8 + S32
template <>
struct Mma<
  gemm::GemmShape<16,8,32>,
  32,
  int8_t,
  layout::RowMajor,
  uint8_t,
  layout::ColumnMajor,
  int,
  layout::RowMajor,
  OpMultiplyAdd> {

  using Shape = gemm::GemmShape<16,8,32>;

  using ElementA = int8_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<int8_t, 16>;

  using ElementB = uint8_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<uint8_t, 8>;

  using ElementC = int;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<int, 4>;

  using Operator = OpMultiplyAdd;
  using ArchTag = arch::Sm80;

  /// Computes multiply-add
  MCTLASS_HOST_DEVICE
  void operator()(
    FragmentC &d,
    FragmentA const &a,
    FragmentB const &b,
    FragmentC const &c
  ) const {

//#if defined(MCTLASS_ARCH_MMA_SM80_ENABLED)
#if 0 //mctlass not support this data-type and shape now

    uint32_t const *A = reinterpret_cast<uint32_t const *>(&a);
    uint32_t const *B = reinterpret_cast<uint32_t const *>(&b);

    int const *C = reinterpret_cast<int const *>(&c);
    int *D = reinterpret_cast<int *>(&d);
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.u8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, "
        "{%8,%9}, {%10,%11,%12,%13};\n"
        : "=r"(D[0]), "=r"(D[1]), "=r"(D[2]), "=r"(D[3])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]),
          "r"(C[0]), "r"(C[1]), "r"(C[2]), "r"(C[3]));
#else
    assert(0);
#endif
  }
};

/// Matrix multiply-add operation: S32 = U8 * U8 + S32
template <>
struct Mma<
  gemm::GemmShape<16,8,32>,
  32,
  uint8_t,
  layout::RowMajor,
  uint8_t,
  layout::ColumnMajor,
  int,
  layout::RowMajor,
  OpMultiplyAdd> {

  using Shape = gemm::GemmShape<16,8,32>;

  using ElementA = uint8_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<uint8_t, 16>;

  using ElementB = uint8_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<uint8_t, 8>;

  using ElementC = int;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<int, 4>;

  using Operator = OpMultiplyAdd;
  using ArchTag = arch::Sm80;

  /// Computes multiply-add
  MCTLASS_HOST_DEVICE
  void operator()(
    FragmentC &d,
    FragmentA const &a,
    FragmentB const &b,
    FragmentC const &c
  ) const {

//#if defined(MCTLASS_ARCH_MMA_SM80_ENABLED)
#if 0 //mctlass not support this data-type and shape now

    uint32_t const *A = reinterpret_cast<uint32_t const *>(&a);
    uint32_t const *B = reinterpret_cast<uint32_t const *>(&b);

    int const *C = reinterpret_cast<int const *>(&c);
    int *D = reinterpret_cast<int *>(&d);
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.u8.u8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, "
        "{%8,%9}, {%10,%11,%12,%13};\n"
        : "=r"(D[0]), "=r"(D[1]), "=r"(D[2]), "=r"(D[3])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]),
          "r"(C[0]), "r"(C[1]), "r"(C[2]), "r"(C[3]));
#else
    assert(0);
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////
//
// Matrix Multiply 16832 - S8 input, S32 accumulation - SATURATE
//
////////////////////////////////////////////////////////////////////////////////

/// Matrix multiply-add operation: S32 = S8 * S8 + S32
template <>
struct Mma<
  gemm::GemmShape<16,8,32>,
  32,
  int8_t,
  layout::RowMajor,
  int8_t,
  layout::ColumnMajor,
  int,
  layout::RowMajor,
  OpMultiplyAddSaturate> {

  using Shape = gemm::GemmShape<16,8,32>;

  using ElementA = int8_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<int8_t, 16>;

  using ElementB = int8_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<int8_t, 8>;

  using ElementC = int;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<int, 4>;

  using Operator = OpMultiplyAdd;
  using ArchTag = arch::Sm80;

  /// Computes multiply-add
  MCTLASS_HOST_DEVICE
  void operator()(
    FragmentC &d,
    FragmentA const &a,
    FragmentB const &b,
    FragmentC const &c
  ) const {

#if defined(MCTLASS_ARCH_MMA_SM80_ENABLED)

  uint32_t const * A = reinterpret_cast<uint32_t const *>(&a);
  uint32_t const * B = reinterpret_cast<uint32_t const *>(&b);

  int const *C = reinterpret_cast<int const *>(&c);
  int *D = reinterpret_cast<int *>(&d);
  // asm volatile(
  //     "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32.satfinite {%0,%1,%2,%3}, "
  //    "{%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
  //    : "=r"(D[0]), "=r"(D[1]), "=r"(D[2]), "=r"(D[3])
  //    : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]),
  //      "r"(C[0]), "r"(C[1]), "r"(C[2]), "r"(C[3]));
  auto temp = __builtin_mxc_mma_16x16x16i8(A[0], B[0], {C[0], C[1], C[2], C[3]});
  D[0] = temp[0];
  D[1] = temp[1];
  D[2] = temp[2];
  D[3] = temp[3];

  temp= __builtin_mxc_mma_16x16x16i8(A[1], B[1], {D[0], D[1], D[2], D[3]});
  D[0] = temp[0];
  D[1] = temp[1];
  D[2] = temp[2];
  D[3] = temp[3];
  D[4] = temp[4];
  D[5] = temp[5];
  D[6] = temp[6];
  D[7] = temp[7];
#endif
  }
};

/// Matrix multiply-add operation: S32 = U8 * S8 + S32
template <>
struct Mma<
  gemm::GemmShape<16,8,32>,
  32,
  uint8_t,
  layout::RowMajor,
  int8_t,
  layout::ColumnMajor,
  int,
  layout::RowMajor,
  OpMultiplyAddSaturate> {

  using Shape = gemm::GemmShape<16,8,32>;

  using ElementA = uint8_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<uint8_t, 16>;

  using ElementB = int8_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<int8_t, 8>;

  using ElementC = int;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<int, 4>;

  using Operator = OpMultiplyAddSaturate;
  using ArchTag = arch::Sm80;

  /// Computes multiply-add
  MCTLASS_HOST_DEVICE
  void operator()(
    FragmentC &d,
    FragmentA const &a,
    FragmentB const &b,
    FragmentC const &c
  ) const {

//#if defined(MCTLASS_ARCH_MMA_SM80_ENABLED)
#if 0 //mctlass not support this data-type and shape now

    uint32_t const *A = reinterpret_cast<uint32_t const *>(&a);
    uint32_t const *B = reinterpret_cast<uint32_t const *>(&b);

    int const *C = reinterpret_cast<int const *>(&c);
    int *D = reinterpret_cast<int *>(&d);
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.u8.s8.s32.satfinite {%0,%1,%2,%3}, "
        "{%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
        : "=r"(D[0]), "=r"(D[1]), "=r"(D[2]), "=r"(D[3])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]),
          "r"(C[0]), "r"(C[1]), "r"(C[2]), "r"(C[3]));
#else
    assert(0);
#endif
  }
};

/// Matrix multiply-add operation: S32 = S8 * U8 + S32
template <>
struct Mma<
  gemm::GemmShape<16,8,32>,
  32,
  int8_t,
  layout::RowMajor,
  uint8_t,
  layout::ColumnMajor,
  int,
  layout::RowMajor,
  OpMultiplyAddSaturate> {

  using Shape = gemm::GemmShape<16,8,32>;

  using ElementA = int8_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<int8_t, 16>;

  using ElementB = uint8_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<uint8_t, 8>;

  using ElementC = int;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<int, 4>;

  using Operator = OpMultiplyAdd;
  using ArchTag = arch::Sm80;

  /// Computes multiply-add
  MCTLASS_HOST_DEVICE
  void operator()(
    FragmentC &d,
    FragmentA const &a,
    FragmentB const &b,
    FragmentC const &c
  ) const {

//#if defined(MCTLASS_ARCH_MMA_SM80_ENABLED)
#if 0 //mctlass not support this data-type and shape now

    uint32_t const *A = reinterpret_cast<uint32_t const *>(&a);
    uint32_t const *B = reinterpret_cast<uint32_t const *>(&b);

    int const *C = reinterpret_cast<int const *>(&c);
    int *D = reinterpret_cast<int *>(&d);
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.u8.s32.satfinite {%0,%1,%2,%3}, "
        "{%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
        : "=r"(D[0]), "=r"(D[1]), "=r"(D[2]), "=r"(D[3])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]),
          "r"(C[0]), "r"(C[1]), "r"(C[2]), "r"(C[3]));
#else
    assert(0);
#endif
  }
};

/// Matrix multiply-add operation: S32 = U8 * U8 + S32
template <>
struct Mma<
  gemm::GemmShape<16,8,32>,
  32,
  uint8_t,
  layout::RowMajor,
  uint8_t,
  layout::ColumnMajor,
  int,
  layout::RowMajor,
  OpMultiplyAddSaturate> {

  using Shape = gemm::GemmShape<16,8,32>;

  using ElementA = uint8_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<uint8_t, 16>;

  using ElementB = uint8_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<uint8_t, 8>;

  using ElementC = int;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<int, 4>;

  using Operator = OpMultiplyAddSaturate;
  using ArchTag = arch::Sm80;

  /// Computes multiply-add
  MCTLASS_HOST_DEVICE
  void operator()(
    FragmentC &d,
    FragmentA const &a,
    FragmentB const &b,
    FragmentC const &c
  ) const {

//#if defined(MCTLASS_ARCH_MMA_SM80_ENABLED)
#if 0 //mctlass not support this data-type and shape now

    uint32_t const *A = reinterpret_cast<uint32_t const *>(&a);
    uint32_t const *B = reinterpret_cast<uint32_t const *>(&b);

    int const *C = reinterpret_cast<int const *>(&c);
    int *D = reinterpret_cast<int *>(&d);
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.u8.u8.s32.satfinite {%0,%1,%2,%3}, "
        "{%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
        : "=r"(D[0]), "=r"(D[1]), "=r"(D[2]), "=r"(D[3])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]),
          "r"(C[0]), "r"(C[1]), "r"(C[2]), "r"(C[3]));
#else
    assert(0);
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////
//
// Matrix Multiply 16864 - S4 input, S32 accumulation
//
////////////////////////////////////////////////////////////////////////////////

/// Matrix multiply-add operation: S32 = S4 * S4 + S32
template <>
struct Mma<
  gemm::GemmShape<16, 8, 64>,
  32,
  mctlass::int4b_t,
  layout::RowMajor,
  mctlass::int4b_t,
  layout::ColumnMajor,
  int,
  layout::RowMajor,
  OpMultiplyAdd> {

  using Shape = gemm::GemmShape<16, 8, 64>;

  using ElementA = mctlass::int4b_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<mctlass::int4b_t, 32>;

  using ElementB = mctlass::int4b_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<mctlass::int4b_t, 16>;

  using ElementC = int;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<int, 4>;

  using Operator = OpMultiplyAdd;
  using ArchTag = arch::Sm80;

  /// Computes multiply-add
  MCTLASS_HOST_DEVICE
  void operator()(
    FragmentC &d,
    FragmentA const &a,
    FragmentB const &b,
    FragmentC const &c
  ) const {

//#if defined(MCTLASS_ARCH_MMA_SM80_ENABLED)
#if 0 //mctlass not support this data-type and shape now

    uint32_t const *A = reinterpret_cast<uint32_t const *>(&a);
    uint32_t const *B = reinterpret_cast<uint32_t const *>(&b);

    int const *C = reinterpret_cast<int const *>(&c);
    int *D = reinterpret_cast<int *>(&d);
    asm volatile(
        "mma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, "
        "{%8,%9}, {%10,%11,%12,%13};\n"
        : "=r"(D[0]), "=r"(D[1]), "=r"(D[2]), "=r"(D[3])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]),
          "r"(C[0]), "r"(C[1]), "r"(C[2]), "r"(C[3]));
#else
    MCTLASS_UNUSED(a);
    MCTLASS_UNUSED(b);
    MCTLASS_UNUSED(c);
    MCTLASS_UNUSED(d);
    assert(0);
#endif
  }
};

/// Matrix multiply-add operation: S32 = U4 * S4 + S32
template <>
struct Mma<
  gemm::GemmShape<16, 8, 64>,
  32,
  mctlass::uint4b_t,
  layout::RowMajor,
  mctlass::int4b_t,
  layout::ColumnMajor,
  int,
  layout::RowMajor,
  OpMultiplyAdd> {

  using Shape = gemm::GemmShape<16, 8, 64>;

  using ElementA = mctlass::uint4b_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<mctlass::uint4b_t, 32>;

  using ElementB = mctlass::int4b_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<mctlass::int4b_t, 16>;

  using ElementC = int;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<int, 4>;

  using Operator = OpMultiplyAdd;
  using ArchTag = arch::Sm80;

  /// Computes multiply-add
  MCTLASS_HOST_DEVICE
  void operator()(
    FragmentC &d,
    FragmentA const &a,
    FragmentB const &b,
    FragmentC const &c
  ) const {

//#if defined(MCTLASS_ARCH_MMA_SM80_ENABLED)
#if 0 //mctlass not support this data-type and shape now

    uint32_t const *A = reinterpret_cast<uint32_t const *>(&a);
    uint32_t const *B = reinterpret_cast<uint32_t const *>(&b);

    int const *C = reinterpret_cast<int const *>(&c);
    int *D = reinterpret_cast<int *>(&d);
    asm volatile(
        "mma.sync.aligned.m16n8k64.row.col.s32.u4.s4.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, "
        "{%8,%9}, {%10,%11,%12,%13};\n"
        : "=r"(D[0]), "=r"(D[1]), "=r"(D[2]), "=r"(D[3])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]),
          "r"(C[0]), "r"(C[1]), "r"(C[2]), "r"(C[3]));
#else
    MCTLASS_UNUSED(a);
    MCTLASS_UNUSED(b);
    MCTLASS_UNUSED(c);
    MCTLASS_UNUSED(d);
    assert(0);
#endif
  }
};

/// Matrix multiply-add operation: S32 = S4 * U4 + S32
template <>
struct Mma<
  gemm::GemmShape<16, 8, 64>,
  32,
  mctlass::int4b_t,
  layout::RowMajor,
  mctlass::uint4b_t,
  layout::ColumnMajor,
  int,
  layout::RowMajor,
  OpMultiplyAdd> {

  using Shape = gemm::GemmShape<16, 8, 64>;

  using ElementA = mctlass::int4b_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<mctlass::int4b_t, 32>;

  using ElementB = mctlass::uint4b_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<mctlass::uint4b_t, 16>;

  using ElementC = int;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<int, 4>;

  using Operator = OpMultiplyAdd;
  using ArchTag = arch::Sm80;

  /// Computes multiply-add
  MCTLASS_HOST_DEVICE
  void operator()(
    FragmentC &d,
    FragmentA const &a,
    FragmentB const &b,
    FragmentC const &c
  ) const {

//#if defined(MCTLASS_ARCH_MMA_SM80_ENABLED)
#if 0 //mctlass not support this data-type and shape now

    uint32_t const *A = reinterpret_cast<uint32_t const *>(&a);
    uint32_t const *B = reinterpret_cast<uint32_t const *>(&b);

    int const *C = reinterpret_cast<int const *>(&c);
    int *D = reinterpret_cast<int *>(&d);
    asm volatile(
        "mma.sync.aligned.m16n8k64.row.col.s32.s4.u4.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, "
        "{%8,%9}, {%10,%11,%12,%13};\n"
        : "=r"(D[0]), "=r"(D[1]), "=r"(D[2]), "=r"(D[3])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]),
          "r"(C[0]), "r"(C[1]), "r"(C[2]), "r"(C[3]));
#else
    MCTLASS_UNUSED(a);
    MCTLASS_UNUSED(b);
    MCTLASS_UNUSED(c);
    MCTLASS_UNUSED(d);
    assert(0);
#endif
  }
};

/// Matrix multiply-add operation: S32 = U4 * U4 + S32
template <>
struct Mma<
  gemm::GemmShape<16, 8, 64>,
  32,
  mctlass::uint4b_t,
  layout::RowMajor,
  mctlass::uint4b_t,
  layout::ColumnMajor,
  int,
  layout::RowMajor,
  OpMultiplyAdd> {

  using Shape = gemm::GemmShape<16, 8, 64>;

  using ElementA = mctlass::uint4b_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<mctlass::uint4b_t, 32>;

  using ElementB = mctlass::uint4b_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<mctlass::uint4b_t, 16>;

  using ElementC = int;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<int, 4>;

  using Operator = OpMultiplyAdd;
  using ArchTag = arch::Sm80;

  /// Computes multiply-add
  MCTLASS_HOST_DEVICE
  void operator()(
    FragmentC &d,
    FragmentA const &a,
    FragmentB const &b,
    FragmentC const &c
  ) const {

//#if defined(MCTLASS_ARCH_MMA_SM80_ENABLED)
#if 0 //mctlass not support this data-type and shape now

    uint32_t const *A = reinterpret_cast<uint32_t const *>(&a);
    uint32_t const *B = reinterpret_cast<uint32_t const *>(&b);

    int const *C = reinterpret_cast<int const *>(&c);
    int *D = reinterpret_cast<int *>(&d);
    asm volatile(
        "mma.sync.aligned.m16n8k64.row.col.s32.u4.u4.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, "
        "{%8,%9}, {%10,%11,%12,%13};\n"
        : "=r"(D[0]), "=r"(D[1]), "=r"(D[2]), "=r"(D[3])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]),
          "r"(C[0]), "r"(C[1]), "r"(C[2]), "r"(C[3]));
#else
    MCTLASS_UNUSED(a);
    MCTLASS_UNUSED(b);
    MCTLASS_UNUSED(c);
    MCTLASS_UNUSED(d);
    assert(0);
#endif
  }
};


////////////////////////////////////////////////////////////////////////////////
//
// Matrix Multiply 16864 - S4 input, S32 accumulation - SATURATE
//
////////////////////////////////////////////////////////////////////////////////

/// Matrix multiply-add operation: S32 = S4 * S4 + S32
template <>
struct Mma<
  gemm::GemmShape<16, 8, 64>,
  32,
  mctlass::int4b_t,
  layout::RowMajor,
  mctlass::int4b_t,
  layout::ColumnMajor,
  int,
  layout::RowMajor,
  OpMultiplyAddSaturate> {

  using Shape = gemm::GemmShape<16, 8, 64>;

  using ElementA = mctlass::int4b_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<mctlass::int4b_t, 32>;

  using ElementB = mctlass::int4b_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<mctlass::int4b_t, 16>;

  using ElementC = int;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<int, 4>;

  using Operator = OpMultiplyAdd;
  using ArchTag = arch::Sm80;

  /// Computes multiply-add
  MCTLASS_HOST_DEVICE
  void operator()(
    FragmentC &d,
    FragmentA const &a,
    FragmentB const &b,
    FragmentC const &c
  ) const {

//#if defined(MCTLASS_ARCH_MMA_SM80_ENABLED)
#if 0 //mctlass not support this data-type and shape now

  uint32_t const * A = reinterpret_cast<uint32_t const *>(&a);
  uint32_t const * B = reinterpret_cast<uint32_t const *>(&b);

  int const *C = reinterpret_cast<int const *>(&c);
  int *D = reinterpret_cast<int *>(&d);
  asm volatile(
      "mma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32.satfinite {%0,%1,%2,%3}, "
      "{%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
      : "=r"(D[0]), "=r"(D[1]), "=r"(D[2]), "=r"(D[3])
      : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]),
        "r"(C[0]), "r"(C[1]), "r"(C[2]), "r"(C[3]));
#else
    MCTLASS_UNUSED(a);
    MCTLASS_UNUSED(b);
    MCTLASS_UNUSED(c);
    MCTLASS_UNUSED(d);
    assert(0);
#endif
  }
};

/// Matrix multiply-add operation: S32 = U4 * S4 + S32
template <>
struct Mma<
  gemm::GemmShape<16, 8, 64>,
  32,
  mctlass::uint4b_t,
  layout::RowMajor,
  mctlass::int4b_t,
  layout::ColumnMajor,
  int,
  layout::RowMajor,
  OpMultiplyAddSaturate> {

  using Shape = gemm::GemmShape<16, 8, 64>;

  using ElementA = mctlass::uint4b_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<mctlass::uint4b_t, 32>;

  using ElementB = mctlass::int4b_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<mctlass::int4b_t, 16>;

  using ElementC = int;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<int, 4>;

  using Operator = OpMultiplyAddSaturate;
  using ArchTag = arch::Sm80;

  /// Computes multiply-add
  MCTLASS_HOST_DEVICE
  void operator()(
    FragmentC &d,
    FragmentA const &a,
    FragmentB const &b,
    FragmentC const &c
  ) const {

//#if defined(MCTLASS_ARCH_MMA_SM80_ENABLED)
#if 0 //mctlass not support this data-type and shape now

    uint32_t const *A = reinterpret_cast<uint32_t const *>(&a);
    uint32_t const *B = reinterpret_cast<uint32_t const *>(&b);

    int const *C = reinterpret_cast<int const *>(&c);
    int *D = reinterpret_cast<int *>(&d);
    asm volatile(
        "mma.sync.aligned.m16n8k64.row.col.s32.u4.s4.s32.satfinite {%0,%1,%2,%3}, "
        "{%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
        : "=r"(D[0]), "=r"(D[1]), "=r"(D[2]), "=r"(D[3])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]),
          "r"(C[0]), "r"(C[1]), "r"(C[2]), "r"(C[3]));
#else
    MCTLASS_UNUSED(a);
    MCTLASS_UNUSED(b);
    MCTLASS_UNUSED(c);
    MCTLASS_UNUSED(d);
    assert(0);
#endif
  }
};

/// Matrix multiply-add operation: S32 = S4 * U4 + S32
template <>
struct Mma<
  gemm::GemmShape<16, 8, 64>,
  32,
  mctlass::int4b_t,
  layout::RowMajor,
  mctlass::uint4b_t,
  layout::ColumnMajor,
  int,
  layout::RowMajor,
  OpMultiplyAddSaturate> {

  using Shape = gemm::GemmShape<16, 8, 64>;

  using ElementA = mctlass::int4b_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<mctlass::int4b_t, 32>;

  using ElementB = mctlass::uint4b_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<mctlass::uint4b_t, 16>;

  using ElementC = int;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<int, 4>;

  using Operator = OpMultiplyAdd;
  using ArchTag = arch::Sm80;

  /// Computes multiply-add
  MCTLASS_HOST_DEVICE
  void operator()(
    FragmentC &d,
    FragmentA const &a,
    FragmentB const &b,
    FragmentC const &c
  ) const {

//#if defined(MCTLASS_ARCH_MMA_SM80_ENABLED)
#if 0 //mctlass not support this data-type and shape now

    uint32_t const *A = reinterpret_cast<uint32_t const *>(&a);
    uint32_t const *B = reinterpret_cast<uint32_t const *>(&b);

    int const *C = reinterpret_cast<int const *>(&c);
    int *D = reinterpret_cast<int *>(&d);
    asm volatile(
        "mma.sync.aligned.m16n8k64.row.col.s32.s4.u4.s32.satfinite {%0,%1,%2,%3}, "
        "{%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
        : "=r"(D[0]), "=r"(D[1]), "=r"(D[2]), "=r"(D[3])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]),
          "r"(C[0]), "r"(C[1]), "r"(C[2]), "r"(C[3]));
#else
    MCTLASS_UNUSED(a);
    MCTLASS_UNUSED(b);
    MCTLASS_UNUSED(c);
    MCTLASS_UNUSED(d);
    assert(0);
#endif
  }
};

/// Matrix multiply-add operation: S32 = U4 * U4 + S32
template <>
struct Mma<
  gemm::GemmShape<16, 8, 64>,
  32,
  mctlass::uint4b_t,
  layout::RowMajor,
  mctlass::uint4b_t,
  layout::ColumnMajor,
  int,
  layout::RowMajor,
  OpMultiplyAddSaturate> {

  using Shape = gemm::GemmShape<16, 8, 64>;

  using ElementA = mctlass::uint4b_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<mctlass::uint4b_t, 32>;

  using ElementB = mctlass::uint4b_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<mctlass::uint4b_t, 16>;

  using ElementC = int;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<int, 4>;

  using Operator = OpMultiplyAddSaturate;
  using ArchTag = arch::Sm80;

  /// Computes multiply-add
  MCTLASS_HOST_DEVICE
  void operator()(
    FragmentC &d,
    FragmentA const &a,
    FragmentB const &b,
    FragmentC const &c
  ) const {

//#if defined(MCTLASS_ARCH_MMA_SM80_ENABLED)
#if 0 //mctlass not support this data-type and shape now

    uint32_t const *A = reinterpret_cast<uint32_t const *>(&a);
    uint32_t const *B = reinterpret_cast<uint32_t const *>(&b);

    int const *C = reinterpret_cast<int const *>(&c);
    int *D = reinterpret_cast<int *>(&d);
    asm volatile(
        "mma.sync.aligned.m16n8k64.row.col.s32.u4.u4.s32.satfinite {%0,%1,%2,%3}, "
        "{%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
        : "=r"(D[0]), "=r"(D[1]), "=r"(D[2]), "=r"(D[3])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]),
          "r"(C[0]), "r"(C[1]), "r"(C[2]), "r"(C[3]));
#else
    MCTLASS_UNUSED(a);
    MCTLASS_UNUSED(b);
    MCTLASS_UNUSED(c);
    MCTLASS_UNUSED(d);
    assert(0);
#endif
  }
};

/// Matrix multiply-add operation: S32 = B1 & B1 + S32
template <>
struct Mma<
  gemm::GemmShape<16,8,256>,
  32,
  mctlass::uint1b_t,
  layout::RowMajor,
  mctlass::uint1b_t,
  layout::ColumnMajor,
  int,
  layout::RowMajor,
  OpMultiplyAdd> {

  using Shape = gemm::GemmShape<16,8,256>;

  using ElementA = mctlass::uint1b_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<mctlass::uint1b_t, 128>;

  using ElementB = mctlass::uint1b_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<mctlass::uint1b_t, 64>;

  using ElementC = int32_t;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<int32_t, 4>;

  using Operator = OpMultiplyAdd;
  using ArchTag = arch::Sm80;

  /// Computes multiply-add
  MCTLASS_HOST_DEVICE
  void operator()(
    FragmentC &d,
    FragmentA const &a,
    FragmentB const &b,
    FragmentC const &c
  ) const {

//#if defined(MCTLASS_ARCH_MMA_SM80_ENABLED)
#if 0 //mctlass not support this data-type and shape now

    uint32_t const *A = reinterpret_cast<uint32_t const *>(&a);
    uint32_t const *B = reinterpret_cast<uint32_t const *>(&b);

    int const *C = reinterpret_cast<int const *>(&c);
    int *D = reinterpret_cast<int *>(&d);
    asm volatile(
        "mma.sync.aligned.m16n8k256.row.col.s32.b1.b1.s32.and.popc {%0,%1,%2,%3}, "
        "{%4,%5,%6,%7}, "
        "{%8,%9}, {%10,%11,%12,%13};\n"
        : "=r"(D[0]), "=r"(D[1]), "=r"(D[2]), "=r"(D[3])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]),
          "r"(C[0]), "r"(C[1]), "r"(C[2]), "r"(C[3]));
#else
    MCTLASS_UNUSED(a);
    MCTLASS_UNUSED(b);
    MCTLASS_UNUSED(c);
    MCTLASS_UNUSED(d);
    assert(0);
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////
//
// Matrix Multiply 168256 - B1 input, S32 accumulation - XOR,POPC
//
////////////////////////////////////////////////////////////////////////////////

/// Matrix multiply-add operation: S32 = B1 & B1 + S32
template <>
struct Mma<
  gemm::GemmShape<16,8,256>,
  32,
  mctlass::uint1b_t,
  layout::RowMajor,
  mctlass::uint1b_t,
  layout::ColumnMajor,
  int,
  layout::RowMajor,
  OpXorPopc> {

  using Shape = gemm::GemmShape<16,8,256>;

  using ElementA = mctlass::uint1b_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<mctlass::uint1b_t, 128>;

  using ElementB = mctlass::uint1b_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<mctlass::uint1b_t, 64>;

  using ElementC = int;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<int, 4>;

  using Operator = OpXorPopc;
  using ArchTag = arch::Sm80;

  /// Computes multiply-add
  MCTLASS_HOST_DEVICE
  void operator()(
    FragmentC &d,
    FragmentA const &a,
    FragmentB const &b,
    FragmentC const &c
  ) const {

//#if defined(MCTLASS_ARCH_MMA_SM80_ENABLED)
#if 0 //mctlass not support this data-type and shape now

    uint32_t const *A = reinterpret_cast<uint32_t const *>(&a);
    uint32_t const *B = reinterpret_cast<uint32_t const *>(&b);

    int const *C = reinterpret_cast<int const *>(&c);
    int *D = reinterpret_cast<int *>(&d);
    asm volatile(
        "mma.sync.aligned.m16n8k256.row.col.s32.b1.b1.s32.xor.popc {%0,%1,%2,%3}, "
        "{%4,%5,%6,%7}, "
        "{%8,%9}, {%10,%11,%12,%13};\n"
        : "=r"(D[0]), "=r"(D[1]), "=r"(D[2]), "=r"(D[3])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]),
          "r"(C[0]), "r"(C[1]), "r"(C[2]), "r"(C[3]));
#else

    MCTLASS_UNUSED(a);
    MCTLASS_UNUSED(b);
    MCTLASS_UNUSED(c);
    MCTLASS_UNUSED(d);
    assert(0);

#endif // defined(MCTLASS_ARCH_MMA_SM80_ENABLED)
  }
};

////////////////////////////////////////////////////////////////////////////////
//
// Matrix Multiply 161616
//
////////////////////////////////////////////////////////////////////////////////

/// Matrix multiply-add operation: I32 = I8 * I8 + I32
template <>
struct Mma<
  gemm::GemmShape<16, 16, 16>,
  32,
  int8_t,
  layout::RowMajor,
  int8_t,
  layout::ColumnMajor,
  int,
  layout::RowMajor,
  OpMultiplyAddSaturate> {

  using Shape = gemm::GemmShape<16, 16, 16>;

  using ElementA = int8_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<int8_t, 4>;

  using ElementB = int8_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<int8_t, 4>;

  using ElementC = int;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<int, 4>;

  using Operator = OpMultiplyAdd;
  using ArchTag = arch::Sm80;

  /// Computes multiply-add
  MCTLASS_HOST_DEVICE
  void operator()(
    FragmentC &d,
    FragmentA const &a,
    FragmentB const &b,
    FragmentC const &c
  ) const {

#if defined(MCTLASS_ARCH_MMA_SM80_ENABLED)
    uint32_t const *A = reinterpret_cast<uint32_t const *>(&a);
    uint32_t const *B = reinterpret_cast<uint32_t const *>(&b);

    int const *C = reinterpret_cast<int const *>(&c);
    int *D = reinterpret_cast<int *>(&d);

    auto temp = __builtin_mxc_mma_16x16x16i8(B[0], A[0], {C[0], C[1], C[2], C[3]});
    D[0] = temp[0];
    D[1] = temp[1];
    D[2] = temp[2];
    D[3] = temp[3];
#else

    MCTLASS_UNUSED(d);
    MCTLASS_UNUSED(a);
    MCTLASS_UNUSED(b);
    MCTLASS_UNUSED(c);
    MCTLASS_NOT_IMPLEMENTED();

#endif
  }
};

template <>
struct Mma<
  gemm::GemmShape<16, 16, 32>,
  32,
  int8_t,
  layout::RowMajor,
  int8_t,
  layout::ColumnMajor,
  int,
  layout::RowMajor,
  OpMultiplyAddSaturate> {

  using Shape = gemm::GemmShape<16, 16, 32>;

  using ElementA = int8_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<int8_t, 8>;

  using ElementB = int8_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<int8_t, 8>;

  using ElementC = int;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<int, 4>;

  using Operator = OpMultiplyAdd;
  using ArchTag = arch::Sm80;

  /// Computes multiply-add
  MCTLASS_HOST_DEVICE
  void operator()(
    FragmentC &d,
    FragmentA const &a,
    FragmentB const &b,
    FragmentC const &c
  ) const {

#if defined(MCTLASS_ARCH_MMA_SM80_ENABLED)
    uint32_t const *A = reinterpret_cast<uint32_t const *>(&a);
    uint32_t const *B = reinterpret_cast<uint32_t const *>(&b);

    int const *C = reinterpret_cast<int const *>(&c);
    int *D = reinterpret_cast<int *>(&d);

    auto temp = __builtin_mxc_mma_16x16x16i8(B[0], A[0], {C[0], C[1], C[2], C[3]});
    auto res = __builtin_mxc_mma_16x16x16i8(B[1], A[1], temp);
    D[0] = res[0];
    D[1] = res[1];
    D[2] = res[2];
    D[3] = res[3];
#else

    MCTLASS_UNUSED(d);
    MCTLASS_UNUSED(a);
    MCTLASS_UNUSED(b);
    MCTLASS_UNUSED(c);
    MCTLASS_NOT_IMPLEMENTED();

#endif
  }
};

/// Matrix multiply-add operation: F16 = F16 * F16 + F16
template <>
struct Mma<
  gemm::GemmShape<16, 16, 16>,
  32,
  half_t,
  layout::RowMajor,
  half_t,
  layout::ColumnMajor,
  half_t,
  layout::RowMajor,
  OpMultiplyAdd> {

  using Shape = gemm::GemmShape<16, 16, 16>;

  using ElementA = half_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<half_t, 4>;

  using ElementB = half_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<half_t, 4>;

  using ElementC = half_t;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<half_t, 4>;

  using Operator = OpMultiplyAdd;

  /// Computes multiply-add
  MCTLASS_HOST_DEVICE
  void operator()(FragmentC &d, FragmentA const &a, FragmentB const &b,
                  FragmentC const &c) const {

#if defined(MCTLASS_ARCH_MMA_SM80_ENABLED)

    auto results = __builtin_mxc_mma_16x16x16f16(
      b.to_macahalf4(),
      a.to_macahalf4(),
      {c[0].get().to_macahalf(), c[1].get().to_macahalf(),
       c[2].get().to_macahalf(), c[3].get().to_macahalf()});

    d[0] = mctlass::half_t(results[0]);
    d[1] = mctlass::half_t(results[1]);
    d[2] = mctlass::half_t(results[2]);
    d[3] = mctlass::half_t(results[3]);
#else

   MCTLASS_UNUSED(d);
   MCTLASS_UNUSED(a);
   MCTLASS_UNUSED(b);
   MCTLASS_UNUSED(c);
   MCTLASS_NOT_IMPLEMENTED();

#endif
  }
};

/// Matrix multiply-add operation: F32 = F16 * F16 + F32
template <>
struct Mma<
  gemm::GemmShape<16, 16, 16>,
  32,
  half_t,
  layout::RowMajor,
  half_t,
  layout::ColumnMajor,
  float,
  layout::RowMajor,
  OpMultiplyAdd> {

  using Shape = gemm::GemmShape<16, 16, 16>;

  using ElementA = half_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<half_t, 4>;

  using ElementB = half_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<half_t, 4>;

  using ElementC = float;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<float, 4>;

  using Operator = OpMultiplyAdd;
  using ArchTag = arch::Sm80;

  /// Computes multiply-add
  MCTLASS_HOST_DEVICE
  void operator()(
    FragmentC &d,
    FragmentA const &a,
    FragmentB const &b,
    FragmentC const &c
  ) const {

#if defined(MCTLASS_ARCH_MMA_SM80_ENABLED)
    auto results = __builtin_mxc_mma_16x16x16f16(
      b.to_macahalf4(),
      a.to_macahalf4(),
      c.to_macafloat4()
    );

    d[0] = results[0];
    d[1] = results[1];
    d[2] = results[2];
    d[3] = results[3];
#else

    MCTLASS_UNUSED(d);
    MCTLASS_UNUSED(a);
    MCTLASS_UNUSED(b);
    MCTLASS_UNUSED(c);
    MCTLASS_NOT_IMPLEMENTED();

#endif
  }
};

////////////////////////////////////////////////////////////////////////////////

/// Matrix multiply-add operation: F16 = F16 * F16 + F16
template <>
struct Mma<
  gemm::GemmShape<16, 16, 16>,
  32,
  bfloat16_t,
  layout::RowMajor,
  bfloat16_t,
  layout::ColumnMajor,
  bfloat16_t,
  layout::RowMajor,
  OpMultiplyAdd> {

  using Shape = gemm::GemmShape<16, 16, 16>;

  using ElementA = bfloat16_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<bfloat16_t, 4>;

  using ElementB = bfloat16_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<bfloat16_t, 4>;

  using ElementC = bfloat16_t;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<bfloat16_t, 4>;

  using Operator = OpMultiplyAdd;

  /// Computes multiply-add
  MCTLASS_HOST_DEVICE
  void operator()(FragmentC &d, FragmentA const &a, FragmentB const &b,
                  FragmentC const &c) const {

#if defined(MCTLASS_ARCH_MMA_SM80_ENABLED)

    auto results = __builtin_mxc_mma_16x16x16bf16(
        b.to_macahalf4(), a.to_macahalf4(), c.to_macafloat4());

    d[0] = mctlass::bfloat16_t(results[0]);
    d[1] = mctlass::bfloat16_t(results[1]);
    d[2] = mctlass::bfloat16_t(results[2]);
    d[3] = mctlass::bfloat16_t(results[3]);
#else

   MCTLASS_UNUSED(d);
   MCTLASS_UNUSED(a);
   MCTLASS_UNUSED(b);
   MCTLASS_UNUSED(c);
   MCTLASS_NOT_IMPLEMENTED();

#endif
  }
};

/// Matrix multiply-add operation: F32 = F16 * F16 + F32
template <>
struct Mma<
  gemm::GemmShape<16, 16, 16>,
  32,
  bfloat16_t,
  layout::RowMajor,
  bfloat16_t,
  layout::ColumnMajor,
  float,
  layout::RowMajor,
  OpMultiplyAdd> {

  using Shape = gemm::GemmShape<16, 16, 16>;

  using ElementA = bfloat16_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<bfloat16_t, 4>;

  using ElementB = bfloat16_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<bfloat16_t, 4>;

  using ElementC = float;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<float, 4>;

  using Operator = OpMultiplyAdd;
  using ArchTag = arch::Sm80;

  /// Computes multiply-add
  MCTLASS_HOST_DEVICE
  void operator()(
    FragmentC &d,
    FragmentA const &a,
    FragmentB const &b,
    FragmentC const &c
  ) const {

#if defined(MCTLASS_ARCH_MMA_SM80_ENABLED)
    auto results = __builtin_mxc_mma_16x16x16bf16(
        b.to_macahalf4(), a.to_macahalf4(), c.to_macafloat4());

    d[0] = results[0];
    d[1] = results[1];
    d[2] = results[2];
    d[3] = results[3];
#else

    MCTLASS_UNUSED(d);
    MCTLASS_UNUSED(a);
    MCTLASS_UNUSED(b);
    MCTLASS_UNUSED(c);
    MCTLASS_NOT_IMPLEMENTED();

#endif
  }
};

////////////////////////////////////////////////////////////////////////////////

} // namespace arch
} // namespace mctlass

/////////////////////////////////////////////////////////////////////////////////////////////////
