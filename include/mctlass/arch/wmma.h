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
    \brief Templates exposing architecture support for warp matrix multiply-add (WMMA) operations
*/

#pragma once

// MCTLASS WMMA does not support clang at present.
//#if !(defined(__clang__) && defined(__MACA__))
#if 1

//#if (__CUDACC_VER_MAJOR__ >= 9)
#if 1
#if defined(__MACA_ARCH__)
   #define MCTLASS_ARCH_WMMA_ENABLED
   #define MCTLASS_ARCH_WMMA_SM70_ENABLED
   #include "__clang_maca_mma_functions.h"
#endif
#endif

//#if (__CUDACC_VER_MAJOR__ >= 10)
#if 0
#if defined(__MACA_ARCH__)
#define MCTLASS_ARCH_INTEGER_MATRIX_MULTIPLY_ENABLED
#define MCTLASS_ARCH_WMMA_SM72_ENABLED
#endif
#endif

//#if (__CUDACC_VER_MAJOR__ >= 10)
#if 0
#if defined(__MACA_ARCH__)
#define MCTLASS_SUBBYTE_INTEGER_MATRIX_MULTIPLY_ENABLED
#define MCTLASS_ARCH_WMMA_SM75_ENABLED
#endif
#endif

#endif //!(defined(__clang__) && defined(__MACA__))

#if defined(MCTLASS_ARCH_WMMA_ENABLED)

// #include <mma.h>
#include "mctlass/arch/mma.h"
#include "mctlass/array.h"
#include "mctlass/numeric_types.h"
#include "mctlass/gemm/gemm.h"


/////////////////////////////////////////////////////////////////////////////////////////////////

namespace mctlass {
namespace arch {

////////////////////////////////////////////////////////////////////////////////////////////////
/// Statically maps mctlass data types => nvcuda::wmma data types
/////////////////////////////////////////////////////////////////////////////////////////////////
template <typename Type_>
struct MctlassToWmmaDataType{
  using Type = Type_;
};

/// Statically maps mctlass::half_t => __half
template<>
struct MctlassToWmmaDataType<mctlass::half_t> {
  using Type = __half;
};

#if defined(__MACA_ARCH__) && (__CUDACC_VER_MAJOR__ >= 11)
template<>
struct MctlassToWmmaDataType<mctlass::bfloat16_t> {
  using Type = maca_bfloat16;
};
#endif

/// Statically maps int8_t => char
template<>
struct MctlassToWmmaDataType<int8_t> {
  using Type = signed char;
};

/// Statically maps uint8_t => char
template<>
struct MctlassToWmmaDataType<uint8_t> {
  using Type = unsigned char;
};

/// Statically maps int32_t => int
template<>
struct MctlassToWmmaDataType<int32_t> {
  using Type = int;
};

//#if defined(MCTLASS_SUBBYTE_INTEGER_MATRIX_MULTIPLY_ENABLED)
#if 0
/// Statically maps mctlass::int4b_t => experimental::precision::s4
template<>
struct MctlassToWmmaDataType<mctlass::int4b_t> {
  using Type = nvcuda::wmma::experimental::precision::s4;
};

/// Statically maps mctlass::uint4b_t => experimental::precision::s4
template<>
struct MctlassToWmmaDataType<mctlass::uint4b_t> {
  using Type = nvcuda::wmma::experimental::precision::u4;
};

/// Statically maps mctlass::uint1b_t => experimental::precision::b1
template<>
struct MctlassToWmmaDataType<mctlass::uint1b_t> {
  using Type = nvcuda::wmma::experimental::precision::b1;
};
#endif

////////////////////////////////////////////////////////////////////////////////////////////////
/// Statically maps mctlass::layout => nvcuda::wmma layout tags
////////////////////////////////////////////////////////////////////////////////////////////////
template <typename Layout_>
struct MctlassToWmmaLayout {
};

/// Statically maps mctlass::layout::RowMajor => wmma::row_major layout tags
template <>
struct MctlassToWmmaLayout<mctlass::layout::RowMajor> {
  using Layout = mxmaca::wmma::row_major;
  static mxmaca::wmma::layout_t const value = mxmaca::wmma::layout_t::mem_row_major;
};

////////////////////////////////////////////////////////////////////////////////////////////////
/// Statically maps mctlass::layout::RowMajor => wmma::row_major layout tags
////////////////////////////////////////////////////////////////////////////////////////////////
template <>
struct MctlassToWmmaLayout<mctlass::layout::ColumnMajor> {
  using Layout = mxmaca::wmma::col_major;
  static mxmaca::wmma::layout_t const value = mxmaca::wmma::layout_t::mem_col_major;
};

////////////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////////////////////
/// Statically maps nvcuda::wmma data types => mctlass data types
/////////////////////////////////////////////////////////////////////////////////////////////////
template <typename Type_>
struct WmmaToMctlassDataType{
  using Type = Type_;
};

/// Statically maps __half => mctlass::half_t
template<>
struct WmmaToMctlassDataType<__half> {
  using Type = mctlass::half_t;
};

#if defined(__MACA_ARCH__) && (__CUDACC_VER_MAJOR__ >= 11)
template<>
struct WmmaToMctlassDataType<maca_bfloat16> {
  using Type = mctlass::bfloat16_t;
};
#endif

////////////////////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////////////////////
// WMMA template structure defines nvcuda::wmma::fragments and static assertion chaeks
// for a specific template paramterized data type (Element[A|B|C]), layout (Layout[A|B|C]), 
// and native wmma size (Shape)
/////////////////////////////////////////////////////////////////////////////////////////////////
template <  
  typename Shape_,                                   ///< Size of the matrix product (concept: GemmShape)
  typename ElementA_,                                ///< Data type of A elements 
  typename LayoutA_,                                 ///< Layout of A matrix (concept: MatrixLayout)  
  typename ElementB_,                                ///< Data type of B elements
  typename LayoutB_,                                 ///< Layout of B matrix (concept: MatrixLayout)  
  typename ElementC_,                                ///< Element type of C matrix  
  typename LayoutC_,                                 /// Layout of C matrix (concept: MatrixLayout)
  typename Operator_ = mctlass::arch::OpMultiplyAdd   ///< Inner product operator (multiply-add, xor.popc)
>
struct Wmma;
/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace arch
} // namespace mctlass

/////////////////////////////////////////////////////////////////////////////////////////////////

//
// Specializations for each compute capability
//
#ifdef MCTLASS_ARCH_WMMA_SM70_ENABLED
#include "mctlass/arch/wmma_sm70.h"
#endif

#ifdef MCTLASS_ARCH_WMMA_SM72_ENABLED
#include "mctlass/arch/wmma_sm72.h"
#endif

#ifdef MCTLASS_ARCH_WMMA_SM75_ENABLED
#include "mctlass/arch/wmma_sm75.h"
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////

#endif //MCTLASS_ARCH_WMMA_ENABLED
