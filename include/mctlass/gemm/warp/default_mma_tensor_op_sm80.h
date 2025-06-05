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
    \brief Default warp-level GEMM operators selected by data type, size, and layouts of operands.
*/

#pragma once

#include "mctlass/mctlass.h"
#include "mctlass/numeric_types.h"
#include "mctlass/arch/mma.h"
#include "mctlass/gemm/warp/mma_tensor_op.h"
#include "mctlass/gemm/warp/mma_tensor_op_fast_f32.h"
#include "mctlass/gemm/warp/default_mma_tensor_op.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace mctlass {
namespace gemm {
namespace warp {

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Partial Specialization - inputs and output types are float - uses BF16 internally
template <
    /// Shape of one matrix production operation (concept: GemmShape)
    typename WarpShape_,
    /// Layout of A matrix (concept: MatrixLayout)
    typename LayoutA,
    /// Layout of B matrix (concept: MatrixLayout)
    typename LayoutB,
    /// Layout of C matrix (concept: MatrixLayout)
    typename LayoutC,
    /// Number of partitions along K dimension
    int PartitionsK,
    /// Store the accumulators in row major or column major.  Row major is used
    /// when output layout is interleaved.
    bool AccumulatorsInRowMajor>
struct DefaultMmaTensorOp<
  WarpShape_, 
  GemmShape<16, 8, 8>, 
  float, LayoutA, 
  float, LayoutB, 
  float, LayoutC, 
  arch::OpMultiplyAddFastBF16, 
  PartitionsK, AccumulatorsInRowMajor> {

  // Uses BF16 internally
  using Policy = mctlass::gemm::warp::MmaTensorOpPolicy<
      mctlass::arch::Mma<
        GemmShape<16, 8, 8>, 
        32, 
        bfloat16_t, mctlass::layout::RowMajor, 
        bfloat16_t, mctlass::layout::ColumnMajor,
        float, mctlass::layout::RowMajor, 
        arch::OpMultiplyAdd
      >,
      mctlass::MatrixShape<1, 1> >;

  // Define the warp-level tensor op
  using Type = mctlass::gemm::warp::MmaTensorOp<
      WarpShape_, float, LayoutA, float, LayoutB, float, LayoutC,
      Policy, PartitionsK, AccumulatorsInRowMajor>;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Partial Specialization - inputs and output types are float - uses F16 internally
template <
    /// Shape of one matrix production operation (concept: GemmShape)
    typename WarpShape_,
    /// Layout of A matrix (concept: MatrixLayout)
    typename LayoutA,
    /// Layout of B matrix (concept: MatrixLayout)
    typename LayoutB,
    /// Layout of C matrix (concept: MatrixLayout)
    typename LayoutC,
    /// Number of partitions along K dimension
    int PartitionsK,
    /// Store the accumulators in row major or column major.  Row major is used
    /// when output layout is interleaved.
    bool AccumulatorsInRowMajor>
struct DefaultMmaTensorOp<
  WarpShape_, 
  GemmShape<16, 8, 8>, 
  float, LayoutA, 
  float, LayoutB, 
  float, LayoutC, 
  arch::OpMultiplyAddFastF16, 
  PartitionsK, AccumulatorsInRowMajor> {

  // Uses F16 internally
  using Policy = mctlass::gemm::warp::MmaTensorOpPolicy<
      mctlass::arch::Mma<
        GemmShape<16, 8, 8>, 
        32, 
        half_t, mctlass::layout::RowMajor, 
        half_t, mctlass::layout::ColumnMajor,
        float, mctlass::layout::RowMajor, 
        arch::OpMultiplyAdd
      >,
      mctlass::MatrixShape<1, 1> >;

  // Define the warp-level tensor op
  using Type = mctlass::gemm::warp::MmaTensorOp<
      WarpShape_, float, LayoutA, float, LayoutB, float, LayoutC,
      Policy, PartitionsK, AccumulatorsInRowMajor>;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Partial Specialization - inputs and output types are float - uses TF32 internally
template <
    /// Shape of one matrix production operation (concept: GemmShape)
    typename WarpShape_,
    /// Shape of target matrix multiply instruction (concept: GemmShape)
    typename InstructionShape_,
    /// Layout of A matrix (concept: MatrixLayout)
    typename LayoutA,
    /// Layout of B matrix (concept: MatrixLayout)
    typename LayoutB,
    /// Layout of C matrix (concept: MatrixLayout)
    typename LayoutC,
    /// Number of partitions along K dimension
    int PartitionsK,
    /// Store the accumulators in row major or column major.  Row major is used
    /// when output layout is interleaved.
    bool AccumulatorsInRowMajor>
struct DefaultMmaTensorOp<
  WarpShape_, 
  InstructionShape_, 
  float, LayoutA, 
  float, LayoutB, 
  float, LayoutC, 
  arch::OpMultiplyAdd, PartitionsK, AccumulatorsInRowMajor> {

  // Uses TF32 internally
  using Policy = mctlass::gemm::warp::MmaTensorOpPolicy<
      mctlass::arch::Mma<
        InstructionShape_, 
        32, 
        float, mctlass::layout::RowMajor, 
        float, mctlass::layout::ColumnMajor,
        float, mctlass::layout::RowMajor, 
        arch::OpMultiplyAdd
      >,
      mctlass::MatrixShape<1, 1> >;

  // Define the warp-level tensor op
  using Type = mctlass::gemm::warp::MmaTensorOp<
      WarpShape_, float, LayoutA, float, LayoutB, float, LayoutC,
      Policy, PartitionsK, AccumulatorsInRowMajor>;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Partial Specialization - inputs and output types are float - uses TF32 internally
template <
    /// Shape of one matrix production operation (concept: GemmShape)
    typename WarpShape_,
    /// Data type of A operand
    typename LayoutA,
    /// Data type of B operand
    typename LayoutB,
    /// Layout of C matrix (concept: MatrixLayout)
    typename LayoutC,
    /// Number of partitions along K dimension
    int PartitionsK,
    /// Store the accumulators in row major or column major.  Row major is used
    /// when output layout is interleaved.
    bool AccumulatorsInRowMajor>
struct DefaultMmaTensorOp<
  WarpShape_, 
  GemmShape<16, 16, 8>, 
  mctlass::tfloat32_t, LayoutA, 
  mctlass::tfloat32_t, LayoutB, 
  float, LayoutC, 
  arch::OpMultiplyAdd, PartitionsK, AccumulatorsInRowMajor> {

  // Uses TF32 internally
  using Policy = mctlass::gemm::warp::MmaTensorOpPolicy<
      mctlass::arch::Mma<
        GemmShape<16, 16, 8>, 
        32, 
        mctlass::tfloat32_t, mctlass::layout::RowMajor, 
        mctlass::tfloat32_t, mctlass::layout::ColumnMajor,
        float, mctlass::layout::RowMajor, 
        arch::OpMultiplyAdd
      >,
      mctlass::MatrixShape<1, 1> >;

  // Define the warp-level tensor op
  using Type = mctlass::gemm::warp::MmaTensorOp<
      WarpShape_, mctlass::tfloat32_t, LayoutA, mctlass::tfloat32_t, LayoutB, float, LayoutC,
      Policy, PartitionsK, AccumulatorsInRowMajor>;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Partial Specialization - inputs and output types are float - uses TF32 for Fast Accurate FP32
template <
    /// Shape of one matrix production operation (concept: GemmShape)
    typename WarpShape_,
    /// Shape of target matrix multiply instruction (concept: GemmShape)
    typename InstructionShape_,
    /// Layout of A matrix (concept: MatrixLayout)
    typename LayoutA,
    /// Layout of B matrix (concept: MatrixLayout)
    typename LayoutB,
    /// Layout of C matrix (concept: MatrixLayout)
    typename LayoutC,
    /// Number of partitions along K dimension
    int PartitionsK,
    /// Store the accumulators in row major or column major.  Row major is used
    /// when output layout is interleaved.
    bool AccumulatorsInRowMajor>
struct DefaultMmaTensorOp<
  WarpShape_, 
  InstructionShape_, 
  float, LayoutA, 
  float, LayoutB, 
  float, LayoutC, 
  arch::OpMultiplyAddFastF32, PartitionsK, AccumulatorsInRowMajor> {

  // Uses TF32 internally
  using Policy = mctlass::gemm::warp::MmaTensorOpPolicy<
      mctlass::arch::Mma<
        InstructionShape_, 
        32, 
        mctlass::tfloat32_t, mctlass::layout::RowMajor, 
        mctlass::tfloat32_t, mctlass::layout::ColumnMajor,
        float, mctlass::layout::RowMajor, 
        arch::OpMultiplyAdd
      >,
      mctlass::MatrixShape<1, 1> >;

  // Define the warp-level tensor op
  using Type = mctlass::gemm::warp::MmaTensorOpFastF32<
      WarpShape_, float, LayoutA, float, LayoutB, float, LayoutC,
      Policy, PartitionsK, AccumulatorsInRowMajor>;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace warp
} // namespace gemm
} // namespace mctlass

/////////////////////////////////////////////////////////////////////////////////////////////////

#include "mctlass/gemm/warp/mma_complex_tensor_op_tile_iterator_sm80.h"

/////////////////////////////////////////////////////////////////////////////////////////////////
