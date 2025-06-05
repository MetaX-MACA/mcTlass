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
    \brief Defines basic properties needed by CTA-level GEMMs assuming
   expectations about data layout of the global memory fragments, data types,
   and internal tile sizes.

      Partial specializations for threadblock::Mma operations targeting TensorOp
   instructions.
*/

#pragma once

#include "mctlass/array.h"
#include "mctlass/mctlass.h"

#include "mctlass/layout/tensor_op_multiplicand_sm75.h"
#include "mctlass/layout/tensor_op_multiplicand_sm80.h"

#include "mctlass/gemm/warp/default_mma_with_reduction_tensor_op.h"
#include "mctlass/gemm/warp/mma_tensor_op_tile_iterator_sm80.h"

#include "mctlass/gemm/threadblock/default_mma_core.h"

#include "mctlass/matrix_shape.h"
#include "mctlass/numeric_types.h"
#include "mctlass/transform/pitch_linear_thread_map.h"
#include "mctlass/transform/threadblock/regular_tile_access_iterator_tensor_op.h"
#include "mctlass/transform/threadblock/regular_tile_access_iterator_tensor_op_sm80.h"
#include "mctlass/transform/threadblock/regular_tile_access_iterator_pitch_linear.h"
#include "mctlass/gemm/threadblock/mma_with_reduction_multistage.h"

////////////////////////////////////////////////////////////////////////////////

namespace mctlass {
namespace gemm {
namespace threadblock {

////////////////////////////////////////////////////////////////////////////////

/// Template defininng default matrix multiply operators inferred from threadblock tile size,
/// global memory data layout, and target math instruction.
template <
    /// Shape of threadblock-scoped matrix multiply operator
    typename Shape_,
    /// Shape of warp-level matrix multiply operator
    typename WarpShape,
    /// Shape of one matrix production operation (concept: GemmShape)
    typename InstructionShape,
    /// Element data type of A operand
    typename ElementA,
    /// Layout of operand A
    typename LayoutA,
    /// Element data type of B operand
    typename ElementB,
    /// Layout of operand B
    typename LayoutB,
    /// Data type of accumulator
    typename ElementC,
    /// Layout of accumulator
    typename LayoutC,
    /// Indicates type of math operator (arch::OpClassSimt or arch::OpClassTensorOp)
    typename OperatorClass,
    ///                                                                                               
    bool ReduceKForA_,
    /// Number of stages
    int Stages = 2,
    /// Operation performed by MMA
    typename Operator = typename platform::conditional<
        (platform::is_same<OperatorClass,
                           mctlass::arch::OpClassTensorOp>::value) &&
            (platform::is_same<ElementA, int8_t>::value ||
             platform::is_same<ElementA, int4b_t>::value ||
             platform::is_same<ElementA, uint8_t>::value ||
             platform::is_same<ElementA, uint4b_t>::value),
        mctlass::arch::OpMultiplyAddSaturate,
        mctlass::arch::OpMultiplyAdd>::type,
    /// Store the accumulators in row major or column major.  Row major is used
    /// when output layout is interleaved.
    bool AccumulatorsInRowMajor = false,
    /// Cache operation of operand A
    mctlass::arch::CacheOperation::Kind CacheOpA =
        mctlass::arch::CacheOperation::Global,
    /// Cache operation of operand B
    mctlass::arch::CacheOperation::Kind CacheOpB =
        mctlass::arch::CacheOperation::Global,
    /// per-element transformation for elements of A
    ComplexTransform TransformA = ComplexTransform::kNone,
    /// per-element transformation for elements of B
    ComplexTransform TransformB = ComplexTransform::kNone,
    bool IsComplex = false// (is_complex<ElementA>::value || is_complex<ElementB>::value)
>
struct DefaultMmaWithReductionCore {
  using Base = DefaultMmaCore<Shape_,
                              WarpShape,
                              InstructionShape,
                              ElementA,
                              LayoutA,
                              ElementB,
                              LayoutB,
                              ElementC,
                              LayoutC,
                              OperatorClass,
                              Stages,
                              Operator,
                              AccumulatorsInRowMajor,
                              CacheOpA,
                              CacheOpB,
                              TransformA,
                              TransformB,
                              IsComplex>;
  using Shape = Shape_;
  using IteratorThreadMapA = typename Base::IteratorThreadMapA;
  using IteratorThreadMapB = typename Base::IteratorThreadMapB;
  using SmemIteratorA = typename Base::SmemIteratorA;
  using SmemIteratorB = typename Base::SmemIteratorB;
  using SmemLayoutA = typename Base::SmemLayoutA;
  using SmemLayoutB = typename Base::SmemLayoutB;
  using WarpCount = typename Base::WarpCount;

  static mctlass::arch::CacheOperation::Kind const kCacheOpA = CacheOpA;
  static mctlass::arch::CacheOperation::Kind const kCacheOpB = CacheOpB;
   
  // Define the warp-level tensor op
  using MmaTensorOp = typename mctlass::gemm::warp::DefaultMmaWithReductionTensorOp<
      WarpShape, InstructionShape, ElementA, SmemLayoutA, ElementB, SmemLayoutB,
      ElementC, LayoutC, Operator, ReduceKForA_, WarpCount::kK>::Type;

  /// Policy used to define MmaPipelined
  using MmaPolicy = MmaPolicy<MmaTensorOp, MatrixShape<0, 0>,
                                        MatrixShape<0, 0>, WarpCount::kK>;
};

////////////////////////////////////////////////////////////////////////////////

}  // namespace threadblock
}  // namespace gemm
}  // namespace mctlass
