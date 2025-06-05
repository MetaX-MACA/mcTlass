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
  \brief Epilogue for threadblock scoped GEMMs and convolution using Tensor Ops.

  The epilogue rearranges the result of a matrix product through shared memory to match canonical
  tensor layouts in global memory. Epilogues support conversion and reduction operations.

*/

#pragma once

#include "mctlass/mctlass.h"
#include "mctlass/numeric_types.h"
#include "mctlass/array.h"

#include "mctlass/gemm/gemm.h"

#include "mctlass/epilogue/thread/linear_combination.h"
#include "mctlass/epilogue/thread/conversion_op.h"
#include "mctlass/epilogue/thread/reduction_op.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace mctlass {
namespace epilogue {
namespace threadblock {

////////////////////////////////////////////////////////////////////////////////

/// Epilogue operator
template <
  typename Shape_,                          ///< Shape of threadblock tile (concept: GemmShape)
  typename WarpMmaOperator_,                ///< Warp-level MMA operator (concept: gemm::warp::MmaTensorOp)
  int PartitionsK,                          ///< Number of partitions of the K dimension
  typename OutputTileIterator_,             ///< Tile iterator reading and writing output tensors
  typename AccumulatorFragmentIterator_,    ///< Fragment iterator selecting accumulators
  typename WarpTileIterator_,               ///< Warp-scoped tile iterator writing accumulators to SMEM
  typename SharedLoadIterator_,             ///< Threadblock-scoped tile iterator loading from SMEM
  typename OutputOp_                        ///< Output operator
>
class EpilogueDirectStore {
public:

  using Shape = Shape_;
  using WarpMmaOperator = WarpMmaOperator_;
  using WarpShape = typename WarpMmaOperator_::Shape;
  static int const kPartitionsK = PartitionsK;
  using OutputTileIterator = OutputTileIterator_;
  using AccumulatorFragmentIterator = AccumulatorFragmentIterator_;
  using WarpTileIterator = WarpTileIterator_;
  using SharedLoadIterator = SharedLoadIterator_;
  using OutputOp = OutputOp_;
  using Padding = MatrixShape<0, 0>;

  using Layout = layout::RowMajor;
  using LongIndex = typename Layout::LongIndex;

  /// The complete warp-level accumulator tile
  using AccumulatorTile = typename AccumulatorFragmentIterator::AccumulatorTile;

  /// Accumulator element
  using ElementAccumulator = typename WarpTileIterator::Element;

  /// Output element
  using ElementOutput = typename OutputTileIterator::Element;

  /// Output access size
  static int const kElementsPerAccess = OutputTileIterator::kElementsPerAccess;

  /// Tensor reference to destination tensor
  using TensorRef = typename OutputTileIterator::TensorRef;

  /// Tensor reference to sync tensor
  using SyncTensorRef = typename mctlass::TensorRef<int, mctlass::layout::PackedVectorLayout>;

  /// Const tensor reference to source tensor
  using ConstTensorRef = typename OutputTileIterator::ConstTensorRef;

  /// Array type used to output
  using OutputAccessType = Array<
    typename OutputTileIterator::Element, OutputTileIterator::kElementsPerAccess>;

  /// Array type used by output functor
  using AccumulatorAccessType = Array<typename WarpTileIterator::Element, OutputTileIterator::kElementsPerAccess>;

  /// Number of warps
  using WarpCount = gemm::GemmShape<
    Shape::kM / WarpShape::kM,
    Shape::kN / WarpShape::kN,
    kPartitionsK
  >;

  /// Use this to control the granularity of one epilogue 'iteration'
  static int const kFragmentsPerIteration = 1;

  static int constexpr kSmemTiles = 1;
  static int constexpr kSmemPointerOffset = 0;

  /// Shared storage allocation needed by the epilogue
  struct SharedStorage { } ;

private:

  // Assume accumulator tile is multipile interleaved 32x32 tile.
  static int const kElementsPerPartial = 4;
  using EleShapePerPatial = typename platform::conditional<
                              platform::is_same<ElementAccumulator, float>::value,
                              MatrixShape<2, 2>,
                              MatrixShape<1, 4> >::type;
  static int const kElementsPerMma = 8;
  static int const kAccumulatorPatials = 2;
  using QuadShapePerPatialMma = MatrixShape<4, 4>;

  static_assert(OutputOp::kCount >= 2,
    "The direct store epilogue for Tensor Ops requires the output functor have kCount >= 2.");

private:

  LongIndex warp_offset;
  int thread_idx;
  int warp_idx;
  int lane_idx;
  int warp_m, warp_n; // warp coordinates within a cta
  int tid_m, tid_n;   // thread coordinates within a warp

public:

  /// Constructor
  MCTLASS_DEVICE
  EpilogueDirectStore(
    SharedStorage &shared_storage,    ///< Shared storage object
    int thread_idx_,                   ///< ID of a thread within the threadblock
    int warp_idx_,                     ///< ID of warp within threadblock
    int lane_idx_                     ///< Id of thread within warp
  ):
    thread_idx(thread_idx_),
    warp_idx(warp_idx_),
    lane_idx(lane_idx_)
  {

    // warp offsetting calculations
    warp_offset = warp_idx * WarpShape::kM * WarpShape::kN;
    int warp_id_mn = warp_idx % (WarpCount::kM * WarpShape::kN);
    warp_m = warp_id_mn % WarpCount::kM;
    warp_n = warp_id_mn / WarpCount::kM;
    MatrixCoord warp_offset_coord(warp_m*WarpShape::kM, warp_n*WarpShape::kN);

    // thread offsetting calculations
    //int quad = (lane_idx >> 2);
    //int lane_in_quad = (lane_idx & 3);
    const int quad = ((lane_idx >> 4) << 2);
    const int lane_in_quad = (lane_idx & 0x7);

    // this seems to be te correct layout
    tid_m = quad;
    //tid_n = 2 * lane_in_quad;
    tid_n = lane_in_quad;

  }

  /// Streams the result to global memory
  MCTLASS_DEVICE
  void operator()(
    OutputOp const &output_op,                    ///< Output operator
    OutputTileIterator destination_iterator,      ///< Tile iterator for destination
    AccumulatorTile const &accumulators,          ///< Complete warp-level accumulator tile
    OutputTileIterator source_iterator) {         ///< Threadblock tile coordinate in GEMM (in units of threadblock tiles)

    if (!output_op.is_source_needed()) {
      compute_source_not_needed_(output_op, destination_iterator, accumulators);
    }
    else {
      compute_source_needed_(output_op, destination_iterator, accumulators, source_iterator);
    }
  }

private:

  /// Streams the result to global memory
  MCTLASS_DEVICE
  void compute_source_needed_(
    OutputOp const &output_op,                    ///< Output operator
    OutputTileIterator destination_iterator,      ///< Tile iterator for destination
    AccumulatorTile const &accumulators,          ///< Complete warp-level accumulator tile
    OutputTileIterator source_iterator) {         ///< Threadblock tile coordinate in GEMM (in units of threadblock tiles)

    const int kAccumBlockN = 4;//2;
    const int kThreadsM = 16;//8;
    const int kThreadsN = 2;//4;
    const int kBlockM = WarpShape::kM / kThreadsM;

    /// Array type used to output
    using OutputAccessType = AlignedArray<ElementOutput, kAccumBlockN>;

    /// Array type passed to the output operator - unused elements are optimized away
    using OutputFragmentType = Array<ElementOutput, OutputOp::kCount>;

    /// Array type used by output functor
    using AccumulatorAccessType = Array<ElementAccumulator, kAccumBlockN>;

    /// Array type used by output functor
    using AccumulatorFragmentType = Array<ElementAccumulator, OutputOp::kCount>;

    AccumulatorAccessType const *accumulator_pair = reinterpret_cast<AccumulatorAccessType const *>(&accumulators);

    MCTLASS_PRAGMA_UNROLL
    for (int accum_m_idx = 0; accum_m_idx < WarpShape::kM / kThreadsM; accum_m_idx++) {

      int accum_m = kThreadsM * accum_m_idx;
      int mL = destination_iterator.threadblock_offset.row() + WarpShape::kM * warp_m + tid_m + accum_m;
      int nL_base = destination_iterator.threadblock_offset.column() + WarpShape::kN * warp_n + tid_n;

      //ElementOutput *output_ptr = destination_iterator.pointer + mL * destination_iterator.stride;
      //ElementOutput *source_ptr = source_iterator.pointer + mL * source_iterator.stride;

      int const kIterationsN = WarpShape::kN / kThreadsN / kAccumBlockN;

      MCTLASS_PRAGMA_UNROLL
      for (int accum_n_idx = 0; accum_n_idx < kIterationsN; accum_n_idx++) {

        int accum_idx = accum_m_idx + kBlockM * accum_n_idx;
        //int accum_n = kThreadsM * accum_n_idx;
        int accum_n = (kThreadsM / 2) * accum_n_idx;

        // mL and nL are logical coordinate in 2D mapping of epilogue's 4D output
        int nL = nL_base + accum_n;

        //bool guard = (mL < destination_iterator.extent.row()) && (nL < destination_iterator.extent.column());
        bool guard = (nL < destination_iterator.extent.column());

        AccumulatorFragmentType accum_fragment;
        reinterpret_cast<AccumulatorAccessType &>(accum_fragment) = accumulator_pair[accum_idx];

        OutputFragmentType output_fragment;

        if(guard) {
          // reinterpret_cast<OutputAccessType &>(output_fragment) =
          //   *reinterpret_cast<OutputAccessType const *>(source_ptr + nL);
          for (int i = 0; i < OutputOp::kCount; ++i) {
            const bool guard_row = ((mL + i) < destination_iterator.extent.row());
            if (guard_row) {
              ElementOutput *source_ptr = source_iterator.pointer + (mL + i) * source_iterator.stride;
              output_fragment[i] = source_ptr[nL];
            }
          }
        }

        // Perform output operator
        output_fragment = output_op(accum_fragment, output_fragment);

        if(guard) {
          // Store
          //*reinterpret_cast<OutputAccessType *>(output_ptr + nL) = reinterpret_cast<OutputAccessType const &>(output_fragment);
          for (int i = 0; i < OutputOp::kCount; ++i) {
            const bool guard_row = ((mL + i) < destination_iterator.extent.row());
            if (guard_row) {
              ElementOutput *output_ptr = destination_iterator.pointer + (mL + i) * destination_iterator.stride;
              output_ptr[nL] = output_fragment[i];
            }
          }
        }
      }
    }
  }

  /// Streams the result to global memory
  MCTLASS_DEVICE
  void compute_source_not_needed_(
    OutputOp const &output_op,                    ///< Output operator
    OutputTileIterator destination_iterator,      ///< Tile iterator for destination
    AccumulatorTile const &accumulators) {         ///< Threadblock tile coordinate in GEMM (in units of threadblock tiles)

    const int kAccumBlockN = 4;//2;
    const int kThreadsM = 16;//8;
    const int kThreadsN = 2;//4;
    const int kBlockM = WarpShape::kM / kThreadsM;

    /// Array type used to output
    using OutputAccessType = AlignedArray<ElementOutput, kAccumBlockN>;

    /// Array type passed to the output operator - unused elements are optimized away
    using OutputFragmentType = Array<ElementOutput, OutputOp::kCount>;

    /// Array type used by output functor
    using AccumulatorAccessType = Array<ElementAccumulator, kAccumBlockN>;

    /// Array type used by output functor
    using AccumulatorFragmentType = Array<ElementAccumulator, OutputOp::kCount>;

    AccumulatorAccessType const *accumulator_pair = reinterpret_cast<AccumulatorAccessType const *>(&accumulators);

    MCTLASS_PRAGMA_UNROLL
    for (int accum_m_idx = 0; accum_m_idx < WarpShape::kM / kThreadsM; accum_m_idx++) {

      int accum_m = kThreadsM * accum_m_idx;
      int mL = destination_iterator.threadblock_offset.row() + WarpShape::kM * warp_m + tid_m + accum_m;
      int nL_base = destination_iterator.threadblock_offset.column() + WarpShape::kN * warp_n + tid_n;

      //ElementOutput *output_ptr = destination_iterator.pointer + mL * destination_iterator.stride;

      int const kIterationsN = WarpShape::kN / kThreadsN / kAccumBlockN;

      MCTLASS_PRAGMA_UNROLL
      for (int accum_n_idx = 0; accum_n_idx < kIterationsN; accum_n_idx++) {

        int accum_idx = accum_m_idx + kBlockM * accum_n_idx;
       // int accum_n = kThreadsM * accum_n_idx;
        int accum_n = (kThreadsM / 2) * accum_n_idx;

        // mL and nL are logical coordinate in 2D mapping of epilogue's 4D output
        int nL = nL_base + accum_n;
        // bool guard = (mL < destination_iterator.extent.row()) && (nL < destination_iterator.extent.column());
        bool guard = (nL < destination_iterator.extent.column());

        AccumulatorFragmentType accum_fragment;
        reinterpret_cast<AccumulatorAccessType &>(accum_fragment) = accumulator_pair[accum_idx];

        OutputFragmentType output_fragment;

        // Perform output operator
        output_fragment = output_op(accum_fragment);
        if(guard) {
          // Store
          // *reinterpret_cast<OutputAccessType *>(output_ptr + nL) =
          //   reinterpret_cast<OutputAccessType const &>(output_fragment);
          for (int i = 0; i < OutputOp::kCount; ++i) {

            const bool guard_row = ((mL + i) < destination_iterator.extent.row());
            if(guard_row) {
              ElementOutput *output_ptr = destination_iterator.pointer + (mL + i) * destination_iterator.stride;
              output_ptr[nL] = output_fragment[i];
            }
          }
        }
      }
    }
  }
};

////////////////////////////////////////////////////////////////////////////////

/// Epilogue operator
template <
  typename Shape_,                          ///< Shape of threadblock tile (concept: GemmShape)
  typename WarpMmaOperator_,                ///< Warp-level MMA operator (concept: gemm::warp::MmaTensorOp)
  int PartitionsK,                          ///< Number of partitions of the K dimension
  typename OutputTileIterator_,             ///< Tile iterator reading and writing output tensors
  typename AccumulatorFragmentIterator_,    ///< Fragment iterator selecting accumulators
  typename WarpTileIterator_,               ///< Warp-scoped tile iterator writing accumulators to SMEM
  typename SharedLoadIterator_,             ///< Threadblock-scoped tile iterator loading from SMEM
  typename OutputOp_                        ///< Output operator
>
class MacaEpilogueDirectStore {
public:

  using Shape = Shape_;
  using WarpMmaOperator = WarpMmaOperator_;
  using WarpShape = typename WarpMmaOperator_::Shape;
  static int const kPartitionsK = PartitionsK;
  using OutputTileIterator = OutputTileIterator_;
  using AccumulatorFragmentIterator = AccumulatorFragmentIterator_;
  using WarpTileIterator = WarpTileIterator_;
  using SharedLoadIterator = SharedLoadIterator_;
  using OutputOp = OutputOp_;
  using Padding = MatrixShape<0, 0>;

  using Layout = layout::RowMajor;
  using LongIndex = typename Layout::LongIndex;
  using LayoutB = typename WarpMmaOperator_::LayoutB;
  using LayoutA = typename WarpMmaOperator_::LayoutA;

  /// The complete warp-level accumulator tile
  using AccumulatorTile = typename AccumulatorFragmentIterator::AccumulatorTile;

  /// Accumulator element
  using ElementAccumulator = typename WarpTileIterator::Element;

  /// Output element
  using ElementOutput = typename OutputTileIterator::Element;

  /// Output access size
  static int const kElementsPerAccess = OutputTileIterator::kElementsPerAccess;

  /// Tensor reference to destination tensor
  using TensorRef = typename OutputTileIterator::TensorRef;

  /// Tensor reference to sync tensor
  using SyncTensorRef = typename mctlass::TensorRef<int, mctlass::layout::PackedVectorLayout>;

  /// Const tensor reference to source tensor
  using ConstTensorRef = typename OutputTileIterator::ConstTensorRef;

  /// Array type used to output
  using OutputAccessType = Array<
    typename OutputTileIterator::Element, OutputTileIterator::kElementsPerAccess>;

  /// Array type used by output functor
  using AccumulatorAccessType = Array<typename WarpTileIterator::Element, OutputTileIterator::kElementsPerAccess>;

  /// Number of warps
  using WarpCount = gemm::GemmShape<
    Shape::kM / WarpShape::kM,
    Shape::kN / WarpShape::kN,
    kPartitionsK
  >;

  /// Use this to control the granularity of one epilogue 'iteration'
  static int const kFragmentsPerIteration = 1;

  static int constexpr kSmemTiles = 1;
  static int constexpr kSmemPointerOffset = 0;

  /// Shared storage allocation needed by the epilogue
  struct SharedStorage {

    //
    // Type definitions
    //

    /// Element type of shared memory
    using Element = typename WarpTileIterator::Element;

    /// Tensor reference to shared memory allocation
    using TensorRef = typename WarpTileIterator::TensorRef;

    /// Layout of shared memory allocation
    using Layout = typename WarpTileIterator::Layout;

    /// Logical shape of the shared memory tile written to by all warps.
    using Shape = MatrixShape<
      WarpCount::kM * WarpTileIterator::Shape::kRow * WarpCount::kK,
      WarpCount::kN * WarpTileIterator::Shape::kColumn
    >;

    /// Shape of the shared memory allocation for the epilogue
    using StorageShape = MatrixShape<
      Shape::kRow * kFragmentsPerIteration,
      Shape::kColumn
    >;

    //
    // Data members
    //

    AlignedBuffer<Element, StorageShape::kCount> storage;

    //
    // Methods
    //

    /// Returns a pointer to the shared memory buffer
    MCTLASS_DEVICE
    Element *data() {
      return storage.data();
    }

    /// Returns a tensor reference to the shared memory buffer
    MCTLASS_DEVICE
    TensorRef reference() {
      return TensorRef(
        storage.data(),
        Layout::packed({StorageShape::kRow, StorageShape::kColumn}));
    }
  };

private:

  // Assume accumulator tile is multipile interleaved 32x32 tile.
  static int const kElementsPerPartial = 4;
  using EleShapePerPatial = typename platform::conditional<
                              platform::is_same<ElementAccumulator, float>::value,
                              MatrixShape<2, 2>,
                              MatrixShape<1, 4> >::type;
  static int const kElementsPerMma = 8;
  static int const kAccumulatorPatials = 2;
  using QuadShapePerPatialMma = MatrixShape<4, 4>;

  static_assert(OutputOp::kCount >= 2,
    "The direct store epilogue for Tensor Ops requires the output functor have kCount >= 2.");

private:

  ElementOutput *sm_pointer_;
  int thread_idx;
  int warp_idx;
  int lane_idx;
  int tid_m, tid_n;   // thread coordinates within a warp

public:

  /// Constructor
  MCTLASS_DEVICE
  MacaEpilogueDirectStore(
    SharedStorage &shared_storage,    ///< Shared storage object
    int thread_idx_,                   ///< ID of a thread within the threadblock
    int warp_idx_,                     ///< ID of warp within threadblock
    int lane_idx_                     ///< Id of thread within warp
  ):
    sm_pointer_(reinterpret_cast<ElementOutput *>(shared_storage.data())),
    thread_idx(thread_idx_),
    warp_idx(warp_idx_),
    lane_idx(lane_idx_)
  {

    int warp_id_mn = warp_idx % (WarpCount::kM * WarpShape::kN);

    tid_m = (lane_idx & 0xf) + WarpShape::kM * (warp_id_mn % WarpCount::kM);
    tid_n = ((lane_idx >> 4) << 2) + WarpShape::kN * (warp_id_mn / WarpCount::kM);
  }

  /// Streams the result to global memory
  MCTLASS_DEVICE
  void operator()(
    OutputOp const &output_op,                    ///< Output operator
    OutputTileIterator destination_iterator,      ///< Tile iterator for destination
    AccumulatorTile const &accumulators,          ///< Complete warp-level accumulator tile
    OutputTileIterator source_iterator) {         ///< Threadblock tile coordinate in GEMM (in units of threadblock tiles)

    if (!output_op.is_source_needed()) {
      if ((platform::is_same<layout::MacaRowMajorTensorOpMultiplicandCongruous4x4Perm<
            sizeof_bits<half_t>::value, int(128 / sizeof(half_t))>, LayoutB>::value) ||
          (platform::is_same<layout::MacaRowMajorTensorOpMultiplicandCongruous4x4Perm<
            sizeof_bits<int8_t>::value, int(128 / sizeof(int8_t))>, LayoutB>::value)) {
        compute_source_not_needed_with_permB_(output_op, destination_iterator, accumulators);
      } else if (platform::is_same<layout::MacaColumnMajorTensorOpMultiplicandCongruous4x4Perm<
                    sizeof_bits<half_t>::value, int(128 / sizeof(half_t))>, LayoutA>::value) {
        compute_source_not_needed_with_permA_(output_op, destination_iterator, accumulators);
      } else {
        compute_source_not_needed_(output_op, destination_iterator, accumulators);
      }
    }
    else {
      compute_source_needed_(output_op, destination_iterator, accumulators, source_iterator);
    }
  }

private:

  /// Streams the result to global memory
  MCTLASS_DEVICE
  void compute_source_needed_(
    OutputOp const &output_op,                    ///< Output operator
    OutputTileIterator destination_iterator,      ///< Tile iterator for destination
    AccumulatorTile const &accumulators,          ///< Complete warp-level accumulator tile
    OutputTileIterator source_iterator) {         ///< Threadblock tile coordinate in GEMM (in units of threadblock tiles)

    const int kAccumBlockN = 4;
    const int kThreadsM = 16;
    const int kThreadsN = 4;
    const int kBlockM = WarpShape::kM / kThreadsM;

    /// Array type used to output
    using OutputAccessType = AlignedArray<ElementOutput, kAccumBlockN>;

    /// Array type passed to the output operator - unused elements are optimized away
    using OutputFragmentType = Array<ElementOutput, OutputOp::kCount>;

    /// Array type used by output functor
    using AccumulatorAccessType = Array<ElementAccumulator, kAccumBlockN>;

    /// Array type used by output functor
    using AccumulatorFragmentType = Array<ElementAccumulator, OutputOp::kCount>;

    AccumulatorAccessType const *accumulator_pair = reinterpret_cast<AccumulatorAccessType const *>(&accumulators);

    int const kIterationsN = WarpShape::kN / kThreadsN / kAccumBlockN;

    MCTLASS_PRAGMA_UNROLL
    for (int accum_m_idx = 0; accum_m_idx < WarpShape::kM / kThreadsM; accum_m_idx++) {

      int mL = destination_iterator.threadblock_offset.row() + tid_m + kThreadsM * accum_m_idx;
      int nL_base = destination_iterator.threadblock_offset.column() + tid_n;

      ElementOutput *output_ptr = destination_iterator.pointer + mL * destination_iterator.stride;
      ElementOutput *source_ptr = source_iterator.pointer + mL * source_iterator.stride;

      MCTLASS_PRAGMA_UNROLL
      for (int accum_n_idx = 0; accum_n_idx < kIterationsN; accum_n_idx++) {

        int accum_idx = accum_m_idx + kBlockM * accum_n_idx;

        // mL and nL are logical coordinate in 2D mapping of epilogue's 4D output
        int nL = nL_base + kThreadsM * accum_n_idx;

        bool guard = (mL < destination_iterator.extent.row()) && (nL < destination_iterator.extent.column());

        AccumulatorFragmentType accum_fragment;
        reinterpret_cast<AccumulatorAccessType &>(accum_fragment) = accumulator_pair[accum_idx];

        OutputFragmentType output_fragment;

        if(guard) {
          reinterpret_cast<OutputAccessType &>(output_fragment) =
            *reinterpret_cast<OutputAccessType const *>(source_ptr + nL);
        }

        // Perform output operator
        output_fragment = output_op(accum_fragment, output_fragment);

        if(guard) {
          // Store
          *reinterpret_cast<OutputAccessType *>(output_ptr + nL) = reinterpret_cast<OutputAccessType const &>(output_fragment);
        }
      }
    }

  }

  /// Streams the result to global memory
  MCTLASS_DEVICE
  void compute_source_not_needed_(
    OutputOp const &output_op,                    ///< Output operator
    OutputTileIterator destination_iterator,      ///< Tile iterator for destination
    AccumulatorTile const &accumulators) {         ///< Threadblock tile coordinate in GEMM (in units of threadblock tiles)

    const int kAccumBlockN = 4;
    const int kThreadsM = 16;
    const int kThreadsN = 4;
    const int kBlockM = WarpShape::kM / kThreadsM;

    /// Array type used to output
    using OutputAccessType = AlignedArray<ElementOutput, kAccumBlockN>;

    /// Array type passed to the output operator - unused elements are optimized away
    using OutputFragmentType = Array<ElementOutput, OutputOp::kCount>;

    /// Array type used by output functor
    using AccumulatorAccessType = Array<ElementAccumulator, kAccumBlockN>;

    /// Array type used by output functor
    using AccumulatorFragmentType = Array<ElementAccumulator, OutputOp::kCount>;

    AccumulatorAccessType const *accumulator_pair = reinterpret_cast<AccumulatorAccessType const *>(&accumulators);

    int const kIterationsN = WarpShape::kN / kThreadsN / kAccumBlockN;

    MCTLASS_PRAGMA_UNROLL
    for (int accum_m_idx = 0; accum_m_idx < WarpShape::kM / kThreadsM; accum_m_idx++) {

      int mL = destination_iterator.threadblock_offset.row() + tid_m + kThreadsM * accum_m_idx;
      int nL_base = destination_iterator.threadblock_offset.column() + tid_n;

      ElementOutput *output_ptr = destination_iterator.pointer + mL * destination_iterator.stride;

      MCTLASS_PRAGMA_UNROLL
      for (int accum_n_idx = 0; accum_n_idx < kIterationsN; accum_n_idx++) {

        int accum_idx = accum_m_idx + kBlockM * accum_n_idx;

        // mL and nL are logical coordinate in 2D mapping of epilogue's 4D output
        int nL = nL_base + kThreadsM * accum_n_idx;

        bool guard = (mL < destination_iterator.extent.row()) && (nL < destination_iterator.extent.column());

        AccumulatorFragmentType accum_fragment;
        reinterpret_cast<AccumulatorAccessType &>(accum_fragment) = accumulator_pair[accum_idx];

        OutputFragmentType output_fragment;

        // Perform output operator
        output_fragment = output_op(accum_fragment);
        if(guard) {
          // Store
          *reinterpret_cast<OutputAccessType *>(output_ptr + nL) =
            reinterpret_cast<OutputAccessType const &>(output_fragment);
        }
      }
    }
  }

  /// Streams the result to global memory
  MCTLASS_DEVICE
  void compute_source_not_needed_with_permB_(
    OutputOp const &output_op,                    ///< Output operator
    OutputTileIterator destination_iterator,      ///< Tile iterator for destination
    AccumulatorTile const &accumulators) {         ///< Threadblock tile coordinate in GEMM (in units of threadblock tiles)

    const int kAccumBlockN = 4;
    const int kThreadsM = 16;
    const int kThreadsN = 4;
    const int kBlockM = WarpShape::kM / kThreadsM;

    /// Array type used to output
    using OutputAccessType = AlignedArray<ElementOutput, kAccumBlockN>;

    /// Array type passed to the output operator - unused elements are optimized away
    using OutputFragmentType = Array<ElementOutput, OutputOp::kCount>;

    /// Array type used by output functor
    using AccumulatorAccessType = Array<ElementAccumulator, kAccumBlockN>;

    /// Array type used by output functor
    using AccumulatorFragmentType = Array<ElementAccumulator, OutputOp::kCount>;

    using PermType = Array<int, 4>;
    using OutputAccessType4 = AlignedArray<ElementOutput, kAccumBlockN * 4>;

    AccumulatorAccessType const *accumulator_pair = reinterpret_cast<AccumulatorAccessType const *>(&accumulators);

    int const kIterationsN = WarpShape::kN / kThreadsN / kAccumBlockN;  // 4

    int mL_base = destination_iterator.threadblock_offset.row() + tid_m;
    int nL_base = destination_iterator.threadblock_offset.column() + tid_n;
    int offset_m;
    int offset_n = (nL_base / 64) * 64 + 4 * ((nL_base & 63) & 15) + (nL_base & 63) / 16;
    int mL_stride = 1;
    if (platform::is_same<layout::MacaColumnMajorTensorOpMultiplicandCongruous4x4Perm<
                    sizeof_bits<half_t>::value, int(128 / sizeof(half_t))>, LayoutA>::value) {
      offset_m = (mL_base / 64) * 64 + 4 * ((mL_base & 63) & 15) + (mL_base & 63) / 16;
    } else {
      offset_m = mL_base;
      mL_stride = kThreadsM;
    }

    MCTLASS_PRAGMA_UNROLL
    for (int accum_m_idx = 0; accum_m_idx < WarpShape::kM / kThreadsM; accum_m_idx++) { // 4
        int mL = offset_m + accum_m_idx * mL_stride;
        ElementOutput *output_ptr = destination_iterator.pointer + mL * destination_iterator.stride + offset_n;

        PermType perm_dst[4];
        PermType perm_src[4];
        perm_src[0] = reinterpret_cast<PermType const &>(accumulator_pair[accum_m_idx]);
        perm_src[1] = reinterpret_cast<PermType const &>(accumulator_pair[accum_m_idx + 4]);
        perm_src[2] = reinterpret_cast<PermType const &>(accumulator_pair[accum_m_idx + 8]);
        perm_src[3] = reinterpret_cast<PermType const &>(accumulator_pair[accum_m_idx + 12]);

        perm_dst[0][0] = perm_src[0][0];
        perm_dst[0][1] = perm_src[1][0];
        perm_dst[0][2] = perm_src[2][0];
        perm_dst[0][3] = perm_src[3][0];
        perm_dst[1][0] = perm_src[0][1];
        perm_dst[1][1] = perm_src[1][1];
        perm_dst[1][2] = perm_src[2][1];
        perm_dst[1][3] = perm_src[3][1];
        perm_dst[2][0] = perm_src[0][2];
        perm_dst[2][1] = perm_src[1][2];
        perm_dst[2][2] = perm_src[2][2];
        perm_dst[2][3] = perm_src[3][2];
        perm_dst[3][0] = perm_src[0][3];
        perm_dst[3][1] = perm_src[1][3];
        perm_dst[3][2] = perm_src[2][3];
        perm_dst[3][3] = perm_src[3][3];

        OutputFragmentType output_fragment[4];
        // Perform output operator
        output_fragment[0] = output_op(reinterpret_cast<AccumulatorAccessType &>(perm_dst[0]));
        output_fragment[1] = output_op(reinterpret_cast<AccumulatorAccessType &>(perm_dst[1]));
        output_fragment[2] = output_op(reinterpret_cast<AccumulatorAccessType &>(perm_dst[2]));
        output_fragment[3] = output_op(reinterpret_cast<AccumulatorAccessType &>(perm_dst[3]));

        *reinterpret_cast<OutputAccessType4 *>(output_ptr) =
                    reinterpret_cast<OutputAccessType4 const &>(output_fragment);
    }
  }

  /// Streams the result to global memory
  MCTLASS_DEVICE
  void compute_source_not_needed_with_permA_(
    OutputOp const &output_op,                    ///< Output operator
    OutputTileIterator destination_iterator,      ///< Tile iterator for destination
    AccumulatorTile const &accumulators) {         ///< Threadblock tile coordinate in GEMM (in units of threadblock tiles)

    const int kAccumBlockN = 4;
    const int kThreadsM = 16;
    const int kThreadsN = 4;
    const int kBlockM = WarpShape::kM / kThreadsM;

    /// Array type used to output
    using OutputAccessType = AlignedArray<ElementOutput, kAccumBlockN>;

    /// Array type passed to the output operator - unused elements are optimized away
    using OutputFragmentType = Array<ElementOutput, OutputOp::kCount>;

    /// Array type used by output functor
    using AccumulatorAccessType = Array<ElementAccumulator, kAccumBlockN>;

    /// Array type used by output functor
    using AccumulatorFragmentType = Array<ElementAccumulator, OutputOp::kCount>;

    AccumulatorAccessType const *accumulator_pair = reinterpret_cast<AccumulatorAccessType const *>(&accumulators);

    int const kIterationsN = WarpShape::kN / kThreadsN / kAccumBlockN;

    int mL_base = destination_iterator.threadblock_offset.row() + tid_m;
    int nL_base = destination_iterator.threadblock_offset.column() + tid_n;
    int offset = (mL_base / 64) * 64 + 4 * ((mL_base & 63) & 15) + (mL_base & 63) / 16;

    ElementOutput *output_ptr[4];
    output_ptr[0] = destination_iterator.pointer + offset * destination_iterator.stride;
    output_ptr[1] = destination_iterator.pointer + (offset + 1) * destination_iterator.stride;
    output_ptr[2] = destination_iterator.pointer + (offset + 2) * destination_iterator.stride;
    output_ptr[3] = destination_iterator.pointer + (offset + 3) * destination_iterator.stride;
    using PermType = Array<int, 4>;
    MCTLASS_PRAGMA_UNROLL
    for (int accum_n_idx = 0; accum_n_idx < kIterationsN; accum_n_idx++) {
      PermType perm_src[4];

      perm_src[0] = reinterpret_cast<PermType const &>(accumulator_pair[accum_n_idx]);
      perm_src[1] = reinterpret_cast<PermType const &>(accumulator_pair[accum_n_idx + 1 * 4]);
      perm_src[2] = reinterpret_cast<PermType const &>(accumulator_pair[accum_n_idx + 2 * 4]);
      perm_src[3] = reinterpret_cast<PermType const &>(accumulator_pair[accum_n_idx + 3 * 4]);

      OutputFragmentType output_fragment[4];

      // Perform output operator
      output_fragment[0] = output_op(reinterpret_cast<AccumulatorAccessType &>(perm_src[0]));
      output_fragment[1] = output_op(reinterpret_cast<AccumulatorAccessType &>(perm_src[1]));
      output_fragment[2] = output_op(reinterpret_cast<AccumulatorAccessType &>(perm_src[2]));
      output_fragment[3] = output_op(reinterpret_cast<AccumulatorAccessType &>(perm_src[3]));

      // Store
      *reinterpret_cast<OutputAccessType *>(output_ptr[accum_n_idx] + nL_base) =
        reinterpret_cast<OutputAccessType const &>(output_fragment[0]);
      *reinterpret_cast<OutputAccessType *>(output_ptr[accum_n_idx] + nL_base + 16) =
        reinterpret_cast<OutputAccessType const &>(output_fragment[1]);
      *reinterpret_cast<OutputAccessType *>(output_ptr[accum_n_idx] + nL_base + 32) =
        reinterpret_cast<OutputAccessType const &>(output_fragment[2]);
      *reinterpret_cast<OutputAccessType *>(output_ptr[accum_n_idx] + nL_base + 48) =
        reinterpret_cast<OutputAccessType const &>(output_fragment[3]);
    }

  }

  MCTLASS_DEVICE
  void compute_source_not_needed_with_smem_permA_(
    OutputOp const &output_op,                    ///< Output operator
    OutputTileIterator destination_iterator,      ///< Tile iterator for destination
    AccumulatorTile const &accumulators) {         ///< Threadblock tile coordinate in GEMM (in units of threadblock tiles)

    const int kAccumBlockN = 4;
    const int kThreadsM = 16;
    const int kThreadsN = 4;
    const int kBlockM = WarpShape::kM / kThreadsM;

    /// Array type used to output
    using OutputAccessType = AlignedArray<ElementOutput, kAccumBlockN>;

    /// Array type passed to the output operator - unused elements are optimized away
    using OutputFragmentType = Array<ElementOutput, OutputOp::kCount>;

    /// Array type used by output functor
    using AccumulatorAccessType = Array<ElementAccumulator, kAccumBlockN>;

    /// Array type used by output functor
    using AccumulatorFragmentType = Array<ElementAccumulator, OutputOp::kCount>;

    AccumulatorAccessType const *accumulator_pair = reinterpret_cast<AccumulatorAccessType const *>(&accumulators);

    int const kIterationsN = WarpShape::kN / kThreadsN / kAccumBlockN;

    int sm_ml_base = tid_m;
    int sm_nl_base = tid_n;
    int sm_offset = (sm_ml_base / 64) * 64 + 4 * ((sm_ml_base & 63) & 15) + (sm_ml_base & 63) / 16;

    ElementOutput *shared_ptr[4];
    shared_ptr[0] = sm_pointer_ + sm_offset * Shape::kN;
    shared_ptr[1] = sm_pointer_ + (sm_offset + 1) * Shape::kN;
    shared_ptr[2] = sm_pointer_ + (sm_offset + 2) * Shape::kN;
    shared_ptr[3] = sm_pointer_ + (sm_offset + 3) * Shape::kN;
    using PermType = Array<int, 4>;
    MCTLASS_PRAGMA_UNROLL
    for (int accum_n_idx = 0; accum_n_idx < kIterationsN; accum_n_idx++) {

      // mL and nL are logical coordinate in 2D mapping of epilogue's 4D output
      int sm_nL = sm_nl_base + kThreadsM * accum_n_idx;

      PermType perm_src[4];
      perm_src[0] = reinterpret_cast<PermType const &>(accumulator_pair[kBlockM * accum_n_idx]);
      perm_src[1] = reinterpret_cast<PermType const &>(accumulator_pair[kBlockM * accum_n_idx + 1]);
      perm_src[2] = reinterpret_cast<PermType const &>(accumulator_pair[kBlockM * accum_n_idx + 2]);
      perm_src[3] = reinterpret_cast<PermType const &>(accumulator_pair[kBlockM * accum_n_idx + 3]);

      OutputFragmentType output_fragment[4];

      // Perform output operator
      output_fragment[0] = output_op(reinterpret_cast<AccumulatorAccessType &>(perm_src[0]));
      output_fragment[1] = output_op(reinterpret_cast<AccumulatorAccessType &>(perm_src[1]));
      output_fragment[2] = output_op(reinterpret_cast<AccumulatorAccessType &>(perm_src[2]));
      output_fragment[3] = output_op(reinterpret_cast<AccumulatorAccessType &>(perm_src[3]));

      // Store
      *reinterpret_cast<OutputAccessType *>(shared_ptr[0] + sm_nL) =
        reinterpret_cast<OutputAccessType const &>(output_fragment[0]);
      *reinterpret_cast<OutputAccessType *>(shared_ptr[1] + sm_nL) =
        reinterpret_cast<OutputAccessType const &>(output_fragment[1]);
      *reinterpret_cast<OutputAccessType *>(shared_ptr[2] + sm_nL) =
        reinterpret_cast<OutputAccessType const &>(output_fragment[2]);
      *reinterpret_cast<OutputAccessType *>(shared_ptr[3] + sm_nL) =
        reinterpret_cast<OutputAccessType const &>(output_fragment[3]);
    }

    __syncthreads();

    MCTLASS_PRAGMA_UNROLL
    for (int accum_m_idx = 0; accum_m_idx < WarpShape::kM / kThreadsM; accum_m_idx++) {
        int mL = destination_iterator.threadblock_offset.row() + tid_m + kThreadsM * accum_m_idx;
        int sm_mL = tid_m + kThreadsM * accum_m_idx;
        int nL_base = destination_iterator.threadblock_offset.column() + tid_n;
        int sm_nL_base = tid_n;
        ElementOutput *output_ptr = destination_iterator.pointer + mL * destination_iterator.stride;
        ElementOutput *shared_ptr = sm_pointer_ + sm_mL * Shape::kN;

        MCTLASS_PRAGMA_UNROLL
        for (int accum_n_idx = 0; accum_n_idx < kIterationsN; accum_n_idx++) {

            int accum_idx = accum_m_idx + kBlockM * accum_n_idx;

            // mL and nL are logical coordinate in 2D mapping of epilogue's 4D output
            int nL = nL_base + kThreadsM * accum_n_idx;
            int sm_nL = sm_nL_base + kThreadsM * accum_n_idx;

            bool guard = (mL < destination_iterator.extent.row()) && (nL < destination_iterator.extent.column());

            if(guard) {
                // Store
                *reinterpret_cast<OutputAccessType *>(output_ptr + nL) =
                    *reinterpret_cast<OutputAccessType *>(shared_ptr + sm_nL);
            }
        }
    }
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace threadblock
} // namespace epilogue
} // namespace mctlass

/////////////////////////////////////////////////////////////////////////////////////////////////
