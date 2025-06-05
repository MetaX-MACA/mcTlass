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
    \brief Defines iterators used by warp-level matrix multiply operations targeting Tensor Cores.
*/

#pragma once

#include "mctlass/mctlass.h"

#include "mctlass/array.h"
#include "mctlass/numeric_types.h"
#include "mctlass/tensor_ref.h"
#include "mctlass/matrix_shape.h"

#include "mctlass/arch/memory_sm75.h"
#include "mctlass/gemm/gemm.h"

#include "mctlass/layout/matrix.h"
#include "mctlass/layout/tensor.h"
#include "mctlass/layout/pitch_linear.h"
#include "mctlass/layout/tensor_op_multiplicand_sm75.h"

#include "mctlass/platform/platform.h"
#include "mctlass/fast_math.h"

#include "mma_tensor_op_tile_precompute_iterator.h"

////////////////////////////////////////////////////////////////////////////////

namespace mctlass {
namespace gemm {
namespace warp {

////////////////////////////////////////////////////////////////////////////////

template <
    /// Size of the matrix to load (concept: MatrixShape)
    typename Shape_,
    /// Operand identity
    Operand Operand,
    /// Data type of A elements
    typename Element_,
    /// Layout of operand
    typename Layout_,
    /// Shape of one matrix production operation (concept: GemmShape)
    typename InstructionShape_,
    /// Delta between *MMA operations (in units of *MMA operations, concept:
    /// MatrixShape)
    int OpDelta_,
    /// Number of threads participating in one matrix operation
    int Threads,
    /// Number of partitions along K dimension
    int PartitionsK_ = 1,
    /// Number of partitions along M dimension
    int PartitionsM_ = 1>
class MmaTensorOpMultiplicandTileIterator;

////////////////////////////////////////////////////////////////////////////////

/// This tile iterator is specialized for 32-thread TensorOps. It uses LDSM to load from shared
/// memory and therefore must be initialized with a TensorRef to shared memory.
///
/// Satisfies:
///   ReadableRandomAccessContiguousTileIteratorConcept
///
template <
    /// Size of the matrix to load (concept: PitchLinearShape)
    typename Shape_,
    /// Identifies A or B multiplicand
    Operand Operand_,
    /// Data type of elements
    typename Element_,
    /// Shape of one matrix product operation (concept: PitchLinearShape)
    typename InstructionShape_,
    /// Interval between adjacent *MMA instructions (in units of MMA
    /// instructions)
    int OpDelta_,
    /// Number of partitions along K dimension
    int PartitionsK_>
class MmaTensorOpMultiplicandTileIterator<
    Shape_, Operand_, Element_,
    mctlass::layout::TensorOpMultiplicandCongruous<sizeof_bits<Element_>::value,
                                                   64>,
    InstructionShape_, OpDelta_, 32, PartitionsK_> {
 public:

  /// Shape of tile to load (concept: PitchLinearShape)
  using Shape = Shape_;

  /// Operand tag
  static Operand const kOperand = Operand_;

  static_assert(kOperand == Operand::kA || kOperand== Operand::kB,
    "MmaTensorOpMultiplicandIterator may only be instantiated for A or B operands to warp-level Mma.");

  /// Element type
  using Element = Element_;

  /// Layout of source tile
  using Layout = mctlass::layout::TensorOpMultiplicandCongruous<
      sizeof_bits<Element_>::value, 64>;

  /// Shape of one matrix product operation (concept: GemmShape)
  using InstructionShape = InstructionShape_;

  /// Delta between *MMA operations (in units of *MMA operations, concept: MatrixShape)
  static int const kOpDelta = OpDelta_;

  /// Number of participating threads
  static int const kThreads = 32;

  /// Number of partitions along K dimension
  static int const kPartitionsK = PartitionsK_;

  /// TensorRef type for loading element from a tensor
  using TensorRef = TensorRef<Element, Layout>;

  /// Index type
  using Index = typename TensorRef::Index;

  /// Long Index type
  using LongIndex = typename TensorRef::LongIndex;

  /// Long Index type
  using StrideIndex = typename TensorRef::Layout::Stride::Index;

  /// Coordinate for an element in the tensor
  using TensorCoord = typename TensorRef::TensorCoord;

  /// Internal structure of iterator - made public to enable introspection
  struct Policy {
    static_assert(
        !(Shape::kContiguous % InstructionShape::kContiguous),
        "Shape of warp-level Mma must be divisible by operator shape.");

    // Determine number of elements along outer dimension per individual LDSM op
    static int const kLdsmOpOuter = Layout::kElementsPerAccess;
    static int const kLdsmOpInner = 8;

    static_assert(!(Shape::kContiguous % kLdsmOpOuter),
      "Shape of warp-level mma must be divisible by LDSM's fundamental tile size.");

    static_assert(!(Shape::kStrided % kLdsmOpInner),
      "Shape of warp-level mma must be divisible by LDSM's fundamental tile size.");

    /// Shape of one individual LDSM instruction
    static int const LdsmShapeStrided =
        InstructionShape::kStrided / kLdsmOpInner;
    static int const LdsmShapeContiguous = 4 / LdsmShapeStrided;
    using LdsmShape =
        layout::PitchLinearShape<LdsmShapeContiguous, LdsmShapeStrided>;

    /// Number and arrangement of LDSM instructions
    using LdsmIterations = layout::PitchLinearShape<
        Shape::kContiguous / Layout::kElementsPerAccess / LdsmShapeContiguous,
        1>;

    /// Number of groups for each tile
    static int const kGroupsPerTile =
        Shape::kStrided / InstructionShape::kStrided;
  };

private:

  /// Not working on this feature at the moment.
  static_assert(kOpDelta == 1,
    "Alternative arrangements not supported at present.");

  /// Number of internal pointers needed to reference shared memory
  static int const kPointerCount =
      Layout::TileShape::kContiguous / Policy::LdsmShape::kContiguous;

  /// Pointer type used for accesses
  using AccessType = Array<Element, Layout::kElementsPerAccess>;

  /// Internal counter used to jump to next K partition
  int k_group_idx_;

public:

  //
  // Derived quantities
  //

  /// Fragment object holding a thread's part of a tile
 using Fragment =
     Array<Element, Shape::kContiguous * InstructionShape::kStrided / kThreads>;

private:

  /// Layout object storing stride values
  StrideIndex stride_;

  /// Shared memory base pointers - not advanced
  AccessType const *pointer_[kPointerCount];

  /// Byte offset incremented as iterator advances
  Index byte_offset_;

public:

  /// Default ctor constructs null iterator
  MCTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator(): stride_(0), byte_offset_(0) { }

  /// Constructor from TensorRef
  MCTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator(
    TensorRef const &ref,
    int lane_id
  ):
    stride_(ref.stride(0) / Layout::kElementsPerAccess), byte_offset_(0),
    k_group_idx_(0) {

    int quad_pair = (lane_id >> 3);
    int quad_quad = (lane_id >> 4);
    int lane_in_quad = (lane_id & 3);
    int lane_in_quad_pair = (lane_id & 7);
    int lane_in_quad_quad = (lane_id & 15);

    MCTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kPointerCount; ++i) {
      int partition_contiguous_idx = -1;
      int access_contiguous_idx = -1;
      int access_strided_idx = -1;

      if (Policy::LdsmShape::kContiguous == 4) {
        // Matrix multiply 1688 A/B
        // Q0 Q1 Q2 Q3 (Q stands for 1 8x128bit block).
        // Four blocks are next to each other in the contiguous dimension.
        partition_contiguous_idx = ((lane_in_quad_pair >> 2) ^ i);
        access_contiguous_idx = (quad_pair ^ lane_in_quad);
        access_strided_idx = lane_in_quad_pair;
      }
      else if (Policy::LdsmShape::kContiguous == 2 &&
                 kOperand == Operand::kA) {
        // Matrix multiply 16816 A
        // Q0 Q1
        // Q2 Q3
        partition_contiguous_idx = ((lane_in_quad_pair >> 2) ^ (i >> 1));
        access_contiguous_idx =
            (((quad_pair & 1) + ((i & 1) << 1)) ^ lane_in_quad);
        access_strided_idx = lane_in_quad_pair + (lane_id >> 4 << 3);
      } else if (Policy::LdsmShape::kContiguous == 2 &&
                 kOperand == Operand::kB) {
        // Matrix multiply 16816 B
        // Q0 Q2
        // Q1 Q3
        partition_contiguous_idx = ((lane_in_quad_pair >> 2) ^ (i >> 1));
        access_contiguous_idx = ((quad_quad + ((i & 1) << 1)) ^ lane_in_quad);
        access_strided_idx = lane_in_quad_quad;
      } else if (Policy::LdsmShape::kContiguous == 1) {
        // Matrix multiply 16832.SP B
        // Q0
        // Q1
        // Q2
        // Q3
        partition_contiguous_idx = ((lane_in_quad_pair >> 2) ^ (i >> 2));
        access_contiguous_idx = ((i & 3) ^ lane_in_quad);
        access_strided_idx = lane_id;
      }

      int access_contiguous =
          partition_contiguous_idx * Layout::PartitionShape::kContiguous +
          access_contiguous_idx;

      int access_strided = access_strided_idx;

      pointer_[i] = reinterpret_cast<AccessType const *>(ref.data()) +
                    access_contiguous + access_strided * stride_;
    }
  }

  /// Adds a pointer offset to internal pointer(s) to advance through memory
  MCTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator &add_pointer_offset(LongIndex offset) {

    byte_offset_ += offset * sizeof(Element);

    return *this;
  }

  /// Advances an iterator along logical dimensions of matrix in units of whole tiles
  MCTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator &add_tile_offset(TensorCoord const &tile_offset) {

    int contiguous_offset = tile_offset.contiguous();
    if (Shape::kContiguous ==
        Layout::PartitionShape::kContiguous * Layout::kElementsPerAccess) {
      if (tile_offset.contiguous() % 2) {
        MCTLASS_PRAGMA_UNROLL
        for (int i = 0; i < kPointerCount / 2; ++i) {
          AccessType const *tmp_pointer = pointer_[i];
          pointer_[i] = pointer_[i + kPointerCount / 2];
          pointer_[i + kPointerCount / 2] = tmp_pointer;
        }
      }
      contiguous_offset = (tile_offset.contiguous() >> 1) << 1;
    }

    int offset = (tile_offset.strided() * InstructionShape::kStrided) *
                     stride_ * Layout::kElementsPerAccess +
                 contiguous_offset * Shape::kContiguous;

    add_pointer_offset(offset);

    return *this;
  }

  /// Advances the iterator along the advance dimension
  MCTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator & operator++() {

    add_tile_offset({0, 1});

    if (kPartitionsK > 1) {
      ++k_group_idx_;
      // Jump to next stage
      if (k_group_idx_ == Policy::kGroupsPerTile) {
        k_group_idx_ = 0;
        add_tile_offset(
            {0, ((kPartitionsK - 1) * Policy::kGroupsPerTile)});
      }
    }

    return *this;
  }

  /// Advances the iterator along the opposite of the advance dimension
  MCTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator & operator--() {
    byte_offset_ -= stride_ * InstructionShape::kStrided * sizeof(Element) *
                    Layout::kElementsPerAccess;

    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of the tensor
  MCTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator & operator+=(TensorCoord const &tile_offset) {
    add_tile_offset(tile_offset);
    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of the tensor
  MCTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator & operator-=(TensorCoord const &tile_offset) {
    add_tile_offset(-tile_offset);
    return *this;
  }

  /// Loads a fragment from memory at the location pointed to by the iterator.
  MCTLASS_HOST_DEVICE
  void load(Fragment &frag) const {

    load_with_byte_offset(frag, 0);
  }

  /// Loads a fragment from memory with additional logical offset
  MCTLASS_DEVICE
  void load_with_byte_offset(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a linear offset in units of bytes
      Index byte_offset) const {

    Array<unsigned, Policy::LdsmShape::kCount> *fetch_ptr =
      reinterpret_cast<Array<unsigned, Policy::LdsmShape::kCount> *>(&frag);

    MCTLASS_PRAGMA_UNROLL
    for (int s = 0; s < Policy::LdsmIterations::kStrided; ++s) {

      MCTLASS_PRAGMA_UNROLL
      for (int c = 0; c < Policy::LdsmIterations::kContiguous; ++c) {

        int access_idx = c + s * Policy::LdsmIterations::kContiguous;

        AccessType const *source_ptr =
            pointer_[c % kPointerCount] +
            Layout::TileShape::kContiguous * (c / kPointerCount) +
            Policy::kLdsmOpInner * Policy::LdsmShape::kStrided * s * stride_;

        char const *source_byte_ptr = reinterpret_cast<char const *>(source_ptr) + byte_offset + byte_offset_;

        mctlass::arch::ldsm<layout::ColumnMajor, Policy::LdsmShape::kCount>(
          fetch_ptr[access_idx],
          source_byte_ptr
        );
      }
    }
  }

  /// Loads a fragment from memory with additional logical offset
  MCTLASS_DEVICE
  void load_with_pointer_offset(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a linear offset
      Index pointer_offset) const {
    load_with_byte_offset(frag, pointer_offset * sizeof(Element));
  }

  /// Loads a fragment from memory with logical offset in units of whole tiles.
  MCTLASS_DEVICE
  void load(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a logical offset in units of whole tiles
      TensorCoord const &tile_offset) const {
    load_with_byte_offset(frag, tile_offset, 0);
  }

  /// Loads a fragment from memory with logical offset in units of whole tiles.
  MCTLASS_DEVICE
  void load(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a logical offset in units of whole tiles
      TensorCoord const &tile_offset,
      /// loads a tile with a logical offset AND a pointer offset
      Index pointer_offset) const {
    load_with_byte_offset(frag, tile_offset, pointer_offset * sizeof(Element));
  }

  /// Loads a fragment from memory with logical offset in units of whole tiles.
  MCTLASS_DEVICE
  void load_with_byte_offset(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a logical offset in units of whole tiles
      TensorCoord const &tile_offset,
      /// loads a tile with a logical offset AND a pointer offset
      Index byte_offset) const {
    Index pointer_offset =
      tile_offset.contiguous() * Shape::kContiguous / Layout::kElementsPerAccess +
      tile_offset.strided() * InstructionShape::kStrided * stride_;

    byte_offset += sizeof(AccessType) * pointer_offset;

    load_with_byte_offset(frag, byte_offset);
  }

  /// Notify the iterator which k-group it is currently pointing to.
  ///
  /// This does not advance the iterator. Rather, it overrides its internal
  /// tracking with constant-valued k-group index to enable the compiler to
  /// fold constants and achieve more efficient code.
  ///
  /// This is used by some nontrivial permuted layouts.
  MCTLASS_DEVICE
  void set_kgroup_index(int k_group) {
    // no op
  }
};

////////////////////////////////////////////////////////////////////////////////

/// This tile iterator is specialized for 32-thread MMA.TF32 NT TensorOps. It
/// uses LDS.32 to load from shared memory and therefore must be initialized
/// with a TensorRef to shared memory.
///
/// Satisfies:
///   ReadableRandomAccessContiguousTileIteratorConcept
///
template <
    /// Size of the matrix to load (concept: PitchLinearShape)
    typename Shape_,
    /// Identifies A or B multiplicand
    Operand Operand_,
    /// Data type of elements
    typename Element_,
    /// Shape of one matrix product operation (concept: PitchLinearShape)
    typename InstructionShape_,
    /// Interval between adjacent *MMA instructions (in units of MMA
    /// instructions)
    int OpDelta_,
    /// Number of partitions along K dimension
    int PartitionsK_>
class MmaTensorOpMultiplicandTileIterator<
    Shape_, Operand_, Element_,
    mctlass::layout::TensorOpMultiplicandCongruous<32, 32>, InstructionShape_,
    OpDelta_, 32, PartitionsK_> {
 public:
  /// Shape of tile to load (concept: PitchLinearShape)
  using Shape = Shape_;

  /// Operand tag
  static Operand const kOperand = Operand_;

  static_assert(kOperand == Operand::kA || kOperand == Operand::kB,
                "MmaTensorOpMultiplicandIterator may only be instantiated for "
                "A or B operands to warp-level Mma.");

  /// Element type
  using Element = Element_;

  /// Layout of source tile
  using Layout = mctlass::layout::TensorOpMultiplicandCongruous<32, 32>;

  /// Shape of one matrix product operation (concept: GemmShape)
  using InstructionShape = InstructionShape_;

  /// Delta between *MMA operations (in units of *MMA operations, concept:
  /// MatrixShape)
  static int const kOpDelta = OpDelta_;

  /// Number of participating threads
  static int const kThreads = 32;

  /// Number of partitions along K dimension
  static int const kPartitionsK = PartitionsK_;

  /// TensorRef type for loading element from a tensor
  using TensorRef = TensorRef<Element, Layout>;

  /// Index type
  using Index = typename TensorRef::Index;

  /// Long Index type
  using LongIndex = typename TensorRef::LongIndex;

  /// Long Index type
  using StrideIndex = typename TensorRef::Layout::Stride::Index;

  /// Coordinate for an element in the tensor
  using TensorCoord = typename TensorRef::TensorCoord;

  /// Internal structure of iterator - made public to enable introspection
  struct Policy {
    static_assert(
        !(Shape::kContiguous % InstructionShape::kContiguous),
        "Shape of warp-level Mma must be divisible by operator shape.");

    // Determine number of elements along outer dimension per individual 32bit
    // shared memory load op.  Every one warp of 32bit shared memory load loads
    // 8x4 elements
    static int const kLdsOpInner = Layout::TileShape::kStrided;
    static int const kLdsOpOuter = kThreads / kLdsOpInner;

    static_assert(!(Shape::kContiguous % kLdsOpOuter),
                  "Shape of warp-level mma must be divisible by 32bit "
                  "fundamental tile size.");

    static_assert(!(Shape::kStrided % kLdsOpInner),
                  "Shape of warp-level mma must be divisible by 32bit "
                  "fundamental tile size.");

    /// Number of 32 bit shared memory load instructions needed by one MMA instruction
    /// 1688  A 2x2
    /// 1688  B 1x2
    /// 16816 B 1x4
    static int const LdsShapeContiguous =
        InstructionShape::kContiguous / kLdsOpOuter;
    static int const LdsShapeStrided = InstructionShape::kStrided / kLdsOpInner;
    using LdsShape =
        layout::PitchLinearShape<LdsShapeContiguous, LdsShapeStrided>;

    /// Number and arrangement of LDS instructions
    using LdsIterations = layout::PitchLinearShape<
        Shape::kContiguous / LdsShapeContiguous / kLdsOpOuter, 1>;

    /// Number of groups for each tile
    static int const kGroupsPerTile =
        Shape::kStrided / InstructionShape::kStrided;
  };

 private:
  /// Not working on this feature at the moment.
  static_assert(kOpDelta == 1,
                "Alternative arrangements not supported at present.");

  /// Number of internal pointers needed to reference shared memory
  static int const kPointerCount = Layout::TileShape::kContiguous *
                                   Layout::kElementsPerAccess /
                                   Policy::kLdsOpOuter;

  /// Vectorized access is not used
  static int const kElementsPerAccess = 1;

  /// Pointer type used for accesses
  using AccessType = Element;

  /// Internal counter used to jump to next K partition
  int k_group_idx_;

 public:
  //
  // Derived quantities
  //

  /// Fragment object holding a thread's part of a tile
  using Fragment =
     Array<Element, Shape::kContiguous * InstructionShape::kStrided / kThreads>;

 private:
  /// Layout object storing stride values
  StrideIndex stride_;

  /// Shared memory base pointers - not advanced
  AccessType const *pointer_[kPointerCount];

  /// Byte offset incremented as iterator advances
  Index byte_offset_;

 public:
  /// Default ctor constructs null iterator
  MCTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator() : stride_(0), byte_offset_(0) {}

  /// Constructor from TensorRef
  MCTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator(TensorRef const &ref, int lane_id)
      : stride_(ref.stride(0)), byte_offset_(0), k_group_idx_(0) {
    MCTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kPointerCount; ++i) {
      int access_strided = lane_id % Policy::kLdsOpInner;
      int access_contiguous = (lane_id / Policy::kLdsOpInner) +
                              (access_strided ^ i) * Policy::kLdsOpOuter;

      pointer_[i] = reinterpret_cast<AccessType const *>(ref.data()) +
                    access_contiguous + access_strided * stride_;
    }
  }

  /// Adds a pointer offset to internal pointer(s) to advance through memory
  MCTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator &add_pointer_offset(LongIndex offset) {
    byte_offset_ += offset * sizeof(Element);

    return *this;
  }

  /// Advances an iterator along logical dimensions of matrix in units of whole
  /// tiles
  MCTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator &add_tile_offset(
      TensorCoord const &tile_offset) {
    int contiguous_offset = tile_offset.contiguous();
    if (Shape::kContiguous ==
        Layout::TileShape::kContiguous * Layout::kElementsPerAccess / 2) {
      if (tile_offset.contiguous() % 2) {
        // Matrix multiply 1688 pointer_[0] <=> pointer_[4] pointer_[1] <=> pointer_[5]
        //           pointer_[2] <=> pointer_[6] pointer_[3] <=> pointer_[7]
        MCTLASS_PRAGMA_UNROLL
        for (int i = 0; i < kPointerCount / 2; ++i) {
          AccessType const *tmp_pointer = pointer_[i];
          pointer_[i] = pointer_[i + kPointerCount / 2];
          pointer_[i + kPointerCount / 2] = tmp_pointer;
        }
      }
      contiguous_offset = (tile_offset.contiguous() >> 1) << 1;
    }

    int offset = (tile_offset.strided() * InstructionShape::kStrided) * stride_ +
                 contiguous_offset * Shape::kContiguous;

    add_pointer_offset(offset);

    return *this;
  }

  /// Advances the iterator along the advance dimension
  MCTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator &operator++() {
    add_tile_offset({0, 1});

    if (kPartitionsK > 1) {
      ++k_group_idx_;
      // Jump to next stage
      if (k_group_idx_ == Policy::kGroupsPerTile) {
        k_group_idx_ = 0;
        add_tile_offset(
            {0, ((kPartitionsK - 1) * Policy::kGroupsPerTile)});
      }
    }

    return *this;
  }

  /// Advances the iterator along the opposite of the advance dimension
  MCTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator &operator--() {
    byte_offset_ -= stride_ * InstructionShape::kStrided * sizeof(Element) *
                    kElementsPerAccess;

    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of
  ///< the tensor
  MCTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator &operator+=(
      TensorCoord const &tile_offset) {
    add_tile_offset(tile_offset);
    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of
  ///< the tensor
  MCTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator &operator-=(
      TensorCoord const &tile_offset) {
    add_tile_offset(-tile_offset);
    return *this;
  }

  /// Loads a fragment from memory at the location pointed to by the iterator.
  MCTLASS_HOST_DEVICE
  void load(Fragment &frag) const { load_with_byte_offset(frag, 0); }

  /// Loads a fragment from memory with additional logical offset
  MCTLASS_DEVICE
  void load_with_byte_offset(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a linear offset in units of bytes
      Index byte_offset) const {
    Element *fetch_ptr = reinterpret_cast<Element *>(&frag);

    MCTLASS_PRAGMA_UNROLL
    for (int s = 0; s < Policy::LdsIterations::kStrided; ++s) {
      MCTLASS_PRAGMA_UNROLL
      for (int c = 0; c < Policy::LdsIterations::kContiguous; ++c) {
        MCTLASS_PRAGMA_UNROLL
        for (int ss = 0; ss < Policy::LdsShape::kStrided; ++ss) {
          MCTLASS_PRAGMA_UNROLL
          for (int cc = 0; cc < Policy::LdsShape::kContiguous; ++cc) {
            int access_idx =
                cc + (ss + (c + s * Policy::LdsIterations::kContiguous) *
                               Policy::LdsShape::kStrided) *
                         Policy::LdsShape::kContiguous;
            int access_idx_contiguous = cc + c * Policy::LdsShape::kContiguous;
            int access_idx_strided =
                (ss + s * Policy::LdsShape::kStrided) * Policy::kLdsOpInner;

            AccessType const *source_ptr =
                pointer_[access_idx_contiguous % kPointerCount] +
                Layout::TileShape::kContiguous * Layout::kElementsPerAccess *
                    (access_idx_contiguous / kPointerCount) +
                access_idx_strided * stride_;

            char const *source_byte_ptr =
                reinterpret_cast<char const *>(source_ptr) + byte_offset +
                byte_offset_;

            fetch_ptr[access_idx] =
                *reinterpret_cast<Element const *>(source_byte_ptr);
          }
        }
      }
    }
  }

  /// Loads a fragment from memory with additional logical offset
  MCTLASS_DEVICE
  void load_with_pointer_offset(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a linear offset
      Index pointer_offset) const {
    load_with_byte_offset(frag, pointer_offset * sizeof(Element));
  }

  /// Loads a fragment from memory with logical offset in units of whole tiles.
  MCTLASS_DEVICE
  void load(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a logical offset in units of whole tiles
      TensorCoord const &tile_offset) const {
    load_with_byte_offset(frag, tile_offset, 0);
  }

  /// Loads a fragment from memory with logical offset in units of whole tiles.
  MCTLASS_DEVICE
  void load(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a logical offset in units of whole tiles
      TensorCoord const &tile_offset,
      /// loads a tile with a logical offset AND a pointer offset
      Index pointer_offset) const {
    load_with_byte_offset(frag, tile_offset, pointer_offset * sizeof(Element));
  }

  /// Loads a fragment from memory with logical offset in units of whole tiles.
  MCTLASS_DEVICE
  void load_with_byte_offset(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a logical offset in units of whole tiles
      TensorCoord const &tile_offset,
      /// loads a tile with a logical offset AND a pointer offset
      Index byte_offset) const {
    Index pointer_offset =
        tile_offset.contiguous() * Shape::kContiguous /
            Layout::kElementsPerAccess +
        tile_offset.strided() * InstructionShape::kStrided * stride_;

    byte_offset += sizeof(AccessType) * pointer_offset;

    load_with_byte_offset(frag, byte_offset);
  }

  /// Notify the iterator which k-group it is currently pointing to.
  ///
  /// This does not advance the iterator. Rather, it overrides its internal
  /// tracking with constant-valued k-group index to enable the compiler to
  /// fold constants and achieve more efficient code.
  ///
  /// This is used by some nontrivial permuted layouts.
  MCTLASS_DEVICE
  void set_kgroup_index(int k_group) {
    // no op
  }
};

template <
    /// Size of the matrix to load (concept: PitchLinearShape)
    typename Shape_,
    /// Identifies A or B multiplicand
    Operand Operand_,
    /// Shape of one matrix product operation (concept: PitchLinearShape)
    typename InstructionShape_,
    /// Interval between adjacent *MMA instructions (in units of MMA
    /// instructions)
    int OpDelta_,
    /// Number of partitions along K dimension
    int PartitionsK_>
class MmaTensorOpMultiplicandTileIterator<
    Shape_, Operand_, mctlass::tfloat32_t,
    mctlass::layout::TensorOpMultiplicandCongruous<32, 32>, InstructionShape_,
    OpDelta_, 32, PartitionsK_> {
 public:
  /// Shape of tile to load (concept: PitchLinearShape)
  using Shape = Shape_;

  /// Operand tag
  static Operand const kOperand = Operand_;

  static_assert(kOperand == Operand::kA || kOperand == Operand::kB,
                "MmaTensorOpMultiplicandIterator may only be instantiated for "
                "A or B operands to warp-level Mma.");

  /// Element type
  using Element = mctlass::tfloat32_t;

  /// Layout of source tile
  using Layout = mctlass::layout::TensorOpMultiplicandCongruous<32, 32>;

  /// Shape of one matrix product operation (concept: GemmShape)
  using InstructionShape = InstructionShape_;

  /// Delta between *MMA operations (in units of *MMA operations, concept:
  /// MatrixShape)
  static int const kOpDelta = OpDelta_;

  /// Number of participating threads
  static int const kThreads = 32;

  /// Number of partitions along K dimension
  static int const kPartitionsK = PartitionsK_;

  /// TensorRef type for loading element from a tensor
  using TensorRef = TensorRef<Element, Layout>;

  /// Index type
  using Index = typename TensorRef::Index;

  /// Long Index type
  using LongIndex = typename TensorRef::LongIndex;

  /// Long Index type
  using StrideIndex = typename TensorRef::Layout::Stride::Index;

  /// Coordinate for an element in the tensor
  using TensorCoord = typename TensorRef::TensorCoord;

  /// Internal structure of iterator - made public to enable introspection
  struct Policy {
    static_assert(
        !(Shape::kContiguous % InstructionShape::kContiguous),
        "Shape of warp-level Mma must be divisible by operator shape.");

    // Determine number of elements along outer dimension per individual 32bit
    // shared memory load op.  Every one warp of 32bit shared memory load loads
    // 8x4 elements
    static int const kLdsOpInner = Layout::TileShape::kStrided;
    static int const kLdsOpOuter = kThreads / kLdsOpInner;

    static_assert(!(Shape::kContiguous % kLdsOpOuter),
                  "Shape of warp-level mma must be divisible by 32bit "
                  "fundamental tile size.");

    static_assert(!(Shape::kStrided % kLdsOpInner),
                  "Shape of warp-level mma must be divisible by 32bit "
                  "fundamental tile size.");

    /// Number of 32 bit shared memory load instructions needed by one MMA instruction
    /// 1688  A 2x2
    /// 1688  B 1x2
    /// 16816 B 1x4
    static int const LdsShapeContiguous =
        InstructionShape::kContiguous / kLdsOpOuter;
    static int const LdsShapeStrided = InstructionShape::kStrided / kLdsOpInner;
    using LdsShape =
        layout::PitchLinearShape<LdsShapeContiguous, LdsShapeStrided>;

    /// Number and arrangement of LDS instructions
    using LdsIterations = layout::PitchLinearShape<
        Shape::kContiguous / LdsShapeContiguous / kLdsOpOuter, 1>;

    /// Number of groups for each tile
    static int const kGroupsPerTile =
        Shape::kStrided / InstructionShape::kStrided;
  };

 private:
  /// Not working on this feature at the moment.
  static_assert(kOpDelta == 1,
                "Alternative arrangements not supported at present.");

  /// Number of internal pointers needed to reference shared memory
  static int const kPointerCount = Layout::TileShape::kContiguous *
                                   Layout::kElementsPerAccess /
                                   Policy::kLdsOpOuter;

  /// Vectorized access is not used
  static int const kElementsPerAccess = 1;

  /// Pointer type used for accesses
  using AccessType = Element;

  /// Internal counter used to jump to next K partition
  int k_group_idx_;

 public:
  //
  // Derived quantities
  //

  /// Fragment object holding a thread's part of a tile
  using Fragment = Array<Element, Shape::kContiguous * InstructionShape::kStrided / kThreads>;

 private:
  /// Layout object storing stride values
  StrideIndex stride_;

  /// Shared memory base pointers - not advanced
  AccessType const *pointer_[kPointerCount];

  /// Byte offset incremented as iterator advances
  Index byte_offset_;

 public:
  /// Default ctor constructs null iterator
  MCTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator() : stride_(0), byte_offset_(0) {}

  /// Constructor from TensorRef
  MCTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator(TensorRef const &ref, int lane_id)
      : stride_(ref.stride(0)), byte_offset_(0), k_group_idx_(0) {
    pointer_[0] = reinterpret_cast<AccessType const *>(ref.data());
  }

  /// Adds a pointer offset to internal pointer(s) to advance through memory
  MCTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator &add_pointer_offset(LongIndex offset) {
    byte_offset_ += offset * sizeof(Element);

    return *this;
  }

  /// Advances an iterator along logical dimensions of matrix in units of whole
  /// tiles
  MCTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator &add_tile_offset(
      TensorCoord const &tile_offset) {
    int contiguous_offset = tile_offset.contiguous();
    if (Shape::kContiguous ==
        Layout::TileShape::kContiguous * Layout::kElementsPerAccess / 2) {
      if (tile_offset.contiguous() % 2) {
        // Matrix multiply 1688 pointer_[0] <=> pointer_[4] pointer_[1] <=> pointer_[5]
        //           pointer_[2] <=> pointer_[6] pointer_[3] <=> pointer_[7]
        MCTLASS_PRAGMA_UNROLL
        for (int i = 0; i < kPointerCount / 2; ++i) {
          AccessType const *tmp_pointer = pointer_[i];
          pointer_[i] = pointer_[i + kPointerCount / 2];
          pointer_[i + kPointerCount / 2] = tmp_pointer;
        }
      }
      contiguous_offset = (tile_offset.contiguous() >> 1) << 1;
    }

    int offset = (tile_offset.strided() * InstructionShape::kStrided) * stride_ +
                 contiguous_offset * Shape::kContiguous;

    add_pointer_offset(offset);

    return *this;
  }

  /// Advances the iterator along the advance dimension
  MCTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator &operator++() {
    add_tile_offset({0, 1});
    if (kPartitionsK > 1) {
      ++k_group_idx_;
      // Jump to next stage
      if (k_group_idx_ == Policy::kGroupsPerTile) {
        k_group_idx_ = 0;
        add_tile_offset(
            {0, ((kPartitionsK - 1) * Policy::kGroupsPerTile)});
      }
    }

    return *this;
  }

  /// Advances the iterator along the opposite of the advance dimension
  MCTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator &operator--() {
    byte_offset_ -= stride_ * InstructionShape::kStrided * sizeof(Element) *
                    kElementsPerAccess;

    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of
  ///< the tensor
  MCTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator &operator+=(
      TensorCoord const &tile_offset) {
    add_tile_offset(tile_offset);
    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of
  ///< the tensor
  MCTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator &operator-=(
      TensorCoord const &tile_offset) {
    add_tile_offset(-tile_offset);
    return *this;
  }

  /// Loads a fragment from memory at the location pointed to by the iterator.
  MCTLASS_HOST_DEVICE
  void load(Fragment &frag) const { load_with_byte_offset(frag, 0); }

  /// Loads a fragment from memory with additional logical offset
  MCTLASS_DEVICE
  void load_with_byte_offset(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a linear offset in units of bytes
      Index byte_offset) const {
    Element *fetch_ptr = reinterpret_cast<Element *>(&frag);
    int lane_id = __lane_id();

    MCTLASS_PRAGMA_UNROLL
    for (int s = 0; s < Policy::LdsIterations::kStrided; ++s) {
      MCTLASS_PRAGMA_UNROLL
      for (int c = 0; c < Policy::LdsIterations::kContiguous; ++c) {
          if(kOperand == Operand::kA) {
           if(InstructionShape::kStrided == 8) {
            unsigned row = (lane_id & 0xf) + c * InstructionShape::kContiguous;
            unsigned col = ((lane_id >> 4) << 1) ^ 0x7;
            unsigned col_1 = col % 4;
            unsigned row_1 = ((row / 8) ^ col_1) * 8 + (row % 8);
            unsigned col_2 = (col - 1) % 4;
            unsigned row_2 = ((row / 8) ^ col_2) * 8 + (row % 8);

            AccessType const *source_ptr = pointer_[0] + row_1 + col * stride_;
            char const *source_byte_ptr = reinterpret_cast<char const *>(source_ptr) +
                                          byte_offset + byte_offset_;

            fetch_ptr[c * 4 + 0] = *reinterpret_cast<Element const *>(source_byte_ptr);
            AccessType const *source_ptr1 = pointer_[0] + row_2 + (col - 1) * stride_;
            char const *source_byte_ptr1 = reinterpret_cast<char const *>(source_ptr1) +
                                           byte_offset + byte_offset_;

            fetch_ptr[c * 4 + 1] = *reinterpret_cast<Element const *>(source_byte_ptr1);
           }
           else {
            unsigned row = (lane_id & 0xf) + c * InstructionShape::kContiguous;
            unsigned col = (lane_id >> 4) ^ 0x3;
            unsigned col_1 = col % 4;
            unsigned row_1 = ((row / 8) ^ col_1) * 8 + (row % 8);

            AccessType const *source_ptr = pointer_[0] + row_1 + col * stride_;
            char const *source_byte_ptr = reinterpret_cast<char const *>(source_ptr) +
                                          byte_offset + byte_offset_;
            fetch_ptr[c * 2 + 0] = *reinterpret_cast<Element const *>(source_byte_ptr);
            fetch_ptr[c * 2 + 1] = static_cast<Element>(0);
           }
         }
         else {
           if(InstructionShape::kStrided == 8) {
            unsigned row = ((lane_id >> 4) << 1) ^ 0x7;
            unsigned col = (lane_id & 0x7) + c * InstructionShape::kStrided;
            unsigned row_1 = row % 4;
            unsigned col_1 = ((col / 8) ^ row_1) * 8 + (col % 8);
            unsigned row_2 = (row - 1) % 4;
            unsigned col_2 = ((col / 8) ^ row_2) * 8 + (col % 8);

            AccessType const *source_ptr = pointer_[0];
            char const *source_byte_ptr = reinterpret_cast<char const *>(source_ptr) +
                                          byte_offset + byte_offset_;

            fetch_ptr[c * 2 + 0] = *(reinterpret_cast<Element const *>(source_byte_ptr) +
                                   col_1 + row * stride_);

            fetch_ptr[c * 2 + 1] = *(reinterpret_cast<Element const *>(source_byte_ptr) +
                                   col_2 + (row - 1) * stride_);
           }
           else {
            unsigned row = (lane_id >> 4) ^ 0x3;
            unsigned col = (lane_id & 0x7) + c * InstructionShape::kContiguous;
            unsigned row_1 = row % 4;
            unsigned col_1 = ((col / 8) ^ row_1) * 8 + (col % 8);

            AccessType const *source_ptr = pointer_[0];
            char const *source_byte_ptr = reinterpret_cast<char const *>(source_ptr) +
                                          byte_offset + byte_offset_;

            fetch_ptr[c] = *(reinterpret_cast<Element const *>(source_byte_ptr) +
                                   col_1 + row * stride_);
           }
         }
      }
    }
    __syncthreads();
  }

  /// Loads a fragment from memory with additional logical offset
  MCTLASS_DEVICE
  void load_with_pointer_offset(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a linear offset
      Index pointer_offset) const {
    load_with_byte_offset(frag, pointer_offset * sizeof(Element));
  }

  /// Loads a fragment from memory with logical offset in units of whole tiles.
  MCTLASS_DEVICE
  void load(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a logical offset in units of whole tiles
      TensorCoord const &tile_offset) const {
    load_with_byte_offset(frag, tile_offset, 0);
  }

  /// Loads a fragment from memory with logical offset in units of whole tiles.
  MCTLASS_DEVICE
  void load(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a logical offset in units of whole tiles
      TensorCoord const &tile_offset,
      /// loads a tile with a logical offset AND a pointer offset
      Index pointer_offset) const {
    load_with_byte_offset(frag, tile_offset, pointer_offset * sizeof(Element));
  }

  /// Loads a fragment from memory with logical offset in units of whole tiles.
  MCTLASS_DEVICE
  void load_with_byte_offset(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a logical offset in units of whole tiles
      TensorCoord const &tile_offset,
      /// loads a tile with a logical offset AND a pointer offset
      Index byte_offset) const {
    Index pointer_offset =
        tile_offset.contiguous() * Shape::kContiguous /
            Layout::kElementsPerAccess +
        tile_offset.strided() * InstructionShape::kStrided * stride_;

    byte_offset += sizeof(AccessType) * pointer_offset;

    load_with_byte_offset(frag, byte_offset);
  }

  /// Notify the iterator which k-group it is currently pointing to.
  ///
  /// This does not advance the iterator. Rather, it overrides its internal
  /// tracking with constant-valued k-group index to enable the compiler to
  /// fold constants and achieve more efficient code.
  ///
  /// This is used by some nontrivial permuted layouts.
  MCTLASS_DEVICE
  void set_kgroup_index(int k_group) {
    // no op
  }
};

///
/// This special template for float is completely same with mctlass:tfloat32_t,
/// It's a very bad implement. We need to refactor them one day.
///
template <
    /// Size of the matrix to load (concept: PitchLinearShape)
    typename Shape_,
    /// Identifies A or B multiplicand
    Operand Operand_,
    /// Shape of one matrix product operation (concept: PitchLinearShape)
    typename InstructionShape_,
    /// Interval between adjacent *MMA instructions (in units of MMA
    /// instructions)
    int OpDelta_,
    /// Number of partitions along K dimension
    int PartitionsK_>
class MmaTensorOpMultiplicandTileIterator<
    Shape_, Operand_, float,
    mctlass::layout::TensorOpMultiplicandCongruous<32, 32>, InstructionShape_,
    OpDelta_, 32, PartitionsK_> {
 public:
  /// Shape of tile to load (concept: PitchLinearShape)
  using Shape = Shape_;

  /// Operand tag
  static Operand const kOperand = Operand_;

  static_assert(kOperand == Operand::kA || kOperand == Operand::kB,
                "MmaTensorOpMultiplicandIterator may only be instantiated for "
                "A or B operands to warp-level Mma.");

  /// Element type
  using Element = float;

  /// Layout of source tile
  using Layout = mctlass::layout::TensorOpMultiplicandCongruous<32, 32>;

  /// Shape of one matrix product operation (concept: GemmShape)
  using InstructionShape = InstructionShape_;

  /// Delta between *MMA operations (in units of *MMA operations, concept:
  /// MatrixShape)
  static int const kOpDelta = OpDelta_;

  /// Number of participating threads
  static int const kThreads = 32;

  /// Number of partitions along K dimension
  static int const kPartitionsK = PartitionsK_;

  /// TensorRef type for loading element from a tensor
  using TensorRef = TensorRef<Element, Layout>;

  /// Index type
  using Index = typename TensorRef::Index;

  /// Long Index type
  using LongIndex = typename TensorRef::LongIndex;

  /// Long Index type
  using StrideIndex = typename TensorRef::Layout::Stride::Index;

  /// Coordinate for an element in the tensor
  using TensorCoord = typename TensorRef::TensorCoord;

  /// Internal structure of iterator - made public to enable introspection
  struct Policy {
    static_assert(
        !(Shape::kContiguous % InstructionShape::kContiguous),
        "Shape of warp-level Mma must be divisible by operator shape.");

    // Determine number of elements along outer dimension per individual 32bit
    // shared memory load op.  Every one warp of 32bit shared memory load loads
    // 8x4 elements
    static int const kLdsOpInner = Layout::TileShape::kStrided;
    static int const kLdsOpOuter = kThreads / kLdsOpInner;

    static_assert(!(Shape::kContiguous % kLdsOpOuter),
                  "Shape of warp-level mma must be divisible by 32bit "
                  "fundamental tile size.");

    static_assert(!(Shape::kStrided % kLdsOpInner),
                  "Shape of warp-level mma must be divisible by 32bit "
                  "fundamental tile size.");

    /// Number of 32 bit shared memory load instructions needed by one MMA instruction
    /// 1688  A 2x2
    /// 1688  B 1x2
    /// 16816 B 1x4
    static int const LdsShapeContiguous =
        InstructionShape::kContiguous / kLdsOpOuter;
    static int const LdsShapeStrided = InstructionShape::kStrided / kLdsOpInner;
    using LdsShape =
        layout::PitchLinearShape<LdsShapeContiguous, LdsShapeStrided>;

    /// Number and arrangement of LDS instructions
    using LdsIterations = layout::PitchLinearShape<
        Shape::kContiguous / LdsShapeContiguous / kLdsOpOuter, 1>;

    /// Number of groups for each tile
    static int const kGroupsPerTile =
        Shape::kStrided / InstructionShape::kStrided;
  };

 private:
  /// Not working on this feature at the moment.
  static_assert(kOpDelta == 1,
                "Alternative arrangements not supported at present.");

  /// Number of internal pointers needed to reference shared memory
  static int const kPointerCount = Layout::TileShape::kContiguous *
                                   Layout::kElementsPerAccess /
                                   Policy::kLdsOpOuter;

  /// Vectorized access is not used
  static int const kElementsPerAccess = 1;

  /// Pointer type used for accesses
  using AccessType = Element;

  /// Internal counter used to jump to next K partition
  int k_group_idx_;

 public:
  //
  // Derived quantities
  //

  /// Fragment object holding a thread's part of a tile
  using Fragment = Array<Element, Shape::kContiguous * InstructionShape::kStrided / kThreads>;

 private:
  /// Layout object storing stride values
  StrideIndex stride_;

  /// Shared memory base pointers - not advanced
  AccessType const *pointer_[kPointerCount];

  /// Byte offset incremented as iterator advances
  Index byte_offset_;

 public:
  /// Default ctor constructs null iterator
  MCTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator() : stride_(0), byte_offset_(0) {}

  /// Constructor from TensorRef
  MCTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator(TensorRef const &ref, int lane_id)
      : stride_(ref.stride(0)), byte_offset_(0), k_group_idx_(0) {
    pointer_[0] = reinterpret_cast<AccessType const *>(ref.data());
  }

  /// Adds a pointer offset to internal pointer(s) to advance through memory
  MCTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator &add_pointer_offset(LongIndex offset) {
    byte_offset_ += offset * sizeof(Element);

    return *this;
  }

  /// Advances an iterator along logical dimensions of matrix in units of whole
  /// tiles
  MCTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator &add_tile_offset(
      TensorCoord const &tile_offset) {
    int contiguous_offset = tile_offset.contiguous();
    if (Shape::kContiguous ==
        Layout::TileShape::kContiguous * Layout::kElementsPerAccess / 2) {
      if (tile_offset.contiguous() % 2) {
        // Matrix multiply 1688 pointer_[0] <=> pointer_[4] pointer_[1] <=> pointer_[5]
        //           pointer_[2] <=> pointer_[6] pointer_[3] <=> pointer_[7]
        MCTLASS_PRAGMA_UNROLL
        for (int i = 0; i < kPointerCount / 2; ++i) {
          AccessType const *tmp_pointer = pointer_[i];
          pointer_[i] = pointer_[i + kPointerCount / 2];
          pointer_[i + kPointerCount / 2] = tmp_pointer;
        }
      }
      contiguous_offset = (tile_offset.contiguous() >> 1) << 1;
    }

    int offset = (tile_offset.strided() * InstructionShape::kStrided) * stride_ +
                 contiguous_offset * Shape::kContiguous;

    add_pointer_offset(offset);

    return *this;
  }

  /// Advances the iterator along the advance dimension
  MCTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator &operator++() {
    add_tile_offset({0, 1});
    if (kPartitionsK > 1) {
      ++k_group_idx_;
      // Jump to next stage
      if (k_group_idx_ == Policy::kGroupsPerTile) {
        k_group_idx_ = 0;
        add_tile_offset(
            {0, ((kPartitionsK - 1) * Policy::kGroupsPerTile)});
      }
    }

    return *this;
  }

  /// Advances the iterator along the opposite of the advance dimension
  MCTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator &operator--() {
    byte_offset_ -= stride_ * InstructionShape::kStrided * sizeof(Element) *
                    kElementsPerAccess;

    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of
  ///< the tensor
  MCTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator &operator+=(
      TensorCoord const &tile_offset) {
    add_tile_offset(tile_offset);
    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of
  ///< the tensor
  MCTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator &operator-=(
      TensorCoord const &tile_offset) {
    add_tile_offset(-tile_offset);
    return *this;
  }

  /// Loads a fragment from memory at the location pointed to by the iterator.
  MCTLASS_HOST_DEVICE
  void load(Fragment &frag) const { load_with_byte_offset(frag, 0); }

  /// Loads a fragment from memory with additional logical offset
  MCTLASS_DEVICE
  void load_with_byte_offset(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a linear offset in units of bytes
      Index byte_offset) const {
    Element *fetch_ptr = reinterpret_cast<Element *>(&frag);
    int lane_id = __lane_id();

    MCTLASS_PRAGMA_UNROLL
    for (int s = 0; s < Policy::LdsIterations::kStrided; ++s) {
      MCTLASS_PRAGMA_UNROLL
      for (int c = 0; c < Policy::LdsIterations::kContiguous; ++c) {
         if(kOperand == Operand::kA) {

            unsigned row = (lane_id & 0xf) + c * InstructionShape::kContiguous;
            unsigned col = ((lane_id >> 4) << 1) ^ 0x7;
            unsigned col_1 = col % 4;
            unsigned row_1 = ((row / 8) ^ col_1) * 8 + (row % 8);
            unsigned col_2 = (col - 1) % 4;
            unsigned row_2 = ((row / 8) ^ col_2) * 8 + (row % 8);

            AccessType const *source_ptr = pointer_[0] + row_1 + col * stride_;
            char const *source_byte_ptr = reinterpret_cast<char const *>(source_ptr) +
                                          byte_offset + byte_offset_;

            fetch_ptr[c * 4 + 0] = *reinterpret_cast<Element const *>(source_byte_ptr);
            AccessType const *source_ptr1 = pointer_[0] + row_2 + (col - 1) * stride_;
            char const *source_byte_ptr1 = reinterpret_cast<char const *>(source_ptr1) +
                                           byte_offset + byte_offset_;

            fetch_ptr[c * 4 + 1] = *reinterpret_cast<Element const *>(source_byte_ptr1);
         } else {
            unsigned row = ((lane_id >> 4) << 1) ^ 0x7;
            unsigned col = (lane_id & 0x7) + c * InstructionShape::kStrided;
            unsigned row_1 = row % 4;
            unsigned col_1 = ((col / 8) ^ row_1) * 8 + (col % 8);
            unsigned row_2 = (row - 1) % 4;
            unsigned col_2 = ((col / 8) ^ row_2) * 8 + (col % 8);

            AccessType const *source_ptr = pointer_[0];
            char const *source_byte_ptr = reinterpret_cast<char const *>(source_ptr) +
                                          byte_offset + byte_offset_;

            fetch_ptr[c * 2 + 0] = *(reinterpret_cast<Element const *>(source_byte_ptr) +
                                   col_1 + row * stride_);

            fetch_ptr[c * 2 + 1] = *(reinterpret_cast<Element const *>(source_byte_ptr) +
                                   col_2 + (row - 1) * stride_);
         }
      }
    }
    __syncthreads();
  }

  /// Loads a fragment from memory with additional logical offset
  MCTLASS_DEVICE
  void load_with_pointer_offset(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a linear offset
      Index pointer_offset) const {
    load_with_byte_offset(frag, pointer_offset * sizeof(Element));
  }

  /// Loads a fragment from memory with logical offset in units of whole tiles.
  MCTLASS_DEVICE
  void load(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a logical offset in units of whole tiles
      TensorCoord const &tile_offset) const {
    load_with_byte_offset(frag, tile_offset, 0);
  }

  /// Loads a fragment from memory with logical offset in units of whole tiles.
  MCTLASS_DEVICE
  void load(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a logical offset in units of whole tiles
      TensorCoord const &tile_offset,
      /// loads a tile with a logical offset AND a pointer offset
      Index pointer_offset) const {
    load_with_byte_offset(frag, tile_offset, pointer_offset * sizeof(Element));
  }

  /// Loads a fragment from memory with logical offset in units of whole tiles.
  MCTLASS_DEVICE
  void load_with_byte_offset(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a logical offset in units of whole tiles
      TensorCoord const &tile_offset,
      /// loads a tile with a logical offset AND a pointer offset
      Index byte_offset) const {
    Index pointer_offset =
        tile_offset.contiguous() * Shape::kContiguous /
            Layout::kElementsPerAccess +
        tile_offset.strided() * InstructionShape::kStrided * stride_;

    byte_offset += sizeof(AccessType) * pointer_offset;

    load_with_byte_offset(frag, byte_offset);
  }

  /// Notify the iterator which k-group it is currently pointing to.
  ///
  /// This does not advance the iterator. Rather, it overrides its internal
  /// tracking with constant-valued k-group index to enable the compiler to
  /// fold constants and achieve more efficient code.
  ///
  /// This is used by some nontrivial permuted layouts.
  MCTLASS_DEVICE
  void set_kgroup_index(int k_group) {
    // no op
  }
};

////////////////////////////////////////////////////////////////////////////////

/// This tile iterator is specialized for 32-thread TensorOps. It uses LDSM to load from shared
/// memory and therefore must be initialized with a TensorRef to shared memory.
///
/// Satisfies:
///   ReadableRandomAccessContiguousTileIteratorConcept
///
template <
    /// Size of the matrix to load (concept: MatrixShape)
    typename Shape_,
    /// Identifies A or B multiplicand
    Operand Operand_,
    /// Data type of elements
    typename Element_,
    /// Shape of one matrix product operation (concept: MatrixShape)
    typename InstructionShape_,
    /// Interval between adjacent *MMA instructions (in units of MMA
    /// instructions)
    int OpDelta_,
    /// Number of partitions along K dimension
    int PartitionsK_>
class MmaTensorOpMultiplicandTileIterator<
    Shape_, Operand_, Element_,
    mctlass::layout::ColumnMajorTensorOpMultiplicandCongruous<
        sizeof_bits<Element_>::value, int(128 / sizeof(Element_))>,
    InstructionShape_, OpDelta_, 32, PartitionsK_> {
 public:

  /// Shape of tile to load (concept: PitchLinearShape)
  using Shape = Shape_;

  /// Operand tag
  static Operand const kOperand = Operand_;

  static_assert(kOperand == Operand::kA,
                "MmaTensorOpMultiplicandIterator for ColumnMajor Congruous may "
                "only be instantiated for A operand to warp-level Mma.");

  /// Element type
  using Element = Element_;

  /// Layout of source tile
  using Layout = mctlass::layout::ColumnMajorTensorOpMultiplicandCongruous<
      sizeof_bits<Element_>::value, int(128 / sizeof(Element_))>;

  /// Shape of one matrix product operation (concept: MatrixShape)
  using InstructionShape = InstructionShape_;

  /// Delta between *MMA operations (in units of *MMA operations, concept: MatrixShape)
  static int const kOpDelta = OpDelta_;

  /// Number of participating threads
  static int const kThreads = 32;

  /// TensorRef type for loading element from a tensor
  using TensorRef = TensorRef<Element, Layout>;

  /// Index type
  using Index = typename TensorRef::Index;

  /// Long Index type
  using LongIndex = typename TensorRef::LongIndex;

  /// Long Index type
  using StrideIndex = typename TensorRef::Layout::Stride::Index;

  /// Coordinate for an element in the tensor
  using TensorCoord = typename TensorRef::TensorCoord;

  /// Underlying tile iterator implementation
  using Base = MmaTensorOpMultiplicandTileIterator<
      layout::PitchLinearShape<Shape::kRow, Shape::kColumn>, kOperand, Element,
      layout::TensorOpMultiplicandCongruous<sizeof_bits<Element_>::value,
                                            int(128 / sizeof(Element_))>,
      layout::PitchLinearShape<InstructionShape::kRow,
                               InstructionShape::kColumn>,
      kOpDelta, kThreads, PartitionsK_>;

 public:

  //
  // Derived quantities
  //

  /// Fragment object holding a thread's part of a tile
  using Fragment = typename Base::Fragment;

private:

  /// Underlying tile iterator
  Base iterator_;

public:

  /// Default ctor constructs null iterator
  MCTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator() { }

  /// Constructor from TensorRef
  MCTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator(
    TensorRef const &ref,
    int lane_id
  ): iterator_({ref.data(), ref.stride()}, lane_id) {
  }

  /// Adds a pointer offset to internal pointer(s) to advance through memory
  MCTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator &add_pointer_offset(LongIndex offset) {

    iterator_.add_pointer_offset(offset);

    return *this;
  }

  /// Advances an iterator along logical dimensions of matrix in units of whole tiles
  MCTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator &add_tile_offset(TensorCoord const &tile_offset) {

    iterator_.add_tile_offset({tile_offset.row(), tile_offset.column()});

    return *this;
  }

  /// Advances the iterator along the advance dimension
  MCTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator & operator++() {

    ++iterator_;

    return *this;
  }

  /// Advances the iterator along the advance dimension
  MCTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator & operator--() {

    --iterator_;

    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of the tensor
  MCTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator & operator+=(TensorCoord const &tile_offset) {
    add_tile_offset(PitchLinearCoord(tile_offset.row(), tile_offset.column()));
    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of the tensor
  MCTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator & operator-=(TensorCoord const &tile_offset) {
    add_tile_offset(-PitchLinearCoord(tile_offset.row(), tile_offset.column()));
    return *this;
  }

  /// Loads a fragment from memory at the location pointed to by the iterator.
  MCTLASS_HOST_DEVICE
  void load(Fragment &frag) const {

    iterator_.load(frag);
  }

  /// Loads a fragment from memory with additional logical offset
  MCTLASS_DEVICE
  void load_with_pointer_offset(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a linear offset
      Index pointer_offset) const {
    iterator_.load_with_pointer_offset(frag, pointer_offset);
  }

  /// Loads a fragment from memory with additional logical offset
  MCTLASS_DEVICE
  void load_with_byte_offset(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a linear offset
      Index byte_offset) const {
    iterator_.load_with_byte_offset(frag, byte_offset);
  }

  /// Loads a fragment from memory with logical offset in units of whole tiles.
  MCTLASS_DEVICE
  void load(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a logical offset in units of whole tiles
      TensorCoord const &tile_offset) const {
    // TODO
  }

  /// Loads a fragment from memory with logical offset in units of whole tiles.
  MCTLASS_DEVICE
  void load(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a logical offset in units of whole tiles
      TensorCoord const &tile_offset,
      /// loads a tile with a logical offset AND a pointer offset
      Index pointer_offset) const {
    // TODO
  }

  /// Loads a fragment from memory with logical offset in units of whole tiles.
  MCTLASS_DEVICE
  void load_with_byte_offset(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a logical offset in units of whole tiles
      TensorCoord const &tile_offset,
      /// loads a tile with a logical offset AND a pointer offset
      Index byte_offset) const {
    iterator_.load_with_byte_offset(
      frag,
      {tile_offset.contiguous(), tile_offset.strided()},
      byte_offset);
  }

  /// Notify the iterator which k-group it is currently pointing to.
  ///
  /// This does not advance the iterator. Rather, it overrides its internal
  /// tracking with constant-valued k-group index to enable the compiler to
  /// fold constants and achieve more efficient code.
  ///
  /// This is used by some nontrivial permuted layouts.
  MCTLASS_DEVICE
  void set_kgroup_index(int k_group) {
    iterator_.set_kgroup_index(k_group);
  }
};

////////////////////////////////////////////////////////////////////////////////

/// This tile iterator is specialized for 32-thread TensorOps. It uses LDSM to load from shared
/// memory and therefore must be initialized with a TensorRef to shared memory.
///
/// Satisfies:
///   ReadableRandomAccessContiguousTileIteratorConcept
///
template <
    /// Size of the matrix to load (concept: MatrixShape)
    typename Shape_,
    /// Identifies A or B multiplicand
    Operand Operand_,
    /// Data type of elements
    typename Element_,
    /// Shape of one matrix product operation (concept: MatrixShape)
    typename InstructionShape_,
    /// Interval between adjacent *MMA instructions (in units of MMA
    /// instructions)
    int OpDelta_,
    /// Number of partitions along K dimension
    int PartitionsK_>
class MmaTensorOpMultiplicandTileIterator<
    Shape_, Operand_, Element_,
    mctlass::layout::RowMajorTensorOpMultiplicandCongruous<
        sizeof_bits<Element_>::value, int(128 / sizeof(Element_))>,
    InstructionShape_, OpDelta_, 32, PartitionsK_> {
 public:

  /// Shape of tile to load (concept: PitchLinearShape)
  using Shape = Shape_;

  /// Operand tag
  static Operand const kOperand = Operand_;

  static_assert(kOperand == Operand::kB,
                "MmaTensorOpMultiplicandIterator for RowMajor Congruous may "
                "only be instantiated for B operand to warp-level Mma.");

  /// Element type
  using Element = Element_;

  /// Layout of source tile
  using Layout = mctlass::layout::RowMajorTensorOpMultiplicandCongruous<
      sizeof_bits<Element_>::value, int(128 / sizeof(Element_))>;

  /// Shape of one matrix product operation (concept: MatrixShape)
  using InstructionShape = InstructionShape_;

  /// Delta between *MMA operations (in units of *MMA operations, concept: MatrixShape)
  static int const kOpDelta = OpDelta_;

  /// Number of participating threads
  static int const kThreads = 32;

  /// TensorRef type for loading element from a tensor
  using TensorRef = TensorRef<Element, Layout>;

  /// Index type
  using Index = typename TensorRef::Index;

  /// Long Index type
  using LongIndex = typename TensorRef::LongIndex;

  /// Coordinate for an element in the tensor
  using TensorCoord = typename TensorRef::TensorCoord;

  /// Underlying tile iterator implementation
  using Base = MmaTensorOpMultiplicandTileIterator<
      layout::PitchLinearShape<Shape::kColumn, Shape::kRow>, kOperand, Element,
      layout::TensorOpMultiplicandCongruous<sizeof_bits<Element_>::value,
                                            int(128 / sizeof(Element_))>,
      layout::PitchLinearShape<InstructionShape::kColumn,
                               InstructionShape::kRow>,
      kOpDelta, kThreads, PartitionsK_>;

 public:

  //
  // Derived quantities
  //

  /// Fragment object holding a thread's part of a tile
  using Fragment = typename Base::Fragment;

private:

  /// Underlying tile iterator
  Base iterator_;

public:

  /// Default ctor constructs null iterator
  MCTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator() { }

  /// Constructor from TensorRef
  MCTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator(
    TensorRef const &ref,
    int lane_id
  ): iterator_({ref.data(), ref.stride()}, lane_id) {
  }

  /// Adds a pointer offset to internal pointer(s) to advance through memory
  MCTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator &add_pointer_offset(LongIndex offset) {

    iterator_.add_pointer_offset(offset);

    return *this;
  }

  /// Advances an iterator along logical dimensions of matrix in units of whole tiles
  MCTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator &add_tile_offset(TensorCoord const &tile_offset) {

    iterator_.add_tile_offset({tile_offset.column(), tile_offset.row()});

    return *this;
  }

  /// Advances the iterator along the advance dimension
  MCTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator & operator++() {

    ++iterator_;

    return *this;
  }

  /// Advances the iterator along the advance dimension
  MCTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator & operator--() {

    --iterator_;

    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of the tensor
  MCTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator & operator+=(TensorCoord const &tile_offset) {
    add_tile_offset(PitchLinearCoord(tile_offset.column(), tile_offset.row()));
    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of the tensor
  MCTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator & operator-=(TensorCoord const &tile_offset) {
    add_tile_offset(-PitchLinearCoord(tile_offset.column(), tile_offset.row()));
    return *this;
  }

  /// Loads a fragment from memory at the location pointed to by the iterator.
  MCTLASS_HOST_DEVICE
  void load(Fragment &frag) const {

    iterator_.load(frag);
  }

  /// Loads a fragment from memory with additional logical offset
  MCTLASS_DEVICE
  void load_with_pointer_offset(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a linear offset
      Index pointer_offset) const {
    iterator_.load_with_pointer_offset(frag, pointer_offset);
  }

  /// Loads a fragment from memory with additional logical offset
  MCTLASS_DEVICE
  void load_with_byte_offset(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a linear offset
      Index byte_offset) const {
    iterator_.load_with_byte_offset(frag, byte_offset);
  }

  /// Loads a fragment from memory with logical offset in units of whole tiles.
  MCTLASS_DEVICE
  void load(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a logical offset in units of whole tiles
      TensorCoord const &tile_offset) const {
    // TODO
  }

  /// Loads a fragment from memory with logical offset in units of whole tiles.
  MCTLASS_DEVICE
  void load(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a logical offset in units of whole tiles
      TensorCoord const &tile_offset,
      /// loads a tile with a logical offset AND a pointer offset
      Index pointer_offset) const {
    // TODO
  }

  /// Loads a fragment from memory with logical offset in units of whole tiles.
  MCTLASS_DEVICE
  void load_with_byte_offset(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a logical offset in units of whole tiles
      TensorCoord const &tile_offset,
      /// loads a tile with a logical offset AND a pointer offset
      Index byte_offset) const {
    iterator_.load_with_byte_offset(
      frag,
      {tile_offset.strided(), tile_offset.contiguous()},
      byte_offset);
  }

  /// Notify the iterator which k-group it is currently pointing to.
  ///
  /// This does not advance the iterator. Rather, it overrides its internal
  /// tracking with constant-valued k-group index to enable the compiler to
  /// fold constants and achieve more efficient code.
  ///
  /// This is used by some nontrivial permuted layouts.
  MCTLASS_DEVICE
  void set_kgroup_index(int k_group) {
    iterator_.set_kgroup_index(k_group);
  }
};

////////////////////////////////////////////////////////////////////////////////

/// This tile iterator is specialized for 32-thread TensorOps. It uses LDSM to
/// load from shared memory and therefore must be initialized with a TensorRef
/// to shared memory.
///
/// Satisfies:
///   ReadableRandomAccessContiguousTileIteratorConcept
///
template <
    /// Size of the matrix to load (concept: PitchLinearShape)
    typename Shape_,
    /// Identifies A or B multiplicand
    Operand Operand_,
    /// Data type of elements
    typename Element_,
    /// Shape of one matrix product operation (concept: PitchLinearShape)
    typename InstructionShape_,
    /// Interval between adjacent *MMA instructions (in units of MMA
    /// instructions)
    int OpDelta_,
    /// Element number when the layout crosses (in units of elements)
    int Crosswise,
    /// Number of partitions along K dimension
    int PartitionsK_>
class MmaTensorOpMultiplicandTileIterator<
    Shape_, Operand_, Element_,
    mctlass::layout::TensorOpMultiplicandCrosswise<sizeof_bits<Element_>::value,
                                                   Crosswise>,
    InstructionShape_, OpDelta_, 32, PartitionsK_> {
 public:
  /// Shape of tile to load (concept: PitchLinearShape)
  using Shape = Shape_;

  /// Operand tag
  static Operand const kOperand = Operand_;

  static_assert(kOperand == Operand::kA || kOperand == Operand::kB,
                "MmaTensorOpMultiplicandIterator may only be instantiated for "
                "A or B operands to warp-level Mma.");

  /// Element type
  using Element = Element_;

  /// Element number when the layout crosses
  static int const kCrosswise = Crosswise;

  /// Layout of source tile
  using Layout = mctlass::layout::TensorOpMultiplicandCrosswise<
      sizeof_bits<Element_>::value, kCrosswise>;

  /// Shape of one matrix product operation (concept: GemmShape)
  using InstructionShape = InstructionShape_;

  /// Delta between *MMA operations (in units of *MMA operations, concept:
  /// MatrixShape)
  static int const kOpDelta = OpDelta_;

  /// Number of participating threads
  static int const kThreads = 32;

  /// Number of partitions along K dimension
  static int const kPartitionsK = PartitionsK_;

  /// TensorRef type for loading element from a tensor
  using TensorRef = TensorRef<Element, Layout>;

  /// Index type
  using Index = typename TensorRef::Index;

  /// Long Index type
  using LongIndex = typename TensorRef::LongIndex;

  /// Long Index type
  using StrideIndex = typename TensorRef::Layout::Stride::Index;

  /// Coordinate for an element in the tensor
  using TensorCoord = typename TensorRef::TensorCoord;

  /// Internal structure of iterator - made public to enable introspection
  struct Policy {
    static_assert(
        !(Shape::kContiguous % InstructionShape::kContiguous),
        "Shape of warp-level Mma must be divisible by operator shape.");

    // Determine number of elements along outer dimension per individual LDSM op
    static int const kLdsmOpOuter = Layout::kElementsPerAccess;
    static int const kLdsmOpInner = 8;

    static_assert(!(Shape::kContiguous % kLdsmOpOuter),
                  "Shape of warp-level mma must be divisible by LDSM's "
                  "fundamental tile size.");

    static_assert(!(Shape::kStrided % kLdsmOpInner),
                  "Shape of warp-level mma must be divisible by LDSM's "
                  "fundamental tile size.");

    /// Shape of one individual LDSM instruction
    static int const LdsmShapeContiguous =
        InstructionShape::kContiguous / kLdsmOpOuter;
    static int const LdsmShapeStrided =
        ((4 / LdsmShapeContiguous * kLdsmOpInner) > Shape::kStrided)
            ? (Shape::kStrided / kLdsmOpInner)
            : (4 / LdsmShapeContiguous);
    using LdsmShape =
        layout::PitchLinearShape<LdsmShapeContiguous, LdsmShapeStrided>;

    /// Number and arrangement of LDSM instructions
    using LdsmIterations =
        layout::PitchLinearShape<1, Shape::kStrided / kLdsmOpInner /
                                        LdsmShape::kStrided>;

    ///
    static int const kGroupsPerTile = Layout::TileShape::kContiguous /
                                      Layout::kFactor / LdsmShape::kContiguous;
  };

 private:
  /// Not working on this feature at the moment.
  static_assert(kOpDelta == 1,
                "Alternative arrangements not supported at present.");

  /// Pointer type used for accesses
  using AccessType = Array<Element, Layout::kElementsPerAccess>;

 public:
  //
  // Derived quantities
  //

  /// Fragment object holding a thread's part of a tile
  using Fragment = Array<Element, Shape::kStrided *
                                      InstructionShape::kContiguous / kThreads>;

 private:

  /// Total number of sections.  The memory is divided into stages.  One stage
  /// can store one tile.  Stage is divided into sections.  Interleaved layout
  /// can have multiple sections in a stage.  The rest layout only has one section
  /// in a stage.
  int sections_;

  /// Layout object storing stride values
  StrideIndex stride_;

  /// Shared memory base pointers - not advanced
  AccessType const *pointer_;

  /// Byte offset incremented as iterator advances
  Index byte_offset_;

  /// Internal counter used to determine when to increment byte offset and when
  /// to XOR it
  int k_group_idx_;

 public:
  /// Default ctor constructs null iterator
  MCTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator()
      : pointer_(nullptr),
        sections_(0),
        stride_(0),
        byte_offset_(0),
        k_group_idx_(0) {}

  /// Constructor from TensorRef
  MCTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator(TensorRef const &ref, int lane_id)
      : pointer_(reinterpret_cast<AccessType const *>(ref.data())),
        sections_(ref.stride(0) / kCrosswise),
        // stride_ = kCrosswise x sections_ x kFactor
        stride_(ref.stride(0) * Layout::kFactor / Layout::kElementsPerAccess),
        byte_offset_(0),
        k_group_idx_(0) {
    // Warp level iterator at most use double buffer to hide latency.  If there
    // are more than 2 sections, every stage should have more than 1 section.

    // Turing silicon requires all 32 threads in a warp provide valid addresses
    // even for LDSM.1 and LDSM.2
//#if defined(__MACA_ARCH__)
#if 0
    lane_id = lane_id % (Policy::LdsmShape::kCount * Policy::kLdsmOpInner);
#endif

    int quad_quad = (lane_id >> 4);
    int quad_pair = (lane_id >> 3);
    int lane_in_pair = (lane_id & 1);
    int lane_in_quad = (lane_id & 3);
    int lane_in_quad_pair = (lane_id & 7);
    int lane_in_quad_quad = (lane_id & 15);

    int partition_contiguous_idx = -1;
    int access_contiguous_idx = -1;
    int access_strided_idx = -1;

    if (Layout::kFactor == 4) {
      // Super Integer matrix multiply Interleaved-32

      int factor_in_partition =
          (Layout::PartitionShape::kContiguous * Layout::kFactor /
           Layout::TileShape::kContiguous);

      if (Policy::LdsmShape::kStrided == Policy::LdsmShape::kCount) {
        // Integer matrix multiply 8816  A/B
        partition_contiguous_idx = lane_in_quad / factor_in_partition;
        access_contiguous_idx = ((lane_in_pair * factor_in_partition) ^
                                 (lane_in_quad_quad / Layout::kFactor));
        access_strided_idx = lane_id / Layout::kFactor;
      }
      else if (Policy::LdsmShape::kStrided ==
                     (Policy::LdsmShape::kCount / 2) &&
                 kOperand == Operand::kA) {
        // Integer matrix multiply 16832 A
        partition_contiguous_idx = lane_in_quad / factor_in_partition;
        access_strided_idx = lane_in_quad_quad / Layout::kFactor;
        access_contiguous_idx =
            ((lane_in_pair * factor_in_partition + quad_quad) ^
             access_strided_idx);
      }
      else if (Policy::LdsmShape::kStrided ==
                     (Policy::LdsmShape::kCount / 2) &&
                 kOperand == Operand::kB) {
        // Integer matrix multiply 16832 B
        partition_contiguous_idx = lane_in_quad / factor_in_partition;
        access_strided_idx = lane_in_quad_pair / Layout::kFactor + quad_quad * 2;
        access_contiguous_idx =
            ((lane_in_pair * factor_in_partition + ((lane_id & 8) >> 3)) ^
             access_strided_idx);
      }
    } else if (Layout::kFactor == 2) {
      // Super Matrix multiply kBlock = 32
      if (Policy::LdsmShape::kStrided == Policy::LdsmShape::kCount) {
        // Matrix multiply 1688 A/B
        // (Q stands for 1 8x128bit block).
        // Q0
        // Q1
        // Q2
        // Q3
        // Four blocks are next to each other in the strided dimension.
        partition_contiguous_idx = (lane_id % Layout::kFactor);
        access_contiguous_idx = (lane_in_quad_pair / Layout::kFactor);
        access_strided_idx = lane_id / Layout::kFactor;
      }
      else if (Policy::LdsmShape::kStrided ==
                     (Policy::LdsmShape::kCount / 2) &&
                 kOperand == Operand::kA) {
        // Matrix multiply 16816|1688.TF32 A
        // Q0 Q2
        // Q1 Q3
        partition_contiguous_idx = (lane_id % Layout::kFactor);
        access_contiguous_idx =
            (quad_quad ^ (lane_in_quad_pair / Layout::kFactor));
        access_strided_idx = (lane_in_quad_quad / Layout::kFactor);
      } else if (Policy::LdsmShape::kStrided ==
                     (Policy::LdsmShape::kCount / 2) &&
                 kOperand == Operand::kB) {
        // Matrix multiply 16816|1688.TF32 B
        // Q0 Q1
        // Q2 Q3
        partition_contiguous_idx = (lane_id % Layout::kFactor);
        access_contiguous_idx =
            ((quad_pair & 1) ^ (lane_in_quad_pair / Layout::kFactor));
        access_strided_idx =
            (lane_in_quad_pair + (lane_id >> 4 << 3)) / Layout::kFactor;
      }
      else if (Policy::LdsmShape::kContiguous == Policy::LdsmShape::kCount) {
        // Matrix multiply 16832.SP B
        // Q0 Q1 Q2 Q3
        partition_contiguous_idx = (lane_id % Layout::kFactor);
        access_contiguous_idx =
            (quad_pair ^ (lane_in_quad_pair / Layout::kFactor));
        access_strided_idx = lane_in_quad_pair / Layout::kFactor;
      }
    } else if (Layout::kFactor == 1) {
      // Super Matrix multiply kBlock = 64
      if (Policy::LdsmShape::kStrided == Policy::LdsmShape::kCount) {
        // Q0
        // Q1
        // Q2
        // Q3
        partition_contiguous_idx = (lane_in_quad_pair >> 2);
        access_contiguous_idx = lane_in_quad;
        access_strided_idx = lane_id;
      }
      else if (Policy::LdsmShape::kStrided ==
                     (Policy::LdsmShape::kCount / 2) &&
                 kOperand == Operand::kA) {
        // Matrix multiply 16816|1688.TF32 A
        // Q0 Q2
        // Q1 Q3
        partition_contiguous_idx = (lane_in_quad_pair >> 2);
        access_contiguous_idx = (quad_quad ^ lane_in_quad);
        access_strided_idx = lane_in_quad_quad;
      } else if (Policy::LdsmShape::kStrided ==
                     (Policy::LdsmShape::kCount / 2) &&
                 kOperand == Operand::kB) {
        // Matrix multiply 16816|1688.TF32 B
        // Q0 Q1
        // Q2 Q3
        partition_contiguous_idx = (lane_in_quad_pair >> 2);
        access_contiguous_idx = ((quad_pair & 1) ^ lane_in_quad);
        access_strided_idx = lane_in_quad_pair + (lane_id >> 4 << 3);
      }
      else if (Policy::LdsmShape::kContiguous == Policy::LdsmShape::kCount) {
        // Matrix multiply 16832.SP B
        // Q0 Q1 Q2 Q3
        partition_contiguous_idx = (lane_in_quad_pair >> 2);
        access_contiguous_idx = (quad_pair ^ lane_in_quad);
        access_strided_idx = lane_in_quad_pair;
      }
    }

    int access_contiguous =
        partition_contiguous_idx * Layout::PartitionShape::kContiguous +
        access_contiguous_idx;

    int access_strided = access_strided_idx;

    byte_offset_ = (access_contiguous + access_strided * stride_) *
                   sizeof_bits<Element>::value * Layout::kElementsPerAccess / 8;
  }

  /// Adds a pointer offset to internal pointer(s) to advance through memory
  MCTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator &add_pointer_offset(LongIndex offset) {
    byte_offset_ += offset * sizeof_bits<Element>::value / 8;

    return *this;
  }

  /// Advances an iterator along logical dimensions of matrix in units of whole
  /// tiles
  MCTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator &add_tile_offset(
      TensorCoord const &tile_offset) {
    int whole_tiles = tile_offset.contiguous() / Policy::kGroupsPerTile;
    int k_groups_delta = tile_offset.contiguous() % Policy::kGroupsPerTile;

    byte_offset_ ^= k_groups_delta * sizeof_bits<Element>::value *
                    Layout::kElementsPerAccess *
                    Policy::LdsmShape::kContiguous / 8;
    pointer_ +=
        tile_offset.strided() * stride_ * Shape::kStrided / Layout::kFactor +
        whole_tiles * stride_ / sections_;
    return *this;
  }

  /// Advances an iterator along logical dimensions of matrix in units of whole
  /// tiles
  MCTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator &add_tile_offset_negative(
      TensorCoord const &tile_offset) {

    int whole_tiles = tile_offset.contiguous() / Policy::kGroupsPerTile;
    int k_groups_delta = tile_offset.contiguous() % Policy::kGroupsPerTile;
    if (k_groups_delta < 0) {
        whole_tiles -= 1;
        k_groups_delta += Policy::kGroupsPerTile;
    }

    if ((Policy::kGroupsPerTile / kPartitionsK) >= 2) {
      byte_offset_ ^= (k_groups_delta & 1) * Policy::LdsmShape::kContiguous *
                        sizeof_bits<Element>::value *
                        Layout::kElementsPerAccess / 8;
    }
    if ((Policy::kGroupsPerTile / kPartitionsK) >= 4) {
      byte_offset_ ^= ((k_groups_delta + (k_group_idx_ & 1)) & 2) *
                        Policy::LdsmShape::kContiguous *
                        sizeof_bits<Element>::value *
                        Layout::kElementsPerAccess / 8;
    }
    if ((Policy::kGroupsPerTile / kPartitionsK) == 8) {
      byte_offset_ ^= ((k_groups_delta + (k_group_idx_ & 3)) & 4) *
                        Policy::LdsmShape::kContiguous *
                        sizeof_bits<Element>::value *
                        Layout::kElementsPerAccess / 8;
    }

    k_group_idx_ += k_groups_delta;
    whole_tiles += k_group_idx_ / (Policy::kGroupsPerTile / kPartitionsK);
    k_group_idx_ = k_group_idx_ % (Policy::kGroupsPerTile / kPartitionsK);

    pointer_ +=
        tile_offset.strided() * stride_ * Shape::kStrided / Layout::kFactor +
        whole_tiles * stride_ / sections_;
    return *this;
  }

  /// Advances the iterator along the advance dimension
  MCTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator &operator++() {

    // Integer matrix multiply 16832 Interleaved-32
    //   NONE
    // Integer matrix multiply 16816 Interleaved-32 || Integer matrix multiply 16816 kblock=32

    // Integer matrix multiply 8816  Interleaved-32
    //   ^1 ^1
    // Matrix multiply 1684.TF32 kblock=16 || Integer matrix multiply 16816 kblock=64
    // Matrix multiply 1688 kblock=32 || Integer matrix multiply 8816 kblock=64
    //   ^1 ^3 ^1 ^3
    // Matrix multiply 1688 kblock=64
    //   ^1 ^3 ^1 ^7 ^1 ^3 ^1 ^7

    // Matrix multiply 16816 kblock=32 | 1688.TF32 kblock=16 || Integer matrix multiply 16832 kblock=64
    //   ^2 ^2
    // Matrix multiply 16816 kblock=64 | 1688.TF32 kblock=32 || Integer matrix multiply 16832 kblock=128
    //   ^2 ^6 ^2 ^6

    if ((Policy::kGroupsPerTile / kPartitionsK) > 1) {
      int mask = ((Policy::kGroupsPerTile / kPartitionsK) == 8)
                     ? 3
                     : (((Policy::kGroupsPerTile / kPartitionsK) == 4) ? 1 : 0);

      if (((k_group_idx_ & mask) % 2) == 0)
        byte_offset_ ^= 1 * Policy::LdsmShape::kContiguous *
                        sizeof_bits<Element>::value *
                        Layout::kElementsPerAccess / 8;
      else if ((k_group_idx_ & mask) == 1)
        byte_offset_ ^= 3 * Policy::LdsmShape::kContiguous *
                        sizeof_bits<Element>::value *
                        Layout::kElementsPerAccess / 8;
      else if ((k_group_idx_ & mask) == 3)
        byte_offset_ ^= 7 * Policy::LdsmShape::kContiguous *
                        sizeof_bits<Element>::value *
                        Layout::kElementsPerAccess / 8;
    }

    k_group_idx_++;

    if (k_group_idx_ == (Policy::kGroupsPerTile / kPartitionsK)) {
      k_group_idx_ = 0;
      add_tile_offset({Policy::kGroupsPerTile, 0});
    }

    return *this;
  }

  /// Advances the iterator along the advance dimension
  MCTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator &operator--() { assert(0); }

  ///< advances in units of whole tiles along the logical coordinate space of
  ///< the tensor
  MCTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator &operator+=(
      TensorCoord const &tile_offset) {
    add_tile_offset(tile_offset);
    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of
  ///< the tensor
  MCTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator &operator-=(
      TensorCoord const &tile_offset) {
    add_tile_offset(-tile_offset);
    return *this;
  }

  /// Loads a fragment from memory at the location pointed to by the iterator.
  MCTLASS_HOST_DEVICE
  void load(Fragment &frag) const { load_with_byte_offset(frag, 0); }

  /// Loads a fragment from memory with additional logical offset
  MCTLASS_DEVICE
  void load_with_byte_offset(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a linear offset in units of bytes
      Index byte_offset) const {
    Array<unsigned, Policy::LdsmShape::kCount> *fetch_ptr =
        reinterpret_cast<Array<unsigned, Policy::LdsmShape::kCount> *>(&frag);

    MCTLASS_PRAGMA_UNROLL
    for (int s = 0; s < Policy::LdsmIterations::kStrided; ++s) {
      MCTLASS_PRAGMA_UNROLL
      for (int c = 0; c < Policy::LdsmIterations::kContiguous; ++c) {
        int access_idx = c + s * Policy::LdsmIterations::kContiguous;

        AccessType const *source_ptr =
            pointer_ + Policy::LdsmShape::kContiguous * c +
            Policy::kLdsmOpInner / Layout::kFactor *
                Policy::LdsmShape::kStrided * s * stride_;

        char const *source_byte_ptr =
            reinterpret_cast<char const *>(source_ptr) + byte_offset +
            byte_offset_;

        mctlass::arch::ldsm<layout::RowMajor, Policy::LdsmShape::kCount>(
            fetch_ptr[access_idx], source_byte_ptr);
      }
    }
  }

  /// Loads a fragment from memory with additional logical offset
  MCTLASS_DEVICE
  void load_with_pointer_offset(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a linear offset
      Index pointer_offset) const {
    load_with_byte_offset(frag, pointer_offset * sizeof(Element));
  }

  /// Loads a fragment from memory with logical offset in units of whole tiles.
  MCTLASS_DEVICE
  void load(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a logical offset in units of whole tiles
      TensorCoord const &tile_offset) const {
    load_with_byte_offset(frag, tile_offset, 0);
  }

  /// Loads a fragment from memory with logical offset in units of whole tiles.
  MCTLASS_DEVICE
  void load(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a logical offset in units of whole tiles
      TensorCoord const &tile_offset,
      /// loads a tile with a logical offset AND a pointer offset
      Index pointer_offset) const {
    load_with_byte_offset(frag, tile_offset, pointer_offset * sizeof(Element));
  }

  /// Loads a fragment from memory with logical offset in units of whole tiles.
  MCTLASS_DEVICE
  void load_with_byte_offset(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a logical offset in units of whole tiles
      TensorCoord const &tile_offset,
      /// loads a tile with a logical offset AND a pointer offset
      Index byte_offset) const {
    Index pointer_offset = tile_offset.contiguous() *
                               InstructionShape::kContiguous /
                               Layout::kElementsPerAccess +
                           tile_offset.strided() * Shape::kStrided * stride_;

    byte_offset += sizeof_bits<AccessType>::value * pointer_offset / 8;

    load_with_byte_offset(frag, byte_offset);
  }

  /// Notify the iterator which k-group it is currently pointing to.
  ///
  /// This does not advance the iterator. Rather, it overrides its internal
  /// tracking with constant-valued k-group index to enable the compiler to
  /// fold constants and achieve more efficient code.
  ///
  /// This is used by some nontrivial permuted layouts.
  MCTLASS_DEVICE
  void set_kgroup_index(int k_group) {
    k_group_idx_ = k_group % (Policy::kGroupsPerTile / kPartitionsK);
  }
};

template <
    /// Size of the matrix to load (concept: PitchLinearShape)
    typename Shape_,
    /// Identifies A or B multiplicand
    Operand Operand_,
    /// Shape of one matrix product operation (concept: PitchLinearShape)
    typename InstructionShape_,
    /// Interval between adjacent *MMA instructions (in units of MMA
    /// instructions)
    int OpDelta_,
    /// Element number when the layout crosses (in units of elements)
    int Crosswise,
    /// Number of partitions along K dimension
    int PartitionsK_>
class MmaTensorOpMultiplicandTileIterator<
    Shape_, Operand_, mctlass::tfloat32_t,
    mctlass::layout::TensorOpMultiplicandCrosswise<sizeof_bits<mctlass::tfloat32_t>::value,
                                                   Crosswise>,
    InstructionShape_, OpDelta_, 32, PartitionsK_> {
 public:
  /// Shape of tile to load (concept: PitchLinearShape)
  using Shape = Shape_;

  /// Operand tag
  static Operand const kOperand = Operand_;

  static_assert(kOperand == Operand::kA || kOperand == Operand::kB,
                "MmaTensorOpMultiplicandIterator may only be instantiated for "
                "A or B operands to warp-level Mma.");

  /// Element type
  using Element = mctlass::tfloat32_t;

  /// Element number when the layout crosses
  static int const kCrosswise = Crosswise;

  /// Layout of source tile
  using Layout = mctlass::layout::TensorOpMultiplicandCrosswise<
      sizeof_bits<mctlass::tfloat32_t>::value, kCrosswise>;

  /// Shape of one matrix product operation (concept: GemmShape)
  using InstructionShape = InstructionShape_;

  /// Delta between *MMA operations (in units of *MMA operations, concept:
  /// MatrixShape)
  static int const kOpDelta = OpDelta_;

  /// Number of participating threads
  static int const kThreads = 32;

  /// Number of partitions along K dimension
  static int const kPartitionsK = PartitionsK_;

  /// TensorRef type for loading element from a tensor
  using TensorRef = TensorRef<Element, Layout>;

  /// Index type
  using Index = typename TensorRef::Index;

  /// Long Index type
  using LongIndex = typename TensorRef::LongIndex;

  /// Long Index type
  using StrideIndex = typename TensorRef::Layout::Stride::Index;

  /// Coordinate for an element in the tensor
  using TensorCoord = typename TensorRef::TensorCoord;

  /// Internal structure of iterator - made public to enable introspection
  struct Policy {
    static_assert(
        !(Shape::kContiguous % InstructionShape::kContiguous),
        "Shape of warp-level Mma must be divisible by operator shape.");

    // Determine number of elements along outer dimension per individual LDSM op
    static int const kLdsmOpOuter = Layout::kElementsPerAccess;
    static int const kLdsmOpInner = 8;

    static_assert(!(Shape::kContiguous % kLdsmOpOuter),
                  "Shape of warp-level mma must be divisible by LDSM's "
                  "fundamental tile size.");

    static_assert(!(Shape::kStrided % kLdsmOpInner),
                  "Shape of warp-level mma must be divisible by LDSM's "
                  "fundamental tile size.");

    /// Shape of one individual LDSM instruction
    static int const LdsmShapeContiguous =
        InstructionShape::kContiguous / kLdsmOpOuter;
    static int const LdsmShapeStrided =
        ((4 / LdsmShapeContiguous * kLdsmOpInner) > Shape::kStrided)
            ? (Shape::kStrided / kLdsmOpInner)
            : (4 / LdsmShapeContiguous);
    using LdsmShape =
        layout::PitchLinearShape<LdsmShapeContiguous, LdsmShapeStrided>;

    /// Number and arrangement of LDSM instructions
    using LdsmIterations =
        layout::PitchLinearShape<1, Shape::kStrided / kLdsmOpInner /
                                        LdsmShape::kStrided>;

    ///
    static int const kGroupsPerTile = Layout::TileShape::kContiguous /
                                      Layout::kFactor / LdsmShape::kContiguous;
  };

 private:
  /// Not working on this feature at the moment.
  static_assert(kOpDelta == 1,
                "Alternative arrangements not supported at present.");

  /// Pointer type used for accesses
  using AccessType = Array<Element, Layout::kElementsPerAccess>;

 public:
  //
  // Derived quantities
  //

  /// Fragment object holding a thread's part of a tile
  using Fragment = Array<Element, Shape::kStrided *
                                      InstructionShape::kContiguous / kThreads>;

 private:

  /// Total number of sections.  The memory is divided into stages.  One stage
  /// can store one tile.  Stage is divided into sections.  Interleaved layout
  /// can have multiple sections in a stage.  The rest layout only has one section
  /// in a stage.
  int sections_;

  /// Layout object storing stride values
  StrideIndex stride_;

  /// Shared memory base pointers - not advanced
  AccessType const *pointer_;

  /// Byte offset incremented as iterator advances
  Index byte_offset_;

  /// Internal counter used to determine when to increment byte offset and when
  /// to XOR it
  int k_group_idx_;

 public:
  /// Default ctor constructs null iterator
  MCTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator()
      : pointer_(nullptr),
        sections_(0),
        stride_(0),
        byte_offset_(0),
        k_group_idx_(0) {}

  /// Constructor from TensorRef
  MCTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator(TensorRef const &ref, int lane_id)
      : pointer_(reinterpret_cast<AccessType const *>(ref.data())),
        sections_(ref.stride(0) / kCrosswise),
        // stride_ = kCrosswise x sections_ x kFactor
        stride_(ref.stride(0) * Layout::kFactor / Layout::kElementsPerAccess),
        byte_offset_(0),
        k_group_idx_(0) {
    // Warp level iterator at most use double buffer to hide latency.  If there
    // are more than 2 sections, every stage should have more than 1 section.

    // Turing silicon requires all 32 threads in a warp provide valid addresses
    // even for LDSM.1 and LDSM.2
//#if defined(__MACA_ARCH__)
#if 0
    lane_id = lane_id % (Policy::LdsmShape::kCount * Policy::kLdsmOpInner);
#endif

    int quad_quad = (lane_id >> 4);
    int quad_pair = (lane_id >> 3);
    int lane_in_pair = (lane_id & 1);
    int lane_in_quad = (lane_id & 3);
    int lane_in_quad_pair = (lane_id & 7);
    int lane_in_quad_quad = (lane_id & 15);

    int partition_contiguous_idx = -1;
    int access_contiguous_idx = -1;
    int access_strided_idx = -1;

    if (Layout::kFactor == 4) {
      // Super Integer matrix multiply Interleaved-32

      int factor_in_partition =
          (Layout::PartitionShape::kContiguous * Layout::kFactor /
           Layout::TileShape::kContiguous);

      if (Policy::LdsmShape::kStrided == Policy::LdsmShape::kCount) {
        // Integer matrix multiply 8816  A/B
        partition_contiguous_idx = lane_in_quad / factor_in_partition;
        access_contiguous_idx = ((lane_in_pair * factor_in_partition) ^
                                 (lane_in_quad_quad / Layout::kFactor));
        access_strided_idx = lane_id / Layout::kFactor;
      }
      else if (Policy::LdsmShape::kStrided ==
                     (Policy::LdsmShape::kCount / 2) &&
                 kOperand == Operand::kA) {
        // Integer matrix multiply 16832 A
        partition_contiguous_idx = lane_in_quad / factor_in_partition;
        access_strided_idx = lane_in_quad_quad / Layout::kFactor;
        access_contiguous_idx =
            ((lane_in_pair * factor_in_partition + quad_quad) ^
             access_strided_idx);
      }
      else if (Policy::LdsmShape::kStrided ==
                     (Policy::LdsmShape::kCount / 2) &&
                 kOperand == Operand::kB) {
        // Integer matrix multiply 16832 B
        partition_contiguous_idx = lane_in_quad / factor_in_partition;
        access_strided_idx = lane_in_quad_pair / Layout::kFactor + quad_quad * 2;
        access_contiguous_idx =
            ((lane_in_pair * factor_in_partition + ((lane_id & 8) >> 3)) ^
             access_strided_idx);
      }
    } else if (Layout::kFactor == 2) {
      // Super Matrix multiply kBlock = 32
      if (Policy::LdsmShape::kStrided == Policy::LdsmShape::kCount) {
        // Matrix multiply 1688 A/B
        // (Q stands for 1 8x128bit block).
        // Q0
        // Q1
        // Q2
        // Q3
        // Four blocks are next to each other in the strided dimension.
        partition_contiguous_idx = (lane_id % Layout::kFactor);
        access_contiguous_idx = (lane_in_quad_pair / Layout::kFactor);
        access_strided_idx = lane_id / Layout::kFactor;
      }
      else if (Policy::LdsmShape::kStrided ==
                     (Policy::LdsmShape::kCount / 2) &&
                 kOperand == Operand::kA) {
        // Matrix multiply 16816|1688.TF32 A
        // Q0 Q2
        // Q1 Q3
        partition_contiguous_idx = (lane_id % Layout::kFactor);
        access_contiguous_idx =
            (quad_quad ^ (lane_in_quad_pair / Layout::kFactor));
        access_strided_idx = (lane_in_quad_quad / Layout::kFactor);
      } else if (Policy::LdsmShape::kStrided ==
                     (Policy::LdsmShape::kCount / 2) &&
                 kOperand == Operand::kB) {
        // Matrix multiply 16816|1688.TF32 B
        // Q0 Q1
        // Q2 Q3
        partition_contiguous_idx = (lane_id % Layout::kFactor);
        access_contiguous_idx =
            ((quad_pair & 1) ^ (lane_in_quad_pair / Layout::kFactor));
        access_strided_idx =
            (lane_in_quad_pair + (lane_id >> 4 << 3)) / Layout::kFactor;
      }
      else if (Policy::LdsmShape::kContiguous == Policy::LdsmShape::kCount) {
        // Matrix multiply 16832.SP B
        // Q0 Q1 Q2 Q3
        partition_contiguous_idx = (lane_id % Layout::kFactor);
        access_contiguous_idx =
            (quad_pair ^ (lane_in_quad_pair / Layout::kFactor));
        access_strided_idx = lane_in_quad_pair / Layout::kFactor;
      }
    } else if (Layout::kFactor == 1) {
      // Super Matrix multiply kBlock = 64
      if (Policy::LdsmShape::kStrided == Policy::LdsmShape::kCount) {
        // Q0
        // Q1
        // Q2
        // Q3
        partition_contiguous_idx = (lane_in_quad_pair >> 2);
        access_contiguous_idx = lane_in_quad;
        access_strided_idx = lane_id;
      }
      else if (Policy::LdsmShape::kStrided ==
                     (Policy::LdsmShape::kCount / 2) &&
                 kOperand == Operand::kA) {
        // Matrix multiply 16816|1688.TF32 A
        // Q0 Q2
        // Q1 Q3
        partition_contiguous_idx = (lane_in_quad_pair >> 2);
        access_contiguous_idx = (quad_quad ^ lane_in_quad);
        access_strided_idx = lane_in_quad_quad;
      } else if (Policy::LdsmShape::kStrided ==
                     (Policy::LdsmShape::kCount / 2) &&
                 kOperand == Operand::kB) {
        // Matrix multiply 16816|1688.TF32 B
        // Q0 Q1
        // Q2 Q3
        partition_contiguous_idx = (lane_in_quad_pair >> 2);
        access_contiguous_idx = ((quad_pair & 1) ^ lane_in_quad);
        access_strided_idx = lane_in_quad_pair + (lane_id >> 4 << 3);
      }
      else if (Policy::LdsmShape::kContiguous == Policy::LdsmShape::kCount) {
        // Matrix multiply 16832.SP B
        // Q0 Q1 Q2 Q3
        partition_contiguous_idx = (lane_in_quad_pair >> 2);
        access_contiguous_idx = (quad_pair ^ lane_in_quad);
        access_strided_idx = lane_in_quad_pair;
      }
    }

    int access_contiguous = partition_contiguous_idx * Layout::PartitionShape::kContiguous +
                            access_contiguous_idx;

    int access_strided = access_strided_idx;
    int byte_offset_o = (access_contiguous + access_strided * stride_) *
                        sizeof_bits<Element>::value * Layout::kElementsPerAccess / 8;

    int row, col;
    if (kOperand == Operand::kA) {
      if(InstructionShape::kContiguous == 8) {
        row = lane_id & 0xf;
        col = ((lane_id >> 4) << 1) ^ 0x7;
      }
      else {
        row = lane_id & 0xf;
        col = (lane_id >> 4) ^ 0x3;
      }
    }
    else {
      if(InstructionShape::kContiguous == 8) {
        row = lane_id & 0x7;
        col = ((lane_id >> 4) << 1) ^ 0x7;
      }
      else {
        row = lane_id & 0x7;
        col = (lane_id >> 4) ^ 0x3;
      }
    }

    TensorCoord coord = make_Coord(col, row);
    Layout lo(ref.stride(0));
    byte_offset_ =  lo(coord) * (sizeof_bits<Element>::value / 8);
  }

  /// Adds a pointer offset to internal pointer(s) to advance through memory
  MCTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator &add_pointer_offset(LongIndex offset) {
    byte_offset_ += offset * sizeof_bits<Element>::value / 8;

    return *this;
  }

  /// Advances an iterator along logical dimensions of matrix in units of whole
  /// tiles
  MCTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator &add_tile_offset(
      TensorCoord const &tile_offset) {
    int whole_tiles = tile_offset.contiguous() / Policy::kGroupsPerTile;
    int k_groups_delta = tile_offset.contiguous() % Policy::kGroupsPerTile;

    byte_offset_ ^= k_groups_delta * sizeof_bits<Element>::value *
                    Layout::kElementsPerAccess *
                    Policy::LdsmShape::kContiguous / 8;
    pointer_ +=
        tile_offset.strided() * stride_ * Shape::kStrided / Layout::kFactor +
        whole_tiles * stride_ / sections_;
    return *this;
  }

  /// Advances an iterator along logical dimensions of matrix in units of whole
  /// tiles
  MCTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator &add_tile_offset_negative(
      TensorCoord const &tile_offset) {

    int whole_tiles = tile_offset.contiguous() / Policy::kGroupsPerTile;
    int k_groups_delta = tile_offset.contiguous() % Policy::kGroupsPerTile;
    if (k_groups_delta < 0) {
        whole_tiles -= 1;
        k_groups_delta += Policy::kGroupsPerTile;
    }

    if ((Policy::kGroupsPerTile / kPartitionsK) >= 2) {
      byte_offset_ ^= (k_groups_delta & 1) * Policy::LdsmShape::kContiguous *
                        sizeof_bits<Element>::value *
                        Layout::kElementsPerAccess / 8;
    }
    if ((Policy::kGroupsPerTile / kPartitionsK) >= 4) {
      byte_offset_ ^= ((k_groups_delta + (k_group_idx_ & 1)) & 2) *
                        Policy::LdsmShape::kContiguous *
                        sizeof_bits<Element>::value *
                        Layout::kElementsPerAccess / 8;
    }
    if ((Policy::kGroupsPerTile / kPartitionsK) == 8) {
      byte_offset_ ^= ((k_groups_delta + (k_group_idx_ & 3)) & 4) *
                        Policy::LdsmShape::kContiguous *
                        sizeof_bits<Element>::value *
                        Layout::kElementsPerAccess / 8;
    }

    k_group_idx_ += k_groups_delta;
    whole_tiles += k_group_idx_ / (Policy::kGroupsPerTile / kPartitionsK);
    k_group_idx_ = k_group_idx_ % (Policy::kGroupsPerTile / kPartitionsK);

    pointer_ +=
        tile_offset.strided() * stride_ * Shape::kStrided / Layout::kFactor +
        whole_tiles * stride_ / sections_;
    return *this;
  }

  /// Advances the iterator along the advance dimension
  MCTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator &operator++() {

    if ((Policy::kGroupsPerTile / kPartitionsK) > 1) {
      int mask = ((Policy::kGroupsPerTile / kPartitionsK) == 8)
                     ? 3
                     : (((Policy::kGroupsPerTile / kPartitionsK) == 4) ? 1 : 0);

      if (((k_group_idx_ & mask) % 2) == 0)
        byte_offset_ ^= 1 * Policy::LdsmShape::kContiguous *
                        sizeof_bits<Element>::value *
                        Layout::kElementsPerAccess / 8;
      else if ((k_group_idx_ & mask) == 1)
        byte_offset_ ^= 3 * Policy::LdsmShape::kContiguous *
                        sizeof_bits<Element>::value *
                        Layout::kElementsPerAccess / 8;
      else if ((k_group_idx_ & mask) == 3)
        byte_offset_ ^= 7 * Policy::LdsmShape::kContiguous *
                        sizeof_bits<Element>::value *
                        Layout::kElementsPerAccess / 8;
    }

    k_group_idx_++;

    if (k_group_idx_ == (Policy::kGroupsPerTile / kPartitionsK)) {
      k_group_idx_ = 0;
      add_tile_offset({Policy::kGroupsPerTile, 0});
    }

    return *this;
  }

  /// Advances the iterator along the advance dimension
  MCTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator &operator--() { assert(0); }

  ///< advances in units of whole tiles along the logical coordinate space of
  ///< the tensor
  MCTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator &operator+=(
      TensorCoord const &tile_offset) {
    add_tile_offset(tile_offset);
    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of
  ///< the tensor
  MCTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator &operator-=(
      TensorCoord const &tile_offset) {
    add_tile_offset(-tile_offset);
    return *this;
  }

  /// Loads a fragment from memory at the location pointed to by the iterator.
  MCTLASS_HOST_DEVICE
  void load(Fragment &frag) const {
    load_with_byte_offset(frag, 0);

  }

  /// Loads a fragment from memory with additional logical offset
  MCTLASS_DEVICE
  void load_with_byte_offset(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a linear offset in units of bytes
      Index byte_offset) const {
    Array<unsigned, Policy::LdsmShape::kCount> *fetch_ptr =
        reinterpret_cast<Array<unsigned, Policy::LdsmShape::kCount> *>(&frag);
    int ldm = stride_ * Layout::kElementsPerAccess / Layout::kFactor;

    MCTLASS_PRAGMA_UNROLL
    for (int s = 0; s < Policy::LdsmIterations::kStrided; ++s) {
      MCTLASS_PRAGMA_UNROLL
      for (int c = 0; c < Policy::LdsmIterations::kContiguous; ++c) {
        int access_idx = c + s * Policy::LdsmIterations::kContiguous;

        AccessType const *source_ptr =
            pointer_ + Policy::LdsmShape::kContiguous * c +
            Policy::kLdsmOpInner / Layout::kFactor *
                Policy::LdsmShape::kStrided * s * stride_;

        char const *source_byte_ptr =
            reinterpret_cast<char const *>(source_ptr) + byte_offset +
            byte_offset_;
        if(kOperand == Operand::kA) {
          if(InstructionShape::kContiguous == 8) {
            mctlass::arch::ldsmAtf32<layout::RowMajor, Policy::LdsmShape::kCount>(
                fetch_ptr[access_idx], source_byte_ptr);
          }
          else {
            int const *p = reinterpret_cast<int const *>(source_byte_ptr);
            float const *pf = reinterpret_cast<float const *>(source_byte_ptr);
            if(4 == Policy::LdsmShape::kCount) {
              int x, y, z, w;
              x = p[0];
              z = p[ldm * 16];
              reinterpret_cast<int4 &>(fetch_ptr[access_idx]) = make_int4(x, y, z, w);
            }
            else {
              int x, y;
              x = p[0];
              reinterpret_cast<int2 &>(fetch_ptr[access_idx]) = make_int2(x, y);
            }
          }
        }
        else {
          if(InstructionShape::kContiguous == 8) {
            mctlass::arch::ldsmBtf32<layout::RowMajor, Policy::LdsmShape::kCount>(
                fetch_ptr[access_idx], source_byte_ptr, ldm);
          }
          else {
            int const *p = reinterpret_cast<int const *>(source_byte_ptr);
            float const *fp = reinterpret_cast<float const *>(source_byte_ptr);
            int x, y, z, w;
            if(4 == Policy::LdsmShape::kCount) {
              x = p[0];
              y = p[8 * ldm];
              z = p[16 * ldm];
              w = p[24 * ldm];
              reinterpret_cast<int4 &>(fetch_ptr[access_idx]) = make_int4(x, y, z, w);
            }
            else {
              x = p[0];
              y = p[8 * ldm];
              reinterpret_cast<int2 &>(fetch_ptr[access_idx]) = make_int2(x, y);
            }
          }
        }
      }
    }
  __syncthreads();
  }

  /// Loads a fragment from memory with additional logical offset
  MCTLASS_DEVICE
  void load_with_pointer_offset(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a linear offset
      Index pointer_offset) const {
    load_with_byte_offset(frag, pointer_offset * sizeof(Element));
  }

  /// Loads a fragment from memory with logical offset in units of whole tiles.
  MCTLASS_DEVICE
  void load(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a logical offset in units of whole tiles
      TensorCoord const &tile_offset) const {
    load_with_byte_offset(frag, tile_offset, 0);
  }

  /// Loads a fragment from memory with logical offset in units of whole tiles.
  MCTLASS_DEVICE
  void load(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a logical offset in units of whole tiles
      TensorCoord const &tile_offset,
      /// loads a tile with a logical offset AND a pointer offset
      Index pointer_offset) const {
    load_with_byte_offset(frag, tile_offset, pointer_offset * sizeof(Element));
  }

  /// Loads a fragment from memory with logical offset in units of whole tiles.
  MCTLASS_DEVICE
  void load_with_byte_offset(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a logical offset in units of whole tiles
      TensorCoord const &tile_offset,
      /// loads a tile with a logical offset AND a pointer offset
      Index byte_offset) const {
    Index pointer_offset = tile_offset.contiguous() *
                               InstructionShape::kContiguous /
                               Layout::kElementsPerAccess +
                           tile_offset.strided() * Shape::kStrided * stride_;

    byte_offset += sizeof_bits<AccessType>::value * pointer_offset / 8;

    load_with_byte_offset(frag, byte_offset);
  }

  /// Notify the iterator which k-group it is currently pointing to.
  ///
  /// This does not advance the iterator. Rather, it overrides its internal
  /// tracking with constant-valued k-group index to enable the compiler to
  /// fold constants and achieve more efficient code.
  ///
  /// This is used by some nontrivial permuted layouts.
  MCTLASS_DEVICE
  void set_kgroup_index(int k_group) {
    k_group_idx_ = k_group % (Policy::kGroupsPerTile / kPartitionsK);
  }
};

///
/// This special template for float is completely same with mctlass:tfloat32_t,
/// It's a very bad implement. We need to refactor them one day.
///
template <
    /// Size of the matrix to load (concept: PitchLinearShape)
    typename Shape_,
    /// Identifies A or B multiplicand
    Operand Operand_,
    /// Shape of one matrix product operation (concept: PitchLinearShape)
    typename InstructionShape_,
    /// Interval between adjacent *MMA instructions (in units of MMA
    /// instructions)
    int OpDelta_,
    /// Element number when the layout crosses (in units of elements)
    int Crosswise,
    /// Number of partitions along K dimension
    int PartitionsK_>
class MmaTensorOpMultiplicandTileIterator<
    Shape_, Operand_, float,
    mctlass::layout::TensorOpMultiplicandCrosswise<sizeof_bits<float>::value,
                                                   Crosswise>,
    InstructionShape_, OpDelta_, 32, PartitionsK_> {
 public:
  /// Shape of tile to load (concept: PitchLinearShape)
  using Shape = Shape_;

  /// Operand tag
  static Operand const kOperand = Operand_;

  static_assert(kOperand == Operand::kA || kOperand == Operand::kB,
                "MmaTensorOpMultiplicandIterator may only be instantiated for "
                "A or B operands to warp-level Mma.");

  /// Element type
  using Element =float;

  /// Element number when the layout crosses
  static int const kCrosswise = Crosswise;

  /// Layout of source tile
  using Layout = mctlass::layout::TensorOpMultiplicandCrosswise<
      sizeof_bits<Element>::value, kCrosswise>;

  /// Shape of one matrix product operation (concept: GemmShape)
  using InstructionShape = InstructionShape_;

  /// Delta between *MMA operations (in units of *MMA operations, concept:
  /// MatrixShape)
  static int const kOpDelta = OpDelta_;

  /// Number of participating threads
  static int const kThreads = 32;

  /// Number of partitions along K dimension
  static int const kPartitionsK = PartitionsK_;

  /// TensorRef type for loading element from a tensor
  using TensorRef = TensorRef<Element, Layout>;

  /// Index type
  using Index = typename TensorRef::Index;

  /// Long Index type
  using LongIndex = typename TensorRef::LongIndex;

  /// Long Index type
  using StrideIndex = typename TensorRef::Layout::Stride::Index;

  /// Coordinate for an element in the tensor
  using TensorCoord = typename TensorRef::TensorCoord;

  /// Internal structure of iterator - made public to enable introspection
  struct Policy {
    static_assert(
        !(Shape::kContiguous % InstructionShape::kContiguous),
        "Shape of warp-level Mma must be divisible by operator shape.");

    // Determine number of elements along outer dimension per individual LDSM op
    static int const kLdsmOpOuter = Layout::kElementsPerAccess;
    static int const kLdsmOpInner = 8;

    static_assert(!(Shape::kContiguous % kLdsmOpOuter),
                  "Shape of warp-level mma must be divisible by LDSM's "
                  "fundamental tile size.");

    static_assert(!(Shape::kStrided % kLdsmOpInner),
                  "Shape of warp-level mma must be divisible by LDSM's "
                  "fundamental tile size.");

    /// Shape of one individual LDSM instruction
    static int const LdsmShapeContiguous =
        InstructionShape::kContiguous / kLdsmOpOuter;
    static int const LdsmShapeStrided =
        ((4 / LdsmShapeContiguous * kLdsmOpInner) > Shape::kStrided)
            ? (Shape::kStrided / kLdsmOpInner)
            : (4 / LdsmShapeContiguous);
    using LdsmShape =
        layout::PitchLinearShape<LdsmShapeContiguous, LdsmShapeStrided>;

    /// Number and arrangement of LDSM instructions
    using LdsmIterations =
        layout::PitchLinearShape<1, Shape::kStrided / kLdsmOpInner /
                                        LdsmShape::kStrided>;

    ///
    static int const kGroupsPerTile = Layout::TileShape::kContiguous /
                                      Layout::kFactor / LdsmShape::kContiguous;
  };

 private:
  /// Not working on this feature at the moment.
  static_assert(kOpDelta == 1,
                "Alternative arrangements not supported at present.");

  /// Pointer type used for accesses
  using AccessType = Array<Element, Layout::kElementsPerAccess>;

 public:
  //
  // Derived quantities
  //

  /// Fragment object holding a thread's part of a tile
  using Fragment = Array<Element, Shape::kStrided *
                                      InstructionShape::kContiguous / kThreads>;

 private:

  /// Total number of sections.  The memory is divided into stages.  One stage
  /// can store one tile.  Stage is divided into sections.  Interleaved layout
  /// can have multiple sections in a stage.  The rest layout only has one section
  /// in a stage.
  int sections_;

  /// Layout object storing stride values
  StrideIndex stride_;

  /// Shared memory base pointers - not advanced
  AccessType const *pointer_;

  /// Byte offset incremented as iterator advances
  Index byte_offset_;

  /// Internal counter used to determine when to increment byte offset and when
  /// to XOR it
  int k_group_idx_;

 public:
  /// Default ctor constructs null iterator
  MCTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator()
      : pointer_(nullptr),
        sections_(0),
        stride_(0),
        byte_offset_(0),
        k_group_idx_(0) {}

  /// Constructor from TensorRef
  MCTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator(TensorRef const &ref, int lane_id)
      : pointer_(reinterpret_cast<AccessType const *>(ref.data())),
        sections_(ref.stride(0) / kCrosswise),
        // stride_ = kCrosswise x sections_ x kFactor
        stride_(ref.stride(0) * Layout::kFactor / Layout::kElementsPerAccess),
        byte_offset_(0),
        k_group_idx_(0) {
    // Warp level iterator at most use double buffer to hide latency.  If there
    // are more than 2 sections, every stage should have more than 1 section.

    // Turing silicon requires all 32 threads in a warp provide valid addresses
    // even for LDSM.1 and LDSM.2
//#if defined(__MACA_ARCH__)
#if 0
    lane_id = lane_id % (Policy::LdsmShape::kCount * Policy::kLdsmOpInner);
#endif

    int row, col;
    if (kOperand == Operand::kA) {
      row = lane_id & 0xf;
      col = ((lane_id >> 4) << 1) ^ 0x7;
    } else {
      row = lane_id & 0x7;
      col = ((lane_id >> 4) << 1) ^ 0x7;
    }

    TensorCoord coord = make_Coord(col, row);
    Layout lo(ref.stride(0));
    byte_offset_ =  lo(coord) * (sizeof_bits<Element>::value / 8);
  }

  /// Adds a pointer offset to internal pointer(s) to advance through memory
  MCTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator &add_pointer_offset(LongIndex offset) {
    byte_offset_ += offset * sizeof_bits<Element>::value / 8;

    return *this;
  }

  /// Advances an iterator along logical dimensions of matrix in units of whole
  /// tiles
  MCTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator &add_tile_offset(
      TensorCoord const &tile_offset) {
    int whole_tiles = tile_offset.contiguous() / Policy::kGroupsPerTile;
    int k_groups_delta = tile_offset.contiguous() % Policy::kGroupsPerTile;

    byte_offset_ ^= k_groups_delta * sizeof_bits<Element>::value *
                    Layout::kElementsPerAccess *
                    Policy::LdsmShape::kContiguous / 8;
    pointer_ +=
        tile_offset.strided() * stride_ * Shape::kStrided / Layout::kFactor +
        whole_tiles * stride_ / sections_;
    return *this;
  }

  /// Advances an iterator along logical dimensions of matrix in units of whole
  /// tiles
  MCTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator &add_tile_offset_negative(
      TensorCoord const &tile_offset) {

    int whole_tiles = tile_offset.contiguous() / Policy::kGroupsPerTile;
    int k_groups_delta = tile_offset.contiguous() % Policy::kGroupsPerTile;
    if (k_groups_delta < 0) {
        whole_tiles -= 1;
        k_groups_delta += Policy::kGroupsPerTile;
    }

    if ((Policy::kGroupsPerTile / kPartitionsK) >= 2) {
      byte_offset_ ^= (k_groups_delta & 1) * Policy::LdsmShape::kContiguous *
                        sizeof_bits<Element>::value *
                        Layout::kElementsPerAccess / 8;
    }
    if ((Policy::kGroupsPerTile / kPartitionsK) >= 4) {
      byte_offset_ ^= ((k_groups_delta + (k_group_idx_ & 1)) & 2) *
                        Policy::LdsmShape::kContiguous *
                        sizeof_bits<Element>::value *
                        Layout::kElementsPerAccess / 8;
    }
    if ((Policy::kGroupsPerTile / kPartitionsK) == 8) {
      byte_offset_ ^= ((k_groups_delta + (k_group_idx_ & 3)) & 4) *
                        Policy::LdsmShape::kContiguous *
                        sizeof_bits<Element>::value *
                        Layout::kElementsPerAccess / 8;
    }

    k_group_idx_ += k_groups_delta;
    whole_tiles += k_group_idx_ / (Policy::kGroupsPerTile / kPartitionsK);
    k_group_idx_ = k_group_idx_ % (Policy::kGroupsPerTile / kPartitionsK);

    pointer_ +=
        tile_offset.strided() * stride_ * Shape::kStrided / Layout::kFactor +
        whole_tiles * stride_ / sections_;
    return *this;
  }

  /// Advances the iterator along the advance dimension
  MCTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator &operator++() {

    if ((Policy::kGroupsPerTile / kPartitionsK) > 1) {
      int mask = ((Policy::kGroupsPerTile / kPartitionsK) == 8)
                     ? 3
                     : (((Policy::kGroupsPerTile / kPartitionsK) == 4) ? 1 : 0);

      if (((k_group_idx_ & mask) % 2) == 0)
        byte_offset_ ^= 1 * Policy::LdsmShape::kContiguous *
                        sizeof_bits<Element>::value *
                        Layout::kElementsPerAccess / 8;
      else if ((k_group_idx_ & mask) == 1)
        byte_offset_ ^= 3 * Policy::LdsmShape::kContiguous *
                        sizeof_bits<Element>::value *
                        Layout::kElementsPerAccess / 8;
      else if ((k_group_idx_ & mask) == 3)
        byte_offset_ ^= 7 * Policy::LdsmShape::kContiguous *
                        sizeof_bits<Element>::value *
                        Layout::kElementsPerAccess / 8;
    }

    k_group_idx_++;

    if (k_group_idx_ == (Policy::kGroupsPerTile / kPartitionsK)) {
      k_group_idx_ = 0;
      add_tile_offset({Policy::kGroupsPerTile, 0});
    }

    return *this;
  }

  /// Advances the iterator along the advance dimension
  MCTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator &operator--() { assert(0); }

  ///< advances in units of whole tiles along the logical coordinate space of
  ///< the tensor
  MCTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator &operator+=(
      TensorCoord const &tile_offset) {
    add_tile_offset(tile_offset);
    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of
  ///< the tensor
  MCTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator &operator-=(
      TensorCoord const &tile_offset) {
    add_tile_offset(-tile_offset);
    return *this;
  }

  /// Loads a fragment from memory at the location pointed to by the iterator.
  MCTLASS_HOST_DEVICE
  void load(Fragment &frag) const {
    load_with_byte_offset(frag, 0);

  }

  /// Loads a fragment from memory with additional logical offset
  MCTLASS_DEVICE
  void load_with_byte_offset(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a linear offset in units of bytes
      Index byte_offset) const {
    Array<unsigned, Policy::LdsmShape::kCount> *fetch_ptr =
        reinterpret_cast<Array<unsigned, Policy::LdsmShape::kCount> *>(&frag);
    int ldm = stride_ * Layout::kElementsPerAccess / Layout::kFactor;

    MCTLASS_PRAGMA_UNROLL
    for (int s = 0; s < Policy::LdsmIterations::kStrided; ++s) {
      MCTLASS_PRAGMA_UNROLL
      for (int c = 0; c < Policy::LdsmIterations::kContiguous; ++c) {
        int access_idx = c + s * Policy::LdsmIterations::kContiguous;

        AccessType const *source_ptr =
            pointer_ + Policy::LdsmShape::kContiguous * c +
            Policy::kLdsmOpInner / Layout::kFactor *
                Policy::LdsmShape::kStrided * s * stride_;

        char const *source_byte_ptr =
            reinterpret_cast<char const *>(source_ptr) + byte_offset +
            byte_offset_;
        if(kOperand == Operand::kA) {
          mctlass::arch::ldsmAtf32<layout::RowMajor, Policy::LdsmShape::kCount>(
              fetch_ptr[access_idx], source_byte_ptr);
        } else {
          mctlass::arch::ldsmBtf32<layout::RowMajor, Policy::LdsmShape::kCount>(
              fetch_ptr[access_idx], source_byte_ptr, ldm);
        }
      }
    }
  }

  /// Loads a fragment from memory with additional logical offset
  MCTLASS_DEVICE
  void load_with_pointer_offset(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a linear offset
      Index pointer_offset) const {
    load_with_byte_offset(frag, pointer_offset * sizeof(Element));
  }

  /// Loads a fragment from memory with logical offset in units of whole tiles.
  MCTLASS_DEVICE
  void load(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a logical offset in units of whole tiles
      TensorCoord const &tile_offset) const {
    load_with_byte_offset(frag, tile_offset, 0);
  }

  /// Loads a fragment from memory with logical offset in units of whole tiles.
  MCTLASS_DEVICE
  void load(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a logical offset in units of whole tiles
      TensorCoord const &tile_offset,
      /// loads a tile with a logical offset AND a pointer offset
      Index pointer_offset) const {
    load_with_byte_offset(frag, tile_offset, pointer_offset * sizeof(Element));
  }

  /// Loads a fragment from memory with logical offset in units of whole tiles.
  MCTLASS_DEVICE
  void load_with_byte_offset(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a logical offset in units of whole tiles
      TensorCoord const &tile_offset,
      /// loads a tile with a logical offset AND a pointer offset
      Index byte_offset) const {
    Index pointer_offset = tile_offset.contiguous() *
                               InstructionShape::kContiguous /
                               Layout::kElementsPerAccess +
                           tile_offset.strided() * Shape::kStrided * stride_;

    byte_offset += sizeof_bits<AccessType>::value * pointer_offset / 8;

    load_with_byte_offset(frag, byte_offset);
  }

  /// Notify the iterator which k-group it is currently pointing to.
  ///
  /// This does not advance the iterator. Rather, it overrides its internal
  /// tracking with constant-valued k-group index to enable the compiler to
  /// fold constants and achieve more efficient code.
  ///
  /// This is used by some nontrivial permuted layouts.
  MCTLASS_DEVICE
  void set_kgroup_index(int k_group) {
    k_group_idx_ = k_group % (Policy::kGroupsPerTile / kPartitionsK);
  }
};

////////////////////////////////////////////////////////////////////////////////

/// This tile iterator is specialized for 32-thread TensorOps. It uses LDSM to
/// load from shared memory and therefore must be initialized with a TensorRef
/// to shared memory.
///
/// Satisfies:
///   ReadableRandomAccessContiguousTileIteratorConcept
///
template <
    /// Size of the matrix to load (concept: PitchLinearShape)
    typename Shape_,
    /// Identifies A or B multiplicand
    Operand Operand_,
    /// Shape of one matrix product operation (concept: PitchLinearShape)
    typename InstructionShape_,
    /// Interval between adjacent *MMA instructions (in units of MMA
    /// instructions)
    int OpDelta_,
    /// Element number when the layout crosses (in units of elements)
    int Crosswise,
    /// Number of partitions along K dimension
    int PartitionsK_>
class MmaTensorOpMultiplicandTileIterator<
    Shape_, Operand_, signed char,
    mctlass::layout::TensorOpMultiplicandCrosswise<sizeof_bits<signed char>::value,
                                                   Crosswise>,
    InstructionShape_, OpDelta_, 32, PartitionsK_> {
 public:
  /// Shape of tile to load (concept: PitchLinearShape)
  using Shape = Shape_;

  /// Operand tag
  static Operand const kOperand = Operand_;

  static_assert(kOperand == Operand::kA || kOperand == Operand::kB,
                "MmaTensorOpMultiplicandIterator may only be instantiated for "
                "A or B operands to warp-level Mma.");

  /// Element type
  using Element = signed char;

  /// Element number when the layout crosses
  static int const kCrosswise = Crosswise;

  /// Layout of source tile
  using Layout = mctlass::layout::TensorOpMultiplicandCrosswise<
      sizeof_bits<Element>::value, kCrosswise>;

  /// Shape of one matrix product operation (concept: GemmShape)
  using InstructionShape = InstructionShape_;

  /// Delta between *MMA operations (in units of *MMA operations, concept:
  /// MatrixShape)
  static int const kOpDelta = OpDelta_;

  /// Number of participating threads
  static int const kThreads = 32;

  /// Number of partitions along K dimension
  static int const kPartitionsK = PartitionsK_;

  /// TensorRef type for loading element from a tensor
  using TensorRef = TensorRef<Element, Layout>;

  /// Index type
  using Index = typename TensorRef::Index;

  /// Long Index type
  using LongIndex = typename TensorRef::LongIndex;

  /// Long Index type
  using StrideIndex = typename TensorRef::Layout::Stride::Index;

  /// Coordinate for an element in the tensor
  using TensorCoord = typename TensorRef::TensorCoord;

  /// Internal structure of iterator - made public to enable introspection
  struct Policy {
    static_assert(
        !(Shape::kContiguous % InstructionShape::kContiguous),
        "Shape of warp-level Mma must be divisible by operator shape.");

    // Determine number of elements along outer dimension per individual LDSM op
    static int const kLdsmOpOuter = Layout::kElementsPerAccess;
    static int const kLdsmOpInner = 8;

    static_assert(!(Shape::kContiguous % kLdsmOpOuter),
                  "Shape of warp-level mma must be divisible by LDSM's "
                  "fundamental tile size.");

    static_assert(!(Shape::kStrided % kLdsmOpInner),
                  "Shape of warp-level mma must be divisible by LDSM's "
                  "fundamental tile size.");
    // Check supported InstructionShape
    static_assert(
    InstructionShape::kContiguous == 16||
    InstructionShape::kContiguous == 32,
    "Supported list of mma operator shape for int8_t multiplicands are: 16x8x16, 16x8x32");

    /// Shape of one individual LDSM instruction
    static int const LdsmShapeContiguous =
        InstructionShape::kContiguous / kLdsmOpOuter;
    static int const LdsmShapeStrided =
        ((4 / LdsmShapeContiguous * kLdsmOpInner) > Shape::kStrided)
            ? (Shape::kStrided / kLdsmOpInner)
            : (4 / LdsmShapeContiguous);

    using LdsmShape =
        layout::PitchLinearShape<LdsmShapeContiguous, LdsmShapeStrided>;

    /// Number and arrangement of LDSM instructions
    using LdsmIterations =
        layout::PitchLinearShape<1, Shape::kStrided / kLdsmOpInner /
                                        LdsmShape::kStrided>;

    ///
    static int const kGroupsPerTile = Layout::TileShape::kContiguous /
                                      Layout::kFactor / LdsmShape::kContiguous;
  };

 private:
  /// Not working on this feature at the moment.
  static_assert(kOpDelta == 1,
                "Alternative arrangements not supported at present.");

  /// Pointer type used for accesses
  using AccessType = Array<Element, Layout::kElementsPerAccess>;

 public:
  //
  // Derived quantities
  //

  /// Fragment object holding a thread's part of a tile
  using Fragment = Array<Element, Shape::kStrided *
                                      InstructionShape::kContiguous / kThreads>;

 private:

  /// Total number of sections.  The memory is divided into stages.  One stage
  /// can store one tile.  Stage is divided into sections.  Interleaved layout
  /// can have multiple sections in a stage.  The rest layout only has one section
  /// in a stage.
  int sections_;

  /// Layout object storing stride values
  StrideIndex stride_;

  /// Shared memory base pointers - not advanced
  AccessType const *pointer_;

  /// Byte offset incremented as iterator advances
  Index byte_offset_;

  int distance_matrix[3];

  /// Internal counter used to determine when to increment byte offset and when
  /// to XOR it
  int k_group_idx_;
  Layout layout_;

 public:
  /// Default ctor constructs null iterator
  MCTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator()
      : pointer_(nullptr),
        sections_(0),
        stride_(0),
        byte_offset_(0),
        k_group_idx_(0) {}

  /// Constructor from TensorRef
  MCTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator(TensorRef const &ref, int lane_id)
      : pointer_(reinterpret_cast<AccessType const *>(ref.data())),
        sections_(ref.stride(0) / kCrosswise),
        // stride_ = kCrosswise x sections_ x kFactor
        stride_(ref.stride(0) * Layout::kFactor / Layout::kElementsPerAccess),
        byte_offset_(0),
        k_group_idx_(0) {
    // Warp level iterator at most use double buffer to hide latency.  If there
    // are more than 2 sections, every stage should have more than 1 section.

    // Turing silicon requires all 32 threads in a warp provide valid addresses
    // even for LDSM.1 and LDSM.2
//#if defined(__MACA_ARCH__)
#if 0
   lane_id = lane_id % (Policy::LdsmShape::kCount * Policy::kLdsmOpInner);
#endif

    int quad_quad = (lane_id >> 4);
    int quad_pair = (lane_id >> 3);
    int lane_in_pair = (lane_id & 1);
    int lane_in_quad = (lane_id & 3);
    int lane_in_quad_pair = (lane_id & 7);
    int lane_in_quad_quad = (lane_id & 15);

    int partition_contiguous_idx = -1;
    int access_contiguous_idx = -1;
    int access_strided_idx = -1;

    if (Layout::kFactor == 4) {
      // Super Integer matrix multiply Interleaved-32

      int factor_in_partition =
          (Layout::PartitionShape::kContiguous * Layout::kFactor /
           Layout::TileShape::kContiguous);

      if (Policy::LdsmShape::kStrided == Policy::LdsmShape::kCount) {
        // Integer matrix multiply 8816  A/B
        partition_contiguous_idx = lane_in_quad / factor_in_partition;
        access_contiguous_idx = ((lane_in_pair * factor_in_partition) ^
                                 (lane_in_quad_quad / Layout::kFactor));
        access_strided_idx = lane_id / Layout::kFactor;
      }
      else if (Policy::LdsmShape::kStrided ==
                     (Policy::LdsmShape::kCount / 2) &&
                 kOperand == Operand::kA) {
        // Integer matrix multiply 16832 A
        partition_contiguous_idx = lane_in_quad / factor_in_partition;
        access_strided_idx = lane_in_quad_quad / Layout::kFactor;
        access_contiguous_idx =
            ((lane_in_pair * factor_in_partition + quad_quad) ^
             access_strided_idx);
      }
      else if (Policy::LdsmShape::kStrided ==
                     (Policy::LdsmShape::kCount / 2) &&
                 kOperand == Operand::kB) {
        // Integer matrix multiply 16832 B
        partition_contiguous_idx = lane_in_quad / factor_in_partition;
        access_strided_idx = lane_in_quad_pair / Layout::kFactor + quad_quad * 2;
        access_contiguous_idx =
            ((lane_in_pair * factor_in_partition + ((lane_id & 8) >> 3)) ^
             access_strided_idx);
      }
    } else if (Layout::kFactor == 2) {
      // Super Matrix multiply kBlock = 32
      if (Policy::LdsmShape::kStrided == Policy::LdsmShape::kCount) {
        // Matrix multiply 1688 A/B
        // (Q stands for 1 8x128bit block).
        // Q0
        // Q1
        // Q2
        // Q3
        // Four blocks are next to each other in the strided dimension.
        partition_contiguous_idx = (lane_id % Layout::kFactor);
        access_contiguous_idx = (lane_in_quad_pair / Layout::kFactor);
        access_strided_idx = lane_id / Layout::kFactor;
      }
      else if (Policy::LdsmShape::kStrided ==
                     (Policy::LdsmShape::kCount / 2) &&
                 kOperand == Operand::kA) {
        // Matrix multiply 16816|1688.TF32 A
        // Q0 Q2
        // Q1 Q3
        partition_contiguous_idx = (lane_id % Layout::kFactor);
        access_contiguous_idx =
            (quad_quad ^ (lane_in_quad_pair / Layout::kFactor));
        access_strided_idx = (lane_in_quad_quad / Layout::kFactor);
      } else if (Policy::LdsmShape::kStrided ==
                     (Policy::LdsmShape::kCount / 2) &&
                 kOperand == Operand::kB) {
        // Matrix multiply 16816|1688.TF32 B
        // Q0 Q1
        // Q2 Q3
        partition_contiguous_idx = (lane_id % Layout::kFactor);
        access_contiguous_idx =
            ((quad_pair & 1) ^ (lane_in_quad_pair / Layout::kFactor));
        access_strided_idx =
            (lane_in_quad_pair + (lane_id >> 4 << 3)) / Layout::kFactor;
      }
      else if (Policy::LdsmShape::kContiguous == Policy::LdsmShape::kCount) {
        // Matrix multiply 16832.SP B
        // Q0 Q1 Q2 Q3
        partition_contiguous_idx = (lane_id % Layout::kFactor);
        access_contiguous_idx =
            (quad_pair ^ (lane_in_quad_pair / Layout::kFactor));
        access_strided_idx = lane_in_quad_pair / Layout::kFactor;
      }
    } else if (Layout::kFactor == 1) {
      // Super Matrix multiply kBlock = 64
      if (Policy::LdsmShape::kStrided == Policy::LdsmShape::kCount) {
        // Q0
        // Q1
        // Q2
        // Q3
        partition_contiguous_idx = (lane_in_quad_pair >> 2);
        access_contiguous_idx = lane_in_quad;
        access_strided_idx = lane_id;
      }
      else if (Policy::LdsmShape::kStrided ==
                     (Policy::LdsmShape::kCount / 2) &&
                 kOperand == Operand::kA) {
        // Matrix multiply 16816|1688.TF32 A
        // Q0 Q2
        // Q1 Q3
        partition_contiguous_idx = (lane_in_quad_pair >> 2);
        access_contiguous_idx = (quad_quad ^ lane_in_quad);
        access_strided_idx = lane_in_quad_quad;
      } else if (Policy::LdsmShape::kStrided ==
                     (Policy::LdsmShape::kCount / 2) &&
                 kOperand == Operand::kB) {
        // Matrix multiply 16816|1688.TF32 B
        // Q0 Q1
        // Q2 Q3
        partition_contiguous_idx = (lane_in_quad_pair >> 2);
        access_contiguous_idx = ((quad_pair & 1) ^ lane_in_quad);
        access_strided_idx = lane_in_quad_pair + (lane_id >> 4 << 3);
      }
      else if (Policy::LdsmShape::kContiguous == Policy::LdsmShape::kCount) {
        // Matrix multiply 16832.SP B
        // Q0 Q1 Q2 Q3
        partition_contiguous_idx = (lane_in_quad_pair >> 2);
        access_contiguous_idx = (quad_pair ^ lane_in_quad);
        access_strided_idx = lane_in_quad_pair;
      }
    }

    int access_contiguous =
        partition_contiguous_idx * Layout::PartitionShape::kContiguous +
        access_contiguous_idx;

    int access_strided = access_strided_idx;

    int byte_offset_o = (access_contiguous + access_strided * stride_) *
                   sizeof_bits<Element>::value * Layout::kElementsPerAccess / 8;

    // For InstructionShape 16x8x16 and 16x8x32
    if (InstructionShape::kContiguous == 16) {
      int x, y, x1, y1, x2, y2, x3, y3;
      x = (((lane_id >> 4) << 2) ^ 0xf) - 3;
      //for matrix A InstructionShape::kStrided=16, for matrix B InstructionShape::kStrided=8
      y = lane_id & (InstructionShape::kStrided - 1);

      if (kOperand == Operand::kA) {
        x1 = x + 16;
        y1 = y;
        x2 = x;
        y2 = y + 16;
        x3 = x + 16;
        y3 = y + 16;
       }
       else {
         x1 = x;
         y1 = y + 8;
         x2 = x;
         y2 = y + 16;
         x3 = x;
         y3 = y + 24;
       }

       TensorCoord coord0 = make_Coord(x, y);
       Layout lo = ref.layout();
       byte_offset_ = lo(coord0) * (sizeof_bits<Element>::value / 8);

       TensorCoord coord1 = make_Coord(x1, y1);
       Index byte_offset_1 = lo(coord1) * (sizeof_bits<Element>::value / 8);
       TensorCoord coord2 = make_Coord(x2, y2);
       Index byte_offset_2 = lo(coord2) * (sizeof_bits<Element>::value / 8);
       TensorCoord coord3 = make_Coord(x3, y3);
       Index byte_offset_3 = lo(coord3) * (sizeof_bits<Element>::value / 8);

       distance_matrix[0] = (byte_offset_1 - byte_offset_) / 4;
       distance_matrix[1] = (byte_offset_2 - byte_offset_) / 4;
       distance_matrix[2] = (byte_offset_3 - byte_offset_) / 4;
    }
    else if (InstructionShape::kContiguous == 32) {
       layout_ = ref.layout();
       byte_offset_ = 0;
    }

  }

  /// Adds a pointer offset to internal pointer(s) to advance through memory
  MCTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator &add_pointer_offset(LongIndex offset) {
    byte_offset_ += offset * sizeof_bits<Element>::value / 8;

    return *this;
  }

  /// Advances an iterator along logical dimensions of matrix in units of whole
  /// tiles
  MCTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator &add_tile_offset(
      TensorCoord const &tile_offset) {
    int whole_tiles = tile_offset.contiguous() / Policy::kGroupsPerTile;
    int k_groups_delta = tile_offset.contiguous() % Policy::kGroupsPerTile;

    byte_offset_ ^= k_groups_delta * sizeof_bits<Element>::value *
                    Layout::kElementsPerAccess *
                    Policy::LdsmShape::kContiguous / 8;
    pointer_ +=
        tile_offset.strided() * stride_ * Shape::kStrided / Layout::kFactor +
        whole_tiles * stride_ / sections_;
    return *this;
  }

  /// Advances an iterator along logical dimensions of matrix in units of whole
  /// tiles
  MCTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator &add_tile_offset_negative(
      TensorCoord const &tile_offset) {

    int whole_tiles = tile_offset.contiguous() / Policy::kGroupsPerTile;
    int k_groups_delta = tile_offset.contiguous() % Policy::kGroupsPerTile;
    if (k_groups_delta < 0) {
        whole_tiles -= 1;
        k_groups_delta += Policy::kGroupsPerTile;
    }

    if ((Policy::kGroupsPerTile / kPartitionsK) >= 2) {
      byte_offset_ ^= (k_groups_delta & 1) * Policy::LdsmShape::kContiguous *
                        sizeof_bits<Element>::value *
                        Layout::kElementsPerAccess / 8;
    }
    if ((Policy::kGroupsPerTile / kPartitionsK) >= 4) {
      byte_offset_ ^= ((k_groups_delta + (k_group_idx_ & 1)) & 2) *
                        Policy::LdsmShape::kContiguous *
                        sizeof_bits<Element>::value *
                        Layout::kElementsPerAccess / 8;
    }
    if ((Policy::kGroupsPerTile / kPartitionsK) == 8) {
      byte_offset_ ^= ((k_groups_delta + (k_group_idx_ & 3)) & 4) *
                        Policy::LdsmShape::kContiguous *
                        sizeof_bits<Element>::value *
                        Layout::kElementsPerAccess / 8;
    }

    k_group_idx_ += k_groups_delta;
    whole_tiles += k_group_idx_ / (Policy::kGroupsPerTile / kPartitionsK);
    k_group_idx_ = k_group_idx_ % (Policy::kGroupsPerTile / kPartitionsK);

    pointer_ +=
        tile_offset.strided() * stride_ * Shape::kStrided / Layout::kFactor +
        whole_tiles * stride_ / sections_;
    return *this;
  }

  /// Advances the iterator along the advance dimension
  MCTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator &operator++() {

    // Integer matrix multiply 16832 Interleaved-32
    //   NONE
    // Integer matrix multiply 16816 Interleaved-32 || Integer matrix multiply 16816 kblock=32

    // Integer matrix multiply 8816  Interleaved-32
    //   ^1 ^1
    // Matrix multiply 1684.TF32 kblock=16 || Integer matrix multiply 16816 kblock=64
    // Matrix multiply 1688 kblock=32 || Integer matrix multiply 8816 kblock=64
    //   ^1 ^3 ^1 ^3
    // Matrix multiply 1688 kblock=64
    //   ^1 ^3 ^1 ^7 ^1 ^3 ^1 ^7

    // Matrix multiply 16816 kblock=32 | 1688.TF32 kblock=16 || Integer matrix multiply 16832 kblock=64
    //   ^2 ^2
    // Matrix multiply 16816 kblock=64 | 1688.TF32 kblock=32 || Integer matrix multiply 16832 kblock=128
    //   ^2 ^6 ^2 ^6
    if (InstructionShape::kContiguous == 32) {
      byte_offset_ += InstructionShape::kContiguous;
    }
    else {
      if ((Policy::kGroupsPerTile / kPartitionsK) > 1) {
        int mask = ((Policy::kGroupsPerTile / kPartitionsK) == 8)
                       ? 3
                       : (((Policy::kGroupsPerTile / kPartitionsK) == 4) ? 1 : 0);

        if (((k_group_idx_ & mask) % 2) == 0)
          byte_offset_ ^= 1 * Policy::LdsmShape::kContiguous *
                          sizeof_bits<Element>::value *
                          Layout::kElementsPerAccess / 8;
        else if ((k_group_idx_ & mask) == 1)
          byte_offset_ ^= 3 * Policy::LdsmShape::kContiguous *
                          sizeof_bits<Element>::value *
                          Layout::kElementsPerAccess / 8;
        else if ((k_group_idx_ & mask) == 3)
          byte_offset_ ^= 7 * Policy::LdsmShape::kContiguous *
                          sizeof_bits<Element>::value *
                          Layout::kElementsPerAccess / 8;
      }

      k_group_idx_++;

      if (k_group_idx_ == (Policy::kGroupsPerTile / kPartitionsK)) {
        k_group_idx_ = 0;
        add_tile_offset({Policy::kGroupsPerTile, 0});
      }
    }

    return *this;
  }

  /// Advances the iterator along the advance dimension
  MCTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator &operator--() { assert(0); }

  ///< advances in units of whole tiles along the logical coordinate space of
  ///< the tensor
  MCTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator &operator+=(
      TensorCoord const &tile_offset) {
    add_tile_offset(tile_offset);
    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of
  ///< the tensor
  MCTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator &operator-=(
      TensorCoord const &tile_offset) {
    add_tile_offset(-tile_offset);
    return *this;
  }

  /// Loads a fragment from memory at the location pointed to by the iterator.
  MCTLASS_HOST_DEVICE
  void load(Fragment &frag) const { load_with_byte_offset(frag, 0); }

  /// Loads a fragment from memory with additional logical offset
  MCTLASS_DEVICE
  void load_with_byte_offset(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a linear offset in units of bytes
      Index byte_offset) const {

    if (InstructionShape::kContiguous == 32) {

      Array<Element, Policy::LdsmShape::kCount> *fetch_element_ptr =
          reinterpret_cast<Array<Element, Policy::LdsmShape::kCount> *>(&frag);
      const int lane_id = __lane_id();
      MCTLASS_PRAGMA_UNROLL
      for (int s = 0; s < Policy::LdsmIterations::kStrided; ++s) {
        MCTLASS_PRAGMA_UNROLL
        for (int c = 0; c < Policy::LdsmIterations::kContiguous; ++c) {

          Element const *ptr = reinterpret_cast<Element const *>(pointer_) + byte_offset;
          Index row = (((lane_id >> 4) << 2) ^ 0xf) + byte_offset_;
          Index col = (lane_id & (InstructionShape::kStrided - 1)) + (s * 16);
          int access_idx = c + s * Policy::LdsmIterations::kContiguous * 4;

          for (int i = 0; i < Policy::LdsmShape::kCount; ++i) {

            Index source_idx = layout_({row - i, col});
            Index source_idx1 = layout_({row - i + 16, col});
            Index source_idx2 = layout_({row - i, col + 8});
            Index source_idx3 = layout_({row - i + 16, col + 8});
            Index dst_idx = Policy::LdsmShape::kCount - i - 1;

            fetch_element_ptr[access_idx][dst_idx] = ptr[source_idx];
            fetch_element_ptr[(access_idx + 1)][dst_idx] = ptr[source_idx1];
            fetch_element_ptr[(access_idx + 2)][dst_idx] = ptr[source_idx2];
            fetch_element_ptr[(access_idx + 3)][dst_idx] = ptr[source_idx3];
          }
        }
      }
    }
    else {

      Array<unsigned, Policy::LdsmShape::kCount> *fetch_ptr =
          reinterpret_cast<Array<unsigned, Policy::LdsmShape::kCount> *>(&frag);
      MCTLASS_PRAGMA_UNROLL
      for (int s = 0; s < Policy::LdsmIterations::kStrided; ++s) {
        MCTLASS_PRAGMA_UNROLL
        for (int c = 0; c < Policy::LdsmIterations::kContiguous; ++c) {
          int access_idx = c + s * Policy::LdsmIterations::kContiguous;
          AccessType const *source_ptr =
              pointer_ + Policy::LdsmShape::kContiguous * c +
              Policy::kLdsmOpInner / Layout::kFactor *
                  Policy::LdsmShape::kStrided * s * stride_;

          char const *source_byte_ptr =
              reinterpret_cast<char const *>(source_ptr) + byte_offset +
              byte_offset_;

          mctlass::arch::ldsmi8<layout::RowMajor, Policy::LdsmShape::kCount>(
              fetch_ptr[access_idx], source_byte_ptr, distance_matrix);
        }
      }

    }
  }

  /// Loads a fragment from memory with additional logical offset
  MCTLASS_DEVICE
  void load_with_pointer_offset(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a linear offset
      Index pointer_offset) const {
    load_with_byte_offset(frag, pointer_offset * sizeof(Element));
  }

  /// Loads a fragment from memory with logical offset in units of whole tiles.
  MCTLASS_DEVICE
  void load(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a logical offset in units of whole tiles
      TensorCoord const &tile_offset) const {
    load_with_byte_offset(frag, tile_offset, 0);
  }

  /// Loads a fragment from memory with logical offset in units of whole tiles.
  MCTLASS_DEVICE
  void load(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a logical offset in units of whole tiles
      TensorCoord const &tile_offset,
      /// loads a tile with a logical offset AND a pointer offset
      Index pointer_offset) const {
    load_with_byte_offset(frag, tile_offset, pointer_offset * sizeof(Element));
  }

  /// Loads a fragment from memory with logical offset in units of whole tiles.
  MCTLASS_DEVICE
  void load_with_byte_offset(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a logical offset in units of whole tiles
      TensorCoord const &tile_offset,
      /// loads a tile with a logical offset AND a pointer offset
      Index byte_offset) const {
    Index pointer_offset = tile_offset.contiguous() *
                               InstructionShape::kContiguous /
                               Layout::kElementsPerAccess +
                           tile_offset.strided() * Shape::kStrided * stride_;

    byte_offset += sizeof_bits<AccessType>::value * pointer_offset / 8;

    load_with_byte_offset(frag, byte_offset);
  }

  /// Notify the iterator which k-group it is currently pointing to.
  ///
  /// This does not advance the iterator. Rather, it overrides its internal
  /// tracking with constant-valued k-group index to enable the compiler to
  /// fold constants and achieve more efficient code.
  ///
  /// This is used by some nontrivial permuted layouts.
  MCTLASS_DEVICE
  void set_kgroup_index(int k_group) {
    k_group_idx_ = k_group % (Policy::kGroupsPerTile / kPartitionsK);
  }
};

////////////////////////////////////////////////////////////////////////////////

/// This tile iterator is specialized for 32-thread TensorOps. It uses LDSM to
/// load from shared memory and therefore must be initialized with a TensorRef
/// to shared memory.
///
/// Satisfies:
///   ReadableRandomAccessContiguousTileIteratorConcept
///
template <
    /// Size of the matrix to load (concept: MatrixShape)
    typename Shape_,
    /// Identifies A or B multiplicand
    Operand Operand_,
    /// Data type of elements
    typename Element_,
    /// Shape of one matrix product operation (concept: MatrixShape)
    typename InstructionShape_,
    /// Interval between adjacent *MMA instructions (in units of MMA
    /// instructions)
    int OpDelta_,
    /// Element number when the layout crosses (in units of elements)
    int Crosswise,
    /// Number of partitions along K dimension
    int PartitionsK_>
class MmaTensorOpMultiplicandTileIterator<
    Shape_, Operand_, Element_,
    mctlass::layout::ColumnMajorTensorOpMultiplicandCrosswise<
        sizeof_bits<Element_>::value, Crosswise>,
    InstructionShape_, OpDelta_, 32, PartitionsK_> {
 public:
  /// Shape of tile to load (concept: PitchLinearShape)
  using Shape = Shape_;

  /// Operand tag
  static Operand const kOperand = Operand_;

  static_assert(kOperand == Operand::kB,
                "MmaTensorOpMultiplicandIterator for ColumnMajor Crosswise may "
                "only be instantiated for B operand to warp-level Mma.");

  /// Element type
  using Element = Element_;

  /// KBlock size
  static int const kCrosswise = Crosswise;

  /// Layout of source tile
  using Layout = mctlass::layout::ColumnMajorTensorOpMultiplicandCrosswise<
      sizeof_bits<Element_>::value, kCrosswise>;

  /// Shape of one matrix product operation (concept: MatrixShape)
  using InstructionShape = InstructionShape_;

  /// Delta between *MMA operations (in units of *MMA operations, concept:
  /// MatrixShape)
  static int const kOpDelta = OpDelta_;

  /// Number of participating threads
  static int const kThreads = 32;

  /// TensorRef type for loading element from a tensor
  using TensorRef = TensorRef<Element, Layout>;

  /// Index type
  using Index = typename TensorRef::Index;

  /// Long Index type
  using LongIndex = typename TensorRef::LongIndex;

  /// Coordinate for an element in the tensor
  using TensorCoord = typename TensorRef::TensorCoord;

  /// Underlying tile iterator implementation
  using Base = MmaTensorOpMultiplicandTileIterator<
      layout::PitchLinearShape<Shape::kRow, Shape::kColumn>, kOperand, Element,
      layout::TensorOpMultiplicandCrosswise<sizeof_bits<Element_>::value,
                                            kCrosswise>,
      layout::PitchLinearShape<InstructionShape::kRow,
                               InstructionShape::kColumn>,
      kOpDelta, kThreads, PartitionsK_>;

 public:
  //
  // Derived quantities
  //

  /// Fragment object holding a thread's part of a tile
  using Fragment = typename Base::Fragment;

 private:
  /// Underlying tile iterator
  Base iterator_;

 public:
  /// Default ctor constructs null iterator
  MCTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator() {}

  /// Constructor from TensorRef
  MCTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator(TensorRef const &ref, int lane_id)
      : iterator_({ref.data(), ref.stride()}, lane_id) {}

  /// Adds a pointer offset to internal pointer(s) to advance through memory
  MCTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator &add_pointer_offset(LongIndex offset) {
    iterator_.add_pointer_offset(offset);

    return *this;
  }

  /// Advances an iterator along logical dimensions of matrix in units of whole
  /// tiles
  MCTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator &add_tile_offset(
      TensorCoord const &tile_offset) {
    iterator_.add_tile_offset({tile_offset.row(), tile_offset.column()});

    return *this;
  }

  /// Advances an iterator along logical dimensions of matrix in units of whole
  /// tiles
  MCTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator &add_tile_offset_negative(
      TensorCoord const &tile_offset) {
    iterator_.add_tile_offset_negative({tile_offset.row(), tile_offset.column()});

    return *this;
  }

  /// Advances the iterator along the advance dimension
  MCTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator &operator++() {
    ++iterator_;

    return *this;
  }

  /// Advances the iterator along the advance dimension
  MCTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator &operator--() {
    --iterator_;

    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of
  ///< the tensor
  MCTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator &operator+=(
      TensorCoord const &tile_offset) {
    add_tile_offset(PitchLinearCoord(tile_offset.row(), tile_offset.column()));
    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of
  ///< the tensor
  MCTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator &operator-=(
      TensorCoord const &tile_offset) {
    add_tile_offset(-PitchLinearCoord(tile_offset.row(), tile_offset.column()));
    return *this;
  }

  /// Loads a fragment from memory at the location pointed to by the iterator.
  MCTLASS_HOST_DEVICE
  void load(Fragment &frag) const { iterator_.load(frag); }

  /// Loads a fragment from memory with additional logical offset
  MCTLASS_DEVICE
  void load_with_pointer_offset(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a linear offset
      Index pointer_offset) const {
    iterator_.load_with_pointer_offset(frag, pointer_offset);
  }

  /// Loads a fragment from memory with additional logical offset
  MCTLASS_DEVICE
  void load_with_byte_offset(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a linear offset
      Index byte_offset) const {
    iterator_.load_with_byte_offset(frag, byte_offset);
  }

  /// Loads a fragment from memory with logical offset in units of whole tiles.
  MCTLASS_DEVICE
  void load(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a logical offset in units of whole tiles
      TensorCoord const &tile_offset) const {
    // TODO
    assert(0);
  }

  /// Loads a fragment from memory with logical offset in units of whole tiles.
  MCTLASS_DEVICE
  void load(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a logical offset in units of whole tiles
      TensorCoord const &tile_offset,
      /// loads a tile with a logical offset AND a pointer offset
      Index pointer_offset) const {
    // TODO
    assert(0);
  }

  /// Loads a fragment from memory with logical offset in units of whole tiles.
  MCTLASS_DEVICE
  void load_with_byte_offset(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a logical offset in units of whole tiles
      TensorCoord const &tile_offset,
      /// loads a tile with a logical offset AND a pointer offset
      Index byte_offset) const {
    iterator_.load_with_byte_offset(
        frag, {tile_offset.contiguous(), tile_offset.strided()}, byte_offset);
  }

  /// Notify the iterator which k-group it is currently pointing to.
  ///
  /// This does not advance the iterator. Rather, it overrides its internal
  /// tracking with constant-valued k-group index to enable the compiler to
  /// fold constants and achieve more efficient code.
  ///
  /// This is used by some nontrivial permuted layouts.
  MCTLASS_DEVICE
  void set_kgroup_index(int k_group) {
    iterator_.set_kgroup_index(k_group);
  }
};

////////////////////////////////////////////////////////////////////////////////

/// This tile iterator is specialized for 32-thread TensorOps. It uses LDSM to
/// load from shared memory and therefore must be initialized with a TensorRef
/// to shared memory.
///
/// Satisfies:
///   ReadableRandomAccessContiguousTileIteratorConcept
///
template <
    /// Size of the matrix to load (concept: MatrixShape)
    typename Shape_,
    /// Identifies A or B multiplicand
    Operand Operand_,
    /// Data type of elements
    typename Element_,
    /// Shape of one matrix product operation (concept: MatrixShape)
    typename InstructionShape_,
    /// Interval between adjacent *MMA instructions (in units of MMA
    /// instructions)
    int OpDelta_,
    /// Element number when the layout crosses (in units of elements)
    int Crosswise,
    /// Number of partitions along K dimension
    int PartitionsK_>
class MmaTensorOpMultiplicandTileIterator<
    Shape_, Operand_, Element_,
    mctlass::layout::RowMajorTensorOpMultiplicandCrosswise<
        sizeof_bits<Element_>::value, Crosswise>,
    InstructionShape_, OpDelta_, 32, PartitionsK_> {
 public:
  /// Shape of tile to load (concept: PitchLinearShape)
  using Shape = Shape_;

  /// Operand tag
  static Operand const kOperand = Operand_;

  static_assert(kOperand == Operand::kA,
                "MmaTensorOpMultiplicandIterator for RowMajor Crosswise may "
                "only be instantiated for A operand to warp-level Mma.");

  /// Element type
  using Element = Element_;

  /// Element number when the layout crosses
  static int const kCrosswise = Crosswise;

  /// Layout of source tile
  using Layout = mctlass::layout::RowMajorTensorOpMultiplicandCrosswise<
      sizeof_bits<Element_>::value, kCrosswise>;

  /// Shape of one matrix product operation (concept: MatrixShape)
  using InstructionShape = InstructionShape_;

  /// Delta between *MMA operations (in units of *MMA operations, concept:
  /// MatrixShape)
  static int const kOpDelta = OpDelta_;

  /// Number of participating threads
  static int const kThreads = 32;

  /// TensorRef type for loading element from a tensor
  using TensorRef = TensorRef<Element, Layout>;

  /// Index type
  using Index = typename TensorRef::Index;

  /// Long Index type
  using LongIndex = typename TensorRef::LongIndex;

  /// Coordinate for an element in the tensor
  using TensorCoord = typename TensorRef::TensorCoord;

  /// Underlying tile iterator implementation
  using Base = MmaTensorOpMultiplicandTileIterator<
      layout::PitchLinearShape<Shape::kColumn, Shape::kRow>, kOperand, Element,
      layout::TensorOpMultiplicandCrosswise<sizeof_bits<Element_>::value,
                                            kCrosswise>,
      layout::PitchLinearShape<InstructionShape::kColumn,
                               InstructionShape::kRow>,
      kOpDelta, kThreads, PartitionsK_>;

 public:
  //
  // Derived quantities
  //

  /// Fragment object holding a thread's part of a tile
  using Fragment = typename Base::Fragment;

 private:
  /// Underlying tile iterator
  Base iterator_;

 public:
  /// Default ctor constructs null iterator
  MCTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator() {}

  /// Constructor from TensorRef
  MCTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator(TensorRef const &ref, int lane_id)
      : iterator_({ref.data(), ref.stride()}, lane_id) {}

  /// Adds a pointer offset to internal pointer(s) to advance through memory
  MCTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator &add_pointer_offset(LongIndex offset) {
    iterator_.add_pointer_offset(offset);

    return *this;
  }

  /// Advances an iterator along logical dimensions of matrix in units of whole
  /// tiles
  MCTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator &add_tile_offset(
      TensorCoord const &tile_offset) {
    iterator_.add_tile_offset({tile_offset.column(), tile_offset.row()});

    return *this;
  }

  /// Advances an iterator along logical dimensions of matrix in units of whole
  /// tiles
  MCTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator &add_tile_offset_negative(
      TensorCoord const &tile_offset) {
    iterator_.add_tile_offset_negative({tile_offset.column(), tile_offset.row()});

    return *this;
  }

  /// Advances the iterator along the advance dimension
  MCTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator &operator++() {
    ++iterator_;

    return *this;
  }

  /// Advances the iterator along the advance dimension
  MCTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator &operator--() {
    --iterator_;

    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of
  ///< the tensor
  MCTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator &operator+=(
      TensorCoord const &tile_offset) {
    add_tile_offset(PitchLinearCoord(tile_offset.column(), tile_offset.row()));
    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of
  ///< the tensor
  MCTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator &operator-=(
      TensorCoord const &tile_offset) {
    add_tile_offset(-PitchLinearCoord(tile_offset.column(), tile_offset.row()));
    return *this;
  }

  /// Loads a fragment from memory at the location pointed to by the iterator.
  MCTLASS_HOST_DEVICE
  void load(Fragment &frag) const { iterator_.load(frag); }

  /// Loads a fragment from memory with additional logical offset
  MCTLASS_DEVICE
  void load_with_pointer_offset(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a linear offset
      Index pointer_offset) const {
    iterator_.load_with_pointer_offset(frag, pointer_offset);
  }

  /// Loads a fragment from memory with additional logical offset
  MCTLASS_DEVICE
  void load_with_byte_offset(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a linear offset
      Index byte_offset) const {
    iterator_.load_with_byte_offset(frag, byte_offset);
  }

  /// Loads a fragment from memory with logical offset in units of whole tiles.
  MCTLASS_DEVICE
  void load(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a logical offset in units of whole tiles
      TensorCoord const &tile_offset) const {
    // TODO
    assert(0);
  }

  /// Loads a fragment from memory with logical offset in units of whole tiles.
  MCTLASS_DEVICE
  void load(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a logical offset in units of whole tiles
      TensorCoord const &tile_offset,
      /// loads a tile with a logical offset AND a pointer offset
      Index pointer_offset) const {
    // TODO
    assert(0);
  }

  /// Loads a fragment from memory with logical offset in units of whole tiles.
  MCTLASS_DEVICE
  void load_with_byte_offset(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a logical offset in units of whole tiles
      TensorCoord const &tile_offset,
      /// loads a tile with a logical offset AND a pointer offset
      Index byte_offset) const {
    iterator_.load_with_byte_offset(
        frag, {tile_offset.strided(), tile_offset.contiguous()}, byte_offset);
  }

  /// Notify the iterator which k-group it is currently pointing to.
  ///
  /// This does not advance the iterator. Rather, it overrides its internal
  /// tracking with constant-valued k-group index to enable the compiler to
  /// fold constants and achieve more efficient code.
  ///
  /// This is used by some nontrivial permuted layouts.
  MCTLASS_DEVICE
  void set_kgroup_index(int k_group) {
    iterator_.set_kgroup_index(k_group);
  }
};

////////////////////////////////////////////////////////////////////////////////

template <
    /// Size of the matrix to load (concept: MatrixShape)
    typename Shape_,
    /// Element type
    typename Element_,
    /// Layout of operand in memory
    typename Layout_,
    /// Shape of one matrix product operation (concept: MatrixShape)
    typename InstructionShape_,
    /// Interval between adjacent *MMA instructions (in units of MMA
    /// instructions, concept: MatrixShape)
    typename OpDelta_>
class MmaTensorOpAccumulatorTileIterator;

////////////////////////////////////////////////////////////////////////////////

/// This tile iterator is specialized for 32-thread TensorOps. It is used to load or store
/// accumulators from memory and is agnostic to layout. It could be faster if it assumed row-major
/// accumulator layout.
///
/// Satisfies:
///   ReadableRandomAccessContiguousTileIteratorConcept |
///   WriteableRandomAccessContiguousTileIteratorConcept
///
template <
    /// Size of the matrix to load (concept: MatrixShape)
    typename Shape_,
    /// Element type
    typename Element_,
    /// Shape of one matrix product operation (concept: MatrixShape)
    typename InstructionShape_,
    /// Interval between adjacent *MMA instructions (in units of MMA
    /// instructions, concept: MatrixShape)
    typename OpDelta_>
class MmaTensorOpAccumulatorTileIterator<
    Shape_, Element_, mctlass::layout::RowMajor, InstructionShape_, OpDelta_> {
 public:

  /// Shape of tile to load (concept: MatrixShape)
  using Shape = Shape_;

  /// Operand tag
  static Operand const kOperand = Operand::kC;

  /// Element type
  using Element = Element_;

  /// Layout of source tile
  using Layout = mctlass::layout::RowMajor;

  /// Shape of one matrix product operation (concept: MatrixShape)
  using InstructionShape = InstructionShape_;

  /// Delta between *MMA operations (in units of *MMA operations, concept: MatrixShape)
  using OpDelta = OpDelta_;

  /// Number of participating threads
  static int const kThreads = 32;

  /// TensorRef type for loading element from a tensor
  using TensorRef = TensorRef<Element, Layout>;

  /// Index type
  using Index = typename TensorRef::Index;

  /// Long Index type
  using LongIndex = typename TensorRef::LongIndex;

  /// Coordinate for an element in the tensor
  using TensorCoord = typename TensorRef::TensorCoord;

  /// Internal structure of iterator - made public to enable introspection
  struct Policy {
    static bool const kDivisible =
        !(Shape::kRow % InstructionShape::kM) &&
            !(Shape::kColumn % InstructionShape::kN);

    static_assert(platform::is_same<TensorCoord, MatrixCoord>::value,
      "Layouts must be defined for logical MatrixCoord coordinate space.");

    /// Number of mma operations performed
    using MmaIterations = MatrixShape<
      (Shape::kRow + InstructionShape::kM - 1) / InstructionShape::kM,
      (Shape::kColumn + InstructionShape::kN - 1) / InstructionShape::kN
    >;
  };

private:

  // Assume accumulator tile is an arrangement of 8-by-8 tiles replicated over the entire
  // shape, with each quad mapped to one row and each thread mapped to 1/4 of the elements
  // of that row. The accumulators within one row are assumed to be consecutive.
 static int const kElementsPerAccess = InstructionShape::kN / 4;
 static int const kRowsPerTile = 8;
 static int const kAccumulatorRows = InstructionShape::kM / kRowsPerTile;

public:

  //
  // Derived quantities
  //

  /// Fragment object holding a thread's part of a tile
  using Fragment = Array<
    Element,
    Policy::MmaIterations::kCount * InstructionShape::kMN / kThreads>;

private:

  /// Reference to output tensor
  TensorRef ref_;

public:

  /// Default ctor constructs null iterator
  MCTLASS_HOST_DEVICE
  MmaTensorOpAccumulatorTileIterator() { }

  /// Constructor from TensorRef
  MCTLASS_HOST_DEVICE
  MmaTensorOpAccumulatorTileIterator(
    TensorRef const &ref,
    int lane_id
  ):
    ref_(ref) {

    // int quad = (lane_id >> 2);
    // int lane_in_quad = (lane_id & 3);
    // MatrixCoord lane_offset(quad, lane_in_quad * kElementsPerAccess);
    int lane_in_quad = lane_id & 0x7;
    int quad = (lane_id >> 4) & 0x3;

    MatrixCoord lane_offset(quad, lane_in_quad);

    ref_.add_coord_offset(lane_offset);
  }

  /// Adds a pointer offset to internal pointer(s) to advance through memory
  MCTLASS_HOST_DEVICE
  MmaTensorOpAccumulatorTileIterator &add_pointer_offset(LongIndex offset) {
    ref_.add_pointer_offset(offset);
    return *this;
  }

  /// Advances an iterator along logical dimensions of matrix in units of whole tiles
  MCTLASS_HOST_DEVICE
  MmaTensorOpAccumulatorTileIterator &add_tile_offset(TensorCoord const &tile_offset) {

    ref_.add_coord_offset(tile_offset * make_Coord(Shape::kRow, Shape::kColumn));

    return *this;
  }

  /// Advances the iterator along the advance dimension
  MCTLASS_HOST_DEVICE
  MmaTensorOpAccumulatorTileIterator & operator++() {
    // deliberate no-op
    return *this;
  }

  /// Advances the iterator along the advance dimension
  MCTLASS_HOST_DEVICE
  MmaTensorOpAccumulatorTileIterator & operator--() {
    // deliberate no-op
    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of the tensor
  MCTLASS_DEVICE
  MmaTensorOpAccumulatorTileIterator & operator+=(TensorCoord const &tile_offset) {
    add_tile_offset(tile_offset);
    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of the tensor
  MCTLASS_DEVICE
  MmaTensorOpAccumulatorTileIterator & operator-=(TensorCoord const &tile_offset) {
    add_tile_offset(-tile_offset);
    return *this;
  }

  /// Loads a fragment from memory at the location pointed to by the iterator.
  MCTLASS_HOST_DEVICE
  void load(Fragment &frag) const {
    load_with_pointer_offset(frag, 0);
  }

  /// Loads a fragment from memory with additional logical offset
  MCTLASS_DEVICE
  void load_with_pointer_offset(
    Fragment &frag,                             ///< fragment to load from the tensor
    Index pointer_offset) const {               ///< loads a tile with a linear offset

    TensorRef offset_ref(ref_);
    offset_ref.add_pointer_offset(pointer_offset);

    MCTLASS_PRAGMA_UNROLL
    for (int mma_n = 0; mma_n < Policy::MmaIterations::kColumn; ++mma_n) {
      MCTLASS_PRAGMA_UNROLL
      for (int mma_m = 0; mma_m < Policy::MmaIterations::kRow; ++mma_m) {

        int mma_accum_start = kAccumulatorRows * kElementsPerAccess *
          (mma_n * Policy::MmaIterations::kRow + mma_m);

        MCTLASS_PRAGMA_UNROLL
        for (int row = 0; row < kAccumulatorRows; ++row) {
          MCTLASS_PRAGMA_UNROLL
          for (int col = 0; col < kElementsPerAccess; ++col) {
            int accum_m = mma_m * InstructionShape::kM * OpDelta::kRow +
                          row * kRowsPerTile;
            int accum_n = mma_n * InstructionShape::kN * OpDelta::kColumn + col;

            frag[mma_accum_start + row * kElementsPerAccess + col] = offset_ref.at({accum_m, accum_n});
          }
        }
      }
    }
  }

  /// Loads a fragment from memory with additional logical offset
  MCTLASS_DEVICE
  void load_with_byte_offset(
    Fragment &frag,                             ///< fragment to load from the tensor
    Index byte_offset) const {                  ///< loads a tile with a linear offset

    load_with_pointer_offset(byte_offset / sizeof(Element));
  }

  /// Loads a fragment from memory with logical offset in units of whole tiles.
  MCTLASS_DEVICE
  void load(
    Fragment &frag,                             ///< fragment to load from the tensor
    TensorCoord const &tile_offset) const {     ///< loads a tile with a logical offset in units of whole tiles

    load(frag, tile_offset, 0);
  }

  /// Loads a fragment from memory with logical offset in units of whole tiles.
  MCTLASS_DEVICE
  void load(
    Fragment &frag,                             ///< fragment to load from the tensor
    TensorCoord const &tile_offset,             ///< loads a tile with a logical offset in units of whole tiles
    Index pointer_offset) const {               ///< loads a tile with a logical offset AND a pointer offset

    load_with_pointer_offset(frag, ref_.offset(tile_offset) + pointer_offset);
  }

  /// Stores a fragment to memory
  MCTLASS_HOST_DEVICE
  void store(Fragment const &frag) const {
    store_with_pointer_offset(frag, 0);
  }

  /// Stores a fragment to memory with additional pointer offset
  MCTLASS_DEVICE
  void store_with_pointer_offset(
    Fragment const &frag,                       ///< fragment to store from the tensor
    Index pointer_offset) const {               ///< store a tile with a linear offset

    TensorRef offset_ref(ref_);
    offset_ref.add_pointer_offset(pointer_offset);

    MCTLASS_PRAGMA_UNROLL
    for (int mma_n = 0; mma_n < Policy::MmaIterations::kColumn; ++mma_n) {
      MCTLASS_PRAGMA_UNROLL
      for (int mma_m = 0; mma_m < Policy::MmaIterations::kRow; ++mma_m) {

        int mma_accum_start = kAccumulatorRows * kElementsPerAccess *
          (mma_n * Policy::MmaIterations::kRow + mma_m);

        MCTLASS_PRAGMA_UNROLL
        for (int row = 0; row < kAccumulatorRows; ++row) {
          MCTLASS_PRAGMA_UNROLL
          for (int col = 0; col < kElementsPerAccess; ++col) {
            // int accum_m = mma_m * InstructionShape::kM * OpDelta::kRow +
            //               row * kRowsPerTile;
            // int accum_n = mma_n * InstructionShape::kN * OpDelta::kColumn + col;
            int accum_m = mma_m * InstructionShape::kM * OpDelta::kRow + row * kRowsPerTile +
                          col * (InstructionShape::kM / kElementsPerAccess);
            int accum_n = mma_n * InstructionShape::kN * OpDelta::kColumn;
            int idx = mma_accum_start + row * kElementsPerAccess + col;

            offset_ref.at({accum_m, accum_n}) = frag[idx];
          }
        }
      }
    }
  }

  /// Stores a fragment to memory with additional pointer offset
  MCTLASS_DEVICE
  void store_with_byte_offset(
    Fragment const &frag,                       ///< fragment to store from the tensor
    Index byte_offset) const {                  ///< store a tile with a linear offset

    store_with_pointer_offset(byte_offset / sizeof(Element));
  }

  /// Stores a fragment to memory with logical offset in units of whole tiles.
  MCTLASS_DEVICE
  void store(
    Fragment &frag,                             ///< fragment to store to the tensor
    TensorCoord const &tile_offset) const {     ///< stores a tile with a logical offset in units of whole tiles

    store(frag, tile_offset, 0);
  }

  /// Stores a fragment from memory with logical offset in units of whole tiles.
  MCTLASS_DEVICE
  void store(
      /// fragment to store to the tensor
      Fragment const &frag,
      /// stores a tile with a logical offset in units of whole tiles
      TensorCoord const &tile_offset,
      /// stores a tile with a logical offset AND a pointer offset
      Index pointer_offset) const {
    store_with_pointer_offset(frag, ref_.offset(tile_offset) + pointer_offset);
  }
};

////////////////////////////////////////////////////////////////////////////////

template <
    /// Size of the matrix to load (concept: MatrixShape)
    typename Shape_,
    /// Shape of one matrix product operation (concept: MatrixShape)
    typename InstructionShape_,
    /// Interval between adjacent *MMA instructions (in units of MMA
    /// instructions, concept: MatrixShape)
    typename OpDelta_>
class MmaTensorOpAccumulatorTileIterator<
    Shape_, int, mctlass::layout::RowMajor, InstructionShape_, OpDelta_> {
 public:

  /// Shape of tile to load (concept: MatrixShape)
  using Shape = Shape_;

  /// Operand tag
  static Operand const kOperand = Operand::kC;

  /// Element type
  using Element = int;

  /// Layout of source tile
  using Layout = mctlass::layout::RowMajor;

  /// Shape of one matrix product operation (concept: MatrixShape)
  using InstructionShape = InstructionShape_;

  /// Delta between *MMA operations (in units of *MMA operations, concept: MatrixShape)
  using OpDelta = OpDelta_;

  /// Number of participating threads
  static int const kThreads = 32;

  /// TensorRef type for loading element from a tensor
  using TensorRef = TensorRef<Element, Layout>;

  /// Index type
  using Index = typename TensorRef::Index;

  /// Long Index type
  using LongIndex = typename TensorRef::LongIndex;

  /// Coordinate for an element in the tensor
  using TensorCoord = typename TensorRef::TensorCoord;

  /// Internal structure of iterator - made public to enable introspection
  struct Policy {
    static bool const kDivisible =
        !(Shape::kRow % InstructionShape::kM) &&
            !(Shape::kColumn % InstructionShape::kN);

    static_assert(platform::is_same<TensorCoord, MatrixCoord>::value,
      "Layouts must be defined for logical MatrixCoord coordinate space.");

    /// Number of mma operations performed
    using MmaIterations = MatrixShape<
      (Shape::kRow + InstructionShape::kM - 1) / InstructionShape::kM,
      (Shape::kColumn + InstructionShape::kN - 1) / InstructionShape::kN
    >;
  };

private:

  // Assume accumulator tile is an arrangement of 8-by-8 tiles replicated over the entire
  // shape, with each quad mapped to one row and each thread mapped to 1/4 of the elements
  // of that row. The accumulators within one row are assumed to be consecutive.
 static int const kElementsPerAccess = InstructionShape::kN / 4;
 static int const kRowsPerTile = 8;
 static int const kAccumulatorRows = InstructionShape::kM / kRowsPerTile;

public:

  //
  // Derived quantities
  //

  /// Fragment object holding a thread's part of a tile
  using Fragment = Array<
    Element,
    Policy::MmaIterations::kCount * InstructionShape::kMN / kThreads>;

private:

  /// Reference to output tensor
  TensorRef ref_;

public:

  /// Default ctor constructs null iterator
  MCTLASS_HOST_DEVICE
  MmaTensorOpAccumulatorTileIterator() { }

  /// Constructor from TensorRef
  MCTLASS_HOST_DEVICE
  MmaTensorOpAccumulatorTileIterator(
    TensorRef const &ref,
    int lane_id
  ):
    ref_(ref) {

    // int quad = (lane_id >> 2);
    // int lane_in_quad = (lane_id & 3);
    // MatrixCoord lane_offset(quad, lane_in_quad * kElementsPerAccess);
    int quad, lane_in_quad;

    quad = ((lane_id >> 4) << 2);
    lane_in_quad = lane_id & 0x7;

    MatrixCoord lane_offset(quad, lane_in_quad);

    ref_.add_coord_offset(lane_offset);
  }

  /// Adds a pointer offset to internal pointer(s) to advance through memory
  MCTLASS_HOST_DEVICE
  MmaTensorOpAccumulatorTileIterator &add_pointer_offset(LongIndex offset) {
    ref_.add_pointer_offset(offset);
    return *this;
  }

  /// Advances an iterator along logical dimensions of matrix in units of whole tiles
  MCTLASS_HOST_DEVICE
  MmaTensorOpAccumulatorTileIterator &add_tile_offset(TensorCoord const &tile_offset) {

    ref_.add_coord_offset(tile_offset * make_Coord(Shape::kRow, Shape::kColumn));

    return *this;
  }

  /// Advances the iterator along the advance dimension
  MCTLASS_HOST_DEVICE
  MmaTensorOpAccumulatorTileIterator & operator++() {
    // deliberate no-op
    return *this;
  }

  /// Advances the iterator along the advance dimension
  MCTLASS_HOST_DEVICE
  MmaTensorOpAccumulatorTileIterator & operator--() {
    // deliberate no-op
    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of the tensor
  MCTLASS_DEVICE
  MmaTensorOpAccumulatorTileIterator & operator+=(TensorCoord const &tile_offset) {
    add_tile_offset(tile_offset);
    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of the tensor
  MCTLASS_DEVICE
  MmaTensorOpAccumulatorTileIterator & operator-=(TensorCoord const &tile_offset) {
   // add_tile_offset(-tile_offset);
    return *this;
  }

  /// Loads a fragment from memory at the location pointed to by the iterator.
  MCTLASS_HOST_DEVICE
  void load(Fragment &frag) const {
    load_with_pointer_offset(frag, 0);
  }

  /// Loads a fragment from memory with additional logical offset
  MCTLASS_DEVICE
  void load_with_pointer_offset(
    Fragment &frag,                             ///< fragment to load from the tensor
    Index pointer_offset) const {               ///< loads a tile with a linear offset

    TensorRef offset_ref(ref_);
    offset_ref.add_pointer_offset(pointer_offset);

    MCTLASS_PRAGMA_UNROLL
    for (int mma_n = 0; mma_n < Policy::MmaIterations::kColumn; ++mma_n) {
      MCTLASS_PRAGMA_UNROLL
      for (int mma_m = 0; mma_m < Policy::MmaIterations::kRow; ++mma_m) {

        int mma_accum_start = kAccumulatorRows * kElementsPerAccess *
          (mma_n * Policy::MmaIterations::kRow + mma_m);

        MCTLASS_PRAGMA_UNROLL
        for (int row = 0; row < kAccumulatorRows; ++row) {
          MCTLASS_PRAGMA_UNROLL
          for (int col = 0; col < kElementsPerAccess; ++col) {
            int accum_m = mma_m * InstructionShape::kM * OpDelta::kRow +
                          row * kRowsPerTile;
            int accum_n = mma_n * InstructionShape::kN * OpDelta::kColumn + col;

            frag[mma_accum_start + row * kElementsPerAccess + col] = offset_ref.at({accum_m, accum_n});
          }
        }
      }
    }
  }

  /// Loads a fragment from memory with additional logical offset
  MCTLASS_DEVICE
  void load_with_byte_offset(
    Fragment &frag,                             ///< fragment to load from the tensor
    Index byte_offset) const {                  ///< loads a tile with a linear offset

    load_with_pointer_offset(byte_offset / sizeof(Element));
  }

  /// Loads a fragment from memory with logical offset in units of whole tiles.
  MCTLASS_DEVICE
  void load(
    Fragment &frag,                             ///< fragment to load from the tensor
    TensorCoord const &tile_offset) const {     ///< loads a tile with a logical offset in units of whole tiles

    load(frag, tile_offset, 0);
  }

  /// Loads a fragment from memory with logical offset in units of whole tiles.
  MCTLASS_DEVICE
  void load(
    Fragment &frag,                             ///< fragment to load from the tensor
    TensorCoord const &tile_offset,             ///< loads a tile with a logical offset in units of whole tiles
    Index pointer_offset) const {               ///< loads a tile with a logical offset AND a pointer offset

    load_with_pointer_offset(frag, ref_.offset(tile_offset) + pointer_offset);
  }

  /// Stores a fragment to memory
  MCTLASS_HOST_DEVICE
  void store(Fragment const &frag) const {
    store_with_pointer_offset(frag, 0);
  }

  /// Stores a fragment to memory with additional pointer offset
  MCTLASS_DEVICE
  void store_with_pointer_offset(
    Fragment const &frag,                       ///< fragment to store from the tensor
    Index pointer_offset) const {               ///< store a tile with a linear offset

    TensorRef offset_ref(ref_);
    offset_ref.add_pointer_offset(pointer_offset);
    //    Policy::MmaIterations::kColumn, Policy::MmaIterations::kRow, kAccumulatorRows, kElementsPerAccess);
    MCTLASS_PRAGMA_UNROLL
    for (int mma_n = 0; mma_n < Policy::MmaIterations::kColumn; ++mma_n) {
      MCTLASS_PRAGMA_UNROLL
      for (int mma_m = 0; mma_m < Policy::MmaIterations::kRow; ++mma_m) {

        int mma_accum_start = kAccumulatorRows * kElementsPerAccess *
          (mma_n * Policy::MmaIterations::kRow + mma_m);

        MCTLASS_PRAGMA_UNROLL
        for (int row = 0; row < kAccumulatorRows; ++row) {
          MCTLASS_PRAGMA_UNROLL
          for (int col = 0; col < kElementsPerAccess; ++col) {
            int accum_m = mma_m * InstructionShape::kM * OpDelta::kRow + row * kAccumulatorRows +
                          col;
            int accum_n = mma_n * InstructionShape::kN * OpDelta::kColumn;
            int idx = mma_accum_start + row * kElementsPerAccess + col;
            offset_ref.at({accum_m, accum_n}) = frag[idx];
          }
        }
      }
    }
  }

  /// Stores a fragment to memory with additional pointer offset
  MCTLASS_DEVICE
  void store_with_byte_offset(
    Fragment const &frag,                       ///< fragment to store from the tensor
    Index byte_offset) const {                  ///< store a tile with a linear offset

    store_with_pointer_offset(byte_offset / sizeof(Element));
  }

  /// Stores a fragment to memory with logical offset in units of whole tiles.
  MCTLASS_DEVICE
  void store(
    Fragment &frag,                             ///< fragment to store to the tensor
    TensorCoord const &tile_offset) const {     ///< stores a tile with a logical offset in units of whole tiles

    store(frag, tile_offset, 0);
  }

  /// Stores a fragment from memory with logical offset in units of whole tiles.
  MCTLASS_DEVICE
  void store(
      /// fragment to store to the tensor
      Fragment const &frag,
      /// stores a tile with a logical offset in units of whole tiles
      TensorCoord const &tile_offset,
      /// stores a tile with a logical offset AND a pointer offset
      Index pointer_offset) const {
    store_with_pointer_offset(frag, ref_.offset(tile_offset) + pointer_offset);
  }
};

template <
    /// Size of the matrix to load (concept: MatrixShape)
    typename Shape_,
    /// Shape of one matrix product operation (concept: MatrixShape)
    typename InstructionShape_,
    /// Interval between adjacent *MMA instructions (in units of MMA
    /// instructions, concept: MatrixShape)
    typename OpDelta_>
class MmaTensorOpAccumulatorTileIterator<
    Shape_, float, mctlass::layout::RowMajor, InstructionShape_, OpDelta_> {
 public:

  /// Shape of tile to load (concept: MatrixShape)
  using Shape = Shape_;

  /// Operand tag
  static Operand const kOperand = Operand::kC;

  /// Element type
  using Element = float;

  /// Layout of source tile
  using Layout = mctlass::layout::RowMajor;

  /// Shape of one matrix product operation (concept: MatrixShape)
  using InstructionShape = InstructionShape_;

  /// Delta between *MMA operations (in units of *MMA operations, concept: MatrixShape)
  using OpDelta = OpDelta_;

  /// Number of participating threads
  static int const kThreads = 32;

  /// TensorRef type for loading element from a tensor
  using TensorRef = TensorRef<Element, Layout>;

  /// Index type
  using Index = typename TensorRef::Index;

  /// Long Index type
  using LongIndex = typename TensorRef::LongIndex;

  /// Coordinate for an element in the tensor
  using TensorCoord = typename TensorRef::TensorCoord;

  /// Internal structure of iterator - made public to enable introspection
  struct Policy {
    static bool const kDivisible =
        !(Shape::kRow % InstructionShape::kM) &&
            !(Shape::kColumn % InstructionShape::kN);

    static_assert(platform::is_same<TensorCoord, MatrixCoord>::value,
      "Layouts must be defined for logical MatrixCoord coordinate space.");

    /// Number of mma operations performed
    using MmaIterations = MatrixShape<
      (Shape::kRow + InstructionShape::kM - 1) / InstructionShape::kM,
      (Shape::kColumn + InstructionShape::kN - 1) / InstructionShape::kN
    >;
  };

private:

  // Assume accumulator tile is an arrangement of 8-by-8 tiles replicated over the entire
  // shape, with each quad mapped to one row and each thread mapped to 1/4 of the elements
  // of that row. The accumulators within one row are assumed to be consecutive.
 static int const kElementsPerAccess = InstructionShape::kN / 4;
 static int const kRowsPerTile = 8;
 static int const kAccumulatorRows = InstructionShape::kM / kRowsPerTile;

public:

  //
  // Derived quantities
  //

  /// Fragment object holding a thread's part of a tile
  using Fragment = Array<
    Element,
    Policy::MmaIterations::kCount * InstructionShape::kMN / kThreads>;

private:

  /// Reference to output tensor
  TensorRef ref_;

public:

  /// Default ctor constructs null iterator
  MCTLASS_HOST_DEVICE
  MmaTensorOpAccumulatorTileIterator() { }

  /// Constructor from TensorRef
  MCTLASS_HOST_DEVICE
  MmaTensorOpAccumulatorTileIterator(
    TensorRef const &ref,
    int lane_id
  ):
    ref_(ref) {
    int quad = (lane_id >> 4) << 2;
    int lane_in_quad = lane_id & 0x7;
    MatrixCoord lane_offset(quad, lane_in_quad);
    ref_.add_coord_offset(lane_offset);
  }

  /// Adds a pointer offset to internal pointer(s) to advance through memory
  MCTLASS_HOST_DEVICE
  MmaTensorOpAccumulatorTileIterator &add_pointer_offset(LongIndex offset) {
    ref_.add_pointer_offset(offset);
    return *this;
  }

  /// Advances an iterator along logical dimensions of matrix in units of whole tiles
  MCTLASS_HOST_DEVICE
  MmaTensorOpAccumulatorTileIterator &add_tile_offset(TensorCoord const &tile_offset) {

    ref_.add_coord_offset(tile_offset * make_Coord(Shape::kRow, Shape::kColumn));

    return *this;
  }

  /// Advances the iterator along the advance dimension
  MCTLASS_HOST_DEVICE
  MmaTensorOpAccumulatorTileIterator & operator++() {
    // deliberate no-op
    return *this;
  }

  /// Advances the iterator along the advance dimension
  MCTLASS_HOST_DEVICE
  MmaTensorOpAccumulatorTileIterator & operator--() {
    // deliberate no-op
    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of the tensor
  MCTLASS_DEVICE
  MmaTensorOpAccumulatorTileIterator & operator+=(TensorCoord const &tile_offset) {
    add_tile_offset(tile_offset);
    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of the tensor
  MCTLASS_DEVICE
  MmaTensorOpAccumulatorTileIterator & operator-=(TensorCoord const &tile_offset) {
    //add_tile_offset(-tile_offset);
    return *this;
  }

  /// Loads a fragment from memory at the location pointed to by the iterator.
  MCTLASS_HOST_DEVICE
  void load(Fragment &frag) const {
    load_with_pointer_offset(frag, 0);
  }

  /// Loads a fragment from memory with additional logical offset
  MCTLASS_DEVICE
  void load_with_pointer_offset(
    Fragment &frag,                             ///< fragment to load from the tensor
    Index pointer_offset) const {               ///< loads a tile with a linear offset

    TensorRef offset_ref(ref_);
    offset_ref.add_pointer_offset(pointer_offset);

    MCTLASS_PRAGMA_UNROLL
    for (int mma_n = 0; mma_n < Policy::MmaIterations::kColumn; ++mma_n) {
      MCTLASS_PRAGMA_UNROLL
      for (int mma_m = 0; mma_m < Policy::MmaIterations::kRow; ++mma_m) {

        int mma_accum_start = kAccumulatorRows * kElementsPerAccess *
          (mma_n * Policy::MmaIterations::kRow + mma_m);

        MCTLASS_PRAGMA_UNROLL
        for (int row = 0; row < kAccumulatorRows; ++row) {
          MCTLASS_PRAGMA_UNROLL
          for (int col = 0; col < kElementsPerAccess; ++col) {
            int accum_m = mma_m * InstructionShape::kM * OpDelta::kRow +
                          row * kRowsPerTile;
            int accum_n = mma_n * InstructionShape::kN * OpDelta::kColumn + col;

            frag[mma_accum_start + row * kElementsPerAccess + col] = offset_ref.at({accum_m, accum_n});
          }
        }
      }
    }
  }

  /// Loads a fragment from memory with additional logical offset
  MCTLASS_DEVICE
  void load_with_byte_offset(
    Fragment &frag,                             ///< fragment to load from the tensor
    Index byte_offset) const {                  ///< loads a tile with a linear offset

    load_with_pointer_offset(byte_offset / sizeof(Element));
  }

  /// Loads a fragment from memory with logical offset in units of whole tiles.
  MCTLASS_DEVICE
  void load(
    Fragment &frag,                             ///< fragment to load from the tensor
    TensorCoord const &tile_offset) const {     ///< loads a tile with a logical offset in units of whole tiles

    load(frag, tile_offset, 0);
  }

  /// Loads a fragment from memory with logical offset in units of whole tiles.
  MCTLASS_DEVICE
  void load(
    Fragment &frag,                             ///< fragment to load from the tensor
    TensorCoord const &tile_offset,             ///< loads a tile with a logical offset in units of whole tiles
    Index pointer_offset) const {               ///< loads a tile with a logical offset AND a pointer offset

    load_with_pointer_offset(frag, ref_.offset(tile_offset) + pointer_offset);
  }

  /// Stores a fragment to memory
  MCTLASS_HOST_DEVICE
  void store(Fragment const &frag) const {
    store_with_pointer_offset(frag, 0);
  }

  /// Stores a fragment to memory with additional pointer offset
  MCTLASS_DEVICE
  void store_with_pointer_offset(
    Fragment const &frag,                       ///< fragment to store from the tensor
    Index pointer_offset) const {               ///< store a tile with a linear offset

    TensorRef offset_ref(ref_);
    offset_ref.add_pointer_offset(pointer_offset);

    MCTLASS_PRAGMA_UNROLL
    for (int mma_n = 0; mma_n < Policy::MmaIterations::kColumn; ++mma_n) {
      MCTLASS_PRAGMA_UNROLL
      for (int mma_m = 0; mma_m < Policy::MmaIterations::kRow; ++mma_m) {
        int mma_accum_start = kAccumulatorRows * kElementsPerAccess *
                              (mma_n * Policy::MmaIterations::kRow);
        MCTLASS_PRAGMA_UNROLL
        for (int row = 0; row < kAccumulatorRows; ++row) {
          MCTLASS_PRAGMA_UNROLL
          for (int col = 0; col < kElementsPerAccess; ++col) {
            int accum_m = mma_m * InstructionShape::kM;
            int accum_n = mma_n * InstructionShape::kN * OpDelta::kColumn;
            int t_idx = row * kElementsPerAccess + col;
            int idx = mma_accum_start + t_idx + mma_m * 4;
             offset_ref.at({accum_m + t_idx, accum_n}) = frag[idx];
          }
        }
      }
    }
  }

  /// Stores a fragment to memory with additional pointer offset
  MCTLASS_DEVICE
  void store_with_byte_offset(
    Fragment const &frag,                       ///< fragment to store from the tensor
    Index byte_offset) const {                  ///< store a tile with a linear offset

    store_with_pointer_offset(byte_offset / sizeof(Element));
  }

  /// Stores a fragment to memory with logical offset in units of whole tiles.
  MCTLASS_DEVICE
  void store(
    Fragment &frag,                             ///< fragment to store to the tensor
    TensorCoord const &tile_offset) const {     ///< stores a tile with a logical offset in units of whole tiles

    store(frag, tile_offset, 0);
  }

  /// Stores a fragment from memory with logical offset in units of whole tiles.
  MCTLASS_DEVICE
  void store(
      /// fragment to store to the tensor
      Fragment const &frag,
      /// stores a tile with a logical offset in units of whole tiles
      TensorCoord const &tile_offset,
      /// stores a tile with a logical offset AND a pointer offset
      Index pointer_offset) const {
    store_with_pointer_offset(frag, ref_.offset(tile_offset) + pointer_offset);
  }
};

////////////////////////////////////////////////////////////////////////////////

/// This tile iterator is specialized for 32-thread TensorOps. It is used to load or store
/// accumulators from memory and is agnostic to layout.
///
/// This iterator is not tested.
///
/// Satisfies:
///   ReadableRandomAccessContiguousTileIteratorConcept |
///   WriteableRandomAccessContiguousTileIteratorConcept
///
template <
    /// Size of the matrix to load (concept: MatrixShape)
    typename Shape_,
    /// Element type
    typename Element_,
    /// Shape of one matrix product operation (concept: MatrixShape)
    typename InstructionShape_,
    /// Interval between adjacent *MMA instructions (in units of MMA
    /// instructions, concept: MatrixShape)
    typename OpDelta_>
class MmaTensorOpAccumulatorTileIterator<
    Shape_, Element_, mctlass::layout::AffineRankN<2>, InstructionShape_, OpDelta_> {
 public:

  /// Shape of tile to load (concept: MatrixShape)
  using Shape = Shape_;

  /// Operand tag
  static Operand const kOperand = Operand::kC;

  /// Element type
  using Element = Element_;

  /// Layout of source tile
  using Layout = mctlass::layout::RowMajor;

  /// Shape of one matrix product operation (concept: MatrixShape)
  using InstructionShape = InstructionShape_;

  /// Delta between *MMA operations (in units of *MMA operations, concept: MatrixShape)
  using OpDelta = OpDelta_;

  /// Number of participating threads
  static int const kThreads = 32;

  /// TensorRef type for loading element from a tensor
  using TensorRef = TensorRef<Element, Layout>;

  /// Index type
  using Index = typename TensorRef::Index;

  /// Long Index type
  using LongIndex = typename TensorRef::LongIndex;

  /// Coordinate for an element in the tensor
  using TensorCoord = typename TensorRef::TensorCoord;

  /// Internal structure of iterator - made public to enable introspection
  struct Policy {
    static bool const kDivisible =
        !(Shape::kRow % InstructionShape::kM) &&
            !(Shape::kColumn % InstructionShape::kN);

    static_assert(platform::is_same<TensorCoord, MatrixCoord>::value,
      "Layouts must be defined for logical MatrixCoord coordinate space.");

    /// Number of mma operations performed
    using MmaIterations = MatrixShape<
      (Shape::kRow + InstructionShape::kM - 1) / InstructionShape::kM,
      (Shape::kColumn + InstructionShape::kN - 1) / InstructionShape::kN
    >;
  };

private:

  // Assume accumulator tile is an arrangement of 8-by-8 tiles replicated over the entire
  // shape, with each quad mapped to one row and each thread mapped to 1/4 of the elements
  // of that row. The accumulators within one row are assumed to be consecutive.
 static int const kElementsPerAccess = InstructionShape::kN / 4;
 static int const kRowsPerTile = 8;
 static int const kAccumulatorRows = InstructionShape::kM / kRowsPerTile;

public:

  //
  // Derived quantities
  //

  /// Fragment object holding a thread's part of a tile
  using Fragment = Array<
    Element,
    Policy::MmaIterations::kCount * InstructionShape::kMN / kThreads>;

private:

  /// Reference to output tensor
  TensorRef ref_;

public:

  /// Default ctor constructs null iterator
  MCTLASS_HOST_DEVICE
  MmaTensorOpAccumulatorTileIterator() { }

  /// Constructor from TensorRef
  MCTLASS_HOST_DEVICE
  MmaTensorOpAccumulatorTileIterator(
    TensorRef const &ref,
    int lane_id
  ):
    ref_(ref) {

    int quad = (lane_id >> 2);
    int lane_in_quad = (lane_id & 3);

    MatrixCoord lane_offset(quad, lane_in_quad * kElementsPerAccess);

    ref_.add_coord_offset(lane_offset);
  }

  /// Adds a pointer offset to internal pointer(s) to advance through memory
  MCTLASS_HOST_DEVICE
  MmaTensorOpAccumulatorTileIterator &add_pointer_offset(LongIndex offset) {
    ref_.add_pointer_offset(offset);
    return *this;
  }

  /// Advances an iterator along logical dimensions of matrix in units of whole tiles
  MCTLASS_HOST_DEVICE
  MmaTensorOpAccumulatorTileIterator &add_tile_offset(TensorCoord const &tile_offset) {

    ref_.add_coord_offset(tile_offset * make_Coord(Shape::kRow, Shape::kColumn));

    return *this;
  }

  /// Advances the iterator along the advance dimension
  MCTLASS_HOST_DEVICE
  MmaTensorOpAccumulatorTileIterator & operator++() {
    // deliberate no-op
    return *this;
  }

  /// Advances the iterator along the advance dimension
  MCTLASS_HOST_DEVICE
  MmaTensorOpAccumulatorTileIterator & operator--() {
    // deliberate no-op
    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of the tensor
  MCTLASS_DEVICE
  MmaTensorOpAccumulatorTileIterator & operator+=(TensorCoord const &tile_offset) {
    add_tile_offset(tile_offset);
    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of the tensor
  MCTLASS_DEVICE
  MmaTensorOpAccumulatorTileIterator & operator-=(TensorCoord const &tile_offset) {
    add_tile_offset(-tile_offset);
    return *this;
  }

  /// Loads a fragment from memory at the location pointed to by the iterator.
  MCTLASS_HOST_DEVICE
  void load(Fragment &frag) const {
    load_with_pointer_offset(frag, 0);
  }

  /// Loads a fragment from memory with additional logical offset
  MCTLASS_DEVICE
  void load_with_pointer_offset(
    Fragment &frag,                             ///< fragment to load from the tensor
    Index pointer_offset) const {               ///< loads a tile with a linear offset

    TensorRef offset_ref(ref_);
    offset_ref.add_pointer_offset(pointer_offset);

    MCTLASS_PRAGMA_UNROLL
    for (int mma_n = 0; mma_n < Policy::MmaIterations::kColumn; ++mma_n) {
      MCTLASS_PRAGMA_UNROLL
      for (int mma_m = 0; mma_m < Policy::MmaIterations::kRow; ++mma_m) {

        int mma_accum_start = kAccumulatorRows * kElementsPerAccess *
          (mma_n * Policy::MmaIterations::kRow + mma_m);

        MCTLASS_PRAGMA_UNROLL
        for (int row = 0; row < kAccumulatorRows; ++row) {
          MCTLASS_PRAGMA_UNROLL
          for (int col = 0; col < kElementsPerAccess; ++col) {
            int accum_m = mma_m * InstructionShape::kM * OpDelta::kRow +
                          row * kRowsPerTile;
            int accum_n = mma_n * InstructionShape::kN * OpDelta::kColumn + col;

            frag[mma_accum_start + row * kElementsPerAccess + col] = offset_ref.at({accum_m, accum_n});
          }
        }
      }
    }
  }

  /// Loads a fragment from memory with additional logical offset
  MCTLASS_DEVICE
  void load_with_byte_offset(
    Fragment &frag,                             ///< fragment to load from the tensor
    Index byte_offset) const {                  ///< loads a tile with a linear offset

    load_with_pointer_offset(byte_offset / sizeof(Element));
  }

  /// Loads a fragment from memory with logical offset in units of whole tiles.
  MCTLASS_DEVICE
  void load(
    Fragment &frag,                             ///< fragment to load from the tensor
    TensorCoord const &tile_offset) const {     ///< loads a tile with a logical offset in units of whole tiles

    load(frag, tile_offset, 0);
  }

  /// Loads a fragment from memory with logical offset in units of whole tiles.
  MCTLASS_DEVICE
  void load(
    Fragment &frag,                             ///< fragment to load from the tensor
    TensorCoord const &tile_offset,             ///< loads a tile with a logical offset in units of whole tiles
    Index pointer_offset) const {               ///< loads a tile with a logical offset AND a pointer offset

    load_with_pointer_offset(frag, ref_.offset(tile_offset) + pointer_offset);
  }

  /// Stores a fragment to memory
  MCTLASS_HOST_DEVICE
  void store(Fragment const &frag) const {
    store_with_pointer_offset(frag, 0);
  }

  /// Stores a fragment to memory with additional pointer offset
  MCTLASS_DEVICE
  void store_with_pointer_offset(
    Fragment const &frag,                       ///< fragment to store from the tensor
    Index pointer_offset) const {               ///< store a tile with a linear offset

    TensorRef offset_ref(ref_);
    offset_ref.add_pointer_offset(pointer_offset);

    MCTLASS_PRAGMA_UNROLL
    for (int mma_n = 0; mma_n < Policy::MmaIterations::kColumn; ++mma_n) {
      MCTLASS_PRAGMA_UNROLL
      for (int mma_m = 0; mma_m < Policy::MmaIterations::kRow; ++mma_m) {

        int mma_accum_start = kAccumulatorRows * kElementsPerAccess *
          (mma_n * Policy::MmaIterations::kRow + mma_m);

        MCTLASS_PRAGMA_UNROLL
        for (int row = 0; row < kAccumulatorRows; ++row) {
          MCTLASS_PRAGMA_UNROLL
          for (int col = 0; col < kElementsPerAccess; ++col) {
            int accum_m = mma_m * InstructionShape::kM * OpDelta::kRow +
                          row * kRowsPerTile;
            int accum_n = mma_n * InstructionShape::kN * OpDelta::kColumn + col;
            int idx = mma_accum_start + row * kElementsPerAccess + col;

            offset_ref.at({accum_m, accum_n}) = frag[idx];
          }
        }
      }
    }
  }

  /// Stores a fragment to memory with additional pointer offset
  MCTLASS_DEVICE
  void store_with_byte_offset(
    Fragment const &frag,                       ///< fragment to store from the tensor
    Index byte_offset) const {                  ///< store a tile with a linear offset

    store_with_pointer_offset(byte_offset / sizeof(Element));
  }

  /// Stores a fragment to memory with logical offset in units of whole tiles.
  MCTLASS_DEVICE
  void store(
    Fragment &frag,                             ///< fragment to store to the tensor
    TensorCoord const &tile_offset) const {     ///< stores a tile with a logical offset in units of whole tiles

    store(frag, tile_offset, 0);
  }

  /// Stores a fragment from memory with logical offset in units of whole tiles.
  MCTLASS_DEVICE
  void store(
      /// fragment to store to the tensor
      Fragment const &frag,
      /// stores a tile with a logical offset in units of whole tiles
      TensorCoord const &tile_offset,
      /// stores a tile with a logical offset AND a pointer offset
      Index pointer_offset) const {
    store_with_pointer_offset(frag, ref_.offset(tile_offset) + pointer_offset);
  }
};

////////////////////////////////////////////////////////////////////////////////

/// This tile iterator is specialized for 32-thread TensorOps. It is used to load or store
/// accumulators from memory and is agnostic to layout. It could be faster if it assumed row-major
/// accumulator layout.
///
/// Satisfies:
///   ReadableRandomAccessContiguousTileIteratorConcept |
///   WriteableRandomAccessContiguousTileIteratorConcept
///
template <
    /// Size of the matrix to load (concept: MatrixShape)
    typename Shape_,
    /// Element type
    typename Element_,
    /// Shape of one matrix product operation (concept: MatrixShape)
    typename InstructionShape_,
    /// Interval between adjacent *MMA instructions (in units of MMA
    /// instructions, concept: MatrixShape)
    typename OpDelta_>
class MmaTensorOpAccumulatorTileIterator<Shape_, Element_,
                                         mctlass::layout::ColumnMajor,
                                         InstructionShape_, OpDelta_> {
 public:

  /// Shape of tile to load (concept: MatrixShape)
  using Shape = Shape_;

  /// Operand tag
  static Operand const kOperand = Operand::kC;

  /// Element type
  using Element = Element_;

  /// Layout of source tile
  using Layout = mctlass::layout::ColumnMajor;

  /// Shape of one matrix product operation (concept: MatrixShape)
  using InstructionShape = InstructionShape_;

  /// Delta between *MMA operations (in units of *MMA operations, concept: MatrixShape)
  using OpDelta = OpDelta_;

  /// Number of participating threads
  static int const kThreads = (InstructionShape::kM == 16
                              && InstructionShape::kN == 16
                              && InstructionShape::kK == 16) ? 64 : 32;

  /// TensorRef type for loading element from a tensor
  using TensorRef = TensorRef<Element, Layout>;

  /// Index type
  using Index = typename TensorRef::Index;

  /// Long Index type
  using LongIndex = typename TensorRef::LongIndex;

  /// Coordinate for an element in the tensor
  using TensorCoord = typename TensorRef::TensorCoord;

  /// Internal structure of iterator - made public to enable introspection
  struct Policy {
    static bool const kDivisible =
        !(Shape::kRow % InstructionShape::kM) &&
            !(Shape::kColumn % InstructionShape::kN);

    static_assert(platform::is_same<TensorCoord, MatrixCoord>::value,
      "Layouts must be defined for logical MatrixCoord coordinate space.");

    /// Number of mma operations performed
    using MmaIterations = MatrixShape<
      (Shape::kRow + InstructionShape::kM - 1) / InstructionShape::kM,
      (Shape::kColumn + InstructionShape::kN - 1) / InstructionShape::kN
    >;
  };

private:

  // Assume accumulator tile is an arrangement of 8-by-8 tiles replicated over the entire
  // shape, with each quad mapped to one row and each thread mapped to 1/4 of the elements
  // of that row. The accumulators within one row are assumed to be consecutive.
 static int const kElementsPerAccess = InstructionShape::kN / 4;
 static int const kRowsPerTile = (InstructionShape::kM == 16
                                  && InstructionShape::kN == 16
                                  && InstructionShape::kK == 16) ? 16 : 8;
 static int const kAccumulatorRows = InstructionShape::kM / kRowsPerTile;

public:

  //
  // Derived quantities
  //

  /// Fragment object holding a thread's part of a tile
  using Fragment = Array<Element,
    Policy::MmaIterations::kCount * InstructionShape::kMN / kThreads>;

private:

  /// Reference to output tensor
  TensorRef ref_;

public:

  /// Default ctor constructs null iterator
  MCTLASS_HOST_DEVICE
  MmaTensorOpAccumulatorTileIterator() { }

  /// Constructor from TensorRef
  MCTLASS_HOST_DEVICE
  MmaTensorOpAccumulatorTileIterator(
    TensorRef const &ref,
    int lane_id
  ):
    ref_(ref) {

    if (platform::is_same<Element, float>::value == true) {
      if (InstructionShape::kM == 16 && InstructionShape::kN == 8
        && (InstructionShape::kK == 16 || InstructionShape::kK == 4 || InstructionShape::kK == 8)) {

        const int quad = (lane_id >> 4) << 2;
        const int lane_in_quad = lane_id & 0x7;
        MatrixCoord lane_offset(quad, lane_in_quad);
        ref_.add_coord_offset(lane_offset);

      }
      else if(InstructionShape::kM == 16 && InstructionShape::kN == 16 && InstructionShape::kK == 16) {

        const int quad = lane_id & 0xf;
        const int lane_in_quad = (lane_id >> 4) << 2;
        MatrixCoord lane_offset(quad, lane_in_quad);
        ref_.add_coord_offset(lane_offset);

      }
      else {
        printf("Not Impl Other InstructionShape for float.\n");
      }
    }
    else if(platform::is_same<Element, int>::value == true) {
      if (InstructionShape::kM == 16 && InstructionShape::kN == 8 && InstructionShape::kK == 32) {

        const int quad = ((lane_id >> 4) << 2);
        const int lane_in_quad = lane_id & 0x7;
        MatrixCoord lane_offset(quad, lane_in_quad);
        ref_.add_coord_offset(lane_offset);

      }
      else {
        const int quad = lane_id & 0xf;
        const int lane_in_quad = (lane_id >> 4) << 2;
        MatrixCoord lane_offset(quad, lane_in_quad);
        ref_.add_coord_offset(lane_offset);
      }
    }
    else {

      int quad = (lane_id >> 2);
      int lane_in_quad = (lane_id & 3);

      MatrixCoord lane_offset(quad, lane_in_quad * kElementsPerAccess);

      ref_.add_coord_offset(lane_offset);
    }

  }

  /// Adds a pointer offset to internal pointer(s) to advance through memory
  MCTLASS_HOST_DEVICE
  MmaTensorOpAccumulatorTileIterator &add_pointer_offset(LongIndex offset) {
    ref_.add_pointer_offset(offset);
    return *this;
  }

  /// Advances an iterator along logical dimensions of matrix in units of whole tiles
  MCTLASS_HOST_DEVICE
  MmaTensorOpAccumulatorTileIterator &add_tile_offset(TensorCoord const &tile_offset) {

    ref_.add_coord_offset(tile_offset * make_Coord(Shape::kRow, Shape::kColumn));

    return *this;
  }

  /// Advances the iterator along the advance dimension
  MCTLASS_HOST_DEVICE
  MmaTensorOpAccumulatorTileIterator & operator++() {
    // deliberate no-op
    return *this;
  }

  /// Advances the iterator along the advance dimension
  MCTLASS_HOST_DEVICE
  MmaTensorOpAccumulatorTileIterator & operator--() {
    // deliberate no-op
    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of the tensor
  MCTLASS_DEVICE
  MmaTensorOpAccumulatorTileIterator & operator+=(TensorCoord const &tile_offset) {
    add_tile_offset(tile_offset);
    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of the tensor
  MCTLASS_DEVICE
  MmaTensorOpAccumulatorTileIterator & operator-=(TensorCoord const &tile_offset) {
    add_tile_offset(-tile_offset);
    return *this;
  }

  /// Loads a fragment from memory at the location pointed to by the iterator.
  MCTLASS_HOST_DEVICE
  void load(Fragment &frag) const {
    load_with_pointer_offset(frag, 0);
  }

  /// Loads a fragment from memory with additional logical offset
  MCTLASS_DEVICE
  void load_with_pointer_offset(
    Fragment &frag,                             ///< fragment to load from the tensor
    Index pointer_offset) const {               ///< loads a tile with a linear offset

    TensorRef offset_ref(ref_);
    offset_ref.add_pointer_offset(pointer_offset);

    MCTLASS_PRAGMA_UNROLL
    for (int mma_n = 0; mma_n < Policy::MmaIterations::kColumn; ++mma_n) {
      MCTLASS_PRAGMA_UNROLL
      for (int mma_m = 0; mma_m < Policy::MmaIterations::kRow; ++mma_m) {

        int mma_accum_start = kAccumulatorRows * kElementsPerAccess *
          (mma_n * Policy::MmaIterations::kRow + mma_m);

        MCTLASS_PRAGMA_UNROLL
        for (int row = 0; row < kAccumulatorRows; ++row) {
          MCTLASS_PRAGMA_UNROLL
          for (int col = 0; col < kElementsPerAccess; ++col) {
            int accum_m = mma_m * InstructionShape::kM * OpDelta::kRow +
                          row * kRowsPerTile;
            int accum_n = mma_n * InstructionShape::kN * OpDelta::kColumn + col;
            int idx = mma_accum_start + row * kElementsPerAccess + col;

            frag[idx] = offset_ref.at({accum_m, accum_n});
          }
        }
      }
    }
  }

  /// Loads a fragment from memory with additional logical offset
  MCTLASS_DEVICE
  void load_with_byte_offset(
    Fragment &frag,                             ///< fragment to load from the tensor
    Index byte_offset) const {                  ///< loads a tile with a linear offset

    load_with_pointer_offset(byte_offset / sizeof(Element));
  }

  /// Loads a fragment from memory with logical offset in units of whole tiles.
  MCTLASS_DEVICE
  void load(
    Fragment &frag,                             ///< fragment to load from the tensor
    TensorCoord const &tile_offset) const {     ///< loads a tile with a logical offset in units of whole tiles

    load(frag, tile_offset, 0);
  }

  /// Loads a fragment from memory with logical offset in units of whole tiles.
  MCTLASS_DEVICE
  void load(
    Fragment &frag,                             ///< fragment to load from the tensor
    TensorCoord const &tile_offset,             ///< loads a tile with a logical offset in units of whole tiles
    Index pointer_offset) const {               ///< loads a tile with a logical offset AND a pointer offset

    load_with_pointer_offset(frag, ref_.offset(tile_offset) + pointer_offset);
  }

  /// Stores a fragment to memory
  MCTLASS_HOST_DEVICE
  void store(Fragment const &frag) const {
    store_with_pointer_offset(frag, 0);
  }

   /// Stores a fragment to memory with additional pointer offset
   /// For m16n16k16
  MCTLASS_DEVICE
  void store_with_pointer_offset_m16n16k16(
    Fragment const &frag,                       ///< fragment to store from the tensor
    Index pointer_offset) const {               ///< store a tile with a linear offset

    TensorRef offset_ref(ref_);
    offset_ref.add_pointer_offset(pointer_offset);

    MCTLASS_PRAGMA_UNROLL
    for (int mma_n = 0; mma_n < Policy::MmaIterations::kColumn; ++mma_n) {
      MCTLASS_PRAGMA_UNROLL
      for (int mma_m = 0; mma_m < Policy::MmaIterations::kRow; ++mma_m) {

        int mma_accum_start = kAccumulatorRows * kElementsPerAccess *
                              (mma_n * Policy::MmaIterations::kRow + mma_m);

        MCTLASS_PRAGMA_UNROLL
        for (int row = 0; row < kAccumulatorRows; ++row) {
          MCTLASS_PRAGMA_UNROLL
          for (int col = 0; col < kElementsPerAccess; ++col) {

            const int accum_m = mma_m * InstructionShape::kM * OpDelta::kRow
                                + row * kElementsPerAccess;
            const int accum_n = mma_n * InstructionShape::kN * OpDelta::kColumn + col;
            const int idx = mma_accum_start + row * kElementsPerAccess + col;
            offset_ref.at({accum_m, accum_n}) = frag[idx];

          }
        }
      }
    }
  }

  /// Stores a fragment to memory with additional pointer offset
  MCTLASS_DEVICE
  void store_with_pointer_offset(
    Fragment const &frag,                       ///< fragment to store from the tensor
    Index pointer_offset) const {               ///< store a tile with a linear offset

    if(InstructionShape::kM == 16 &&
      InstructionShape::kN == 16 && InstructionShape::kK == 16) {
        store_with_pointer_offset_m16n16k16(frag,pointer_offset);
        return;
    }

    TensorRef offset_ref(ref_);
    offset_ref.add_pointer_offset(pointer_offset);

    MCTLASS_PRAGMA_UNROLL
    for (int mma_n = 0; mma_n < Policy::MmaIterations::kColumn; ++mma_n) {
      MCTLASS_PRAGMA_UNROLL
      for (int mma_m = 0; mma_m < Policy::MmaIterations::kRow; ++mma_m) {

        int mma_accum_start = kAccumulatorRows * kElementsPerAccess *
          (mma_n * Policy::MmaIterations::kRow + mma_m);
        if (InstructionShape::kM == 16 && InstructionShape::kN == 8 && InstructionShape::kK == 4) {
          mma_accum_start = kAccumulatorRows * kElementsPerAccess *
            (mma_n * Policy::MmaIterations::kRow);
        }

        MCTLASS_PRAGMA_UNROLL
        for (int row = 0; row < kAccumulatorRows; ++row) {
          MCTLASS_PRAGMA_UNROLL
          for (int col = 0; col < kElementsPerAccess; ++col) {
            if (platform::is_same<Element, float>::value == true) {
              if (InstructionShape::kM == 16 && InstructionShape::kN == 8
                && (InstructionShape::kK == 16 || InstructionShape::kK == 8)) {
                const int accum_m = mma_m * InstructionShape::kM * OpDelta::kRow + row * kElementsPerAccess + col;
                const int accum_n = mma_n * InstructionShape::kN * OpDelta::kColumn;
                const int idx = mma_accum_start + row * kElementsPerAccess + col;
                offset_ref.at({accum_m, accum_n}) = frag[idx];
              }
              else if (InstructionShape::kM == 16 && InstructionShape::kN == 8 && InstructionShape::kK == 4) {
                int accum_m = mma_m * InstructionShape::kM;
                int accum_n = mma_n * InstructionShape::kN * OpDelta::kColumn;
                int t_idx = row * kElementsPerAccess + col;
                int idx = mma_accum_start + t_idx + mma_m * 4;
                offset_ref.at({accum_m + t_idx, accum_n}) = frag[idx];
              }
              else {
                printf("Not Impl other InstructionShape with float Now.\n");
              }
            }
            else  if (platform::is_same<Element, int>::value == true) {
              if (InstructionShape::kM == 16 && InstructionShape::kN == 8 && InstructionShape::kK == 32) {
                const int accum_m = mma_m * InstructionShape::kM * OpDelta::kRow + row * kElementsPerAccess + col;
                const int accum_n = mma_n * InstructionShape::kN * OpDelta::kColumn;
                const int idx = mma_accum_start + row * kElementsPerAccess + col;
                offset_ref.at({accum_m, accum_n}) = frag[idx];
              }
              else {
                printf("Not Impl other InstructionShape with int Now.\n");
              }
            }
            else {
              int accum_m = mma_m * InstructionShape::kM * OpDelta::kRow +
                            row * kRowsPerTile;
              int accum_n = mma_n * InstructionShape::kN * OpDelta::kColumn + col;
              int idx = mma_accum_start + row * kElementsPerAccess + col;

              offset_ref.at({accum_m, accum_n}) = frag[idx];
            }
          }
        }
      }
    }
  }

  /// Stores a fragment to memory with additional pointer offset
  MCTLASS_DEVICE
  void store_with_byte_offset(
    Fragment const &frag,                       ///< fragment to store from the tensor
    Index byte_offset) const {                  ///< store a tile with a linear offset

    store_with_pointer_offset(byte_offset / sizeof(Element));
  }

  /// Stores a fragment to memory with logical offset in units of whole tiles.
  MCTLASS_DEVICE
  void store(
    Fragment &frag,                             ///< fragment to store to the tensor
    TensorCoord const &tile_offset) const {     ///< stores a tile with a logical offset in units of whole tiles

    store(frag, tile_offset, 0);
  }

  /// Stores a fragment from memory with logical offset in units of whole tiles.
  MCTLASS_DEVICE
  void store(
      /// fragment to store to the tensor
      Fragment const &frag,
      /// stores a tile with a logical offset in units of whole tiles
      TensorCoord const &tile_offset,
      /// stores a tile with a logical offset AND a pointer offset
      Index pointer_offset) const {
    store_with_pointer_offset(frag, ref_.offset(tile_offset) + pointer_offset);
  }
};

////////////////////////////////////////////////////////////////////////////////

/// This tile iterator is specialized for 32-thread TensorOps. It is used to load or store
/// accumulators from memory and is agnostic to layout. It could be faster if it assumed row-major
/// accumulator layout.
///
/// Satisfies:
///   ReadableRandomAccessContiguousTileIteratorConcept |
///   WriteableRandomAccessContiguousTileIteratorConcept
///

template <
    /// Size of the matrix to load (concept: MatrixShape)
    typename Shape_,
    /// Element typ
    typename Element_,
    /// Shape of one matrix product operation (concept: MatrixShape)
    typename InstructionShape_,
    /// Interval between adjacent *MMA instructions (in units of MMA
    /// instructions, concept: MatrixShape)
    typename OpDelta_,
    /// Interleaved N
    int InterleavedN>
class MmaTensorOpAccumulatorTileIterator<
    Shape_, Element_, mctlass::layout::ColumnMajorInterleaved<InterleavedN>,
    InstructionShape_, OpDelta_> {
 public:

  /// Shape of tile to load (concept: MatrixShape)
  using Shape = Shape_;

  /// Operand tag
  static Operand const kOperand = Operand::kC;

  /// Element type
  using Element = Element_;

  /// Layout of source tile
  using Layout = mctlass::layout::ColumnMajorInterleaved<InterleavedN>;

  /// Shape of one matrix product operation (concept: MatrixShape)
  using InstructionShape = InstructionShape_;

  /// Delta between *MMA operations (in units of *MMA operations, concept: MatrixShape)
  using OpDelta = OpDelta_;

  /// Number of participating threads
  static int const kThreads = 32;

  /// TensorRef type for loading element from a tensor
  using TensorRef = TensorRef<Element, Layout>;

  /// Index type
  using Index = typename TensorRef::Index;

  /// Long Index type
  using LongIndex = typename TensorRef::LongIndex;

  /// Coordinate for an element in the tensor
  using TensorCoord = typename TensorRef::TensorCoord;

  /// Internal structure of iterator - made public to enable introspection
  struct Policy {
    static_assert(
        !(Shape::kRow % InstructionShape::kM) &&
            !(Shape::kColumn % InstructionShape::kN),
        "Shape of warp-level Mma must be divisible by operator shape.");

    static_assert(platform::is_same<TensorCoord, MatrixCoord>::value,
      "Layouts must be defined for logical MatrixCoord coordinate space.");

    /// Number of mma operations performed
    using MmaIterations = MatrixShape<Shape::kRow / InstructionShape::kM,
                                      Shape::kColumn / InstructionShape::kN>;
  };

private:

  static int const kElementsPerAccess = 2;

public:

  //
  // Derived quantities
  //

  using AccessType = Array<Element, kElementsPerAccess>;

  /// Fragment object holding a thread's part of a tile
  using Fragment = Array<Element, Shape::kCount / kThreads>;

private:

  /// Reference to output tensor
  TensorRef ref_;

public:

  /// Default ctor constructs null iterator
  MCTLASS_HOST_DEVICE
  MmaTensorOpAccumulatorTileIterator() { }

  /// Constructor from TensorRef
  MCTLASS_HOST_DEVICE
  MmaTensorOpAccumulatorTileIterator(
    TensorRef const &ref,
    int lane_id
  ):
    ref_(ref) {

    int quad = (lane_id >> 2);
    int lane_in_quad = (lane_id & 3);

    MatrixCoord lane_offset(quad, lane_in_quad * kElementsPerAccess);

    ref_.add_coord_offset(lane_offset);
  }

  /// Adds a pointer offset to internal pointer(s) to advance through memory
  MCTLASS_HOST_DEVICE
  MmaTensorOpAccumulatorTileIterator &add_pointer_offset(LongIndex offset) {
    ref_.add_pointer_offset(offset);
    return *this;
  }

  /// Advances an iterator along logical dimensions of matrix in units of whole tiles
  MCTLASS_HOST_DEVICE
  MmaTensorOpAccumulatorTileIterator &add_tile_offset(TensorCoord const &tile_offset) {

    ref_.add_coord_offset(tile_offset * make_Coord(Shape::kRow, Shape::kColumn));

    return *this;
  }

  /// Advances the iterator along the advance dimension
  MCTLASS_HOST_DEVICE
  MmaTensorOpAccumulatorTileIterator & operator++() {
    // deliberate no-op
    return *this;
  }

  /// Advances the iterator along the advance dimension
  MCTLASS_HOST_DEVICE
  MmaTensorOpAccumulatorTileIterator & operator--() {
    // deliberate no-op
    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of the tensor
  MCTLASS_DEVICE
  MmaTensorOpAccumulatorTileIterator & operator+=(TensorCoord const &tile_offset) {
    add_tile_offset(tile_offset);
    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of the tensor
  MCTLASS_DEVICE
  MmaTensorOpAccumulatorTileIterator & operator-=(TensorCoord const &tile_offset) {
    add_tile_offset(-tile_offset);
    return *this;
  }

  /// Loads a fragment from memory at the location pointed to by the iterator.
  MCTLASS_HOST_DEVICE
  void load(Fragment &frag) const {
    load_with_pointer_offset(frag, 0);
  }

  /// Loads a fragment from memory with additional logical offset
  MCTLASS_DEVICE
  void load_with_pointer_offset(
    Fragment &frag,                             ///< fragment to load from the tensor
    Index pointer_offset) const {               ///< loads a tile with a linear offset

    TensorRef offset_ref(ref_);
    offset_ref.add_pointer_offset(pointer_offset);

    AccessType* frag_ptr = reinterpret_cast<AccessType *>(&frag);

    MCTLASS_PRAGMA_UNROLL
    for (int mma_n = 0; mma_n < Policy::MmaIterations::kColumn; ++mma_n) {
      MCTLASS_PRAGMA_UNROLL
      for (int mma_m = 0; mma_m < Policy::MmaIterations::kRow; ++mma_m) {
        int accum_m = mma_m * InstructionShape::kM;
        int accum_n = mma_n * InstructionShape::kN;

        int idx = mma_m + mma_n * Policy::MmaIterations::kRow;

        AccessType* access_ptr = reinterpret_cast<AccessType *>(offset_ref.data() +
          offset_ref.offset(TensorCoord(accum_m, accum_n)));

        frag_ptr[idx] = access_ptr[0];
      }
    }
  }

  /// Loads a fragment from memory with additional logical offset
  MCTLASS_DEVICE
  void load_with_byte_offset(
    Fragment &frag,                             ///< fragment to load from the tensor
    Index byte_offset) const {                  ///< loads a tile with a linear offset

    load_with_pointer_offset(byte_offset / sizeof(Element));
  }

  /// Loads a fragment from memory with logical offset in units of whole tiles.
  MCTLASS_DEVICE
  void load(
    Fragment &frag,                             ///< fragment to load from the tensor
    TensorCoord const &tile_offset) const {     ///< loads a tile with a logical offset in units of whole tiles

    load(frag, tile_offset, 0);
  }

  /// Loads a fragment from memory with logical offset in units of whole tiles.
  MCTLASS_DEVICE
  void load(
    Fragment &frag,                             ///< fragment to load from the tensor
    TensorCoord const &tile_offset,             ///< loads a tile with a logical offset in units of whole tiles
    Index pointer_offset) const {               ///< loads a tile with a logical offset AND a pointer offset

    load_with_pointer_offset(frag, ref_.offset(tile_offset) + pointer_offset);
  }

  /// Stores a fragment to memory
  MCTLASS_HOST_DEVICE
  void store(Fragment const &frag) const {
    store_with_pointer_offset(frag, 0);
  }

  /// Stores a fragment to memory with additional pointer offset
  MCTLASS_DEVICE
  void store_with_pointer_offset(
    Fragment const &frag,                       ///< fragment to store from the tensor
    Index pointer_offset) const {               ///< store a tile with a linear offset

    TensorRef offset_ref(ref_);
    offset_ref.add_pointer_offset(pointer_offset);

    AccessType const *frag_ptr = reinterpret_cast<AccessType const*>(&frag);

    MCTLASS_PRAGMA_UNROLL
    for (int mma_n = 0; mma_n < Policy::MmaIterations::kColumn; ++mma_n) {
      MCTLASS_PRAGMA_UNROLL
      for (int mma_m = 0; mma_m < Policy::MmaIterations::kRow; ++mma_m) {
        int accum_m = mma_m * InstructionShape::kM;
        int accum_n = mma_n * InstructionShape::kN;

        int idx = mma_m + mma_n * Policy::MmaIterations::kRow;

        AccessType* access_ptr = reinterpret_cast<AccessType *>(offset_ref.data() +
                                 offset_ref.offset(TensorCoord(accum_m, accum_n)));

        access_ptr[0] = frag_ptr[idx];
      }
    }
  }

  /// Stores a fragment to memory with additional pointer offset
  MCTLASS_DEVICE
  void store_with_byte_offset(
    Fragment const &frag,                       ///< fragment to store from the tensor
    Index byte_offset) const {                  ///< store a tile with a linear offset

    store_with_pointer_offset(byte_offset / sizeof(Element));
  }

  /// Stores a fragment to memory with logical offset in units of whole tiles.
  MCTLASS_DEVICE
  void store(
    Fragment &frag,                             ///< fragment to store to the tensor
    TensorCoord const &tile_offset) const {     ///< stores a tile with a logical offset in units of whole tiles

    store(frag, tile_offset, 0);
  }

  /// Stores a fragment from memory with logical offset in units of whole tiles.
  MCTLASS_DEVICE
  void store(
      /// fragment to store to the tensor
      Fragment const &frag,
      /// stores a tile with a logical offset in units of whole tiles
      TensorCoord const &tile_offset,
      /// stores a tile with a logical offset AND a pointer offset
      Index pointer_offset) const {
    store_with_pointer_offset(frag, ref_.offset(tile_offset) + pointer_offset);
  }
};

////////////////////////////////////////////////////////////////////////////////

/// This tile iterator is specialized for 32-thread TensorOps. It is used to load or store
/// accumulators from memory and is agnostic to layout. It could be faster if it assumed row-major
/// accumulator layout.
///
/// Satisfies:
///   ReadableRandomAccessContiguousTileIteratorConcept |
///   WriteableRandomAccessContiguousTileIteratorConcept
///

template <
    /// Size of the matrix to load (concept: MatrixShape)
    typename Shape_,
    /// Element typ
    typename Element_,
    /// Shape of one matrix product operation (concept: MatrixShape)
    typename InstructionShape_,
    /// Interval between adjacent *MMA instructions (in units of MMA
    /// instructions, concept: MatrixShape)
    typename OpDelta_,
    /// Interleaved N
    int InterleavedN>
class MmaTensorOpAccumulatorTileIterator<
    Shape_, Element_, mctlass::layout::TensorNCxHWx<InterleavedN>,
    InstructionShape_, OpDelta_> {
 public:

  /// Shape of tile to load (concept: MatrixShape)
  using Shape = Shape_;

  /// Operand tag
  static Operand const kOperand = Operand::kC;

  /// Element type
  using Element = int8_t;

  /// Layout of source tile
  using Layout = mctlass::layout::TensorNCxHWx<InterleavedN>;

  /// Shape of one matrix product operation (concept: MatrixShape)
  using InstructionShape = InstructionShape_;

  /// Delta between *MMA operations (in units of *MMA operations, concept: MatrixShape)
  using OpDelta = OpDelta_;

  /// Number of participating threads
  static int const kThreads = 32;

  /// TensorRef type for loading element from a tensor
  using TensorRef = TensorRef<Element, Layout>;

  /// Index type
  using Index = typename TensorRef::Index;

  /// Long Index type
  using LongIndex = typename TensorRef::LongIndex;

  /// Long Index type
  using StrideIndex = typename TensorRef::Layout::Stride::Index;

  /// Coordinate for an element in the tensor
  using TensorCoord = typename TensorRef::TensorCoord;

  /// Internal structure of iterator - made public to enable introspection
  struct Policy {
    static_assert(
        !(Shape::kRow % InstructionShape::kM) &&
            !(Shape::kColumn % InstructionShape::kN),
        "Shape of warp-level Mma must be divisible by operator shape.");

    /// Number of elements in strided dimension that each STG writes
    static int const kStridedPerSTG = 8;

    /// Factor to calculate reorder index to pack accumulator.
    static int const kPackedFactor = Shape::kColumn / 32;

    /// Number of mma operations performed
    using MmaIterations = MatrixShape<Shape::kRow / kStridedPerSTG,
                                      Shape::kColumn / InterleavedN>;
  };

private:

  static int const kElementsPerAccess = InterleavedN / 4;

public:

  //
  // Derived quantities
  //

  struct alignas((kElementsPerAccess * sizeof_bits<Element>::value / 8)) AccessType {
      Array<Element, kElementsPerAccess> storage;
  };

  /// Fragment object holding a thread's part of a tile
  using Fragment = Array<int32_t, Shape::kCount / kThreads>;

private:

  /// Reference to output tensor
  TensorRef ref_;

  /// Row offset index globally
  LongIndex global_offset_row_;

  /// Column offset index globally
  LongIndex global_offset_col_;

  /// Output tensor size
  TensorCoord extent_;

  /// Alpha
  float alpha_;

  /// Beta
  float beta_;

public:

  /// Default ctor constructs null iterator
  MCTLASS_HOST_DEVICE
  MmaTensorOpAccumulatorTileIterator() { }

  /// Constructor from TensorRef
  MCTLASS_HOST_DEVICE
  MmaTensorOpAccumulatorTileIterator(
    TensorRef const &ref,
    int const lane_id,
    TensorCoord extent,
    float alpha = 1.0f,
    float beta = 0.0f
  ):
    ref_(ref),
    extent_(extent),
    alpha_(alpha),
    beta_(beta) {

    int quad = (lane_id >> 2);
    int lane_in_quad = (lane_id & 3);

    global_offset_row_ = quad;

    global_offset_col_ = lane_in_quad * kElementsPerAccess;
  }

  /// Adds a pointer offset to internal pointer(s) to advance through memory
  MCTLASS_HOST_DEVICE
  MmaTensorOpAccumulatorTileIterator &add_pointer_offset(LongIndex offset) {
    ref_.add_pointer_offset(offset);
    return *this;
  }

  /// Advances an iterator along logical dimensions of matrix in units of whole tiles
  MCTLASS_HOST_DEVICE
  MmaTensorOpAccumulatorTileIterator &add_tile_offset(MatrixCoord const &tile_offset) {

    global_offset_row_ += tile_offset.row() * Shape::kRow;

    global_offset_col_ += tile_offset.column() * Shape::kColumn;

    return *this;
  }

  /// Advances the iterator along the advance dimension
  MCTLASS_HOST_DEVICE
  MmaTensorOpAccumulatorTileIterator & operator++() {
    // deliberate no-op
    return *this;
  }

  /// Advances the iterator along the advance dimension
  MCTLASS_HOST_DEVICE
  MmaTensorOpAccumulatorTileIterator & operator--() {
    // deliberate no-op
    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of the tensor
  MCTLASS_DEVICE
  MmaTensorOpAccumulatorTileIterator & operator+=(TensorCoord const &tile_offset) {
    add_tile_offset(tile_offset);
    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of the tensor
  MCTLASS_DEVICE
  MmaTensorOpAccumulatorTileIterator & operator-=(TensorCoord const &tile_offset) {
    add_tile_offset(-tile_offset);
    return *this;
  }

  /// Loads a fragment from memory at the location pointed to by the iterator.
  MCTLASS_HOST_DEVICE
  void load(Fragment &frag) const {
    load_with_pointer_offset(frag);
  }

  /// Loads a fragment from memory with additional logical offset
  MCTLASS_DEVICE
  void load_with_pointer_offset(
    Fragment &frag,                             ///< fragment to load from the tensor
    Index pointer_offset) const {               ///< loads a tile with a linear offset

    TensorRef offset_ref(ref_);
    offset_ref.add_pointer_offset(pointer_offset);

    AccessType* frag_ptr = reinterpret_cast<AccessType *>(&frag);

    MCTLASS_PRAGMA_UNROLL
    for (int mma_n = 0; mma_n < Policy::MmaIterations::kN; ++mma_n) {
      MCTLASS_PRAGMA_UNROLL
      for (int mma_m = 0; mma_m < Policy::MmaIterations::kM; ++mma_m) {
        int accum_m = mma_m * InstructionShape::kM;
        int accum_n = mma_n * InstructionShape::kN;

        int idx = mma_m + mma_n * Policy::MmaIterations::kM;

        AccessType* access_ptr = reinterpret_cast<AccessType *>(offset_ref.data() +
                                 accum_m * offset_ref.stride(0) + accum_n);

        frag_ptr[idx] = access_ptr[0];
      }
    }
  }

  /// Loads a fragment from memory with additional logical offset
  MCTLASS_DEVICE
  void load_with_byte_offset(
    Fragment &frag,                             ///< fragment to load from the tensor
    Index byte_offset) const {                  ///< loads a tile with a linear offset

    load_with_pointer_offset(byte_offset / sizeof(Element));
  }

  /// Loads a fragment from memory with logical offset in units of whole tiles.
  MCTLASS_DEVICE
  void load(
    Fragment &frag,                             ///< fragment to load from the tensor
    TensorCoord const &tile_offset) const {     ///< loads a tile with a logical offset in units of whole tiles

    load(frag, tile_offset, 0);
  }

  /// Loads a fragment from memory with logical offset in units of whole tiles.
  MCTLASS_DEVICE
  void load(
    Fragment &frag,                             ///< fragment to load from the tensor
    TensorCoord const &tile_offset,             ///< loads a tile with a logical offset in units of whole tiles
    Index pointer_offset) const {               ///< loads a tile with a logical offset AND a pointer offset

    load_with_pointer_offset(frag, ref_.offset(tile_offset) + pointer_offset);
  }

  /// Stores a fragment to memory
  MCTLASS_HOST_DEVICE
  void store(Fragment const &frag) const {
    store_with_pointer_offset(frag, 0);
  }

  /// Stores a fragment to memory with additional pointer offset
  MCTLASS_DEVICE
  void store_with_pointer_offset(
    Fragment const &frag,                       ///< fragment to store from the tensor
    Index pointer_offset) const {               ///< store a tile with a linear offset

    TensorRef offset_ref(ref_);
    offset_ref.add_pointer_offset(pointer_offset);

    Array<float, Shape::kCount / kThreads> output_frag_f;
    Array<Element, Shape::kCount / kThreads> output_frag;

    LongIndex pq = extent_.h() * extent_.w();

    LongIndex extent_row = extent_.n() * pq;
    LongIndex extent_col = extent_.c();

    LongIndex k_major = (global_offset_col_ / InterleavedN) * pq;
    Index k_minor = global_offset_col_ % InterleavedN;
    LongIndex k_offset = k_major * InterleavedN + k_minor;
    LongIndex k_offset_delta = pq * InterleavedN;

    LongIndex stride_n = pq * extent_.c();

    Index n;
    LongIndex pq_rem;

    unsigned int pq_mul, pq_shr;
    find_divisor(pq_mul, pq_shr, pq);

    if(beta_ == 0.0f) {
      MCTLASS_PRAGMA_UNROLL
      for(int i = 0; i < frag.size(); ++i) {
        output_frag_f[i] = frag[i];
      }

      if(InstructionShape::kM == Policy::kStridedPerSTG) {
        MCTLASS_PRAGMA_UNROLL
        for(int i = 0; i < frag.size(); ++i) {
          output_frag[i] = (Element)(output_frag_f[i] * alpha_);
        }
      } else {
        MCTLASS_PRAGMA_UNROLL
        for(int i = 0; i < frag.size(); ++i) {
          int map_i = (i / (16 * Policy::kPackedFactor)) * (16 * Policy::kPackedFactor)
                    + (i % (8 * Policy::kPackedFactor)) / 2 * 4
                    + (i % (8 * Policy::kPackedFactor)) % 2
                    + (i / (8 * Policy::kPackedFactor)) % 2 * 2;
          output_frag[i] = (Element)(output_frag_f[map_i] * alpha_);
        }
      }

      AccessType const *frag_ptr = reinterpret_cast<AccessType const*>(&output_frag);

      MCTLASS_PRAGMA_UNROLL
      for (int mma_m = 0; mma_m < Policy::MmaIterations::kRow; ++mma_m) {
        int accum_m = mma_m * Policy::kStridedPerSTG;

        fast_divmod(n, pq_rem, global_offset_row_ + accum_m, pq, pq_mul, pq_shr);
        LongIndex offset_m = n * stride_n + k_offset + pq_rem * InterleavedN;

        MCTLASS_PRAGMA_UNROLL
        for (int mma_n = 0; mma_n < Policy::MmaIterations::kColumn; ++mma_n) {

          int accum_n = mma_n * InterleavedN;

          int idx = mma_n + mma_m * Policy::MmaIterations::kColumn;

          if((global_offset_row_ + accum_m < extent_row) && (global_offset_col_ + accum_n < extent_col)) {
            AccessType* access_ptr = reinterpret_cast<AccessType *>(offset_ref.data() +
                                                                    offset_m + mma_n * k_offset_delta);

            access_ptr[0] = frag_ptr[idx];
          }
        }
      }
    } else {
      if(InstructionShape::kM == Policy::kStridedPerSTG) {
        MCTLASS_PRAGMA_UNROLL
        for(int i = 0; i < frag.size(); ++i) {
          output_frag_f[i] = frag[i];
        }
      } else {
        MCTLASS_PRAGMA_UNROLL
        for(int i = 0; i < frag.size(); ++i) {
          int map_i = (i / (16 * Policy::kPackedFactor)) * (16 * Policy::kPackedFactor)
                    + (i % (8 * Policy::kPackedFactor)) / 2 * 4
                    + (i % (8 * Policy::kPackedFactor)) % 2
                    + (i / (8 * Policy::kPackedFactor)) % 2 * 2;
          output_frag_f[i] = frag[map_i];
        }
      }

      AccessType const *frag_ptr = reinterpret_cast<AccessType const*>(&output_frag);

      Array<Element, kElementsPerAccess> ref_frag;
      AccessType *ref_frag_ptr = reinterpret_cast<AccessType *>(&ref_frag);

      MCTLASS_PRAGMA_UNROLL
      for (int mma_m = 0; mma_m < Policy::MmaIterations::kRow; ++mma_m) {
        int accum_m = mma_m * Policy::kStridedPerSTG;

        fast_divmod(n, pq_rem, global_offset_row_ + accum_m, pq, pq_mul, pq_shr);
        LongIndex offset_m = n * stride_n + k_offset + pq_rem * InterleavedN;

        MCTLASS_PRAGMA_UNROLL
        for (int mma_n = 0; mma_n < Policy::MmaIterations::kColumn; ++mma_n) {

          int accum_n = mma_n * InterleavedN;

          int idx = mma_n + mma_m * Policy::MmaIterations::kColumn;

          if((global_offset_row_ + accum_m < extent_row) && (global_offset_col_ + accum_n < extent_col)) {
            AccessType* access_ptr = reinterpret_cast<AccessType *>(offset_ref.data() +
                                                                    offset_m + mma_n * k_offset_delta);

            ref_frag_ptr[0] = access_ptr[0];

            MCTLASS_PRAGMA_UNROLL
            for(int i = 0; i < kElementsPerAccess; ++i) {
              output_frag[idx * kElementsPerAccess + i] = Element(alpha_ * output_frag_f[idx * kElementsPerAccess + i]
                                                                + beta_ * ref_frag[i]);
            }

            access_ptr[0] = frag_ptr[idx];
          }
        }
      }
    }
  }

  /// Stores a fragment to memory with additional pointer offset
  MCTLASS_DEVICE
  void store_with_byte_offset(
    Fragment const &frag,                       ///< fragment to store from the tensor
    Index byte_offset) const {                  ///< store a tile with a linear offset

    store_with_pointer_offset(byte_offset / sizeof(Element));
  }

  /// Stores a fragment to memory with logical offset in units of whole tiles.
  MCTLASS_DEVICE
  void store(
    Fragment &frag,                             ///< fragment to store to the tensor
    TensorCoord const &tile_offset) const {     ///< stores a tile with a logical offset in units of whole tiles

    store(frag, tile_offset, 0);
  }

  /// Stores a fragment from memory with logical offset in units of whole tiles.
  MCTLASS_DEVICE
  void store(
      /// fragment to store to the tensor
      Fragment const &frag,
      /// stores a tile with a logical offset in units of whole tiles
      TensorCoord const &tile_offset,
      /// stores a tile with a logical offset AND a pointer offset
      Index pointer_offset) const {
    store_with_pointer_offset(frag, ref_.offset(tile_offset) + pointer_offset);
  }
};

template <
    /// Size of the matrix to load (concept: PitchLinearShape)
    typename Shape_,
    /// Identifies A or B multiplicand
    Operand Operand_,
    /// Shape of one matrix product operation (concept: PitchLinearShape)
    typename InstructionShape_,
    /// Interval between adjacent *MMA instructions (in units of MMA
    /// instructions)
    int OpDelta_,
    /// Number of partitions along K dimension
    int PartitionsK_>
class MmaTensorOpMultiplicandTileIterator<
    Shape_, Operand_, half_t,
    mctlass::layout::TensorOpMultiplicandCongruous<sizeof_bits<half_t>::value,
                                                   64>,
    InstructionShape_, OpDelta_, 32, PartitionsK_> {
 public:

  /// Shape of tile to load (concept: PitchLinearShape)
  using Shape = Shape_;

  /// Operand tag
  static Operand const kOperand = Operand_;

  static_assert(kOperand == Operand::kA || kOperand== Operand::kB,
    "MmaTensorOpMultiplicandIterator may only be instantiated for A or B operands to warp-level Mma.");

  /// Element type
  using Element = half_t;

  /// Layout of source tile
  using Layout = mctlass::layout::TensorOpMultiplicandCongruous<
      sizeof_bits<Element>::value, 64>;

  /// Shape of one matrix product operation (concept: GemmShape)
  using InstructionShape = InstructionShape_;

  /// Delta between *MMA operations (in units of *MMA operations, concept: MatrixShape)
  static int const kOpDelta = OpDelta_;

  /// Number of participating threads
  static int const kThreads = 32;

  /// Number of partitions along K dimension
  static int const kPartitionsK = PartitionsK_;

  /// TensorRef type for loading element from a tensor
  using TensorRef = TensorRef<Element, Layout>;

  /// Index type
  using Index = typename TensorRef::Index;

  /// Long Index type
  using LongIndex = typename TensorRef::LongIndex;

  /// Long Index type
  using StrideIndex = typename TensorRef::Layout::Stride::Index;

  /// Coordinate for an element in the tensor
  using TensorCoord = typename TensorRef::TensorCoord;

  /// Internal structure of iterator - made public to enable introspection
  struct Policy {
    static_assert(
        !(Shape::kContiguous % InstructionShape::kContiguous),
        "Shape of warp-level Mma must be divisible by operator shape.");

    // Determine number of elements along outer dimension per individual LDSM op
    static int const kLdsmOpOuter_16x8x16 = Layout::kElementsPerAccess;
    //static int const kLdsmOpInner = 8;
    static int const kLdsmOpInner_16x8x16 = kOperand == Operand::kA ? 8 : 4;

    static int const kLdsmOpInner_16x8x8 = Layout::TileShape::kStrided;
    static int const kLdsmOpOuter_16x8x8 = kThreads / kLdsmOpInner_16x8x8;

    static_assert(!(Shape::kContiguous % kLdsmOpOuter_16x8x16),
      "Shape of warp-level mma must be divisible by LDSM's fundamental tile size.");

    static_assert(!(Shape::kStrided % kLdsmOpInner_16x8x16),
      "Shape of warp-level mma must be divisible by LDSM's fundamental tile size.");

    static_assert(!(Shape::kContiguous % kLdsmOpOuter_16x8x8),
      "Shape of warp-level mma must be divisible by LDSM's fundamental tile size.");

    static_assert(!(Shape::kStrided % kLdsmOpInner_16x8x8),
      "Shape of warp-level mma must be divisible by LDSM's fundamental tile size.");

    /// Shape of one individual LDSM instruction
    static int const LdsmShapeStrided_16x8x16 =
        InstructionShape::kStrided / kLdsmOpInner_16x8x16;
    static int const LdsmShapeContiguous_16x8x16 = 4 / LdsmShapeStrided_16x8x16;
    static int const LdsmShapeContiguous_16x8x8 =
        InstructionShape::kContiguous / kLdsmOpOuter_16x8x8;
    static int const LdsmShapeStrided_16x8x8 = InstructionShape::kStrided / kLdsmOpInner_16x8x8;

    using LdsmShape = typename platform::conditional<InstructionShape::kStrided == 8,
                               layout::PitchLinearShape<LdsmShapeContiguous_16x8x8, LdsmShapeStrided_16x8x8>,
                               layout::PitchLinearShape<LdsmShapeContiguous_16x8x16, LdsmShapeStrided_16x8x16>>::type;

    using LdsmIterations = typename platform::conditional<InstructionShape::kStrided == 8,
          layout::PitchLinearShape<Shape::kContiguous / LdsmShapeContiguous_16x8x8 / kLdsmOpOuter_16x8x8, 1>,
          layout::PitchLinearShape<Shape::kContiguous / Layout::kElementsPerAccess / LdsmShapeContiguous_16x8x16, 1>>::type;

    /// Number of groups for each tile
    static int const kGroupsPerTile =
        Shape::kStrided / InstructionShape::kStrided;
  };

private:

  /// Not working on this feature at the moment.
  static_assert(kOpDelta == 1,
    "Alternative arrangements not supported at present.");

  /// Number of internal pointers needed to reference shared memory
  static int const kPointerCount = Layout::TileShape::kContiguous *
                                   Layout::kElementsPerAccess /
                                   Policy::kLdsmOpOuter_16x8x8;

  /// Pointer type used for accesses
  using AccessType = typename platform::conditional<InstructionShape::kStrided == 8,
                     Element, Array<Element, Layout::kElementsPerAccess>>::type;
   /// Vectorized access is not used
  static int const kElementsPerAccess = 1;

  /// Internal counter used to jump to next K partition
  int k_group_idx_;

public:

  //
  // Derived quantities
  //

  /// Fragment object holding a thread's part of a tile
 using Fragment =
     Array<Element, Shape::kContiguous * InstructionShape::kStrided / kThreads>;

private:

  /// Layout object storing stride values
  StrideIndex stride_;

  /// Shared memory base pointers - not advanced
  AccessType const *pointer_[kPointerCount];

  /// Byte offset incremented as iterator advances
  Index byte_offset_;
  Layout layout_;
  Index tile_row_offset;
  Index tile_col_offset;

public:

  /// Default ctor constructs null iterator
  MCTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator(): stride_(0), byte_offset_(0), tile_row_offset(0), tile_col_offset(0) { }

  /// Constructor from TensorRef
  MCTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator(
    TensorRef const &ref,
    int lane_id
  ):
    stride_(ref.stride(0)), byte_offset_(0),
    k_group_idx_(0) {

    layout_ = ref.layout();
    pointer_[0] = reinterpret_cast<AccessType const *>(ref.data());
    tile_row_offset = 0;
    tile_col_offset = 0;
    if (InstructionShape::kStrided != 8) {
      stride_ /= Layout::kElementsPerAccess;
    }
  }

  /// Adds a pointer offset to internal pointer(s) to advance through memory
  MCTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator &add_pointer_offset(LongIndex offset) {

    byte_offset_ += offset * sizeof(Element);

    return *this;
  }

  /// Advances an iterator along logical dimensions of matrix in units of whole tiles
  MCTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator &add_tile_offset(TensorCoord const &tile_offset) {
    if (InstructionShape::kStrided == 8) {
      int contiguous_offset = tile_offset.contiguous();
      if (Shape::kContiguous ==
          Layout::TileShape::kContiguous * Layout::kElementsPerAccess / 2) {
        if (tile_offset.contiguous() % 2) {
          MCTLASS_PRAGMA_UNROLL
          for (int i = 0; i < kPointerCount / 2; ++i) {
            AccessType const *tmp_pointer = pointer_[i];
            pointer_[i] = pointer_[i + kPointerCount / 2];
            pointer_[i + kPointerCount / 2] = tmp_pointer;
          }
        }
        contiguous_offset = (tile_offset.contiguous() >> 1) << 1;
      }
      int offset = (tile_offset.strided() * InstructionShape::kStrided) * stride_ +
                   contiguous_offset * Shape::kContiguous;
      add_pointer_offset(offset);
    }
    else {
      tile_row_offset += tile_offset.contiguous() * Shape::kContiguous;
      tile_col_offset += tile_offset.strided() * InstructionShape::kStrided;
    }
    return *this;

  }

  /// Advances the iterator along the advance dimension
  MCTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator & operator++() {
    if (InstructionShape::kStrided == 8) {
      add_tile_offset({0, 1});
      if (kPartitionsK > 1) {
        ++k_group_idx_;
        // Jump to next stage
        if (k_group_idx_ == Policy::kGroupsPerTile) {
          k_group_idx_ = 0;
          add_tile_offset(
              {0, ((kPartitionsK - 1) * Policy::kGroupsPerTile)});
        }
      }
    }
    else {
      //For RowMajorXXX, the Matrices was transposed,so for Operand::kA and Operand::kB can use same variable.
      byte_offset_ += InstructionShape::kStrided;
    }
    return *this;
  }

  /// Advances the iterator along the opposite of the advance dimension
  MCTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator & operator--() {
    if (InstructionShape::kStrided == 8) {
      byte_offset_ -= stride_ * InstructionShape::kStrided * sizeof(Element) *
                      kElementsPerAccess;
    }
    else {
      byte_offset_ -= stride_ * InstructionShape::kStrided * sizeof(Element) *
                      Layout::kElementsPerAccess;
    }
    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of the tensor
  MCTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator & operator+=(TensorCoord const &tile_offset) {
    add_tile_offset(tile_offset);
    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of the tensor
  MCTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator & operator-=(TensorCoord const &tile_offset) {
    add_tile_offset(-tile_offset);
    return *this;
  }

  /// Loads a fragment from memory at the location pointed to by the iterator.
  MCTLASS_HOST_DEVICE
  void load(Fragment &frag) const {

    load_with_byte_offset(frag, 0);
  }

  /// Loads a fragment from memory with additional logical offset
  MCTLASS_DEVICE
  void load_with_byte_offset(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a linear offset in units of bytes
      Index byte_offset) const {

    // Array<unsigned, Policy::LdsmShape::kCount> *fetch_ptr =
    //   reinterpret_cast<Array<unsigned, Policy::LdsmShape::kCount> *>(&frag);
    Element *fetch_ptr = reinterpret_cast<Element *>(&frag);
    const size_t fetch_size = kOperand == Operand::kA ? Policy::LdsmShape::kCount * 2 : Policy::LdsmShape::kCount;
    Array<Element, fetch_size> *fetch_element_ptr = reinterpret_cast<Array<Element, fetch_size> *>(&frag);

    const int lane_id = __lane_id();
    MCTLASS_PRAGMA_UNROLL
    for (int s = 0; s < Policy::LdsmIterations::kStrided; ++s) {

      MCTLASS_PRAGMA_UNROLL
      for (int c = 0; c < Policy::LdsmIterations::kContiguous; ++c) {

        int access_idx = c + s * Policy::LdsmIterations::kContiguous;

        /* AccessType const *source_ptr =
            pointer_[c % kPointerCount] +
            Layout::TileShape::kContiguous * (c / kPointerCount) +
            Policy::kLdsmOpInner * Policy::LdsmShape::kStrided * s * stride_;

        char const *source_byte_ptr = reinterpret_cast<char const *>(source_ptr) + byte_offset + byte_offset_;
        mctlass::arch::ldsm<layout::ColumnMajor, Policy::LdsmShape::kCount>(
          fetch_ptr[access_idx],
          source_byte_ptr
        ); */
        if (InstructionShape::kStrided == 8) {
          if(kOperand == Operand::kA) {
            int row = (lane_id & 0xf) + c * InstructionShape::kContiguous;
            int col = ((lane_id >> 4) << 1) ^ 0x7;
            for (int i=0; i < 2; i++) {
              Index idx = layout_(MatrixCoord(row, col - i));
              AccessType const *source_ptr = pointer_[0] + idx;
              char const *source_byte_ptr = reinterpret_cast<char const *>(source_ptr) +
                                          byte_offset + byte_offset_;
              fetch_ptr[c * 4 + i] = *reinterpret_cast<Element const *>(source_byte_ptr);
            }
          }
          else {
            int col = ((lane_id >> 4) << 1) ^ 0x7;
            int row = (lane_id & 0x7) + c * InstructionShape::kStrided;
            for (int i=0; i < 2; i++) {
              Index idx = layout_(MatrixCoord(row, col - i));
              AccessType const *source_ptr = pointer_[0] + idx;
              char const *source_byte_ptr = reinterpret_cast<char const *>(source_ptr) +
                                          byte_offset + byte_offset_;
              fetch_ptr[c * 2 + i] = *reinterpret_cast<Element const *>(source_byte_ptr);
            }
          }
        }
        else {
          int row = (lane_id & (InstructionShape::kContiguous - 1)) + c * InstructionShape::kContiguous + tile_row_offset;
          int col = (((lane_id >> 4) << 2) ^ (InstructionShape::kStrided - 1)) + s * InstructionShape::kStrided + byte_offset_ + tile_col_offset;
          Element const *ptr = reinterpret_cast<Element const *>(pointer_[0]) + byte_offset;
          for (int i = 0; i < Policy::LdsmShape::kCount; ++i) {
            Index idx = layout_(MatrixCoord(row, col - i));
            fetch_element_ptr[access_idx][i] = ptr[idx];
          }
        }
      }
    }
  }

  /// Loads a fragment from memory with additional logical offset
  MCTLASS_DEVICE
  void load_with_pointer_offset(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a linear offset
      Index pointer_offset) const {
    //load_with_byte_offset(frag, pointer_offset * sizeof(Element));
    load_with_byte_offset(frag, pointer_offset);
  }

  /// Loads a fragment from memory with logical offset in units of whole tiles.
  MCTLASS_DEVICE
  void load(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a logical offset in units of whole tiles
      TensorCoord const &tile_offset) const {
    load_with_byte_offset(frag, tile_offset, 0);
  }

  /// Loads a fragment from memory with logical offset in units of whole tiles.
  MCTLASS_DEVICE
  void load(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a logical offset in units of whole tiles
      TensorCoord const &tile_offset,
      /// loads a tile with a logical offset AND a pointer offset
      Index pointer_offset) const {
    load_with_byte_offset(frag, tile_offset, pointer_offset * sizeof(Element));
  }

  /// Loads a fragment from memory with logical offset in units of whole tiles.
  MCTLASS_DEVICE
  void load_with_byte_offset(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a logical offset in units of whole tiles
      TensorCoord const &tile_offset,
      /// loads a tile with a logical offset AND a pointer offset
      Index byte_offset) const {
    Index pointer_offset =
      tile_offset.contiguous() * Shape::kContiguous / Layout::kElementsPerAccess +
      tile_offset.strided() * InstructionShape::kStrided * stride_;

    byte_offset += sizeof(AccessType) * pointer_offset;

    load_with_byte_offset(frag, byte_offset);
  }

  /// Notify the iterator which k-group it is currently pointing to.
  ///
  /// This does not advance the iterator. Rather, it overrides its internal
  /// tracking with constant-valued k-group index to enable the compiler to
  /// fold constants and achieve more efficient code.
  ///
  /// This is used by some nontrivial permuted layouts.
  MCTLASS_DEVICE
  void set_kgroup_index(int k_group) {
    // no op
  }
};

template <
    /// Size of the matrix to load (concept: PitchLinearShape)
    typename Shape_,
    /// Identifies A or B multiplicand
    Operand Operand_,
    /// Shape of one matrix product operation (concept: PitchLinearShape)
    typename InstructionShape_,
    /// Interval between adjacent *MMA instructions (in units of MMA
    /// instructions)
    int OpDelta_,
    /// Element number when the layout crosses (in units of elements)
    int Crosswise,
    /// Number of partitions along K dimension
    int PartitionsK_>
class MmaTensorOpMultiplicandTileIterator<
    Shape_, Operand_, half_t,
    mctlass::layout::TensorOpMultiplicandCrosswise<sizeof_bits<half_t>::value,
                                                   Crosswise>,
    InstructionShape_, OpDelta_, 32, PartitionsK_> {
 public:
  /// Shape of tile to load (concept: PitchLinearShape)
  using Shape = Shape_;

  /// Operand tag
  static Operand const kOperand = Operand_;

  static_assert(kOperand == Operand::kA || kOperand == Operand::kB,
                "MmaTensorOpMultiplicandIterator may only be instantiated for "
                "A or B operands to warp-level Mma.");

  /// Element type
  using Element = half_t;

  /// Element number when the layout crosses
  static int const kCrosswise = Crosswise;

  /// Layout of source tile
  using Layout = mctlass::layout::TensorOpMultiplicandCrosswise<
      sizeof_bits<Element>::value, kCrosswise>;

  /// Shape of one matrix product operation (concept: GemmShape)
  using InstructionShape = InstructionShape_;

  /// Delta between *MMA operations (in units of *MMA operations, concept:
  /// MatrixShape)
  static int const kOpDelta = OpDelta_;

  /// Number of participating threads
  static int const kThreads = 32;

  /// Number of partitions along K dimension
  static int const kPartitionsK = PartitionsK_;

  /// TensorRef type for loading element from a tensor
  using TensorRef = TensorRef<Element, Layout>;

  /// Index type
  using Index = typename TensorRef::Index;

  /// Long Index type
  using LongIndex = typename TensorRef::LongIndex;

  /// Long Index type
  using StrideIndex = typename TensorRef::Layout::Stride::Index;

  /// Coordinate for an element in the tensor
  using TensorCoord = typename TensorRef::TensorCoord;

  /// Internal structure of iterator - made public to enable introspection
  struct Policy {
    static_assert(
        !(Shape::kContiguous % InstructionShape::kContiguous),
        "Shape of warp-level Mma must be divisible by operator shape.");

    // Determine number of elements along outer dimension per individual LDSM op
    static int const kLdsmOpOuter = Layout::kElementsPerAccess;
    static int const kLdsmOpInner_16x8x16 = kOperand == Operand::kA ? 8 : 4;
    static int const kLdsmOpInner_16x8x8 = 4;

    static_assert(!(Shape::kContiguous % kLdsmOpOuter),
                  "Shape of warp-level mma must be divisible by LDSM's "
                  "fundamental tile size.");

    static_assert(!(Shape::kStrided % kLdsmOpInner_16x8x16),
                  "Shape of warp-level mma must be divisible by LDSM's "
                  "fundamental tile size.");

    static_assert(!(Shape::kStrided % kLdsmOpInner_16x8x8),
                  "Shape of warp-level mma must be divisible by LDSM's "
                  "fundamental tile size.");

    /// Shape of one individual LDSM instruction
    static int const LdsmShapeContiguous =
        InstructionShape::kContiguous / kLdsmOpOuter;
    static int const LdsmShapeStrided_16x8x16 =
        ((4 / LdsmShapeContiguous * kLdsmOpInner_16x8x16) > Shape::kStrided)
            ? (Shape::kStrided / kLdsmOpInner_16x8x16)
            : (4 / LdsmShapeContiguous);
    static int const LdsmShapeStrided_16x8x8 =
        ((4 / LdsmShapeContiguous * kLdsmOpInner_16x8x8) > Shape::kStrided)
            ? (Shape::kStrided / kLdsmOpInner_16x8x8)
            : (4 / LdsmShapeContiguous);
    using LdsmShape = typename platform::conditional<InstructionShape::kContiguous == 8,
                               layout::PitchLinearShape<LdsmShapeContiguous, LdsmShapeStrided_16x8x8>,
                               layout::PitchLinearShape<LdsmShapeContiguous, LdsmShapeStrided_16x8x16>>::type;
    /// Number and arrangement of LDSM instructions
    using LdsmIterations = typename platform::conditional<InstructionShape::kContiguous == 8,
          layout::PitchLinearShape<1, Shape::kStrided / kLdsmOpInner_16x8x8 / LdsmShape::kStrided>,
          layout::PitchLinearShape<1, Shape::kStrided / kLdsmOpInner_16x8x16 / LdsmShape::kStrided>>::type;

    ///
    static int const kGroupsPerTile = Layout::TileShape::kContiguous /
                                      Layout::kFactor / LdsmShape::kContiguous;
  };

 private:
  /// Not working on this feature at the moment.
  static_assert(kOpDelta == 1,
                "Alternative arrangements not supported at present.");

  /// Pointer type used for accesses
  using AccessType = Array<Element, Layout::kElementsPerAccess>;

 public:
  //
  // Derived quantities
  //

  /// Fragment object holding a thread's part of a tile
  using Fragment = Array<Element, Shape::kStrided *
                                      InstructionShape::kContiguous / kThreads>;

 private:

  /// Total number of sections.  The memory is divided into stages.  One stage
  /// can store one tile.  Stage is divided into sections.  Interleaved layout
  /// can have multiple sections in a stage.  The rest layout only has one section
  /// in a stage.
  int sections_;

  /// Layout object storing stride values
  StrideIndex stride_;

  /// Shared memory base pointers - not advanced
  AccessType const *pointer_;

  /// Byte offset incremented as iterator advances
  Index byte_offset_;

  /// Internal counter used to determine when to increment byte offset and when
  /// to XOR it
  int k_group_idx_;

  Layout layout_;

 public:
  /// Default ctor constructs null iterator
  MCTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator()
      : pointer_(nullptr),
        sections_(0),
        stride_(0),
        byte_offset_(0),
        k_group_idx_(0) {}

  /// Constructor from TensorRef
  MCTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator(TensorRef const &ref, int lane_id)
      : pointer_(reinterpret_cast<AccessType const *>(ref.data())),
        sections_(ref.stride(0) / kCrosswise),
        // stride_ = kCrosswise x sections_ x kFactor
        stride_(ref.stride(0) * Layout::kFactor / Layout::kElementsPerAccess),
        byte_offset_(0),
        k_group_idx_(0) {

    layout_ = ref.layout();
    byte_offset_ = 0;
    if (InstructionShape::kContiguous == 8) {
      int row, col;
      if (kOperand == Operand::kA) {
        row = lane_id & 0xf;
        col = ((lane_id >> 4) << 1) ^ 0x7;
      }
      else {
        row = lane_id & 0x7;
        col = ((lane_id >> 4) << 1) ^ 0x7;
      }
      TensorCoord coord = make_Coord(col, row);
      Layout lo(ref.stride(0));
      byte_offset_ =  lo(coord) * sizeof_bits<Element>::value / 8;
    }
  }

  /// Adds a pointer offset to internal pointer(s) to advance through memory
  MCTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator &add_pointer_offset(LongIndex offset) {
    byte_offset_ += offset * sizeof_bits<Element>::value / 8;

    return *this;
  }

  /// Advances an iterator along logical dimensions of matrix in units of whole
  /// tiles
  MCTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator &add_tile_offset(
      TensorCoord const &tile_offset) {
    int whole_tiles = tile_offset.contiguous() / Policy::kGroupsPerTile;
    int k_groups_delta = tile_offset.contiguous() % Policy::kGroupsPerTile;

    byte_offset_ ^= k_groups_delta * sizeof_bits<Element>::value *
                    Layout::kElementsPerAccess *
                    Policy::LdsmShape::kContiguous / 8;
    pointer_ +=
        tile_offset.strided() * stride_ * Shape::kStrided / Layout::kFactor +
        whole_tiles * stride_ / sections_;
    return *this;
  }

  /// Advances an iterator along logical dimensions of matrix in units of whole
  /// tiles
  MCTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator &add_tile_offset_negative(
      TensorCoord const &tile_offset) {

    int whole_tiles = tile_offset.contiguous() / Policy::kGroupsPerTile;
    int k_groups_delta = tile_offset.contiguous() % Policy::kGroupsPerTile;
    if (k_groups_delta < 0) {
        whole_tiles -= 1;
        k_groups_delta += Policy::kGroupsPerTile;
    }

    if ((Policy::kGroupsPerTile / kPartitionsK) >= 2) {
      byte_offset_ ^= (k_groups_delta & 1) * Policy::LdsmShape::kContiguous *
                        sizeof_bits<Element>::value *
                        Layout::kElementsPerAccess / 8;
    }
    if ((Policy::kGroupsPerTile / kPartitionsK) >= 4) {
      byte_offset_ ^= ((k_groups_delta + (k_group_idx_ & 1)) & 2) *
                        Policy::LdsmShape::kContiguous *
                        sizeof_bits<Element>::value *
                        Layout::kElementsPerAccess / 8;
    }
    if ((Policy::kGroupsPerTile / kPartitionsK) == 8) {
      byte_offset_ ^= ((k_groups_delta + (k_group_idx_ & 3)) & 4) *
                        Policy::LdsmShape::kContiguous *
                        sizeof_bits<Element>::value *
                        Layout::kElementsPerAccess / 8;
    }

    k_group_idx_ += k_groups_delta;
    whole_tiles += k_group_idx_ / (Policy::kGroupsPerTile / kPartitionsK);
    k_group_idx_ = k_group_idx_ % (Policy::kGroupsPerTile / kPartitionsK);

    pointer_ +=
        tile_offset.strided() * stride_ * Shape::kStrided / Layout::kFactor +
        whole_tiles * stride_ / sections_;
    return *this;
  }

  /// Advances the iterator along the advance dimension
  MCTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator &operator++() {
    if (InstructionShape::kContiguous == 8) {
      if ((Policy::kGroupsPerTile / kPartitionsK) > 1) {
      int mask = ((Policy::kGroupsPerTile / kPartitionsK) == 8)
                     ? 3
                     : (((Policy::kGroupsPerTile / kPartitionsK) == 4) ? 1 : 0);

      if (((k_group_idx_ & mask) % 2) == 0)
        byte_offset_ ^= 1 * Policy::LdsmShape::kContiguous *
                        sizeof_bits<Element>::value *
                        Layout::kElementsPerAccess / 8;
      else if ((k_group_idx_ & mask) == 1)
        byte_offset_ ^= 3 * Policy::LdsmShape::kContiguous *
                        sizeof_bits<Element>::value *
                        Layout::kElementsPerAccess / 8;
      else if ((k_group_idx_ & mask) == 3)
        byte_offset_ ^= 7 * Policy::LdsmShape::kContiguous *
                        sizeof_bits<Element>::value *
                        Layout::kElementsPerAccess / 8;
      }
      k_group_idx_++;

      if (k_group_idx_ == (Policy::kGroupsPerTile / kPartitionsK)) {
        k_group_idx_ = 0;
        add_tile_offset({Policy::kGroupsPerTile, 0});
      }
    }
    else {
      byte_offset_ += InstructionShape::kContiguous;
    }
    return *this;

  }

  /// Advances the iterator along the advance dimension
  MCTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator &operator--() { assert(0); }

  ///< advances in units of whole tiles along the logical coordinate space of
  ///< the tensor
  MCTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator &operator+=(
      TensorCoord const &tile_offset) {
    add_tile_offset(tile_offset);
  }

  ///< advances in units of whole tiles along the logical coordinate space of
  ///< the tensor
  MCTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator &operator-=(
      TensorCoord const &tile_offset) {
    if (InstructionShape::kContiguous == 8) {
      add_tile_offset(-tile_offset);
    }
    else {
      printf("Not Impl Now.\n");
    //add_tile_offset(-tile_offset);
    }
    return *this;
  }

  /// Loads a fragment from memory at the location pointed to by the iterator.
  MCTLASS_HOST_DEVICE
  void load(Fragment &frag) const { load_with_byte_offset(frag, 0); }

  /// Loads a fragment from memory with additional logical offset
  MCTLASS_DEVICE
  void load_with_byte_offset(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a linear offset in units of bytes
      Index byte_offset) const {

    // Array<unsigned, Policy::LdsmShape::kCount> *fetch_ptr =
    //     reinterpret_cast<Array<unsigned, Policy::LdsmShape::kCount> *>(&frag);
    Array<Element, Policy::LdsmShape::kCount> *fetch_ptr =
        reinterpret_cast<Array<Element, Policy::LdsmShape::kCount> *>(&frag);
    const size_t frag_fetch_size = kOperand == Operand::kA ? Policy::LdsmShape::kCount * 2 : Policy::LdsmShape::kCount;
    Array<Element, frag_fetch_size> *fetch_element_ptr =
        reinterpret_cast<Array<Element, frag_fetch_size> *>(&frag);
    int ldm = stride_ * Layout::kElementsPerAccess / Layout::kFactor;

    const int lane_id = __lane_id();

    MCTLASS_PRAGMA_UNROLL
    for (int s = 0; s < Policy::LdsmIterations::kStrided; ++s) {
      MCTLASS_PRAGMA_UNROLL
      for (int c = 0; c < Policy::LdsmIterations::kContiguous; ++c) {
        int access_idx = c + s * Policy::LdsmIterations::kContiguous;
        int kLdsmOpInner = 0;
        if (InstructionShape::kContiguous == 8) {
          kLdsmOpInner = Policy::kLdsmOpInner_16x8x8;
        }
        else {
          kLdsmOpInner = Policy::kLdsmOpInner_16x8x16;
        }
        AccessType const *source_ptr =
            pointer_ + Policy::LdsmShape::kContiguous * c +
            kLdsmOpInner / Layout::kFactor *
                Policy::LdsmShape::kStrided * s * stride_;

        // char const *source_byte_ptr =
        //     reinterpret_cast<char const *>(source_ptr) + byte_offset +
        //     byte_offset_;

        // mctlass::arch::ldsm<layout::RowMajor, Policy::LdsmShape::kCount>(
        //     fetch_ptr[access_idx], source_byte_ptr);
        if (InstructionShape::kContiguous == 8) {
          char const *source_byte_ptr =
            reinterpret_cast<char const *>(source_ptr) + byte_offset +
            byte_offset_;
          if(kOperand == Operand::kA){
            fetch_ptr[access_idx][0] = *reinterpret_cast<Element const *>(source_byte_ptr);
            fetch_ptr[access_idx][1] = *(reinterpret_cast<Element const *>(source_byte_ptr) - 1);
          }
          else {
            Element const *p = reinterpret_cast<Element const *>(source_byte_ptr);
            Element const *p0 = p - 1;
            Element const *p1 = p0 + 8 * ldm;
            fetch_ptr[access_idx][0] = p[0];
            fetch_ptr[access_idx][1] = p0[0];
            fetch_ptr[access_idx][2] = p1[1];
            fetch_ptr[access_idx][3] = p1[0];
          }
        }
        else {
          Element const *ptr = reinterpret_cast<Element const *>(source_ptr) + byte_offset / sizeof(Element);
          Index row = ((lane_id >> 4) << 2) ^ (InstructionShape::kContiguous - 1) + byte_offset_;
          Index col = lane_id & (InstructionShape::kStrided - 1);
          for (int i = 0; i < Policy::LdsmShape::kCount; ++i) {
            Index idx = layout_(MatrixCoord(row - i, col));
            fetch_element_ptr[access_idx][i] = ptr[idx];
          }
       }
      }
    }
  }

  /// Loads a fragment from memory with additional logical offset
  MCTLASS_DEVICE
  void load_with_pointer_offset(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a linear offset
      Index pointer_offset) const {

    load_with_byte_offset(frag, pointer_offset * sizeof(Element));
  }

  /// Loads a fragment from memory with logical offset in units of whole tiles.
  MCTLASS_DEVICE
  void load(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a logical offset in units of whole tiles
      TensorCoord const &tile_offset) const {
    load_with_byte_offset(frag, tile_offset, 0);
  }

  /// Loads a fragment from memory with logical offset in units of whole tiles.
  MCTLASS_DEVICE
  void load(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a logical offset in units of whole tiles
      TensorCoord const &tile_offset,
      /// loads a tile with a logical offset AND a pointer offset
      Index pointer_offset) const {
    load_with_byte_offset(frag, tile_offset, pointer_offset * sizeof(Element));
  }

  /// Loads a fragment from memory with logical offset in units of whole tiles.
  MCTLASS_DEVICE
  void load_with_byte_offset(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a logical offset in units of whole tiles
      TensorCoord const &tile_offset,
      /// loads a tile with a logical offset AND a pointer offset
      Index byte_offset) const {
    Index pointer_offset = tile_offset.contiguous() *
                               InstructionShape::kContiguous /
                               Layout::kElementsPerAccess +
                           tile_offset.strided() * Shape::kStrided * stride_;

    byte_offset += sizeof_bits<AccessType>::value * pointer_offset / 8;

    load_with_byte_offset(frag, byte_offset);
  }

  /// Notify the iterator which k-group it is currently pointing to.
  ///
  /// This does not advance the iterator. Rather, it overrides its internal
  /// tracking with constant-valued k-group index to enable the compiler to
  /// fold constants and achieve more efficient code.
  ///
  /// This is used by some nontrivial permuted layouts.
  MCTLASS_DEVICE
  void set_kgroup_index(int k_group) {

    k_group_idx_ = k_group % (Policy::kGroupsPerTile / kPartitionsK);
  }
};

// This impl is for m16n16k16i8.
// Can't be merged with float<16,16,16> because of ambiguous partial specializations
template <
    /// Size of the matrix to load (concept: MatrixShape)
    typename Shape_,
    /// Interval between adjacent *MMA instructions (in units of MMA
    /// instructions, concept: MatrixShape)
    typename OpDelta_>
class MmaTensorOpAccumulatorTileIterator<
    Shape_, int, mctlass::layout::RowMajor, gemm::GemmShape<16, 16, 16>, OpDelta_> {
 public:

  /// Shape of tile to load (concept: MatrixShape)
  using Shape = Shape_;

  /// Operand tag
  static Operand const kOperand = Operand::kC;

  /// Element type
  using Element = int;

  /// Layout of source tile
  using Layout = mctlass::layout::RowMajor;

  /// Shape of one matrix product operation (concept: MatrixShape)
  using InstructionShape = gemm::GemmShape<16, 16, 16>;

  /// Delta between *MMA operations (in units of *MMA operations, concept: MatrixShape)
  using OpDelta = OpDelta_;

  /// Number of participating threads
  static int const kThreads = 64;

  /// TensorRef type for loading element from a tensor
  using TensorRef = TensorRef<Element, Layout>;

  /// Index type
  using Index = typename TensorRef::Index;

  /// Long Index type
  using LongIndex = typename TensorRef::LongIndex;

  /// Coordinate for an element in the tensor
  using TensorCoord = typename TensorRef::TensorCoord;

  /// Internal structure of iterator - made public to enable introspection
  struct Policy {
    static bool const kDivisible =
        !(Shape::kRow % InstructionShape::kM) &&
            !(Shape::kColumn % InstructionShape::kN);

    static_assert(platform::is_same<TensorCoord, MatrixCoord>::value,
      "Layouts must be defined for logical MatrixCoord coordinate space.");

    /// Number of mma operations performed
    using MmaIterations = MatrixShape<
      (Shape::kRow + InstructionShape::kM - 1) / InstructionShape::kM,
      (Shape::kColumn + InstructionShape::kN - 1) / InstructionShape::kN
    >;
  };

private:

  // Assume accumulator tile is an arrangement of 8-by-8 tiles replicated over the entire
  // shape, with each quad mapped to one row and each thread mapped to 1/4 of the elements
  // of that row. The accumulators within one row are assumed to be consecutive.
 static int const kElementsPerAccess = InstructionShape::kN / 4;
 static int const kRowsPerTile = 16;
 static int const kAccumulatorRows = InstructionShape::kM / kRowsPerTile;

public:

  //
  // Derived quantities
  //

  /// Fragment object holding a thread's part of a tile
  using Fragment = Array<
    Element,
    Policy::MmaIterations::kCount * InstructionShape::kMN / kThreads>;

private:

  /// Reference to output tensor
  TensorRef ref_;

public:

  /// Default ctor constructs null iterator
  MCTLASS_HOST_DEVICE
  MmaTensorOpAccumulatorTileIterator() { }

  /// Constructor from TensorRef
  MCTLASS_HOST_DEVICE
  MmaTensorOpAccumulatorTileIterator(
    TensorRef const &ref,
    int lane_id
  ):
    ref_(ref) {

    int lane_in_quad = (lane_id >> 4) << 2;
    int quad = lane_id & 0xf;
    MatrixCoord lane_offset(quad, lane_in_quad);
    ref_.add_coord_offset(lane_offset);

  }

  /// Adds a pointer offset to internal pointer(s) to advance through memory
  MCTLASS_HOST_DEVICE
  MmaTensorOpAccumulatorTileIterator &add_pointer_offset(LongIndex offset) {
    ref_.add_pointer_offset(offset);
    return *this;
  }

  /// Advances an iterator along logical dimensions of matrix in units of whole tiles
  MCTLASS_HOST_DEVICE
  MmaTensorOpAccumulatorTileIterator &add_tile_offset(TensorCoord const &tile_offset) {

    ref_.add_coord_offset(tile_offset * make_Coord(Shape::kRow, Shape::kColumn));

    return *this;
  }

  /// Advances the iterator along the advance dimension
  MCTLASS_HOST_DEVICE
  MmaTensorOpAccumulatorTileIterator & operator++() {
    // deliberate no-op
    return *this;
  }

  /// Advances the iterator along the advance dimension
  MCTLASS_HOST_DEVICE
  MmaTensorOpAccumulatorTileIterator & operator--() {
    // deliberate no-op
    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of the tensor
  MCTLASS_DEVICE
  MmaTensorOpAccumulatorTileIterator & operator+=(TensorCoord const &tile_offset) {
    add_tile_offset(tile_offset);
    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of the tensor
  MCTLASS_DEVICE
  MmaTensorOpAccumulatorTileIterator & operator-=(TensorCoord const &tile_offset) {
    printf("Not Impl Now.\n");
    //add_tile_offset(-tile_offset);
    return *this;
  }

  /// Loads a fragment from memory at the location pointed to by the iterator.
  MCTLASS_HOST_DEVICE
  void load(Fragment &frag) const {
    load_with_pointer_offset(frag, 0);
  }

  /// Loads a fragment from memory with additional logical offset
  MCTLASS_DEVICE
  void load_with_pointer_offset(
    Fragment &frag,                             ///< fragment to load from the tensor
    Index pointer_offset) const {               ///< loads a tile with a linear offset

    TensorRef offset_ref(ref_);
    offset_ref.add_pointer_offset(pointer_offset);

    MCTLASS_PRAGMA_UNROLL
    for (int mma_n = 0; mma_n < Policy::MmaIterations::kColumn; ++mma_n) {
      MCTLASS_PRAGMA_UNROLL
      for (int mma_m = 0; mma_m < Policy::MmaIterations::kRow; ++mma_m) {

        int mma_accum_start = kAccumulatorRows * kElementsPerAccess *
          (mma_n * Policy::MmaIterations::kRow + mma_m);

        MCTLASS_PRAGMA_UNROLL
        for (int row = 0; row < kAccumulatorRows; ++row) {
          MCTLASS_PRAGMA_UNROLL
          for (int col = 0; col < kElementsPerAccess; ++col) {
            int accum_m = mma_m * InstructionShape::kM * OpDelta::kRow +
                          row * kRowsPerTile;
            int accum_n = mma_n * InstructionShape::kN * OpDelta::kColumn + col;

            frag[mma_accum_start + row * kElementsPerAccess + col] = offset_ref.at({accum_m, accum_n});
          }
        }
      }
    }
  }

  /// Loads a fragment from memory with additional logical offset
  MCTLASS_DEVICE
  void load_with_byte_offset(
    Fragment &frag,                             ///< fragment to load from the tensor
    Index byte_offset) const {                  ///< loads a tile with a linear offset

    load_with_pointer_offset(byte_offset / sizeof(Element));
  }

  /// Loads a fragment from memory with logical offset in units of whole tiles.
  MCTLASS_DEVICE
  void load(
    Fragment &frag,                             ///< fragment to load from the tensor
    TensorCoord const &tile_offset) const {     ///< loads a tile with a logical offset in units of whole tiles

    load(frag, tile_offset, 0);
  }

  /// Loads a fragment from memory with logical offset in units of whole tiles.
  MCTLASS_DEVICE
  void load(
    Fragment &frag,                             ///< fragment to load from the tensor
    TensorCoord const &tile_offset,             ///< loads a tile with a logical offset in units of whole tiles
    Index pointer_offset) const {               ///< loads a tile with a logical offset AND a pointer offset

    load_with_pointer_offset(frag, ref_.offset(tile_offset) + pointer_offset);
  }

  /// Stores a fragment to memory
  MCTLASS_HOST_DEVICE
  void store(Fragment const &frag) const {
    store_with_pointer_offset(frag, 0);
  }

  /// Stores a fragment to memory with additional pointer offset
  MCTLASS_DEVICE
  void store_with_pointer_offset(
    Fragment const &frag,                       ///< fragment to store from the tensor
    Index pointer_offset) const {               ///< store a tile with a linear offset

    TensorRef offset_ref(ref_);
    offset_ref.add_pointer_offset(pointer_offset);

    MCTLASS_PRAGMA_UNROLL
    for (int mma_n = 0; mma_n < Policy::MmaIterations::kColumn; ++mma_n) {
      MCTLASS_PRAGMA_UNROLL
      for (int mma_m = 0; mma_m < Policy::MmaIterations::kRow; ++mma_m) {

        int mma_accum_start = kAccumulatorRows * kElementsPerAccess *
          (mma_n * Policy::MmaIterations::kRow + mma_m);

        MCTLASS_PRAGMA_UNROLL
        for (int row = 0; row < kAccumulatorRows; ++row) {
          MCTLASS_PRAGMA_UNROLL
          for (int col = 0; col < kElementsPerAccess; ++col) {

            int accum_m = mma_m * InstructionShape::kM * OpDelta::kRow + row * kElementsPerAccess;
            int accum_n = mma_n * InstructionShape::kN * OpDelta::kColumn + col;

            int idx = mma_accum_start + row * kElementsPerAccess + col;

            offset_ref.at({accum_m, accum_n}) = frag[idx];

          }
        }
      }

    }

  }

  /// Stores a fragment to memory with additional pointer offset
  MCTLASS_DEVICE
  void store_with_byte_offset(
    Fragment const &frag,                       ///< fragment to store from the tensor
    Index byte_offset) const {                  ///< store a tile with a linear offset

    store_with_pointer_offset(byte_offset / sizeof(Element));
  }

  /// Stores a fragment to memory with logical offset in units of whole tiles.
  MCTLASS_DEVICE
  void store(
    Fragment &frag,                             ///< fragment to store to the tensor
    TensorCoord const &tile_offset) const {     ///< stores a tile with a logical offset in units of whole tiles

    store(frag, tile_offset, 0);
  }

  /// Stores a fragment from memory with logical offset in units of whole tiles.
  MCTLASS_DEVICE
  void store(
      /// fragment to store to the tensor
      Fragment const &frag,
      /// stores a tile with a logical offset in units of whole tiles
      TensorCoord const &tile_offset,
      /// stores a tile with a logical offset AND a pointer offset
      Index pointer_offset) const {
    store_with_pointer_offset(frag, ref_.offset(tile_offset) + pointer_offset);
  }
};

template <
    /// Size of the matrix to load (concept: MatrixShape)
    typename Shape_,
    /// Interval between adjacent *MMA instructions (in units of MMA
    /// instructions, concept: MatrixShape)
    typename OpDelta_>
class MmaTensorOpAccumulatorTileIterator<
    Shape_, int, mctlass::layout::RowMajor, gemm::GemmShape<16, 16, 32>, OpDelta_> {
 public:

  /// Shape of tile to load (concept: MatrixShape)
  using Shape = Shape_;

  /// Operand tag
  static Operand const kOperand = Operand::kC;

  /// Element type
  using Element = int;

  /// Layout of source tile
  using Layout = mctlass::layout::RowMajor;

  /// Shape of one matrix product operation (concept: MatrixShape)
  using InstructionShape = gemm::GemmShape<16, 16, 32>;

  /// Delta between *MMA operations (in units of *MMA operations, concept: MatrixShape)
  using OpDelta = OpDelta_;

  /// Number of participating threads
  static int const kThreads = 64;

  /// TensorRef type for loading element from a tensor
  using TensorRef = TensorRef<Element, Layout>;

  /// Index type
  using Index = typename TensorRef::Index;

  /// Long Index type
  using LongIndex = typename TensorRef::LongIndex;

  /// Coordinate for an element in the tensor
  using TensorCoord = typename TensorRef::TensorCoord;

  /// Underlying tile iterator implementation
  using Base = MmaTensorOpAccumulatorTileIterator<
      Shape_, int, mctlass::layout::RowMajor,
      gemm::GemmShape<16, 16, 16>, OpDelta_>;

private:

  /// Underlying tile iterator
  Base iterator_;

public:

  //
  // Derived quantities
  //

  /// Fragment object holding a thread's part of a tile
  using Fragment = typename Base::Fragment;

private:

  /// Reference to output tensor
  TensorRef ref_;

public:

  /// Default ctor constructs null iterator
  MCTLASS_HOST_DEVICE
  MmaTensorOpAccumulatorTileIterator() { }

  /// Constructor from TensorRef
  MCTLASS_HOST_DEVICE
  MmaTensorOpAccumulatorTileIterator(
    TensorRef const &ref,
    int lane_id
  ): iterator_({ref.data(), ref.stride()}, lane_id) {
  }

  /// Adds a pointer offset to internal pointer(s) to advance through memory
  MCTLASS_HOST_DEVICE
  MmaTensorOpAccumulatorTileIterator &add_pointer_offset(LongIndex offset) {
    ref_.add_pointer_offset(offset);
    return *this;
  }

  /// Advances an iterator along logical dimensions of matrix in units of whole tiles
  MCTLASS_HOST_DEVICE
  MmaTensorOpAccumulatorTileIterator &add_tile_offset(TensorCoord const &tile_offset) {
    ref_.add_coord_offset(tile_offset * make_Coord(Shape::kRow, Shape::kColumn));
    return *this;
  }

  /// Advances the iterator along the advance dimension
  MCTLASS_HOST_DEVICE
  MmaTensorOpAccumulatorTileIterator & operator++() {
    // deliberate no-op
    return *this;
  }

  /// Advances the iterator along the advance dimension
  MCTLASS_HOST_DEVICE
  MmaTensorOpAccumulatorTileIterator & operator--() {
    // deliberate no-op
    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of the tensor
  MCTLASS_DEVICE
  MmaTensorOpAccumulatorTileIterator & operator+=(TensorCoord const &tile_offset) {
    add_tile_offset(tile_offset);
    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of the tensor
  MCTLASS_DEVICE
  MmaTensorOpAccumulatorTileIterator & operator-=(TensorCoord const &tile_offset) {
    printf("Not Impl Now.\n");
    //add_tile_offset(-tile_offset);
    return *this;
  }

  /// Loads a fragment from memory at the location pointed to by the iterator.
  MCTLASS_HOST_DEVICE
  void load(Fragment &frag) const {
    load_with_pointer_offset(frag, 0);
  }

  /// Loads a fragment from memory with additional logical offset
  MCTLASS_DEVICE
  void load_with_pointer_offset(
    Fragment &frag,                             ///< fragment to load from the tensor
    Index pointer_offset) const {               ///< loads a tile with a linear offset
    iterator_.load_with_pointer_offset(frag, pointer_offset);
  }

  /// Loads a fragment from memory with additional logical offset
  MCTLASS_DEVICE
  void load_with_byte_offset(
    Fragment &frag,                             ///< fragment to load from the tensor
    Index byte_offset) const {                  ///< loads a tile with a linear offset

    load_with_pointer_offset(byte_offset / sizeof(Element));
  }

  /// Loads a fragment from memory with logical offset in units of whole tiles.
  MCTLASS_DEVICE
  void load(
    Fragment &frag,                             ///< fragment to load from the tensor
    TensorCoord const &tile_offset) const {     ///< loads a tile with a logical offset in units of whole tiles

    load(frag, tile_offset, 0);
  }

  /// Loads a fragment from memory with logical offset in units of whole tiles.
  MCTLASS_DEVICE
  void load(
    Fragment &frag,                             ///< fragment to load from the tensor
    TensorCoord const &tile_offset,             ///< loads a tile with a logical offset in units of whole tiles
    Index pointer_offset) const {               ///< loads a tile with a logical offset AND a pointer offset

    load_with_pointer_offset(frag, ref_.offset(tile_offset) + pointer_offset);
  }

  /// Stores a fragment to memory
  MCTLASS_HOST_DEVICE
  void store(Fragment const &frag) const {
    store_with_pointer_offset(frag, 0);
  }

  /// Stores a fragment to memory with additional pointer offset
  MCTLASS_DEVICE
  void store_with_pointer_offset(
    Fragment const &frag,                       ///< fragment to store from the tensor
    Index pointer_offset) const {               ///< store a tile with a linear offset
    iterator_.store_with_pointer_offset(frag, pointer_offset);
  }

  /// Stores a fragment to memory with additional pointer offset
  MCTLASS_DEVICE
  void store_with_byte_offset(
    Fragment const &frag,                       ///< fragment to store from the tensor
    Index byte_offset) const {                  ///< store a tile with a linear offset

    store_with_pointer_offset(byte_offset / sizeof(Element));
  }

  /// Stores a fragment to memory with logical offset in units of whole tiles.
  MCTLASS_DEVICE
  void store(
    Fragment &frag,                             ///< fragment to store to the tensor
    TensorCoord const &tile_offset) const {     ///< stores a tile with a logical offset in units of whole tiles

    store(frag, tile_offset, 0);
  }

  /// Stores a fragment from memory with logical offset in units of whole tiles.
  MCTLASS_DEVICE
  void store(
      /// fragment to store to the tensor
      Fragment const &frag,
      /// stores a tile with a logical offset in units of whole tiles
      TensorCoord const &tile_offset,
      /// stores a tile with a logical offset AND a pointer offset
      Index pointer_offset) const {
    store_with_pointer_offset(frag, ref_.offset(tile_offset) + pointer_offset);
  }
};

template <
    /// Size of the matrix to load (concept: MatrixShape)
    typename Shape_,
    /// Interval between adjacent *MMA instructions (in units of MMA
    /// instructions, concept: MatrixShape)
    typename OpDelta_>
class MmaTensorOpAccumulatorTileIterator<
    Shape_, float, mctlass::layout::RowMajor, gemm::GemmShape<16, 8, 16>, OpDelta_> {
 public:

  /// Shape of tile to load (concept: MatrixShape)
  using Shape = Shape_;

  /// Operand tag
  static Operand const kOperand = Operand::kC;

  /// Element type
  using Element = float;

  /// Layout of source tile
  using Layout = mctlass::layout::RowMajor;

  /// Shape of one matrix product operation (concept: MatrixShape)
  using InstructionShape = gemm::GemmShape<16, 8, 16>;

  /// Delta between *MMA operations (in units of *MMA operations, concept: MatrixShape)
  using OpDelta = OpDelta_;

  /// Number of participating threads
  static int const kThreads = 32;

  /// TensorRef type for loading element from a tensor
  using TensorRef = TensorRef<Element, Layout>;

  /// Index type
  using Index = typename TensorRef::Index;

  /// Long Index type
  using LongIndex = typename TensorRef::LongIndex;

  /// Coordinate for an element in the tensor
  using TensorCoord = typename TensorRef::TensorCoord;

  /// Internal structure of iterator - made public to enable introspection
  struct Policy {
    static bool const kDivisible =
        !(Shape::kRow % InstructionShape::kM) &&
            !(Shape::kColumn % InstructionShape::kN);

    static_assert(platform::is_same<TensorCoord, MatrixCoord>::value,
      "Layouts must be defined for logical MatrixCoord coordinate space.");

    /// Number of mma operations performed
    using MmaIterations = MatrixShape<
      (Shape::kRow + InstructionShape::kM - 1) / InstructionShape::kM,
      (Shape::kColumn + InstructionShape::kN - 1) / InstructionShape::kN
    >;
  };

private:

  // Assume accumulator tile is an arrangement of 8-by-8 tiles replicated over the entire
  // shape, with each quad mapped to one row and each thread mapped to 1/4 of the elements
  // of that row. The accumulators within one row are assumed to be consecutive.
 static int const kElementsPerAccess = InstructionShape::kN / 4;
 static int const kRowsPerTile = 8;
 static int const kAccumulatorRows = InstructionShape::kM / kRowsPerTile;

public:

  //
  // Derived quantities
  //

  /// Fragment object holding a thread's part of a tile
  using Fragment = Array<
    Element,
    Policy::MmaIterations::kCount * InstructionShape::kMN / kThreads>;

private:

  /// Reference to output tensor
  TensorRef ref_;

public:

  /// Default ctor constructs null iterator
  MCTLASS_HOST_DEVICE
  MmaTensorOpAccumulatorTileIterator() { }

  /// Constructor from TensorRef
  MCTLASS_HOST_DEVICE
  MmaTensorOpAccumulatorTileIterator(
    TensorRef const &ref,
    int lane_id
  ):
    ref_(ref) {

    int quad = (lane_id >> 4) << 2;
    int lane_in_quad = lane_id & 0x7;
    MatrixCoord lane_offset(quad, lane_in_quad);
    ref_.add_coord_offset(lane_offset);

  }

  /// Adds a pointer offset to internal pointer(s) to advance through memory
  MCTLASS_HOST_DEVICE
  MmaTensorOpAccumulatorTileIterator &add_pointer_offset(LongIndex offset) {
    ref_.add_pointer_offset(offset);
    return *this;
  }

  /// Advances an iterator along logical dimensions of matrix in units of whole tiles
  MCTLASS_HOST_DEVICE
  MmaTensorOpAccumulatorTileIterator &add_tile_offset(TensorCoord const &tile_offset) {

    ref_.add_coord_offset(tile_offset * make_Coord(Shape::kRow, Shape::kColumn));

    return *this;
  }

  /// Advances the iterator along the advance dimension
  MCTLASS_HOST_DEVICE
  MmaTensorOpAccumulatorTileIterator & operator++() {
    // deliberate no-op
    return *this;
  }

  /// Advances the iterator along the advance dimension
  MCTLASS_HOST_DEVICE
  MmaTensorOpAccumulatorTileIterator & operator--() {
    // deliberate no-op
    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of the tensor
  MCTLASS_DEVICE
  MmaTensorOpAccumulatorTileIterator & operator+=(TensorCoord const &tile_offset) {
    add_tile_offset(tile_offset);
    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of the tensor
  MCTLASS_DEVICE
  MmaTensorOpAccumulatorTileIterator & operator-=(TensorCoord const &tile_offset) {
    printf("Not Impl Now.\n");
    //add_tile_offset(-tile_offset);
    return *this;
  }

  /// Loads a fragment from memory at the location pointed to by the iterator.
  MCTLASS_HOST_DEVICE
  void load(Fragment &frag) const {
    load_with_pointer_offset(frag, 0);
  }

  /// Loads a fragment from memory with additional logical offset
  MCTLASS_DEVICE
  void load_with_pointer_offset(
    Fragment &frag,                             ///< fragment to load from the tensor
    Index pointer_offset) const {               ///< loads a tile with a linear offset

    TensorRef offset_ref(ref_);
    offset_ref.add_pointer_offset(pointer_offset);

    MCTLASS_PRAGMA_UNROLL
    for (int mma_n = 0; mma_n < Policy::MmaIterations::kColumn; ++mma_n) {
      MCTLASS_PRAGMA_UNROLL
      for (int mma_m = 0; mma_m < Policy::MmaIterations::kRow; ++mma_m) {

        int mma_accum_start = kAccumulatorRows * kElementsPerAccess *
          (mma_n * Policy::MmaIterations::kRow + mma_m);

        MCTLASS_PRAGMA_UNROLL
        for (int row = 0; row < kAccumulatorRows; ++row) {
          MCTLASS_PRAGMA_UNROLL
          for (int col = 0; col < kElementsPerAccess; ++col) {
            int accum_m = mma_m * InstructionShape::kM * OpDelta::kRow +
                          row * kRowsPerTile;
            int accum_n = mma_n * InstructionShape::kN * OpDelta::kColumn + col;

            frag[mma_accum_start + row * kElementsPerAccess + col] = offset_ref.at({accum_m, accum_n});
          }
        }
      }
    }
  }

  /// Loads a fragment from memory with additional logical offset
  MCTLASS_DEVICE
  void load_with_byte_offset(
    Fragment &frag,                             ///< fragment to load from the tensor
    Index byte_offset) const {                  ///< loads a tile with a linear offset

    load_with_pointer_offset(byte_offset / sizeof(Element));
  }

  /// Loads a fragment from memory with logical offset in units of whole tiles.
  MCTLASS_DEVICE
  void load(
    Fragment &frag,                             ///< fragment to load from the tensor
    TensorCoord const &tile_offset) const {     ///< loads a tile with a logical offset in units of whole tiles

    load(frag, tile_offset, 0);
  }

  /// Loads a fragment from memory with logical offset in units of whole tiles.
  MCTLASS_DEVICE
  void load(
    Fragment &frag,                             ///< fragment to load from the tensor
    TensorCoord const &tile_offset,             ///< loads a tile with a logical offset in units of whole tiles
    Index pointer_offset) const {               ///< loads a tile with a logical offset AND a pointer offset

    load_with_pointer_offset(frag, ref_.offset(tile_offset) + pointer_offset);
  }

  /// Stores a fragment to memory
  MCTLASS_HOST_DEVICE
  void store(Fragment const &frag) const {
    store_with_pointer_offset(frag, 0);
  }

  /// Stores a fragment to memory with additional pointer offset
  MCTLASS_DEVICE
  void store_with_pointer_offset(
    Fragment const &frag,                       ///< fragment to store from the tensor
    Index pointer_offset) const {               ///< store a tile with a linear offset

    TensorRef offset_ref(ref_);
    offset_ref.add_pointer_offset(pointer_offset);

    MCTLASS_PRAGMA_UNROLL
    for (int mma_n = 0; mma_n < Policy::MmaIterations::kColumn; ++mma_n) {
      MCTLASS_PRAGMA_UNROLL
      for (int mma_m = 0; mma_m < Policy::MmaIterations::kRow; ++mma_m) {

        int mma_accum_start = kAccumulatorRows * kElementsPerAccess *
          (mma_n * Policy::MmaIterations::kRow + mma_m);

        MCTLASS_PRAGMA_UNROLL
        for (int row = 0; row < kAccumulatorRows; ++row) {
          MCTLASS_PRAGMA_UNROLL
          for (int col = 0; col < kElementsPerAccess; ++col) {

            int accum_m = mma_m * InstructionShape::kM * OpDelta::kRow + row * kElementsPerAccess + col;
            int accum_n = mma_n * InstructionShape::kN * OpDelta::kColumn;

            int idx = mma_accum_start + row * kElementsPerAccess + col;

            offset_ref.at({accum_m, accum_n}) = frag[idx];

          }
        }
      }

    }

  }

  /// Stores a fragment to memory with additional pointer offset
  MCTLASS_DEVICE
  void store_with_byte_offset(
    Fragment const &frag,                       ///< fragment to store from the tensor
    Index byte_offset) const {                  ///< store a tile with a linear offset

    store_with_pointer_offset(byte_offset / sizeof(Element));
  }

  /// Stores a fragment to memory with logical offset in units of whole tiles.
  MCTLASS_DEVICE
  void store(
    Fragment &frag,                             ///< fragment to store to the tensor
    TensorCoord const &tile_offset) const {     ///< stores a tile with a logical offset in units of whole tiles

    store(frag, tile_offset, 0);
  }

  /// Stores a fragment from memory with logical offset in units of whole tiles.
  MCTLASS_DEVICE
  void store(
      /// fragment to store to the tensor
      Fragment const &frag,
      /// stores a tile with a logical offset in units of whole tiles
      TensorCoord const &tile_offset,
      /// stores a tile with a logical offset AND a pointer offset
      Index pointer_offset) const {
    store_with_pointer_offset(frag, ref_.offset(tile_offset) + pointer_offset);
  }
};


template <
    /// Size of the matrix to load (concept: MatrixShape)
    typename Shape_,
    /// Interval between adjacent *MMA instructions (in units of MMA
    /// instructions, concept: MatrixShape)
    typename OpDelta_>
class MmaTensorOpAccumulatorTileIterator<
    Shape_, float, mctlass::layout::RowMajor, gemm::GemmShape<16, 16, 16>, OpDelta_> {
 public:

  /// Shape of tile to load (concept: MatrixShape)
  using Shape = Shape_;

  /// Operand tag
  static Operand const kOperand = Operand::kC;

  /// Element type
  using Element = float;

  /// Layout of source tile
  using Layout = mctlass::layout::RowMajor;

  /// Shape of one matrix product operation (concept: MatrixShape)
  using InstructionShape = gemm::GemmShape<16, 16, 16>;

  /// Delta between *MMA operations (in units of *MMA operations, concept: MatrixShape)
  using OpDelta = OpDelta_;

  /// Number of participating threads
  static int const kThreads = 64;

  /// TensorRef type for loading element from a tensor
  using TensorRef = TensorRef<Element, Layout>;

  /// Index type
  using Index = typename TensorRef::Index;

  /// Long Index type
  using LongIndex = typename TensorRef::LongIndex;

  /// Coordinate for an element in the tensor
  using TensorCoord = typename TensorRef::TensorCoord;

  /// Internal structure of iterator - made public to enable introspection
  struct Policy {
    static bool const kDivisible =
        !(Shape::kRow % InstructionShape::kM) &&
            !(Shape::kColumn % InstructionShape::kN);

    static_assert(platform::is_same<TensorCoord, MatrixCoord>::value,
      "Layouts must be defined for logical MatrixCoord coordinate space.");

    /// Number of mma operations performed
    using MmaIterations = MatrixShape<
      (Shape::kRow + InstructionShape::kM - 1) / InstructionShape::kM,
      (Shape::kColumn + InstructionShape::kN - 1) / InstructionShape::kN
    >;
  };

private:

  // Assume accumulator tile is an arrangement of 8-by-8 tiles replicated over the entire
  // shape, with each quad mapped to one row and each thread mapped to 1/4 of the elements
  // of that row. The accumulators within one row are assumed to be consecutive.
 static int const kElementsPerAccess = InstructionShape::kN / 4;
 static int const kRowsPerTile = 16;
 static int const kAccumulatorRows = InstructionShape::kM / kRowsPerTile;

public:

  //
  // Derived quantities
  //

  /// Fragment object holding a thread's part of a tile
  using Fragment = Array<
    Element,
    Policy::MmaIterations::kCount * InstructionShape::kMN / kThreads>;

private:

  /// Reference to output tensor
  TensorRef ref_;

public:

  /// Default ctor constructs null iterator
  MCTLASS_HOST_DEVICE
  MmaTensorOpAccumulatorTileIterator() { }

  /// Constructor from TensorRef
  MCTLASS_HOST_DEVICE
  MmaTensorOpAccumulatorTileIterator(
    TensorRef const &ref,
    int lane_id
  ):
    ref_(ref) {

    int lane_in_quad = (lane_id >> 4) << 2;
    int quad = lane_id & 0xf;
    MatrixCoord lane_offset(quad,lane_in_quad);
    ref_.add_coord_offset(lane_offset);

  }

  /// Adds a pointer offset to internal pointer(s) to advance through memory
  MCTLASS_HOST_DEVICE
  MmaTensorOpAccumulatorTileIterator &add_pointer_offset(LongIndex offset) {
    ref_.add_pointer_offset(offset);
    return *this;
  }

  /// Advances an iterator along logical dimensions of matrix in units of whole tiles
  MCTLASS_HOST_DEVICE
  MmaTensorOpAccumulatorTileIterator &add_tile_offset(TensorCoord const &tile_offset) {

    ref_.add_coord_offset(tile_offset * make_Coord(Shape::kRow, Shape::kColumn));

    return *this;
  }

  /// Advances the iterator along the advance dimension
  MCTLASS_HOST_DEVICE
  MmaTensorOpAccumulatorTileIterator & operator++() {
    // deliberate no-op
    return *this;
  }

  /// Advances the iterator along the advance dimension
  MCTLASS_HOST_DEVICE
  MmaTensorOpAccumulatorTileIterator & operator--() {
    // deliberate no-op
    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of the tensor
  MCTLASS_DEVICE
  MmaTensorOpAccumulatorTileIterator & operator+=(TensorCoord const &tile_offset) {
    add_tile_offset(tile_offset);
    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of the tensor
  MCTLASS_DEVICE
  MmaTensorOpAccumulatorTileIterator & operator-=(TensorCoord const &tile_offset) {
    add_tile_offset(make_Coord(-tile_offset.row(), tile_offset.column()));
    return *this;
  }

  /// Loads a fragment from memory at the location pointed to by the iterator.
  MCTLASS_HOST_DEVICE
  void load(Fragment &frag) const {
    load_with_pointer_offset(frag, 0);
  }

  /// Loads a fragment from memory with additional logical offset
  MCTLASS_DEVICE
  void load_with_pointer_offset(
    Fragment &frag,                             ///< fragment to load from the tensor
    Index pointer_offset) const {               ///< loads a tile with a linear offset

    TensorRef offset_ref(ref_);
    offset_ref.add_pointer_offset(pointer_offset);

    MCTLASS_PRAGMA_UNROLL
    for (int mma_n = 0; mma_n < Policy::MmaIterations::kColumn; ++mma_n) {
      MCTLASS_PRAGMA_UNROLL
      for (int mma_m = 0; mma_m < Policy::MmaIterations::kRow; ++mma_m) {

        int mma_accum_start = kAccumulatorRows * kElementsPerAccess *
          (mma_n * Policy::MmaIterations::kRow + mma_m);

        MCTLASS_PRAGMA_UNROLL
        for (int row = 0; row < kAccumulatorRows; ++row) {
          MCTLASS_PRAGMA_UNROLL
          for (int col = 0; col < kElementsPerAccess; ++col) {
            int accum_m = mma_m * InstructionShape::kM * OpDelta::kRow +
                          row * kRowsPerTile;
            int accum_n = mma_n * InstructionShape::kN * OpDelta::kColumn + col;

            frag[mma_accum_start + row * kElementsPerAccess + col] = offset_ref.at({accum_m, accum_n});
          }
        }
      }
    }
  }

  /// Loads a fragment from memory with additional logical offset
  MCTLASS_DEVICE
  void load_with_byte_offset(
    Fragment &frag,                             ///< fragment to load from the tensor
    Index byte_offset) const {                  ///< loads a tile with a linear offset

    load_with_pointer_offset(byte_offset / sizeof(Element));
  }

  /// Loads a fragment from memory with logical offset in units of whole tiles.
  MCTLASS_DEVICE
  void load(
    Fragment &frag,                             ///< fragment to load from the tensor
    TensorCoord const &tile_offset) const {     ///< loads a tile with a logical offset in units of whole tiles

    load(frag, tile_offset, 0);
  }

  /// Loads a fragment from memory with logical offset in units of whole tiles.
  MCTLASS_DEVICE
  void load(
    Fragment &frag,                             ///< fragment to load from the tensor
    TensorCoord const &tile_offset,             ///< loads a tile with a logical offset in units of whole tiles
    Index pointer_offset) const {               ///< loads a tile with a logical offset AND a pointer offset

    load_with_pointer_offset(frag, ref_.offset(tile_offset) + pointer_offset);
  }

  /// Stores a fragment to memory
  MCTLASS_HOST_DEVICE
  void store(Fragment const &frag) const {
    store_with_pointer_offset(frag, 0);
  }

  /// Stores a fragment to memory with additional pointer offset
  MCTLASS_DEVICE
  void store_with_pointer_offset(
    Fragment const &frag,                       ///< fragment to store from the tensor
    Index pointer_offset) const {               ///< store a tile with a linear offset

    TensorRef offset_ref(ref_);
    offset_ref.add_pointer_offset(pointer_offset);

    MCTLASS_PRAGMA_UNROLL
    for (int mma_n = 0; mma_n < Policy::MmaIterations::kColumn; ++mma_n) {
      MCTLASS_PRAGMA_UNROLL
      for (int mma_m = 0; mma_m < Policy::MmaIterations::kRow; ++mma_m) {

        int mma_accum_start = kAccumulatorRows * kElementsPerAccess *
          (mma_n * Policy::MmaIterations::kRow + mma_m);

        MCTLASS_PRAGMA_UNROLL
        for (int row = 0; row < kAccumulatorRows; ++row) {
          MCTLASS_PRAGMA_UNROLL
          for (int col = 0; col < kElementsPerAccess; ++col) {

            int accum_m = mma_m * InstructionShape::kM * OpDelta::kRow + row * kElementsPerAccess;
            int accum_n = mma_n * InstructionShape::kN * OpDelta::kColumn + col;

            int idx = mma_accum_start + row * kElementsPerAccess + col;

            offset_ref.at({accum_m, accum_n}) = frag[idx];

          }
        }
      }

    }

  }

  /// Stores a fragment to memory with additional pointer offset
  MCTLASS_DEVICE
  void store_with_byte_offset(
    Fragment const &frag,                       ///< fragment to store from the tensor
    Index byte_offset) const {                  ///< store a tile with a linear offset

    store_with_pointer_offset(byte_offset / sizeof(Element));
  }

  /// Stores a fragment to memory with logical offset in units of whole tiles.
  MCTLASS_DEVICE
  void store(
    Fragment &frag,                             ///< fragment to store to the tensor
    TensorCoord const &tile_offset) const {     ///< stores a tile with a logical offset in units of whole tiles

    store(frag, tile_offset, 0);
  }

  /// Stores a fragment from memory with logical offset in units of whole tiles.
  MCTLASS_DEVICE
  void store(
      /// fragment to store to the tensor
      Fragment const &frag,
      /// stores a tile with a logical offset in units of whole tiles
      TensorCoord const &tile_offset,
      /// stores a tile with a logical offset AND a pointer offset
      Index pointer_offset) const {
    store_with_pointer_offset(frag, ref_.offset(tile_offset) + pointer_offset);
  }
};

template <
    /// Size of the matrix to load (concept: MatrixShape)
    typename Shape_,
    /// Interval between adjacent *MMA instructions (in units of MMA
    /// instructions, concept: MatrixShape)
    typename OpDelta_>
class MmaTensorOpAccumulatorTileIterator<
    Shape_, float, mctlass::layout::RowMajor, gemm::GemmShape<16, 16, 8>, OpDelta_> {
 public:

  /// Shape of tile to load (concept: MatrixShape)
  using Shape = Shape_;

  /// Operand tag
  static Operand const kOperand = Operand::kC;

  /// Element type
  using Element = float;

  /// Layout of source tile
  using Layout = mctlass::layout::RowMajor;

  /// Shape of one matrix product operation (concept: MatrixShape)
  using InstructionShape = gemm::GemmShape<16, 16, 8>;

  /// Delta between *MMA operations (in units of *MMA operations, concept: MatrixShape)
  using OpDelta = OpDelta_;

  /// Number of participating threads
  static int const kThreads = 64;

  /// TensorRef type for loading element from a tensor
  using TensorRef = TensorRef<Element, Layout>;

  /// Index type
  using Index = typename TensorRef::Index;

  /// Long Index type
  using LongIndex = typename TensorRef::LongIndex;

  /// Coordinate for an element in the tensor
  using TensorCoord = typename TensorRef::TensorCoord;

  /// Internal structure of iterator - made public to enable introspection
  struct Policy {
    static bool const kDivisible =
        !(Shape::kRow % InstructionShape::kM) &&
            !(Shape::kColumn % InstructionShape::kN);

    static_assert(platform::is_same<TensorCoord, MatrixCoord>::value,
      "Layouts must be defined for logical MatrixCoord coordinate space.");

    /// Number of mma operations performed
    using MmaIterations = MatrixShape<
      (Shape::kRow + InstructionShape::kM - 1) / InstructionShape::kM,
      (Shape::kColumn + InstructionShape::kN - 1) / InstructionShape::kN
    >;
  };

private:

  // Assume accumulator tile is an arrangement of 8-by-8 tiles replicated over the entire
  // shape, with each quad mapped to one row and each thread mapped to 1/4 of the elements
  // of that row. The accumulators within one row are assumed to be consecutive.
 static int const kElementsPerAccess = InstructionShape::kN / 4;
 static int const kRowsPerTile = 16;
 static int const kAccumulatorRows = InstructionShape::kM / kRowsPerTile;

public:

  //
  // Derived quantities
  //

  /// Fragment object holding a thread's part of a tile
  using Fragment = Array<
    Element,
    Policy::MmaIterations::kCount * InstructionShape::kMN / kThreads>;

private:

  /// Reference to output tensor
  TensorRef ref_;

public:

  /// Default ctor constructs null iterator
  MCTLASS_HOST_DEVICE
  MmaTensorOpAccumulatorTileIterator() { }

  /// Constructor from TensorRef
  MCTLASS_HOST_DEVICE
  MmaTensorOpAccumulatorTileIterator(
    TensorRef const &ref,
    int lane_id
  ):
    ref_(ref) {

    int lane_in_quad = (lane_id >> 4) << 2;
    int quad = lane_id & 0xf;
    MatrixCoord lane_offset(quad,lane_in_quad);
    ref_.add_coord_offset(lane_offset);

  }

  /// Adds a pointer offset to internal pointer(s) to advance through memory
  MCTLASS_HOST_DEVICE
  MmaTensorOpAccumulatorTileIterator &add_pointer_offset(LongIndex offset) {
    ref_.add_pointer_offset(offset);
    return *this;
  }

  /// Advances an iterator along logical dimensions of matrix in units of whole tiles
  MCTLASS_HOST_DEVICE
  MmaTensorOpAccumulatorTileIterator &add_tile_offset(TensorCoord const &tile_offset) {

    ref_.add_coord_offset(tile_offset * make_Coord(Shape::kRow, Shape::kColumn));

    return *this;
  }

  /// Advances the iterator along the advance dimension
  MCTLASS_HOST_DEVICE
  MmaTensorOpAccumulatorTileIterator & operator++() {
    // deliberate no-op
    return *this;
  }

  /// Advances the iterator along the advance dimension
  MCTLASS_HOST_DEVICE
  MmaTensorOpAccumulatorTileIterator & operator--() {
    // deliberate no-op
    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of the tensor
  MCTLASS_DEVICE
  MmaTensorOpAccumulatorTileIterator & operator+=(TensorCoord const &tile_offset) {
    add_tile_offset(tile_offset);
    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of the tensor
  MCTLASS_DEVICE
  MmaTensorOpAccumulatorTileIterator & operator-=(TensorCoord const &tile_offset) {
    add_tile_offset(make_Coord(-tile_offset.row(), tile_offset.column()));
    return *this;
  }

  /// Loads a fragment from memory at the location pointed to by the iterator.
  MCTLASS_HOST_DEVICE
  void load(Fragment &frag) const {
    load_with_pointer_offset(frag, 0);
  }

  /// Loads a fragment from memory with additional logical offset
  MCTLASS_DEVICE
  void load_with_pointer_offset(
    Fragment &frag,                             ///< fragment to load from the tensor
    Index pointer_offset) const {               ///< loads a tile with a linear offset

    TensorRef offset_ref(ref_);
    offset_ref.add_pointer_offset(pointer_offset);

    MCTLASS_PRAGMA_UNROLL
    for (int mma_n = 0; mma_n < Policy::MmaIterations::kColumn; ++mma_n) {
      MCTLASS_PRAGMA_UNROLL
      for (int mma_m = 0; mma_m < Policy::MmaIterations::kRow; ++mma_m) {

        int mma_accum_start = kAccumulatorRows * kElementsPerAccess *
          (mma_n * Policy::MmaIterations::kRow + mma_m);

        MCTLASS_PRAGMA_UNROLL
        for (int row = 0; row < kAccumulatorRows; ++row) {
          MCTLASS_PRAGMA_UNROLL
          for (int col = 0; col < kElementsPerAccess; ++col) {
            int accum_m = mma_m * InstructionShape::kM * OpDelta::kRow +
                          row * kRowsPerTile;
            int accum_n = mma_n * InstructionShape::kN * OpDelta::kColumn + col;

            frag[mma_accum_start + row * kElementsPerAccess + col] = offset_ref.at({accum_m, accum_n});
          }
        }
      }
    }
  }

  /// Loads a fragment from memory with additional logical offset
  MCTLASS_DEVICE
  void load_with_byte_offset(
    Fragment &frag,                             ///< fragment to load from the tensor
    Index byte_offset) const {                  ///< loads a tile with a linear offset

    load_with_pointer_offset(byte_offset / sizeof(Element));
  }

  /// Loads a fragment from memory with logical offset in units of whole tiles.
  MCTLASS_DEVICE
  void load(
    Fragment &frag,                             ///< fragment to load from the tensor
    TensorCoord const &tile_offset) const {     ///< loads a tile with a logical offset in units of whole tiles

    load(frag, tile_offset, 0);
  }

  /// Loads a fragment from memory with logical offset in units of whole tiles.
  MCTLASS_DEVICE
  void load(
    Fragment &frag,                             ///< fragment to load from the tensor
    TensorCoord const &tile_offset,             ///< loads a tile with a logical offset in units of whole tiles
    Index pointer_offset) const {               ///< loads a tile with a logical offset AND a pointer offset

    load_with_pointer_offset(frag, ref_.offset(tile_offset) + pointer_offset);
  }

  /// Stores a fragment to memory
  MCTLASS_HOST_DEVICE
  void store(Fragment const &frag) const {
    store_with_pointer_offset(frag, 0);
  }

  /// Stores a fragment to memory with additional pointer offset
  MCTLASS_DEVICE
  void store_with_pointer_offset(
    Fragment const &frag,                       ///< fragment to store from the tensor
    Index pointer_offset) const {               ///< store a tile with a linear offset

    TensorRef offset_ref(ref_);
    offset_ref.add_pointer_offset(pointer_offset);

    MCTLASS_PRAGMA_UNROLL
    for (int mma_n = 0; mma_n < Policy::MmaIterations::kColumn; ++mma_n) {
      MCTLASS_PRAGMA_UNROLL
      for (int mma_m = 0; mma_m < Policy::MmaIterations::kRow; ++mma_m) {

        int mma_accum_start = kAccumulatorRows * kElementsPerAccess *
          (mma_n * Policy::MmaIterations::kRow + mma_m);

        MCTLASS_PRAGMA_UNROLL
        for (int row = 0; row < kAccumulatorRows; ++row) {
          MCTLASS_PRAGMA_UNROLL
          for (int col = 0; col < kElementsPerAccess; ++col) {

            int accum_m = mma_m * InstructionShape::kM * OpDelta::kRow + row * kElementsPerAccess;
            int accum_n = mma_n * InstructionShape::kN * OpDelta::kColumn + col;

            int idx = mma_accum_start + row * kElementsPerAccess + col;

            offset_ref.at({accum_m, accum_n}) = frag[idx];

          }
        }
      }

    }

  }

  /// Stores a fragment to memory with additional pointer offset
  MCTLASS_DEVICE
  void store_with_byte_offset(
    Fragment const &frag,                       ///< fragment to store from the tensor
    Index byte_offset) const {                  ///< store a tile with a linear offset

    store_with_pointer_offset(byte_offset / sizeof(Element));
  }

  /// Stores a fragment to memory with logical offset in units of whole tiles.
  MCTLASS_DEVICE
  void store(
    Fragment &frag,                             ///< fragment to store to the tensor
    TensorCoord const &tile_offset) const {     ///< stores a tile with a logical offset in units of whole tiles

    store(frag, tile_offset, 0);
  }

  /// Stores a fragment from memory with logical offset in units of whole tiles.
  MCTLASS_DEVICE
  void store(
      /// fragment to store to the tensor
      Fragment const &frag,
      /// stores a tile with a logical offset in units of whole tiles
      TensorCoord const &tile_offset,
      /// stores a tile with a logical offset AND a pointer offset
      Index pointer_offset) const {
    store_with_pointer_offset(frag, ref_.offset(tile_offset) + pointer_offset);
  }
};

////////////////////////////////////////////////////////////////////////////////

/// This tile iterator is specialized for 64-thread TensorOps. It's used to load from shared
/// memory and therefore must be initialized with a TensorRef to shared memory.
///
/// Satisfies:
///   ReadableRandomAccessContiguousTileIteratorConcept
///
template <
    /// Size of the matrix to load (concept: PitchLinearShape)
    typename Shape_,
    /// Identifies A or B multiplicand
    Operand Operand_,
    /// Data type of elements
    typename Element_,
    /// Shape of one matrix product operation (concept: PitchLinearShape)
    typename InstructionShape_,
    /// Interval between adjacent *MMA instructions (in units of MMA
    /// instructions)
    int OpDelta_,
    /// Number of partitions along K dimension
    int PartitionsK_>
class MmaTensorOpMultiplicandTileIterator<
    Shape_, Operand_, Element_,
    mctlass::layout::MacaTensorOpMultiplicandCongruous<sizeof_bits<Element_>::value,
                                                       int(128 / sizeof(Element_))>,
    InstructionShape_, OpDelta_, 64, PartitionsK_> {
 public:

  /// Shape of tile to load (concept: PitchLinearShape)
  using Shape = Shape_;

  /// Operand tag
  static Operand const kOperand = Operand_;

  static_assert(kOperand == Operand::kA || kOperand== Operand::kB,
    "MmaTensorOpMultiplicandIterator may only be instantiated for A or B operands to warp-level Mma.");

  /// Element type
  using Element = Element_;

  /// Layout of source tile
  using Layout = mctlass::layout::MacaTensorOpMultiplicandCongruous<
      sizeof_bits<Element_>::value, int(128 / sizeof(Element_))>;

  /// Shape of one matrix product operation (concept: GemmShape)
  using InstructionShape = InstructionShape_;

  /// Delta between *MMA operations (in units of *MMA operations, concept: MatrixShape)
  static int const kOpDelta = OpDelta_;

  /// Number of participating threads
  static int const kThreads = 64;

  /// Number of partitions along K dimension
  static int const kPartitionsK = PartitionsK_;

  /// TensorRef type for loading element from a tensor
  using TensorRef = TensorRef<Element, Layout>;

  /// Index type
  using Index = typename TensorRef::Index;

  /// Long Index type
  using LongIndex = typename TensorRef::LongIndex;

  /// Long Index type
  using StrideIndex = typename TensorRef::Layout::Stride::Index;

  /// Coordinate for an element in the tensor
  using TensorCoord = typename TensorRef::TensorCoord;

  /// Internal structure of iterator - made public to enable introspection
  struct Policy {
    static_assert(
        !(Shape::kContiguous % InstructionShape::kContiguous),
        "Shape of warp-level Mma must be divisible by operator shape.");

    // Determine number of elements along outer dimension per individual LDSM op
    static int const kLdsmOpOuter = Layout::kElementsPerAccess;
    static int const kLdsmOpInner = (sizeof(Element_) == 1) ? 4 : 8;

    static_assert(!(Shape::kContiguous % kLdsmOpOuter),
      "Shape of warp-level mma must be divisible by LDSM's fundamental tile size.");

    static_assert(!(Shape::kStrided % kLdsmOpInner),
      "Shape of warp-level mma must be divisible by LDSM's fundamental tile size.");

    /// Shape of one individual LDSM instruction
    static int const LdsmShapeStrided =
        InstructionShape::kStrided / kLdsmOpInner;
    static int const LdsmShapeContiguous = 4 / LdsmShapeStrided;
    using LdsmShape =
        layout::PitchLinearShape<LdsmShapeContiguous, LdsmShapeStrided>;

    /// Number and arrangement of LDSM instructions
    using LdsmIterations = layout::PitchLinearShape<
        Shape::kContiguous / Layout::kElementsPerAccess / LdsmShapeContiguous,
        1>;

    /// Number of groups for each tile
    static int const kGroupsPerTile =
        Shape::kStrided / InstructionShape::kStrided;
  };

private:

  /// Not working on this feature at the moment.
  static_assert(kOpDelta == 1,
    "Alternative arrangements not supported at present.");

  /// Number of internal pointers needed to reference shared memory
  static int const kPointerCount =
      Policy::LdsmIterations::kStrided * Policy::LdsmIterations::kContiguous;

  /// Pointer type used for accesses
  using AccessType = Array<Element, Layout::kElementsPerAccess>;

  /// Internal counter used to jump to next K partition
  int k_group_idx_;

public:

  //
  // Derived quantities
  //

  /// Fragment object holding a thread's part of a tile
 using Fragment =
     Array<Element, Shape::kContiguous * InstructionShape::kStrided / kThreads>;
 using Fragment_int =
     Array<int, Shape::kContiguous * InstructionShape::kStrided / kThreads>;

private:

  /// Layout object storing stride values
  StrideIndex stride_;

  /// Shared memory base pointers - not advanced
  AccessType const *pointer_[kPointerCount];

  /// Byte offset incremented as iterator advances
  Index byte_offset_;

  int bank1_;

public:

  /// Default ctor constructs null iterator
  MCTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator(): stride_(0), byte_offset_(0) { }

  /// Constructor from TensorRef
  MCTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator(
    TensorRef const &ref,
    int lane_id
  ):
    stride_(ref.stride(0)), byte_offset_(0),
    k_group_idx_(0) {

    bank1_ = lane_id & 1;

    MCTLASS_PRAGMA_UNROLL
    for (int s = 0; s < Policy::LdsmIterations::kStrided; ++s) {

      MCTLASS_PRAGMA_UNROLL
      for (int c = 0; c < Policy::LdsmIterations::kContiguous; ++c) {

        int access_idx = c + s * Policy::LdsmIterations::kContiguous;

        int row = (lane_id & (InstructionShape::kContiguous - 1))
                    + c * InstructionShape::kContiguous;
        int col = (((lane_id >> 4) << 2) ^ (InstructionShape::kStrided - 1)) - 3
                    + s * InstructionShape::kStrided;
        Element const *source_ptr = reinterpret_cast<Element const *>(ref.data())
                    + (col * stride_ + row);
        pointer_[access_idx] = reinterpret_cast<AccessType const *>(source_ptr);
      }
    }

  }

  /// Adds a pointer offset to internal pointer(s) to advance through memory
  MCTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator &add_pointer_offset(LongIndex offset) {

    byte_offset_ += offset * sizeof(Element);

    return *this;
  }

  /// Advances an iterator along logical dimensions of matrix in units of whole tiles
  MCTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator &add_tile_offset(TensorCoord const &tile_offset) {

    LongIndex offset = tile_offset.strided() * InstructionShape::kStrided * stride_
                        + tile_offset.contiguous() * Shape::kContiguous;

    byte_offset_ += offset * sizeof(Element);

    return *this;
  }

  /// Advances the iterator along the advance dimension
  MCTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator & operator++() {

    byte_offset_ += stride_ * InstructionShape::kStrided * sizeof(Element);
    return *this;
  }

  /// Advances the iterator along the opposite of the advance dimension
  MCTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator & operator--() {
    byte_offset_ -= stride_ * InstructionShape::kStrided * sizeof(Element);

    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of the tensor
  MCTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator & operator+=(TensorCoord const &tile_offset) {
    add_tile_offset(tile_offset);
    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of the tensor
  MCTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator & operator-=(TensorCoord const &tile_offset) {
    add_tile_offset(-tile_offset);
    return *this;
  }

  MCTLASS_DEVICE
  void fragment_int_conversion(const Fragment_int &frag_in, Fragment &frag) const {
    using LoadIntType = Array<int, Policy::LdsmShape::kCount>;
    using LoadType = Array<int, Policy::LdsmShape::kCount / 2>;

    const LoadIntType *fetch_int_element_ptr = reinterpret_cast<const LoadIntType *>(&frag_in);
    LoadType *fetch_element_ptr = reinterpret_cast<LoadType *>(&frag);

    int hi;
    int lo;

    MCTLASS_PRAGMA_UNROLL
    for (int s = 0; s < Policy::LdsmIterations::kStrided; ++s) {

      MCTLASS_PRAGMA_UNROLL
      for (int c = 0; c < Policy::LdsmIterations::kContiguous; ++c) {
        int access_idx = c + s * Policy::LdsmIterations::kContiguous;
        for (int i = 0; i < Policy::LdsmShape::kCount / 2; ++i) {
          lo = __builtin_mxc_byte_perm(fetch_int_element_ptr[access_idx][2 * i + 1],
                                       fetch_int_element_ptr[access_idx][2 * i], 0x05040100);
          hi = __builtin_mxc_byte_perm(fetch_int_element_ptr[access_idx][2 * i + 1],
                                       fetch_int_element_ptr[access_idx][2 * i], 0x07060302);
          fetch_element_ptr[access_idx][i] = (bank1_)? hi : lo;
        }
      }
    }
  }

  MCTLASS_HOST_DEVICE
  void load(Fragment_int &frag) const {

    load_with_byte_offset(frag, 0);
  }

  MCTLASS_DEVICE
  void load_with_byte_offset(
      /// fragment to load from the tensor
      Fragment_int &frag,
      /// loads a tile with a linear offset in units of bytes
      Index byte_offset) const {

    Array<int, Policy::LdsmShape::kCount> *fetch_element_ptr =
        reinterpret_cast<Array<int, Policy::LdsmShape::kCount> *>(&frag);

    MCTLASS_PRAGMA_UNROLL
    for (int s = 0; s < Policy::LdsmIterations::kStrided; ++s) {

      MCTLASS_PRAGMA_UNROLL
      for (int c = 0; c < Policy::LdsmIterations::kContiguous; ++c) {

        int access_idx = c + s * Policy::LdsmIterations::kContiguous;
        char const *source_ptr = reinterpret_cast<char const *>(pointer_[access_idx])
                                  + byte_offset + byte_offset_;
        // int const *ptr = reinterpret_cast<int const *>(source_ptr);
        Element const *ptr = reinterpret_cast<Element const *>(source_ptr) - bank1_;

        for (int i = 0; i < Policy::LdsmShape::kCount; ++i) {
          Index offset = i * stride_;
          fetch_element_ptr[access_idx][i] = *(int *)(&(ptr[offset]));
        }
      }
    }

  }

  /// Loads a fragment from memory at the location pointed to by the iterator.
  MCTLASS_HOST_DEVICE
  void load(Fragment &frag) const {

    load_with_byte_offset(frag, 0);
  }

  /// Loads a fragment from memory with additional logical offset
  MCTLASS_DEVICE
  void load_with_byte_offset(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a linear offset in units of bytes
      Index byte_offset) const {

    Array<Element, Policy::LdsmShape::kCount> *fetch_element_ptr =
        reinterpret_cast<Array<Element, Policy::LdsmShape::kCount> *>(&frag);

    MCTLASS_PRAGMA_UNROLL
    for (int s = 0; s < Policy::LdsmIterations::kStrided; ++s) {

      MCTLASS_PRAGMA_UNROLL
      for (int c = 0; c < Policy::LdsmIterations::kContiguous; ++c) {

        int access_idx = c + s * Policy::LdsmIterations::kContiguous;
        char const *source_ptr = reinterpret_cast<char const *>(pointer_[access_idx])
                                  + byte_offset + byte_offset_;
        Element const *ptr = reinterpret_cast<Element const *>(source_ptr);

        for (int i = 0; i < Policy::LdsmShape::kCount; ++i) {
          Index offset = i * stride_;
          fetch_element_ptr[access_idx][i] = ptr[offset];
        }

      }
    }

  }

  /// Loads a fragment from memory with additional logical offset
  MCTLASS_DEVICE
  void load_with_pointer_offset(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a linear offset
      Index pointer_offset) const {
    load_with_byte_offset(frag, pointer_offset * sizeof(Element));
  }

  /// Loads a fragment from memory with logical offset in units of whole tiles.
  MCTLASS_DEVICE
  void load(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a logical offset in units of whole tiles
      TensorCoord const &tile_offset) const {
    load_with_byte_offset(frag, tile_offset, 0);
  }

  /// Loads a fragment from memory with logical offset in units of whole tiles.
  MCTLASS_DEVICE
  void load(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a logical offset in units of whole tiles
      TensorCoord const &tile_offset,
      /// loads a tile with a logical offset AND a pointer offset
      Index pointer_offset) const {
    load_with_byte_offset(frag, tile_offset, pointer_offset * sizeof(Element));
  }

  /// Loads a fragment from memory with logical offset in units of whole tiles.
  MCTLASS_DEVICE
  void load_with_byte_offset(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a logical offset in units of whole tiles
      TensorCoord const &tile_offset,
      /// loads a tile with a logical offset AND a pointer offset
      Index byte_offset) const {
    Index pointer_offset =
      tile_offset.contiguous() * Shape::kContiguous / Layout::kElementsPerAccess +
      tile_offset.strided() * InstructionShape::kStrided * stride_ / Layout::kElementsPerAccess;

    byte_offset += sizeof(AccessType) * pointer_offset;

    load_with_byte_offset(frag, byte_offset);
  }

  /// Notify the iterator which k-group it is currently pointing to.
  ///
  /// This does not advance the iterator. Rather, it overrides its internal
  /// tracking with constant-valued k-group index to enable the compiler to
  /// fold constants and achieve more efficient code.
  ///
  /// This is used by some nontrivial permuted layouts.
  MCTLASS_DEVICE
  void set_kgroup_index(int k_group) {
    // no op
  }
};

////////////////////////////////////////////////////////////////////////////////

/// This tile iterator is specialized for 64-thread TensorOps. It's used to load from shared
/// memory and therefore must be initialized with a TensorRef to shared memory.
///
/// Satisfies:
///   ReadableRandomAccessContiguousTileIteratorConcept
///
template <
    /// Size of the matrix to load (concept: MatrixShape)
    typename Shape_,
    /// Identifies A or B multiplicand
    Operand Operand_,
    /// Data type of elements
    typename Element_,
    /// Shape of one matrix product operation (concept: MatrixShape)
    typename InstructionShape_,
    /// Interval between adjacent *MMA instructions (in units of MMA
    /// instructions)
    int OpDelta_,
    /// Number of partitions along K dimension
    int PartitionsK_>
class MmaTensorOpMultiplicandTileIterator<
    Shape_, Operand_, Element_,
    mctlass::layout::MacaColumnMajorTensorOpMultiplicandCongruous<
        sizeof_bits<Element_>::value, int(128 / sizeof(Element_))>,
    InstructionShape_, OpDelta_, 32, PartitionsK_> {
 public:

  /// Shape of tile to load (concept: PitchLinearShape)
  using Shape = Shape_;

  /// Operand tag
  static Operand const kOperand = Operand_;

  static_assert(kOperand == Operand::kA,
                "MmaTensorOpMultiplicandIterator for ColumnMajor Congruous may "
                "only be instantiated for A operand to warp-level Mma.");

  /// Element type
  using Element = Element_;

  /// Layout of source tile
  using Layout = mctlass::layout::MacaColumnMajorTensorOpMultiplicandCongruous<
      sizeof_bits<Element_>::value, int(128 / sizeof(Element_))>;

  /// Shape of one matrix product operation (concept: MatrixShape)
  using InstructionShape = InstructionShape_;

  /// Delta between *MMA operations (in units of *MMA operations, concept: MatrixShape)
  static int const kOpDelta = OpDelta_;

  /// Number of participating threads
  static int const kThreads = 64;

  /// TensorRef type for loading element from a tensor
  using TensorRef = TensorRef<Element, Layout>;

  /// Index type
  using Index = typename TensorRef::Index;

  /// Long Index type
  using LongIndex = typename TensorRef::LongIndex;

  /// Long Index type
  using StrideIndex = typename TensorRef::Layout::Stride::Index;

  /// Coordinate for an element in the tensor
  using TensorCoord = typename TensorRef::TensorCoord;

  /// Underlying tile iterator implementation
  using Base = MmaTensorOpMultiplicandTileIterator<
      layout::PitchLinearShape<Shape::kRow, Shape::kColumn>, kOperand, Element,
      layout::MacaTensorOpMultiplicandCongruous<sizeof_bits<Element_>::value,
                                            int(128 / sizeof(Element_))>,
      layout::PitchLinearShape<InstructionShape::kRow,
                               InstructionShape::kColumn>,
      kOpDelta, kThreads, PartitionsK_>;

 public:

  //
  // Derived quantities
  //

  /// Fragment object holding a thread's part of a tile
  using Fragment = typename Base::Fragment;

private:

  /// Underlying tile iterator
  Base iterator_;

public:

  /// Default ctor constructs null iterator
  MCTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator() { }

  /// Constructor from TensorRef
  MCTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator(
    TensorRef const &ref,
    int lane_id
  ): iterator_({ref.data(), ref.stride()}, lane_id) {
  }

  /// Adds a pointer offset to internal pointer(s) to advance through memory
  MCTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator &add_pointer_offset(LongIndex offset) {

    iterator_.add_pointer_offset(offset);

    return *this;
  }

  /// Advances an iterator along logical dimensions of matrix in units of whole tiles
  MCTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator &add_tile_offset(TensorCoord const &tile_offset) {

    iterator_.add_tile_offset({tile_offset.row(), tile_offset.column()});

    return *this;
  }

  /// Advances the iterator along the advance dimension
  MCTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator & operator++() {

    ++iterator_;

    return *this;
  }

  /// Advances the iterator along the advance dimension
  MCTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator & operator--() {

    --iterator_;

    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of the tensor
  MCTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator & operator+=(TensorCoord const &tile_offset) {
    add_tile_offset(PitchLinearCoord(tile_offset.row(), tile_offset.column()));
    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of the tensor
  MCTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator & operator-=(TensorCoord const &tile_offset) {
    add_tile_offset(-PitchLinearCoord(tile_offset.row(), tile_offset.column()));
    return *this;
  }

  /// Loads a fragment from memory at the location pointed to by the iterator.
  MCTLASS_HOST_DEVICE
  void load(Fragment &frag) const {

    iterator_.load(frag);
  }

  /// Loads a fragment from memory with additional logical offset
  MCTLASS_DEVICE
  void load_with_pointer_offset(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a linear offset
      Index pointer_offset) const {
    iterator_.load_with_pointer_offset(frag, pointer_offset);
  }

  /// Loads a fragment from memory with additional logical offset
  MCTLASS_DEVICE
  void load_with_byte_offset(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a linear offset
      Index byte_offset) const {
    iterator_.load_with_byte_offset(frag, byte_offset);
  }

  /// Loads a fragment from memory with logical offset in units of whole tiles.
  MCTLASS_DEVICE
  void load(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a logical offset in units of whole tiles
      TensorCoord const &tile_offset) const {
    // TODO
  }

  /// Loads a fragment from memory with logical offset in units of whole tiles.
  MCTLASS_DEVICE
  void load(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a logical offset in units of whole tiles
      TensorCoord const &tile_offset,
      /// loads a tile with a logical offset AND a pointer offset
      Index pointer_offset) const {
    // TODO
  }

  /// Loads a fragment from memory with logical offset in units of whole tiles.
  MCTLASS_DEVICE
  void load_with_byte_offset(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a logical offset in units of whole tiles
      TensorCoord const &tile_offset,
      /// loads a tile with a logical offset AND a pointer offset
      Index byte_offset) const {
    iterator_.load_with_byte_offset(
      frag,
      {tile_offset.contiguous(), tile_offset.strided()},
      byte_offset);
  }

  /// Notify the iterator which k-group it is currently pointing to.
  ///
  /// This does not advance the iterator. Rather, it overrides its internal
  /// tracking with constant-valued k-group index to enable the compiler to
  /// fold constants and achieve more efficient code.
  ///
  /// This is used by some nontrivial permuted layouts.
  MCTLASS_DEVICE
  void set_kgroup_index(int k_group) {
    iterator_.set_kgroup_index(k_group);
  }
};

////////////////////////////////////////////////////////////////////////////////

/// This tile iterator is specialized for 64-thread TensorOps. It's used to load from shared
/// memory and therefore must be initialized with a TensorRef to shared memory.
///
/// Satisfies:
///   ReadableRandomAccessContiguousTileIteratorConcept
///
template <
    /// Size of the matrix to load (concept: MatrixShape)
    typename Shape_,
    /// Identifies A or B multiplicand
    Operand Operand_,
    /// Data type of elements
    typename Element_,
    /// Shape of one matrix product operation (concept: MatrixShape)
    typename InstructionShape_,
    /// Interval between adjacent *MMA instructions (in units of MMA
    /// instructions)
    int OpDelta_,
    /// Number of partitions along K dimension
    int PartitionsK_>
class MmaTensorOpMultiplicandTileIterator<
    Shape_, Operand_, Element_,
    mctlass::layout::MacaRowMajorTensorOpMultiplicandCongruous<
        sizeof_bits<Element_>::value, int(128 / sizeof(Element_))>,
    InstructionShape_, OpDelta_, 32, PartitionsK_> {
 public:

  /// Shape of tile to load (concept: PitchLinearShape)
  using Shape = Shape_;

  /// Operand tag
  static Operand const kOperand = Operand_;

  static_assert(kOperand == Operand::kB,
                "MmaTensorOpMultiplicandIterator for RowMajor Congruous may "
                "only be instantiated for B operand to warp-level Mma.");

  /// Element type
  using Element = Element_;

  /// Layout of source tile
  using Layout = mctlass::layout::MacaRowMajorTensorOpMultiplicandCongruous<
      sizeof_bits<Element_>::value, int(128 / sizeof(Element_))>;

  /// Shape of one matrix product operation (concept: MatrixShape)
  using InstructionShape = InstructionShape_;

  /// Delta between *MMA operations (in units of *MMA operations, concept: MatrixShape)
  static int const kOpDelta = OpDelta_;

  /// Number of participating threads
  static int const kThreads = 64;

  /// TensorRef type for loading element from a tensor
  using TensorRef = TensorRef<Element, Layout>;

  /// Index type
  using Index = typename TensorRef::Index;

  /// Long Index type
  using LongIndex = typename TensorRef::LongIndex;

  /// Coordinate for an element in the tensor
  using TensorCoord = typename TensorRef::TensorCoord;

  /// Underlying tile iterator implementation
  using Base = MmaTensorOpMultiplicandTileIterator<
      layout::PitchLinearShape<Shape::kColumn, Shape::kRow>, kOperand, Element,
      layout::MacaTensorOpMultiplicandCongruous<sizeof_bits<Element_>::value,
                                            int(128 / sizeof(Element_))>,
      layout::PitchLinearShape<InstructionShape::kColumn,
                               InstructionShape::kRow>,
      kOpDelta, kThreads, PartitionsK_>;

 public:

  //
  // Derived quantities
  //

  /// Fragment object holding a thread's part of a tile
  using Fragment = typename Base::Fragment;
  using Fragment_int = typename Base::Fragment_int;

private:

  /// Underlying tile iterator
  Base iterator_;

public:

  /// Default ctor constructs null iterator
  MCTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator() { }

  /// Constructor from TensorRef
  MCTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator(
    TensorRef const &ref,
    int lane_id
  ): iterator_({ref.data(), ref.stride()}, lane_id) {
  }

  /// Adds a pointer offset to internal pointer(s) to advance through memory
  MCTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator &add_pointer_offset(LongIndex offset) {

    iterator_.add_pointer_offset(offset);

    return *this;
  }

  /// Advances an iterator along logical dimensions of matrix in units of whole tiles
  MCTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator &add_tile_offset(TensorCoord const &tile_offset) {

    iterator_.add_tile_offset({tile_offset.column(), tile_offset.row()});

    return *this;
  }

  /// Advances the iterator along the advance dimension
  MCTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator & operator++() {

    ++iterator_;

    return *this;
  }

  /// Advances the iterator along the advance dimension
  MCTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator & operator--() {

    --iterator_;

    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of the tensor
  MCTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator & operator+=(TensorCoord const &tile_offset) {
    add_tile_offset(PitchLinearCoord(tile_offset.column(), tile_offset.row()));
    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of the tensor
  MCTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator & operator-=(TensorCoord const &tile_offset) {
    add_tile_offset(-PitchLinearCoord(tile_offset.column(), tile_offset.row()));
    return *this;
  }

  MCTLASS_DEVICE
  void fragment_int_conversion(const Fragment_int &frag_in, Fragment &frag) const {
    iterator_.fragment_int_conversion(frag_in, frag);
  }

 /// Loads a fragment from memory at the location pointed to by the iterator.
  MCTLASS_HOST_DEVICE
  void load(Fragment_int &frag) const {

    iterator_.load(frag);
  }

  /// Loads a fragment from memory with additional logical offset
  MCTLASS_DEVICE
  void load_with_pointer_offset(
      /// fragment to load from the tensor
      Fragment_int &frag,
      /// loads a tile with a linear offset
      Index pointer_offset) const {
    iterator_.load_with_pointer_offset(frag, pointer_offset);
  }

  /// Loads a fragment from memory at the location pointed to by the iterator.
  MCTLASS_HOST_DEVICE
  void load(Fragment &frag) const {

    iterator_.load(frag);
  }

  /// Loads a fragment from memory with additional logical offset
  MCTLASS_DEVICE
  void load_with_pointer_offset(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a linear offset
      Index pointer_offset) const {
    iterator_.load_with_pointer_offset(frag, pointer_offset);
  }

  /// Loads a fragment from memory with additional logical offset
  MCTLASS_DEVICE
  void load_with_byte_offset(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a linear offset
      Index byte_offset) const {
    iterator_.load_with_byte_offset(frag, byte_offset);
  }

  /// Loads a fragment from memory with logical offset in units of whole tiles.
  MCTLASS_DEVICE
  void load(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a logical offset in units of whole tiles
      TensorCoord const &tile_offset) const {
    // TODO
  }

  /// Loads a fragment from memory with logical offset in units of whole tiles.
  MCTLASS_DEVICE
  void load(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a logical offset in units of whole tiles
      TensorCoord const &tile_offset,
      /// loads a tile with a logical offset AND a pointer offset
      Index pointer_offset) const {
    // TODO
  }

  /// Loads a fragment from memory with logical offset in units of whole tiles.
  MCTLASS_DEVICE
  void load_with_byte_offset(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a logical offset in units of whole tiles
      TensorCoord const &tile_offset,
      /// loads a tile with a logical offset AND a pointer offset
      Index byte_offset) const {
    iterator_.load_with_byte_offset(
      frag,
      {tile_offset.strided(), tile_offset.contiguous()},
      byte_offset);
  }

  /// Notify the iterator which k-group it is currently pointing to.
  ///
  /// This does not advance the iterator. Rather, it overrides its internal
  /// tracking with constant-valued k-group index to enable the compiler to
  /// fold constants and achieve more efficient code.
  ///
  /// This is used by some nontrivial permuted layouts.
  MCTLASS_DEVICE
  void set_kgroup_index(int k_group) {
    iterator_.set_kgroup_index(k_group);
  }
};

////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////

/// This tile iterator is specialized for 64-thread TensorOps. It's used to
/// load from shared memory and therefore must be initialized with a TensorRef
/// to shared memory.
///
/// Satisfies:
///   ReadableRandomAccessContiguousTileIteratorConcept
///
template <
    /// Size of the matrix to load (concept: PitchLinearShape)
    typename Shape_,
    /// Identifies A or B multiplicand
    Operand Operand_,
    /// Data type of elements
    typename Element_,
    /// Shape of one matrix product operation (concept: PitchLinearShape)
    typename InstructionShape_,
    /// Interval between adjacent *MMA instructions (in units of MMA
    /// instructions)
    int OpDelta_,
    /// Element number when the layout crosses (in units of elements)
    int Crosswise,
    /// Number of partitions along K dimension
    int PartitionsK_>
class MmaTensorOpMultiplicandTileIterator<
    Shape_, Operand_, Element_,
    mctlass::layout::TensorOpMultiplicandCrosswise<sizeof_bits<Element_>::value,
                                                   Crosswise>,
    InstructionShape_, OpDelta_, 64, PartitionsK_> {
 public:
  /// Shape of tile to load (concept: PitchLinearShape)
  using Shape = Shape_;

  /// Operand tag
  static Operand const kOperand = Operand_;

  static_assert(kOperand == Operand::kA || kOperand == Operand::kB,
                "MmaTensorOpMultiplicandIterator may only be instantiated for "
                "A or B operands to warp-level Mma.");

  /// Element type
  using Element = Element_;

  /// Element number when the layout crosses
  static int const kCrosswise = Crosswise;

  /// Layout of source tile
  using Layout = mctlass::layout::TensorOpMultiplicandCrosswise<
      sizeof_bits<Element_>::value, kCrosswise>;

  /// Shape of one matrix product operation (concept: GemmShape)
  using InstructionShape = InstructionShape_;

  /// Delta between *MMA operations (in units of *MMA operations, concept:
  /// MatrixShape)
  static int const kOpDelta = OpDelta_;

  /// Number of participating threads
  static int const kThreads = 64;

  /// Number of partitions along K dimension
  static int const kPartitionsK = PartitionsK_;

  /// TensorRef type for loading element from a tensor
  using TensorRef = TensorRef<Element, Layout>;

  /// Index type
  using Index = typename TensorRef::Index;

  /// Long Index type
  using LongIndex = typename TensorRef::LongIndex;

  /// Long Index type
  using StrideIndex = typename TensorRef::Layout::Stride::Index;

  /// Coordinate for an element in the tensor
  using TensorCoord = typename TensorRef::TensorCoord;

  /// Internal structure of iterator - made public to enable introspection
  struct Policy {
    static_assert(
        !(Shape::kContiguous % InstructionShape::kContiguous),
        "Shape of warp-level Mma must be divisible by operator shape.");

    // Determine number of elements along outer dimension per individual LDSM op
    static int const kLdsmOpOuter = Layout::kElementsPerAccess;
    static int const kLdsmOpInner = (sizeof(Element)==1) ? 4 : 8;

    static_assert(!(Shape::kContiguous % kLdsmOpOuter),
                  "Shape of warp-level mma must be divisible by LDSM's "
                  "fundamental tile size.");

    static_assert(!(Shape::kStrided % kLdsmOpInner),
                  "Shape of warp-level mma must be divisible by LDSM's "
                  "fundamental tile size.");

    /// Shape of one individual LDSM instruction
    static int const LdsmShapeContiguous =
        InstructionShape::kContiguous / kLdsmOpOuter;
    static int const LdsmShapeStrided =
        ((4 / LdsmShapeContiguous * kLdsmOpInner) > Shape::kStrided)
            ? (Shape::kStrided / kLdsmOpInner)
            : (4 / LdsmShapeContiguous);
    using LdsmShape =
        layout::PitchLinearShape<LdsmShapeContiguous, LdsmShapeStrided>;

    /// Number and arrangement of LDSM instructions
    using LdsmIterations =
        layout::PitchLinearShape<1, Shape::kStrided / kLdsmOpInner /
                                        LdsmShape::kStrided>;

    ///
    static int const kGroupsPerTile = Layout::TileShape::kContiguous /
                                      Layout::kFactor / LdsmShape::kContiguous;
  };

 private:
  /// Not working on this feature at the moment.
  static_assert(kOpDelta == 1,
                "Alternative arrangements not supported at present.");

  /// Pointer type used for accesses
  using AccessType = Array<Element, Layout::kElementsPerAccess>;

 public:
  //
  // Derived quantities
  //

  /// Fragment object holding a thread's part of a tile
  using Fragment = Array<Element, Shape::kStrided *
                                      InstructionShape::kContiguous / kThreads>;
  using Fragment_int =
      Array<int, Shape::kStrided * InstructionShape::kContiguous / kThreads>;

 private:

  /// Total number of sections.  The memory is divided into stages.  One stage
  /// can store one tile.  Stage is divided into sections.  Interleaved layout
  /// can have multiple sections in a stage.  The rest layout only has one section
  /// in a stage.
  int sections_;

  /// Layout object storing stride values
  StrideIndex stride_;

  /// Shared memory base pointers - not advanced
  AccessType const *pointer_;

  /// Byte offset incremented as iterator advances
  Index byte_offset_;

  /// Internal counter used to determine when to increment byte offset and when
  /// to XOR it
  int k_group_idx_;

 public:
  /// Default ctor constructs null iterator
  MCTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator()
      : pointer_(nullptr),
        sections_(0),
        stride_(0),
        byte_offset_(0),
        k_group_idx_(0) {}

  /// Constructor from TensorRef
  MCTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator(TensorRef const &ref, int lane_id)
      : pointer_(reinterpret_cast<AccessType const *>(ref.data())),
        sections_(ref.stride(0) / kCrosswise),
        // stride_ = kCrosswise x sections_ x kFactor
        stride_(ref.stride(0) * Layout::kFactor / Layout::kElementsPerAccess),
        byte_offset_(0),
        k_group_idx_(0) {

    Index row = (((lane_id >> 4) << 2) ^ (InstructionShape::kContiguous - 1)) - 3;
    Index col = lane_id & (InstructionShape::kStrided - 1);

    byte_offset_ = ref.layout()(TensorCoord({row, col})) * (sizeof_bits<Element>::value / 8);

  }

  /// Adds a pointer offset to internal pointer(s) to advance through memory
  MCTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator &add_pointer_offset(LongIndex offset) {
    byte_offset_ += offset * sizeof_bits<Element>::value / 8;

    return *this;
  }

  /// Advances an iterator along logical dimensions of matrix in units of whole
  /// tiles
  MCTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator &add_tile_offset(
      TensorCoord const &tile_offset) {
    int whole_tiles = tile_offset.contiguous() / Policy::kGroupsPerTile;
    int k_groups_delta = tile_offset.contiguous() % Policy::kGroupsPerTile;

    byte_offset_ ^= k_groups_delta * sizeof_bits<Element>::value *
                    Layout::kElementsPerAccess *
                    Policy::LdsmShape::kContiguous / 8;

    pointer_ +=
      tile_offset.strided() * stride_ * Shape::kStrided / Layout::kFactor +
      whole_tiles * stride_ / sections_;

    return *this;
  }

  /// Advances an iterator along logical dimensions of matrix in units of whole
  /// tiles
  MCTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator &add_tile_offset_negative(
      TensorCoord const &tile_offset) {

    int whole_tiles = tile_offset.contiguous() / Policy::kGroupsPerTile;
    int k_groups_delta = tile_offset.contiguous() % Policy::kGroupsPerTile;
    if (k_groups_delta < 0) {
        whole_tiles -= 1;
        k_groups_delta += Policy::kGroupsPerTile;
    }

    if ((Policy::kGroupsPerTile / kPartitionsK) >= 2) {
      byte_offset_ ^= (k_groups_delta & 1) * Policy::LdsmShape::kContiguous *
                        sizeof_bits<Element>::value *
                        Layout::kElementsPerAccess / 8;
    }
    if ((Policy::kGroupsPerTile / kPartitionsK) >= 4) {
      byte_offset_ ^= ((k_groups_delta + (k_group_idx_ & 1)) & 2) *
                        Policy::LdsmShape::kContiguous *
                        sizeof_bits<Element>::value *
                        Layout::kElementsPerAccess / 8;
    }
    if ((Policy::kGroupsPerTile / kPartitionsK) == 8) {
      byte_offset_ ^= ((k_groups_delta + (k_group_idx_ & 3)) & 4) *
                        Policy::LdsmShape::kContiguous *
                        sizeof_bits<Element>::value *
                        Layout::kElementsPerAccess / 8;
    }

    k_group_idx_ += k_groups_delta;
    whole_tiles += k_group_idx_ / (Policy::kGroupsPerTile / kPartitionsK);
    k_group_idx_ = k_group_idx_ % (Policy::kGroupsPerTile / kPartitionsK);

    pointer_ +=
        tile_offset.strided() * stride_ * Shape::kStrided / Layout::kFactor +
        whole_tiles * stride_ / sections_;
    return *this;
  }

  /// Advances the iterator along the advance dimension
  MCTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator &operator++() {


    if ((Policy::kGroupsPerTile / kPartitionsK) > 1) {
      int mask = ((Policy::kGroupsPerTile / kPartitionsK) == 8)
                     ? 3
                     : (((Policy::kGroupsPerTile / kPartitionsK) == 4) ? 1 : 0);
      if (((k_group_idx_ & mask) % 2) == 0)
        byte_offset_ ^= 1 * Policy::LdsmShape::kContiguous *
                        sizeof_bits<Element>::value *
                        Layout::kElementsPerAccess / 8;
      else if ((k_group_idx_ & mask) == 1)
        byte_offset_ ^= 3 * Policy::LdsmShape::kContiguous *
                        sizeof_bits<Element>::value *
                        Layout::kElementsPerAccess / 8;
      else if ((k_group_idx_ & mask) == 3)
        byte_offset_ ^= 7 * Policy::LdsmShape::kContiguous *
                        sizeof_bits<Element>::value *
                        Layout::kElementsPerAccess / 8;
    }

    k_group_idx_++;

    if (k_group_idx_ == (Policy::kGroupsPerTile / kPartitionsK)) {
      k_group_idx_ = 0;
      add_tile_offset({Policy::kGroupsPerTile, 0});
    }

    return *this;
  }

  /// Advances the iterator along the advance dimension
  MCTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator &operator--() { assert(0); }

  ///< advances in units of whole tiles along the logical coordinate space of
  ///< the tensor
  MCTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator &operator+=(
      TensorCoord const &tile_offset) {
    add_tile_offset(tile_offset);
    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of
  ///< the tensor
  MCTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator &operator-=(
      TensorCoord const &tile_offset) {
    add_tile_offset(-tile_offset);
    return *this;
  }

  /// Loads a fragment from memory at the location pointed to by the iterator.
  MCTLASS_HOST_DEVICE
  void load(Fragment &frag) const { load_with_byte_offset(frag, 0); }

  /// Loads a fragment from memory with additional logical offset
  MCTLASS_DEVICE
  void load_with_byte_offset(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a linear offset in units of bytes
      Index byte_offset) const {

    using LoadType = typename platform::conditional<sizeof(Element) ==1,
        Array<int8_t, 4>, Array<float, Policy::LdsmShape::kCount / 2>>::type;
    LoadType *fetch_ptr = reinterpret_cast<LoadType *>(&frag);

    MCTLASS_PRAGMA_UNROLL
    for (int s = 0; s < Policy::LdsmIterations::kStrided; ++s) {
      MCTLASS_PRAGMA_UNROLL
      for (int c = 0; c < Policy::LdsmIterations::kContiguous; ++c) {
        int access_idx = c + s * Policy::LdsmIterations::kContiguous;
        AccessType const *source_ptr =
            pointer_ + Policy::LdsmShape::kContiguous * c +
            Policy::kLdsmOpInner / Layout::kFactor *
                Policy::LdsmShape::kStrided * s * stride_;
        char const *source_byte_ptr =
            reinterpret_cast<char const *>(source_ptr) + byte_offset +
            byte_offset_;

        fetch_ptr[access_idx] = *reinterpret_cast<LoadType const *>(source_byte_ptr);

      }
    }
  }

  /// Loads a fragment from memory with additional logical offset
  MCTLASS_DEVICE
  void load_with_pointer_offset(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a linear offset
      Index pointer_offset) const {
    load_with_byte_offset(frag, pointer_offset * sizeof(Element));
  }

  /// Loads a fragment from memory with logical offset in units of whole tiles.
  MCTLASS_DEVICE
  void load(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a logical offset in units of whole tiles
      TensorCoord const &tile_offset) const {
    load_with_byte_offset(frag, tile_offset, 0);
  }

  /// Loads a fragment from memory with logical offset in units of whole tiles.
  MCTLASS_DEVICE
  void load(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a logical offset in units of whole tiles
      TensorCoord const &tile_offset,
      /// loads a tile with a logical offset AND a pointer offset
      Index pointer_offset) const {
    load_with_byte_offset(frag, tile_offset, pointer_offset * sizeof(Element));
  }

  /// Loads a fragment from memory with logical offset in units of whole tiles.
  MCTLASS_DEVICE
  void load_with_byte_offset(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a logical offset in units of whole tiles
      TensorCoord const &tile_offset,
      /// loads a tile with a logical offset AND a pointer offset
      Index byte_offset) const {
    Index pointer_offset = tile_offset.contiguous() *
                               InstructionShape::kContiguous /
                               Layout::kElementsPerAccess +
                           tile_offset.strided() * Shape::kStrided * stride_;

    byte_offset += sizeof_bits<AccessType>::value * pointer_offset / 8;

    load_with_byte_offset(frag, byte_offset);
  }

  /// Notify the iterator which k-group it is currently pointing to.
  ///
  /// This does not advance the iterator. Rather, it overrides its internal
  /// tracking with constant-valued k-group index to enable the compiler to
  /// fold constants and achieve more efficient code.
  ///
  /// This is used by some nontrivial permuted layouts.
  MCTLASS_DEVICE
  void set_kgroup_index(int k_group) {
    k_group_idx_ = k_group % (Policy::kGroupsPerTile / kPartitionsK);
  }
};

////////////////////////////////////////////////////////////////////////////////

/// This tile iterator is specialized for 64-thread TensorOps. It's used to
/// load from shared memory and therefore must be initialized with a TensorRef
/// to shared memory.
///
/// Satisfies:
///   ReadableRandomAccessContiguousTileIteratorConcept
///
template <
    /// Size of the matrix to load (concept: MatrixShape)
    typename Shape_,
    /// Identifies A or B multiplicand
    Operand Operand_,
    /// Data type of elements
    typename Element_,
    /// Shape of one matrix product operation (concept: MatrixShape)
    typename InstructionShape_,
    /// Interval between adjacent *MMA instructions (in units of MMA
    /// instructions)
    int OpDelta_,
    /// Element number when the layout crosses (in units of elements)
    int Crosswise,
    /// Number of partitions along K dimension
    int PartitionsK_>
class MmaTensorOpMultiplicandTileIterator<
    Shape_, Operand_, Element_,
    mctlass::layout::MacaColumnMajorTensorOpMultiplicandCrosswise<
        sizeof_bits<Element_>::value, Crosswise>,
    InstructionShape_, OpDelta_, 32, PartitionsK_> {
 public:
  /// Shape of tile to load (concept: PitchLinearShape)
  using Shape = Shape_;

  /// Operand tag
  static Operand const kOperand = Operand_;

  static_assert(kOperand == Operand::kB,
                "MmaTensorOpMultiplicandIterator for ColumnMajor Crosswise may "
                "only be instantiated for B operand to warp-level Mma.");

  /// Element type
  using Element = Element_;

  /// KBlock size
  static int const kCrosswise = Crosswise;

  /// Layout of source tile
  using Layout = mctlass::layout::MacaColumnMajorTensorOpMultiplicandCrosswise<
      sizeof_bits<Element_>::value, kCrosswise>;

  /// Shape of one matrix product operation (concept: MatrixShape)
  using InstructionShape = InstructionShape_;

  /// Delta between *MMA operations (in units of *MMA operations, concept:
  /// MatrixShape)
  static int const kOpDelta = OpDelta_;

  /// Number of participating threads
  static int const kThreads = 64;

  /// TensorRef type for loading element from a tensor
  using TensorRef = TensorRef<Element, Layout>;

  /// Index type
  using Index = typename TensorRef::Index;

  /// Long Index type
  using LongIndex = typename TensorRef::LongIndex;

  /// Coordinate for an element in the tensor
  using TensorCoord = typename TensorRef::TensorCoord;

  /// Underlying tile iterator implementation
  using Base = MmaTensorOpMultiplicandTileIterator<
      layout::PitchLinearShape<Shape::kRow, Shape::kColumn>, kOperand, Element,
      layout::TensorOpMultiplicandCrosswise<sizeof_bits<Element_>::value,
                                            kCrosswise>,
      layout::PitchLinearShape<InstructionShape::kRow,
                               InstructionShape::kColumn>,
      kOpDelta, kThreads, PartitionsK_>;

 public:
  //
  // Derived quantities
  //

  /// Fragment object holding a thread's part of a tile
  using Fragment = typename Base::Fragment;

 private:
  /// Underlying tile iterator
  Base iterator_;

 public:
  /// Default ctor constructs null iterator
  MCTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator() {}

  /// Constructor from TensorRef
  MCTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator(TensorRef const &ref, int lane_id)
      : iterator_({ref.data(), ref.stride()}, lane_id) {}

  /// Adds a pointer offset to internal pointer(s) to advance through memory
  MCTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator &add_pointer_offset(LongIndex offset) {
    iterator_.add_pointer_offset(offset);

    return *this;
  }

  /// Advances an iterator along logical dimensions of matrix in units of whole
  /// tiles
  MCTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator &add_tile_offset(
      TensorCoord const &tile_offset) {
    iterator_.add_tile_offset({tile_offset.row(), tile_offset.column()});

    return *this;
  }

  /// Advances an iterator along logical dimensions of matrix in units of whole
  /// tiles
  MCTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator &add_tile_offset_negative(
      TensorCoord const &tile_offset) {
    iterator_.add_tile_offset_negative({tile_offset.row(), tile_offset.column()});

    return *this;
  }

  /// Advances the iterator along the advance dimension
  MCTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator &operator++() {
    ++iterator_;

    return *this;
  }

  /// Advances the iterator along the advance dimension
  MCTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator &operator--() {
    --iterator_;

    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of
  ///< the tensor
  MCTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator &operator+=(
      TensorCoord const &tile_offset) {
    add_tile_offset(PitchLinearCoord(tile_offset.row(), tile_offset.column()));
    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of
  ///< the tensor
  MCTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator &operator-=(
      TensorCoord const &tile_offset) {
    add_tile_offset(-PitchLinearCoord(tile_offset.row(), tile_offset.column()));
    return *this;
  }

  /// Loads a fragment from memory at the location pointed to by the iterator.
  MCTLASS_HOST_DEVICE
  void load(Fragment &frag) const { iterator_.load(frag); }

  /// Loads a fragment from memory with additional logical offset
  MCTLASS_DEVICE
  void load_with_pointer_offset(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a linear offset
      Index pointer_offset) const {
    iterator_.load_with_pointer_offset(frag, pointer_offset);
  }

  /// Loads a fragment from memory with additional logical offset
  MCTLASS_DEVICE
  void load_with_byte_offset(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a linear offset
      Index byte_offset) const {
    iterator_.load_with_byte_offset(frag, byte_offset);
  }

  /// Loads a fragment from memory with logical offset in units of whole tiles.
  MCTLASS_DEVICE
  void load(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a logical offset in units of whole tiles
      TensorCoord const &tile_offset) const {
    // TODO
    assert(0);
  }

  /// Loads a fragment from memory with logical offset in units of whole tiles.
  MCTLASS_DEVICE
  void load(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a logical offset in units of whole tiles
      TensorCoord const &tile_offset,
      /// loads a tile with a logical offset AND a pointer offset
      Index pointer_offset) const {
    // TODO
    assert(0);
  }

  /// Loads a fragment from memory with logical offset in units of whole tiles.
  MCTLASS_DEVICE
  void load_with_byte_offset(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a logical offset in units of whole tiles
      TensorCoord const &tile_offset,
      /// loads a tile with a logical offset AND a pointer offset
      Index byte_offset) const {
    iterator_.load_with_byte_offset(
        frag, {tile_offset.contiguous(), tile_offset.strided()}, byte_offset);
  }

  /// Notify the iterator which k-group it is currently pointing to.
  ///
  /// This does not advance the iterator. Rather, it overrides its internal
  /// tracking with constant-valued k-group index to enable the compiler to
  /// fold constants and achieve more efficient code.
  ///
  /// This is used by some nontrivial permuted layouts.
  MCTLASS_DEVICE
  void set_kgroup_index(int k_group) {
    iterator_.set_kgroup_index(k_group);
  }
};

////////////////////////////////////////////////////////////////////////////////

/// This tile iterator is specialized for 32-thread TensorOps. It uses LDSM to
/// load from shared memory and therefore must be initialized with a TensorRef
/// to shared memory.
///
/// Satisfies:
///   ReadableRandomAccessContiguousTileIteratorConcept
///
template <
    /// Size of the matrix to load (concept: MatrixShape)
    typename Shape_,
    /// Identifies A or B multiplicand
    Operand Operand_,
    /// Data type of elements
    typename Element_,
    /// Shape of one matrix product operation (concept: MatrixShape)
    typename InstructionShape_,
    /// Interval between adjacent *MMA instructions (in units of MMA
    /// instructions)
    int OpDelta_,
    /// Element number when the layout crosses (in units of elements)
    int Crosswise,
    /// Number of partitions along K dimension
    int PartitionsK_>
class MmaTensorOpMultiplicandTileIterator<
    Shape_, Operand_, Element_,
    mctlass::layout::MacaRowMajorTensorOpMultiplicandCrosswise<
        sizeof_bits<Element_>::value, Crosswise>,
    InstructionShape_, OpDelta_, 32, PartitionsK_> {
 public:
  /// Shape of tile to load (concept: PitchLinearShape)
  using Shape = Shape_;

  /// Operand tag
  static Operand const kOperand = Operand_;

  static_assert(kOperand == Operand::kA,
                "MmaTensorOpMultiplicandIterator for RowMajor Crosswise may "
                "only be instantiated for A operand to warp-level Mma.");

  /// Element type
  using Element = Element_;

  /// Element number when the layout crosses
  static int const kCrosswise = Crosswise;

  /// Layout of source tile
  using Layout = mctlass::layout::MacaRowMajorTensorOpMultiplicandCrosswise<
      sizeof_bits<Element_>::value, kCrosswise>;

  /// Shape of one matrix product operation (concept: MatrixShape)
  using InstructionShape = InstructionShape_;

  /// Delta between *MMA operations (in units of *MMA operations, concept:
  /// MatrixShape)
  static int const kOpDelta = OpDelta_;

  /// Number of participating threads
  static int const kThreads = 64;

  /// TensorRef type for loading element from a tensor
  using TensorRef = TensorRef<Element, Layout>;

  /// Index type
  using Index = typename TensorRef::Index;

  /// Long Index type
  using LongIndex = typename TensorRef::LongIndex;

  /// Coordinate for an element in the tensor
  using TensorCoord = typename TensorRef::TensorCoord;

  /// Underlying tile iterator implementation
  using Base = MmaTensorOpMultiplicandTileIterator<
      layout::PitchLinearShape<Shape::kColumn, Shape::kRow>, kOperand, Element,
      layout::TensorOpMultiplicandCrosswise<sizeof_bits<Element_>::value,
                                            kCrosswise>,
      layout::PitchLinearShape<InstructionShape::kColumn,
                               InstructionShape::kRow>,
      kOpDelta, kThreads, PartitionsK_>;

 public:
  //
  // Derived quantities
  //

  /// Fragment object holding a thread's part of a tile
  using Fragment = typename Base::Fragment;

 private:
  /// Underlying tile iterator
  Base iterator_;

 public:
  /// Default ctor constructs null iterator
  MCTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator() {}

  /// Constructor from TensorRef
  MCTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator(TensorRef const &ref, int lane_id)
      : iterator_({ref.data(), ref.stride()}, lane_id) {}

  /// Adds a pointer offset to internal pointer(s) to advance through memory
  MCTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator &add_pointer_offset(LongIndex offset) {
    iterator_.add_pointer_offset(offset);

    return *this;
  }

  /// Advances an iterator along logical dimensions of matrix in units of whole
  /// tiles
  MCTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator &add_tile_offset(
      TensorCoord const &tile_offset) {
    iterator_.add_tile_offset({tile_offset.column(), tile_offset.row()});

    return *this;
  }

  /// Advances an iterator along logical dimensions of matrix in units of whole
  /// tiles
  MCTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator &add_tile_offset_negative(
      TensorCoord const &tile_offset) {
    iterator_.add_tile_offset_negative({tile_offset.column(), tile_offset.row()});

    return *this;
  }

  /// Advances the iterator along the advance dimension
  MCTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator &operator++() {
    ++iterator_;

    return *this;
  }

  /// Advances the iterator along the advance dimension
  MCTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator &operator--() {
    --iterator_;

    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of
  ///< the tensor
  MCTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator &operator+=(
      TensorCoord const &tile_offset) {
    add_tile_offset(PitchLinearCoord(tile_offset.column(), tile_offset.row()));
    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of
  ///< the tensor
  MCTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator &operator-=(
      TensorCoord const &tile_offset) {
    add_tile_offset(-PitchLinearCoord(tile_offset.column(), tile_offset.row()));
    return *this;
  }

  /// Loads a fragment from memory at the location pointed to by the iterator.
  MCTLASS_HOST_DEVICE
  void load(Fragment &frag) const { iterator_.load(frag); }

  /// Loads a fragment from memory with additional logical offset
  MCTLASS_DEVICE
  void load_with_pointer_offset(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a linear offset
      Index pointer_offset) const {
    iterator_.load_with_pointer_offset(frag, pointer_offset);
  }

  /// Loads a fragment from memory with additional logical offset
  MCTLASS_DEVICE
  void load_with_byte_offset(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a linear offset
      Index byte_offset) const {
    iterator_.load_with_byte_offset(frag, byte_offset);
  }

  /// Loads a fragment from memory with logical offset in units of whole tiles.
  MCTLASS_DEVICE
  void load(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a logical offset in units of whole tiles
      TensorCoord const &tile_offset) const {
    // TODO
    assert(0);
  }

  /// Loads a fragment from memory with logical offset in units of whole tiles.
  MCTLASS_DEVICE
  void load(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a logical offset in units of whole tiles
      TensorCoord const &tile_offset,
      /// loads a tile with a logical offset AND a pointer offset
      Index pointer_offset) const {
    // TODO
    assert(0);
  }

  /// Loads a fragment from memory with logical offset in units of whole tiles.
  MCTLASS_DEVICE
  void load_with_byte_offset(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a logical offset in units of whole tiles
      TensorCoord const &tile_offset,
      /// loads a tile with a logical offset AND a pointer offset
      Index byte_offset) const {
    iterator_.load_with_byte_offset(
        frag, {tile_offset.strided(), tile_offset.contiguous()}, byte_offset);
  }

  /// Notify the iterator which k-group it is currently pointing to.
  ///
  /// This does not advance the iterator. Rather, it overrides its internal
  /// tracking with constant-valued k-group index to enable the compiler to
  /// fold constants and achieve more efficient code.
  ///
  /// This is used by some nontrivial permuted layouts.
  MCTLASS_DEVICE
  void set_kgroup_index(int k_group) {
    iterator_.set_kgroup_index(k_group);
  }
};

////////////////////////////////////////////////////////////////////////////////

/// This tile iterator is specialized for 64-thread TensorOps. It's used to
/// load from shared memory and therefore must be initialized with a TensorRef
/// to shared memory.
///
/// Satisfies:
///   ReadableRandomAccessContiguousTileIteratorConcept
///
template <
    /// Size of the matrix to load (concept: PitchLinearShape)
    typename Shape_,
    /// Identifies A or B multiplicand
    Operand Operand_,
    /// Data type of elements
    typename Element_,
    /// Shape of one matrix product operation (concept: PitchLinearShape)
    typename InstructionShape_,
    /// Interval between adjacent *MMA instructions (in units of MMA
    /// instructions)
    int OpDelta_,
    /// Element number when the layout crosses (in units of elements)
    int Crosswise,
    /// Number of partitions along K dimension
    int PartitionsK_,
    /// Number of partitions along M dimension
    int PartitionsM_
    >
class MmaTensorOpMultiplicandTileIterator<
    Shape_, Operand_, Element_,
    mctlass::layout::MacaTensorOpMultiplicandCrosswise<sizeof_bits<Element_>::value, Crosswise>,
    InstructionShape_, OpDelta_, 64, PartitionsK_, PartitionsM_> {
 public:
  /// Shape of tile to load (concept: PitchLinearShape)
  using Shape = Shape_;

  /// Operand tag
  static Operand const kOperand = Operand_;

  static_assert(kOperand == Operand::kA || kOperand == Operand::kB,
                "MmaTensorOpMultiplicandIterator may only be instantiated for "
                "A or B operands to warp-level Mma.");

  /// Element type
  using Element = Element_;

  /// Element number when the layout crosses
  static int const kCrosswise = Crosswise;

  /// Layout of source tile
  using Layout = mctlass::layout::MacaTensorOpMultiplicandCrosswise<sizeof_bits<Element_>::value,
                 Crosswise>;

  /// Shape of one matrix product operation (concept: GemmShape)
  using InstructionShape = InstructionShape_;

  /// Delta between *MMA operations (in units of *MMA operations, concept:
  /// MatrixShape)
  static int const kOpDelta = OpDelta_;

  /// Number of participating threads
  static int const kThreads = 64;

  /// Number of partitions along K dimension
  static int const kPartitionsK = PartitionsK_;

  /// Number of partitions along M dimension
  static int const PartitionsM = PartitionsM_;

  /// TensorRef type for loading element from a tensor
  using TensorRef = TensorRef<Element, Layout>;

  /// Index type
  using Index = typename TensorRef::Index;

  /// Long Index type
  using LongIndex = typename TensorRef::LongIndex;

  /// Long Index type
  using StrideIndex = typename TensorRef::Layout::Stride::Index;

  /// Coordinate for an element in the tensor
  using TensorCoord = typename TensorRef::TensorCoord;

  /// Internal structure of iterator - made public to enable introspection
  struct Policy {
    static_assert(
        !(Shape::kContiguous % InstructionShape::kContiguous),
        "Shape of warp-level Mma must be divisible by operator shape.");

    // Determine number of elements along outer dimension per individual LDSM op
    static int const kLdsmOpOuter = Layout::kElementsPerAccess;
    static int const kLdsmOpInner = (sizeof(Element) ==1) ? 4 : 8;

    static_assert(!(Shape::kContiguous % kLdsmOpOuter),
                  "Shape of warp-level mma must be divisible by LDSM's "
                  "fundamental tile size.");

    static_assert(!(Shape::kStrided % kLdsmOpInner),
                  "Shape of warp-level mma must be divisible by LDSM's "
                  "fundamental tile size.");

    /// Shape of one individual LDSM instruction
    static int const LdsmShapeContiguous =
        InstructionShape::kContiguous / kLdsmOpOuter;
    static int const LdsmShapeStrided =
        ((4 / LdsmShapeContiguous * kLdsmOpInner) > Shape::kStrided)
            ? (Shape::kStrided / kLdsmOpInner)
            : (4 / LdsmShapeContiguous);
    using LdsmShape =
        layout::PitchLinearShape<LdsmShapeContiguous, LdsmShapeStrided>;

    /// Number and arrangement of LDSM instructions
    using LdsmIterations =
        layout::PitchLinearShape<1, Shape::kStrided / kLdsmOpInner /
                                        LdsmShape::kStrided>;

    ///
    static int const kGroupsPerTile = Layout::TileShape::kContiguous /
                                      Layout::kFactor / LdsmShape::kContiguous;
  };

 private:
  /// Not working on this feature at the moment.
  static_assert(kOpDelta == 1,
                "Alternative arrangements not supported at present.");

  /// Pointer type used for accesses
  using AccessType = Array<Element, Layout::kElementsPerAccess>;

 public:
  //
  // Derived quantities
  //

  /// Fragment object holding a thread's part of a tile
  using Fragment = Array<Element, Shape::kStrided *
                                      InstructionShape::kContiguous / kThreads>;

 private:

  /// Total number of sections.  The memory is divided into stages.  One stage
  /// can store one tile.  Stage is divided into sections.  Interleaved layout
  /// can have multiple sections in a stage.  The rest layout only has one section
  /// in a stage.
  int sections_;

  /// Layout object storing stride values
  StrideIndex stride_;

  /// Shared memory base pointers - not advanced
  AccessType const *pointer_;

  /// Byte offset incremented as iterator advances
  Index byte_offset_;

  /// Internal counter used to determine when to increment byte offset and when
  /// to XOR it
  int k_group_idx_;

 public:
  /// Default ctor constructs null iterator
  MCTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator()
      : pointer_(nullptr),
        sections_(0),
        stride_(0),
        byte_offset_(0),
        k_group_idx_(0) {}

  /// Constructor from TensorRef
  MCTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator(TensorRef const &ref, int lane_id)
      : pointer_(reinterpret_cast<AccessType const *>(ref.data())),
        sections_(ref.stride(0) / kCrosswise),
        // stride_ = kCrosswise x sections_ x kFactor
        stride_(ref.stride(0) * Layout::kFactor / Layout::kElementsPerAccess),
        byte_offset_(0),
        k_group_idx_(0) {

    Index row = (((lane_id >> 4) << 2) ^ (InstructionShape::kContiguous - 1)) - 3;
    Index col = lane_id & (InstructionShape::kStrided - 1);

    byte_offset_ = ref.layout()(TensorCoord({row, col})) * (sizeof_bits<Element>::value / 8);

  }

  /// Adds a pointer offset to internal pointer(s) to advance through memory
  MCTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator &add_pointer_offset(LongIndex offset) {
    byte_offset_ += offset * sizeof_bits<Element>::value / 8;

    return *this;
  }

  /// Advances an iterator along logical dimensions of matrix in units of whole
  /// tiles
  MCTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator &add_tile_offset(
      TensorCoord const &tile_offset) {
    int k_groups_delta = tile_offset.contiguous() % Policy::kGroupsPerTile;

    byte_offset_ ^= k_groups_delta * sizeof_bits<Element>::value *
                    Layout::kElementsPerAccess *
                    Policy::LdsmShape::kContiguous / 8;

    pointer_ +=
      tile_offset.strided() * stride_ * Shape::kStrided / Layout::kFactor;
    return *this;
  }

  /// Advances an iterator along logical dimensions of matrix in units of whole
  /// tiles
  MCTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator &add_tile_offset_negative(
      TensorCoord const &tile_offset) {

    int whole_tiles = tile_offset.contiguous() / Policy::kGroupsPerTile;
    int k_groups_delta = tile_offset.contiguous() % Policy::kGroupsPerTile;
    if (k_groups_delta < 0) {
        whole_tiles -= 1;
        k_groups_delta += Policy::kGroupsPerTile;
    }

    if ((Policy::kGroupsPerTile / kPartitionsK) >= 2) {
      byte_offset_ ^= (k_groups_delta & 1) * Policy::LdsmShape::kContiguous *
                        sizeof_bits<Element>::value *
                        Layout::kElementsPerAccess / 8;
    }
    if ((Policy::kGroupsPerTile / kPartitionsK) >= 4) {
      byte_offset_ ^= ((k_groups_delta + (k_group_idx_ & 1)) & 2) *
                        Policy::LdsmShape::kContiguous *
                        sizeof_bits<Element>::value *
                        Layout::kElementsPerAccess / 8;
    }
    if ((Policy::kGroupsPerTile / kPartitionsK) == 8) {
      byte_offset_ ^= ((k_groups_delta + (k_group_idx_ & 3)) & 4) *
                        Policy::LdsmShape::kContiguous *
                        sizeof_bits<Element>::value *
                        Layout::kElementsPerAccess / 8;
    }

    k_group_idx_ += k_groups_delta;
    whole_tiles += k_group_idx_ / (Policy::kGroupsPerTile / kPartitionsK);
    k_group_idx_ = k_group_idx_ % (Policy::kGroupsPerTile / kPartitionsK);

    pointer_ +=
        tile_offset.strided() * stride_ * Shape::kStrided / Layout::kFactor +
        whole_tiles * stride_ / sections_;
    return *this;
  }

  /// Advances the iterator along the advance dimension
  MCTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator &operator++() {

    if ((Policy::kGroupsPerTile / kPartitionsK) > 1) {
      int mask = ((Policy::kGroupsPerTile / kPartitionsK) == 8)
                     ? 3
                     : (((Policy::kGroupsPerTile / kPartitionsK) == 4) ? 1 : 0);
      if (((k_group_idx_ & mask) % 2) == 0)
        byte_offset_ ^= 1 * Policy::LdsmShape::kContiguous *
                        sizeof_bits<Element>::value *
                        Layout::kElementsPerAccess / 8;
      else if ((k_group_idx_ & mask) == 1)
        byte_offset_ ^= 3 * Policy::LdsmShape::kContiguous *
                        sizeof_bits<Element>::value *
                        Layout::kElementsPerAccess / 8;
      else if ((k_group_idx_ & mask) == 3)
        byte_offset_ ^= 7 * Policy::LdsmShape::kContiguous *
                        sizeof_bits<Element>::value *
                        Layout::kElementsPerAccess / 8;
    }

    k_group_idx_++;

    if (k_group_idx_ == (Policy::kGroupsPerTile / kPartitionsK)) {
      k_group_idx_ = 0;
      add_tile_offset({0, PartitionsM});
    }

    return *this;
  }

  /// Advances the iterator along the advance dimension
  MCTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator &operator--() { assert(0); }

  ///< advances in units of whole tiles along the logical coordinate space of
  ///< the tensor
  MCTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator &operator+=(
      TensorCoord const &tile_offset) {
    add_tile_offset(tile_offset);
    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of
  ///< the tensor
  MCTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator &operator-=(
      TensorCoord const &tile_offset) {
    add_tile_offset(-tile_offset);
    return *this;
  }

  /// Loads a fragment from memory at the location pointed to by the iterator.
  MCTLASS_HOST_DEVICE
  void load(Fragment &frag) const { load_with_byte_offset(frag, 0); }

  /// Loads a fragment from memory with additional logical offset
  MCTLASS_DEVICE
  void load_with_byte_offset(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a linear offset in units of bytes
      Index byte_offset) const {

    using LoadType = typename platform::conditional<sizeof(Element) ==1,
        Array<int8_t, Policy::LdsmShape::kCount>,
        Array<float, Policy::LdsmShape::kCount / 2>>::type;
    LoadType *fetch_ptr = reinterpret_cast<LoadType *>(&frag);

    MCTLASS_PRAGMA_UNROLL
    for (int s = 0; s < Policy::LdsmIterations::kStrided; ++s) {
      MCTLASS_PRAGMA_UNROLL
      for (int c = 0; c < Policy::LdsmIterations::kContiguous; ++c) {
        int access_idx = c + s * Policy::LdsmIterations::kContiguous;
        AccessType const *source_ptr =
            pointer_ + Policy::LdsmShape::kContiguous * c +
            Policy::kLdsmOpInner / Layout::kFactor *
                Policy::LdsmShape::kStrided * s * stride_;
        char const *source_byte_ptr =
            reinterpret_cast<char const *>(source_ptr) + byte_offset +
            byte_offset_;

        fetch_ptr[access_idx] = *reinterpret_cast<LoadType const *>(source_byte_ptr);
      }
    }
  }

  /// Loads a fragment from memory with additional logical offset
  MCTLASS_DEVICE
  void load_with_pointer_offset(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a linear offset
      Index pointer_offset) const {
    load_with_byte_offset(frag, pointer_offset * sizeof(Element));
  }

  /// Loads a fragment from memory with logical offset in units of whole tiles.
  MCTLASS_DEVICE
  void load(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a logical offset in units of whole tiles
      TensorCoord const &tile_offset) const {
    load_with_byte_offset(frag, tile_offset, 0);
  }

  /// Loads a fragment from memory with logical offset in units of whole tiles.
  MCTLASS_DEVICE
  void load(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a logical offset in units of whole tiles
      TensorCoord const &tile_offset,
      /// loads a tile with a logical offset AND a pointer offset
      Index pointer_offset) const {
    load_with_byte_offset(frag, tile_offset, pointer_offset * sizeof(Element));
  }

  /// Loads a fragment from memory with logical offset in units of whole tiles.
  MCTLASS_DEVICE
  void load_with_byte_offset(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a logical offset in units of whole tiles
      TensorCoord const &tile_offset,
      /// loads a tile with a logical offset AND a pointer offset
      Index byte_offset) const {
    Index pointer_offset = tile_offset.contiguous() *
                               InstructionShape::kContiguous /
                               Layout::kElementsPerAccess +
                           tile_offset.strided() * Shape::kStrided * stride_;

    byte_offset += sizeof_bits<AccessType>::value * pointer_offset / 8;

    load_with_byte_offset(frag, byte_offset);
  }

  /// Notify the iterator which k-group it is currently pointing to.
  ///
  /// This does not advance the iterator. Rather, it overrides its internal
  /// tracking with constant-valued k-group index to enable the compiler to
  /// fold constants and achieve more efficient code.
  ///
  /// This is used by some nontrivial permuted layouts.
  MCTLASS_DEVICE
  void set_kgroup_index(int k_group) {
    k_group_idx_ = k_group % (Policy::kGroupsPerTile / kPartitionsK);
  }
};

/// This tile iterator is specialized for 32-thread TensorOps. It uses LDSM to
/// load from shared memory and therefore must be initialized with a TensorRef
/// to shared memory.
///
/// Satisfies:
///   ReadableRandomAccessContiguousTileIteratorConcept
///
template <
    /// Size of the matrix to load (concept: MatrixShape)
    typename Shape_,
    /// Identifies A or B multiplicand
    Operand Operand_,
    /// Data type of elements
    typename Element_,
    /// Shape of one matrix product operation (concept: MatrixShape)
    typename InstructionShape_,
    /// Interval between adjacent *MMA instructions (in units of MMA
    /// instructions)
    int OpDelta_,
    /// Element number when the layout crosses (in units of elements)
    int Crosswise,
    /// Number of partitions along K dimension
    int PartitionsK_,
    /// Number of partitions along M dimension
    int PartitionsM_>
class MmaTensorOpMultiplicandTileIterator<
    Shape_, Operand_, Element_,
    mctlass::layout::MacaRowMajorTensorOpMultiplicandCpAsyncCrosswise<
        sizeof_bits<Element_>::value, Crosswise>,
    InstructionShape_, OpDelta_, 32, PartitionsK_, PartitionsM_> {
 public:
  /// Shape of tile to load (concept: PitchLinearShape)
  using Shape = Shape_;

  /// Operand tag
  static Operand const kOperand = Operand_;

  static_assert(kOperand == Operand::kA,
                "MmaTensorOpMultiplicandIterator for RowMajor Crosswise may "
                "only be instantiated for A operand to warp-level Mma.");

  /// Element type
  using Element = Element_;

  /// Element number when the layout crosses
  static int const kCrosswise = Crosswise;

  /// Layout of source tile
  using Layout = mctlass::layout::MacaRowMajorTensorOpMultiplicandCpAsyncCrosswise<
      sizeof_bits<Element_>::value, kCrosswise>;

  /// Shape of one matrix product operation (concept: MatrixShape)
  using InstructionShape = InstructionShape_;

  /// Delta between *MMA operations (in units of *MMA operations, concept:
  /// MatrixShape)
  static int const kOpDelta = OpDelta_;

  /// Number of participating threads
  static int const kThreads = 64;

  /// TensorRef type for loading element from a tensor
  using TensorRef = TensorRef<Element, Layout>;

  /// Index type
  using Index = typename TensorRef::Index;

  /// Long Index type
  using LongIndex = typename TensorRef::LongIndex;

  /// Coordinate for an element in the tensor
  using TensorCoord = typename TensorRef::TensorCoord;

  /// Underlying tile iterator implementation
  using Base = MmaTensorOpMultiplicandTilePrecomputeIterator<
      layout::PitchLinearShape<Shape::kColumn, Shape::kRow>, kOperand, Element,
      layout::MacaTensorOpMultiplicandCrosswise<sizeof_bits<Element_>::value,
                                            kCrosswise>,
      layout::PitchLinearShape<InstructionShape::kColumn,
                               InstructionShape::kRow>,
      kOpDelta, kThreads, PartitionsK_, PartitionsM_>;

 public:
  //
  // Derived quantities
  //

  /// Fragment object holding a thread's part of a tile
  using Fragment = typename Base::Fragment;

 private:
  /// Underlying tile iterator
  Base iterator_;

 public:
  /// Default ctor constructs null iterator
  MCTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator() {}

  /// Constructor from TensorRef
  MCTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator(TensorRef const &ref, int lane_id)
      : iterator_({ref.data(), ref.stride()}, lane_id) {}

  /// Adds a pointer offset to internal pointer(s) to advance through memory
  MCTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator &add_pointer_offset(LongIndex offset) {
    iterator_.add_pointer_offset(offset);

    return *this;
  }

  /// Advances an iterator along logical dimensions of matrix in units of whole
  /// tiles
  MCTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator &add_tile_offset(
      TensorCoord const &tile_offset) {
    iterator_.add_tile_offset({tile_offset.column(), tile_offset.row()});

    return *this;
  }

  /// Advances an iterator along logical dimensions of matrix in units of whole
  /// tiles
  MCTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator &add_tile_offset_negative(
      TensorCoord const &tile_offset) {
    iterator_.add_tile_offset_negative({tile_offset.column(), tile_offset.row()});

    return *this;
  }

  /// Advances the iterator along the advance dimension
  MCTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator &operator++() {
    ++iterator_;

    return *this;
  }

  /// Advances the iterator along the advance dimension
  MCTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator &operator--() {
    --iterator_;

    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of
  ///< the tensor
  MCTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator &operator+=(
      TensorCoord const &tile_offset) {
    add_tile_offset(PitchLinearCoord(tile_offset.column(), tile_offset.row()));
    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of
  ///< the tensor
  MCTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator &operator-=(
      TensorCoord const &tile_offset) {
    add_tile_offset(-PitchLinearCoord(tile_offset.column(), tile_offset.row()));
    return *this;
  }

  MCTLASS_DEVICE
  void pre_compute_pointer(int warp_idx_m = 0){
    iterator_.pre_compute_pointer_row_major();
  }

  /// Loads a fragment from memory at the location pointed to by the iterator.
  MCTLASS_HOST_DEVICE
  void load(Fragment &frag) { iterator_.load(frag); }

  MCTLASS_HOST_DEVICE
  void load_with_index(Fragment &frag,int index) { iterator_.load_with_index_A(frag,index); }

  /// Loads a fragment from memory with additional logical offset
  MCTLASS_DEVICE
  void load_with_pointer_offset(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a linear offset
      Index pointer_offset) const {
    iterator_.load_with_pointer_offset(frag, pointer_offset);
  }

  /// Loads a fragment from memory with additional logical offset
  MCTLASS_DEVICE
  void load_with_byte_offset(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a linear offset
      Index byte_offset) const {
    iterator_.load_with_byte_offset(frag, byte_offset);
  }

  /// Loads a fragment from memory with logical offset in units of whole tiles.
  MCTLASS_DEVICE
  void load(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a logical offset in units of whole tiles
      TensorCoord const &tile_offset) const {
    // TODO
    assert(0);
  }

  /// Loads a fragment from memory with logical offset in units of whole tiles.
  MCTLASS_DEVICE
  void load(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a logical offset in units of whole tiles
      TensorCoord const &tile_offset,
      /// loads a tile with a logical offset AND a pointer offset
      Index pointer_offset) const {
    // TODO
    assert(0);
  }

  /// Loads a fragment from memory with logical offset in units of whole tiles.
  MCTLASS_DEVICE
  void load_with_byte_offset(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a logical offset in units of whole tiles
      TensorCoord const &tile_offset,
      /// loads a tile with a logical offset AND a pointer offset
      Index byte_offset) const {
    iterator_.load_with_byte_offset(
        frag, {tile_offset.strided(), tile_offset.contiguous()}, byte_offset);
  }

  /// Notify the iterator which k-group it is currently pointing to.
  ///
  /// This does not advance the iterator. Rather, it overrides its internal
  /// tracking with constant-valued k-group index to enable the compiler to
  /// fold constants and achieve more efficient code.
  ///
  /// This is used by some nontrivial permuted layouts.
  MCTLASS_DEVICE
  void set_kgroup_index(int k_group) {
    iterator_.set_kgroup_index(k_group);
  }
};

/// This tile iterator is specialized for 32-thread TensorOps. It uses LDSM to
/// load from shared memory and therefore must be initialized with a TensorRef
/// to shared memory.
///
/// Satisfies:
///   ReadableRandomAccessContiguousTileIteratorConcept
///
template <
    /// Size of the matrix to load (concept: MatrixShape)
    typename Shape_,
    /// Identifies A or B multiplicand
    Operand Operand_,
    /// Data type of elements
    typename Element_,
    /// Shape of one matrix product operation (concept: MatrixShape)
    typename InstructionShape_,
    /// Interval between adjacent *MMA instructions (in units of MMA
    /// instructions)
    int OpDelta_,
    /// Element number when the layout crosses (in units of elements)
    int Crosswise,
    /// Number of partitions along K dimension
    int PartitionsK_,
    /// Number of partitions along M dimension
    int PartitionsM_>
class MmaTensorOpMultiplicandTileIterator<
    Shape_, Operand_, Element_,
    mctlass::layout::MacaColumnMajorTensorOpMultiplicandCpAsyncCrosswise<
        sizeof_bits<Element_>::value, Crosswise>,
    InstructionShape_, OpDelta_, 32, PartitionsK_, PartitionsM_> {
 public:
  /// Shape of tile to load (concept: PitchLinearShape)
  using Shape = Shape_;

  /// Operand tag
  static Operand const kOperand = Operand_;

  static_assert(kOperand == Operand::kB,
                "MmaTensorOpMultiplicandIterator for RowMajor Crosswise may "
                "only be instantiated for A operand to warp-level Mma.");

  /// Element type
  using Element = Element_;

  /// Element number when the layout crosses
  static int const kCrosswise = Crosswise;

  /// Layout of source tile
  using Layout = mctlass::layout::MacaColumnMajorTensorOpMultiplicandCpAsyncCrosswise<
      sizeof_bits<Element_>::value, kCrosswise>;

  /// Shape of one matrix product operation (concept: MatrixShape)
  using InstructionShape = InstructionShape_;

  /// Delta between *MMA operations (in units of *MMA operations, concept:
  /// MatrixShape)
  static int const kOpDelta = OpDelta_;

  /// Number of participating threads
  static int const kThreads = 64;

  /// TensorRef type for loading element from a tensor
  using TensorRef = TensorRef<Element, Layout>;

  /// Index type
  using Index = typename TensorRef::Index;

  /// Long Index type
  using LongIndex = typename TensorRef::LongIndex;

  /// Coordinate for an element in the tensor
  using TensorCoord = typename TensorRef::TensorCoord;

  /// Underlying tile iterator implementation
  using Base = MmaTensorOpMultiplicandTilePrecomputeIterator<
      layout::PitchLinearShape<Shape::kRow, Shape::kColumn>, kOperand, Element,
      layout::MacaTensorOpMultiplicandCrosswise<sizeof_bits<Element_>::value,
                                            kCrosswise>,
      layout::PitchLinearShape<InstructionShape::kRow,
                               InstructionShape::kColumn>,
      kOpDelta, kThreads, PartitionsK_, PartitionsM_>;

 public:
  //
  // Derived quantities
  //

  /// Fragment object holding a thread's part of a tile
  using Fragment = typename Base::Fragment;

 private:
  /// Underlying tile iterator
  Base iterator_;

 public:
  /// Default ctor constructs null iterator
  MCTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator() {}

  /// Constructor from TensorRef
  MCTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator(TensorRef const &ref, int lane_id)
      : iterator_({ref.data(), ref.stride()}, lane_id) {}

  /// Adds a pointer offset to internal pointer(s) to advance through memory
  MCTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator &add_pointer_offset(LongIndex offset) {

    iterator_.add_pointer_offset(offset);

    return *this;
  }

  /// Advances an iterator along logical dimensions of matrix in units of whole
  /// tiles
  MCTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator &add_tile_offset(
      TensorCoord const &tile_offset) {
    iterator_.add_tile_offset({tile_offset.row(), tile_offset.column()});

    return *this;
  }

  /// Advances an iterator along logical dimensions of matrix in units of whole
  /// tiles
  MCTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator &add_tile_offset_negative(
      TensorCoord const &tile_offset) {
    iterator_.add_tile_offset_negative({tile_offset.row(), tile_offset.column()});

    return *this;
  }

  /// Advances the iterator along the advance dimension
  MCTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator &operator++() {
    ++iterator_;

    return *this;
  }

  /// Advances the iterator along the advance dimension
  MCTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator &operator--() {
    --iterator_;

    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of
  ///< the tensor
  MCTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator &operator+=(
      TensorCoord const &tile_offset) {
    add_tile_offset(PitchLinearCoord(tile_offset.row(), tile_offset.column()));
    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of
  ///< the tensor
  MCTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator &operator-=(
      TensorCoord const &tile_offset) {
    add_tile_offset(-PitchLinearCoord(tile_offset.row(), tile_offset.column()));
    return *this;
  }

  MCTLASS_DEVICE
  void pre_compute_pointer(int warp_idx_n = 0){
    iterator_.pre_compute_pointer_col_major(warp_idx_n);
  }
  /// Loads a fragment from memory at the location pointed to by the iterator.
  MCTLASS_HOST_DEVICE
  void load(Fragment &frag)  { iterator_.load(frag); }

  MCTLASS_HOST_DEVICE
  void load_with_index(Fragment &frag,int index) { iterator_.load_with_index_B(frag,index); }

  /// Loads a fragment from memory with additional logical offset
  MCTLASS_DEVICE
  void load_with_pointer_offset(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a linear offset
      Index pointer_offset) const {
    iterator_.load_with_pointer_offset(frag, pointer_offset);
  }

  /// Loads a fragment from memory with additional logical offset
  MCTLASS_DEVICE
  void load_with_byte_offset(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a linear offset
      Index byte_offset) const {
    iterator_.load_with_byte_offset(frag, byte_offset);
  }

  /// Loads a fragment from memory with logical offset in units of whole tiles.
  MCTLASS_DEVICE
  void load(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a logical offset in units of whole tiles
      TensorCoord const &tile_offset) const {
    // TODO
    assert(0);
  }

  /// Loads a fragment from memory with logical offset in units of whole tiles.
  MCTLASS_DEVICE
  void load(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a logical offset in units of whole tiles
      TensorCoord const &tile_offset,
      /// loads a tile with a logical offset AND a pointer offset
      Index pointer_offset) const {
    // TODO
    assert(0);
  }

  /// Loads a fragment from memory with logical offset in units of whole tiles.
  MCTLASS_DEVICE
  void load_with_byte_offset(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a logical offset in units of whole tiles
      TensorCoord const &tile_offset,
      /// loads a tile with a logical offset AND a pointer offset
      Index byte_offset) const {
    iterator_.load_with_byte_offset(
        frag, {tile_offset.contiguous(), tile_offset.strided()}, byte_offset);
  }

  /// Notify the iterator which k-group it is currently pointing to.
  ///
  /// This does not advance the iterator. Rather, it overrides its internal
  /// tracking with constant-valued k-group index to enable the compiler to
  /// fold constants and achieve more efficient code.
  ///
  /// This is used by some nontrivial permuted layouts.
  MCTLASS_DEVICE
  void set_kgroup_index(int k_group) {
    iterator_.set_kgroup_index(k_group);
  }
};

/// This tile iterator is specialized for 32-thread TensorOps. It uses LDSM to
/// load from shared memory and therefore must be initialized with a TensorRef
/// to shared memory.
///
/// Satisfies:
///   ReadableRandomAccessContiguousTileIteratorConcept
///
template <
    /// Size of the matrix to load (concept: MatrixShape)
    typename Shape_,
    /// Identifies A or B multiplicand
    Operand Operand_,
    /// Data type of elements
    typename Element_,
    /// Shape of one matrix product operation (concept: MatrixShape)
    typename InstructionShape_,
    /// Interval between adjacent *MMA instructions (in units of MMA
    /// instructions)
    int OpDelta_,
    /// Element number when the layout crosses (in units of elements)
    int Crosswise,
    /// Number of partitions along K dimension
    int PartitionsK_,
    /// Number of partitions along M dimension
    int PartitionsM_>
class MmaTensorOpMultiplicandTileIterator<
    Shape_, Operand_, Element_,
    mctlass::layout::MacaRowMajorTensorOpMultiplicandConflictFreeCrosswise<
        sizeof_bits<Element_>::value, Crosswise>,
    InstructionShape_, OpDelta_, 32, PartitionsK_, PartitionsM_> {
 public:
  /// Shape of tile to load (concept: PitchLinearShape)
  using Shape = Shape_;

  /// Operand tag
  static Operand const kOperand = Operand_;

  static_assert(kOperand == Operand::kA,
                "MmaTensorOpMultiplicandIterator for RowMajor Crosswise may "
                "only be instantiated for A operand to warp-level Mma.");

  /// Element type
  using Element = Element_;

  /// Element number when the layout crosses
  static int const kCrosswise = Crosswise;

  /// Layout of source tile
  using Layout = mctlass::layout::MacaRowMajorTensorOpMultiplicandConflictFreeCrosswise<
      sizeof_bits<Element_>::value, kCrosswise>;

  /// Shape of one matrix product operation (concept: MatrixShape)
  using InstructionShape = InstructionShape_;

  /// Delta between *MMA operations (in units of *MMA operations, concept:
  /// MatrixShape)
  static int const kOpDelta = OpDelta_;

  /// Number of participating threads
  static int const kThreads = 64;

  /// TensorRef type for loading element from a tensor
  using TensorRef = TensorRef<Element, Layout>;

  /// Index type
  using Index = typename TensorRef::Index;

  /// Long Index type
  using LongIndex = typename TensorRef::LongIndex;

  /// Coordinate for an element in the tensor
  using TensorCoord = typename TensorRef::TensorCoord;

  /// Underlying tile iterator implementation
  using Base = MmaTensorOpMultiplicandTilePrecomputeIterator<
      layout::PitchLinearShape<Shape::kColumn, Shape::kRow>, kOperand, Element,
      layout::MacaTensorOpMultiplicandConflictFreeCrosswise<sizeof_bits<Element_>::value,
                                            kCrosswise>,
      layout::PitchLinearShape<InstructionShape::kColumn,
                               InstructionShape::kRow>,
      kOpDelta, kThreads, PartitionsK_, PartitionsM_>;

 public:
  //
  // Derived quantities
  //

  /// Fragment object holding a thread's part of a tile
  using Fragment = typename Base::Fragment;

 private:
  /// Underlying tile iterator
  Base iterator_;

 public:
  /// Default ctor constructs null iterator
  MCTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator() {}

  /// Constructor from TensorRef
  MCTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator(TensorRef const &ref, int lane_id)
      : iterator_({ref.data(), ref.stride()}, lane_id) {}

  /// Adds a pointer offset to internal pointer(s) to advance through memory
  MCTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator &add_pointer_offset(LongIndex offset) {
    iterator_.add_pointer_offset(offset);

    return *this;
  }

  /// Advances an iterator along logical dimensions of matrix in units of whole
  /// tiles
  MCTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator &add_tile_offset(
      TensorCoord const &tile_offset) {
    iterator_.add_tile_offset({tile_offset.column(), tile_offset.row()});

    return *this;
  }

  /// Advances an iterator along logical dimensions of matrix in units of whole
  /// tiles
  MCTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator &add_tile_offset_negative(
      TensorCoord const &tile_offset) {
    iterator_.add_tile_offset_negative({tile_offset.column(), tile_offset.row()});

    return *this;
  }

  /// Advances the iterator along the advance dimension
  MCTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator &operator++() {
    ++iterator_;

    return *this;
  }

  /// Advances the iterator along the advance dimension
  MCTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator &operator--() {
    --iterator_;

    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of
  ///< the tensor
  MCTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator &operator+=(
      TensorCoord const &tile_offset) {
    add_tile_offset(PitchLinearCoord(tile_offset.column(), tile_offset.row()));
    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of
  ///< the tensor
  MCTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator &operator-=(
      TensorCoord const &tile_offset) {
    add_tile_offset(-PitchLinearCoord(tile_offset.column(), tile_offset.row()));
    return *this;
  }

  MCTLASS_DEVICE
  void pre_compute_pointer(int warp_idx_m = 0){
    iterator_.pre_compute_pointer_row_major(warp_idx_m);
  }

  /// Loads a fragment from memory at the location pointed to by the iterator.
  MCTLASS_HOST_DEVICE
  void load(Fragment &frag) { iterator_.load(frag); }

  MCTLASS_HOST_DEVICE
  void load_with_index(Fragment &frag,int index) { iterator_.load_with_index_A(frag,index); }

  /// Loads a fragment from memory with additional logical offset
  MCTLASS_DEVICE
  void load_with_pointer_offset(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a linear offset
      Index pointer_offset) const {
    iterator_.load_with_pointer_offset(frag, pointer_offset);
  }

  /// Loads a fragment from memory with additional logical offset
  MCTLASS_DEVICE
  void load_with_byte_offset(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a linear offset
      Index byte_offset) const {
    iterator_.load_with_byte_offset(frag, byte_offset);
  }

  /// Loads a fragment from memory with logical offset in units of whole tiles.
  MCTLASS_DEVICE
  void load(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a logical offset in units of whole tiles
      TensorCoord const &tile_offset) const {
    // TODO
    assert(0);
  }

  /// Loads a fragment from memory with logical offset in units of whole tiles.
  MCTLASS_DEVICE
  void load(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a logical offset in units of whole tiles
      TensorCoord const &tile_offset,
      /// loads a tile with a logical offset AND a pointer offset
      Index pointer_offset) const {
    // TODO
    assert(0);
  }

  /// Loads a fragment from memory with logical offset in units of whole tiles.
  MCTLASS_DEVICE
  void load_with_byte_offset(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a logical offset in units of whole tiles
      TensorCoord const &tile_offset,
      /// loads a tile with a logical offset AND a pointer offset
      Index byte_offset) const {
    iterator_.load_with_byte_offset(
        frag, {tile_offset.strided(), tile_offset.contiguous()}, byte_offset);
  }

  /// Notify the iterator which k-group it is currently pointing to.
  ///
  /// This does not advance the iterator. Rather, it overrides its internal
  /// tracking with constant-valued k-group index to enable the compiler to
  /// fold constants and achieve more efficient code.
  ///
  /// This is used by some nontrivial permuted layouts.
  MCTLASS_DEVICE
  void set_kgroup_index(int k_group) {
    iterator_.set_kgroup_index(k_group);
  }
};

/// This tile iterator is specialized for 32-thread TensorOps. It uses LDSM to
/// load from shared memory and therefore must be initialized with a TensorRef
/// to shared memory.
///
/// Satisfies:
///   ReadableRandomAccessContiguousTileIteratorConcept
///
template <
    /// Size of the matrix to load (concept: MatrixShape)
    typename Shape_,
    /// Identifies A or B multiplicand
    Operand Operand_,
    /// Data type of elements
    typename Element_,
    /// Shape of one matrix product operation (concept: MatrixShape)
    typename InstructionShape_,
    /// Interval between adjacent *MMA instructions (in units of MMA
    /// instructions)
    int OpDelta_,
    /// Element number when the layout crosses (in units of elements)
    int Crosswise,
    /// Number of partitions along K dimension
    int PartitionsK_,
    /// Number of partitions along M dimension
    int PartitionsM_>
class MmaTensorOpMultiplicandTileIterator<
    Shape_, Operand_, Element_,
    mctlass::layout::MacaColumnMajorTensorOpMultiplicandConflictFreeCrosswise<
        sizeof_bits<Element_>::value, Crosswise>,
    InstructionShape_, OpDelta_, 32, PartitionsK_, PartitionsM_> {
 public:
  /// Shape of tile to load (concept: PitchLinearShape)
  using Shape = Shape_;

  /// Operand tag
  static Operand const kOperand = Operand_;

  static_assert(kOperand == Operand::kB,
                "MmaTensorOpMultiplicandIterator for RowMajor Crosswise may "
                "only be instantiated for A operand to warp-level Mma.");

  /// Element type
  using Element = Element_;

  /// Element number when the layout crosses
  static int const kCrosswise = Crosswise;

  /// Layout of source tile
  using Layout = mctlass::layout::MacaColumnMajorTensorOpMultiplicandConflictFreeCrosswise<
      sizeof_bits<Element_>::value, kCrosswise>;

  /// Shape of one matrix product operation (concept: MatrixShape)
  using InstructionShape = InstructionShape_;

  /// Delta between *MMA operations (in units of *MMA operations, concept:
  /// MatrixShape)
  static int const kOpDelta = OpDelta_;

  /// Number of participating threads
  static int const kThreads = 64;

  /// TensorRef type for loading element from a tensor
  using TensorRef = TensorRef<Element, Layout>;

  /// Index type
  using Index = typename TensorRef::Index;

  /// Long Index type
  using LongIndex = typename TensorRef::LongIndex;

  /// Coordinate for an element in the tensor
  using TensorCoord = typename TensorRef::TensorCoord;

  /// Underlying tile iterator implementation
  using Base = MmaTensorOpMultiplicandTilePrecomputeIterator<
      layout::PitchLinearShape<Shape::kRow, Shape::kColumn>, kOperand, Element,
      layout::MacaTensorOpMultiplicandConflictFreeCrosswise<sizeof_bits<Element_>::value,
                                            kCrosswise>,
      layout::PitchLinearShape<InstructionShape::kRow,
                               InstructionShape::kColumn>,
      kOpDelta, kThreads, PartitionsK_, PartitionsM_>;

 public:
  //
  // Derived quantities
  //

  /// Fragment object holding a thread's part of a tile
  using Fragment = typename Base::Fragment;

 private:
  /// Underlying tile iterator
  Base iterator_;

 public:
  /// Default ctor constructs null iterator
  MCTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator() {}

  /// Constructor from TensorRef
  MCTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator(TensorRef const &ref, int lane_id)
      : iterator_({ref.data(), ref.stride()}, lane_id) {}

  /// Adds a pointer offset to internal pointer(s) to advance through memory
  MCTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator &add_pointer_offset(LongIndex offset) {

    iterator_.add_pointer_offset(offset);

    return *this;
  }

  /// Advances an iterator along logical dimensions of matrix in units of whole
  /// tiles
  MCTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator &add_tile_offset(
      TensorCoord const &tile_offset) {
    iterator_.add_tile_offset({tile_offset.row(), tile_offset.column()});

    return *this;
  }

  /// Advances an iterator along logical dimensions of matrix in units of whole
  /// tiles
  MCTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator &add_tile_offset_negative(
      TensorCoord const &tile_offset) {
    iterator_.add_tile_offset_negative({tile_offset.row(), tile_offset.column()});

    return *this;
  }

  /// Advances the iterator along the advance dimension
  MCTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator &operator++() {
    ++iterator_;

    return *this;
  }

  /// Advances the iterator along the advance dimension
  MCTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator &operator--() {
    --iterator_;

    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of
  ///< the tensor
  MCTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator &operator+=(
      TensorCoord const &tile_offset) {
    add_tile_offset(PitchLinearCoord(tile_offset.row(), tile_offset.column()));
    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of
  ///< the tensor
  MCTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator &operator-=(
      TensorCoord const &tile_offset) {
    add_tile_offset(-PitchLinearCoord(tile_offset.row(), tile_offset.column()));
    return *this;
  }

  MCTLASS_DEVICE
  void pre_compute_pointer(int warp_idx_n = 0){
    iterator_.pre_compute_pointer_col_major(warp_idx_n);
  }
  /// Loads a fragment from memory at the location pointed to by the iterator.
  MCTLASS_HOST_DEVICE
  void load(Fragment &frag)  { iterator_.load(frag); }

  MCTLASS_HOST_DEVICE
  void load_with_index(Fragment &frag,int index) { iterator_.load_with_index_B(frag,index); }

  /// Loads a fragment from memory with additional logical offset
  MCTLASS_DEVICE
  void load_with_pointer_offset(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a linear offset
      Index pointer_offset) const {
    iterator_.load_with_pointer_offset(frag, pointer_offset);
  }

  /// Loads a fragment from memory with additional logical offset
  MCTLASS_DEVICE
  void load_with_byte_offset(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a linear offset
      Index byte_offset) const {
    iterator_.load_with_byte_offset(frag, byte_offset);
  }

  /// Loads a fragment from memory with logical offset in units of whole tiles.
  MCTLASS_DEVICE
  void load(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a logical offset in units of whole tiles
      TensorCoord const &tile_offset) const {
    // TODO
    assert(0);
  }

  /// Loads a fragment from memory with logical offset in units of whole tiles.
  MCTLASS_DEVICE
  void load(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a logical offset in units of whole tiles
      TensorCoord const &tile_offset,
      /// loads a tile with a logical offset AND a pointer offset
      Index pointer_offset) const {
    // TODO
    assert(0);
  }

  /// Loads a fragment from memory with logical offset in units of whole tiles.
  MCTLASS_DEVICE
  void load_with_byte_offset(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a logical offset in units of whole tiles
      TensorCoord const &tile_offset,
      /// loads a tile with a logical offset AND a pointer offset
      Index byte_offset) const {
    iterator_.load_with_byte_offset(
        frag, {tile_offset.contiguous(), tile_offset.strided()}, byte_offset);
  }

  /// Notify the iterator which k-group it is currently pointing to.
  ///
  /// This does not advance the iterator. Rather, it overrides its internal
  /// tracking with constant-valued k-group index to enable the compiler to
  /// fold constants and achieve more efficient code.
  ///
  /// This is used by some nontrivial permuted layouts.
  MCTLASS_DEVICE
  void set_kgroup_index(int k_group) {
    iterator_.set_kgroup_index(k_group);
  }
};

/// This tile iterator is specialized for 32-thread TensorOps. It uses LDSM to
/// load from shared memory and therefore must be initialized with a TensorRef
/// to shared memory.
///
/// Satisfies:
///   ReadableRandomAccessContiguousTileIteratorConcept
///
template <
    /// Size of the matrix to load (concept: MatrixShape)
    typename Shape_,
    /// Identifies A or B multiplicand
    Operand Operand_,
    /// Data type of elements
    typename Element_,
    /// Shape of one matrix product operation (concept: MatrixShape)
    typename InstructionShape_,
    /// Interval between adjacent *MMA instructions (in units of MMA
    /// instructions)
    int OpDelta_,
    /// Element number when the layout crosses (in units of elements)
    int Crosswise,
    /// Number of partitions along K dimension
    int PartitionsK_,
    /// Number of partitions along M dimension
    int PartitionsM_>
class MmaTensorOpMultiplicandTileIterator<
    Shape_, Operand_, Element_,
    mctlass::layout::MacaRowMajorTensorOpMultiplicandMultipleLdgCrosswise<
        sizeof_bits<Element_>::value, Crosswise>,
    InstructionShape_, OpDelta_, 32, PartitionsK_, PartitionsM_> {
 public:
  /// Shape of tile to load (concept: PitchLinearShape)
  using Shape = Shape_;

  /// Operand tag
  static Operand const kOperand = Operand_;

  static_assert(kOperand == Operand::kA,
                "MmaTensorOpMultiplicandIterator for RowMajor Crosswise may "
                "only be instantiated for A operand to warp-level Mma.");

  /// Element type
  using Element = Element_;

  /// Element number when the layout crosses
  static int const kCrosswise = Crosswise;

  /// Layout of source tile
  using Layout = mctlass::layout::MacaRowMajorTensorOpMultiplicandMultipleLdgCrosswise<
      sizeof_bits<Element_>::value, kCrosswise>;

  /// Shape of one matrix product operation (concept: MatrixShape)
  using InstructionShape = InstructionShape_;

  /// Delta between *MMA operations (in units of *MMA operations, concept:
  /// MatrixShape)
  static int const kOpDelta = OpDelta_;

  /// Number of participating threads
  static int const kThreads = 64;

  /// TensorRef type for loading element from a tensor
  using TensorRef = TensorRef<Element, Layout>;

  /// Index type
  using Index = typename TensorRef::Index;

  /// Long Index type
  using LongIndex = typename TensorRef::LongIndex;

  /// Coordinate for an element in the tensor
  using TensorCoord = typename TensorRef::TensorCoord;

  /// Underlying tile iterator implementation
  using Base = MmaTensorOpMultiplicandTilePrecomputeIterator<
      layout::PitchLinearShape<Shape::kColumn, Shape::kRow>, kOperand, Element,
      layout::MacaTensorOpMultiplicandConflictFreeCrosswise<sizeof_bits<Element_>::value,
                                            kCrosswise>,
      layout::PitchLinearShape<InstructionShape::kColumn,
                               InstructionShape::kRow>,
      kOpDelta, kThreads, PartitionsK_, PartitionsM_>;

 public:
  //
  // Derived quantities
  //

  /// Fragment object holding a thread's part of a tile
  using Fragment = typename Base::Fragment;

 private:
  /// Underlying tile iterator
  Base iterator_;

 public:
  /// Default ctor constructs null iterator
  MCTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator() {}

  /// Constructor from TensorRef
  MCTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator(TensorRef const &ref, int lane_id)
      : iterator_({ref.data(), ref.stride()}, lane_id) {}

  /// Adds a pointer offset to internal pointer(s) to advance through memory
  MCTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator &add_pointer_offset(LongIndex offset) {
    iterator_.add_pointer_offset(offset);

    return *this;
  }

  /// Advances an iterator along logical dimensions of matrix in units of whole
  /// tiles
  MCTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator &add_tile_offset(
      TensorCoord const &tile_offset) {
    iterator_.add_tile_offset({tile_offset.column(), tile_offset.row()});

    return *this;
  }

  /// Advances an iterator along logical dimensions of matrix in units of whole
  /// tiles
  MCTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator &add_tile_offset_negative(
      TensorCoord const &tile_offset) {
    iterator_.add_tile_offset_negative({tile_offset.column(), tile_offset.row()});

    return *this;
  }

  /// Advances the iterator along the advance dimension
  MCTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator &operator++() {
    ++iterator_;

    return *this;
  }

  /// Advances the iterator along the advance dimension
  MCTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator &operator--() {
    --iterator_;

    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of
  ///< the tensor
  MCTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator &operator+=(
      TensorCoord const &tile_offset) {
    add_tile_offset(PitchLinearCoord(tile_offset.column(), tile_offset.row()));
    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of
  ///< the tensor
  MCTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator &operator-=(
      TensorCoord const &tile_offset) {
    add_tile_offset(-PitchLinearCoord(tile_offset.column(), tile_offset.row()));
    return *this;
  }

  MCTLASS_DEVICE
  void pre_compute_pointer(int warp_idx_m = 0, int tile_k = 0){
    iterator_.pre_compute_pointer_row_major(warp_idx_m, tile_k);
  }

  /// Loads a fragment from memory at the location pointed to by the iterator.
  MCTLASS_HOST_DEVICE
  void load(Fragment &frag) { iterator_.load(frag); }

  MCTLASS_HOST_DEVICE
  void load_with_index(Fragment &frag,int index) { iterator_.load_with_index_A(frag,index); }

  /// Loads a fragment from memory with additional logical offset
  MCTLASS_DEVICE
  void load_with_pointer_offset(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a linear offset
      Index pointer_offset) const {
    iterator_.load_with_pointer_offset(frag, pointer_offset);
  }

  /// Loads a fragment from memory with additional logical offset
  MCTLASS_DEVICE
  void load_with_byte_offset(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a linear offset
      Index byte_offset) const {
    iterator_.load_with_byte_offset(frag, byte_offset);
  }

  /// Loads a fragment from memory with logical offset in units of whole tiles.
  MCTLASS_DEVICE
  void load(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a logical offset in units of whole tiles
      TensorCoord const &tile_offset) const {
    // TODO
    assert(0);
  }

  /// Loads a fragment from memory with logical offset in units of whole tiles.
  MCTLASS_DEVICE
  void load(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a logical offset in units of whole tiles
      TensorCoord const &tile_offset,
      /// loads a tile with a logical offset AND a pointer offset
      Index pointer_offset) const {
    // TODO
    assert(0);
  }

  /// Loads a fragment from memory with logical offset in units of whole tiles.
  MCTLASS_DEVICE
  void load_with_byte_offset(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a logical offset in units of whole tiles
      TensorCoord const &tile_offset,
      /// loads a tile with a logical offset AND a pointer offset
      Index byte_offset) const {
    iterator_.load_with_byte_offset(
        frag, {tile_offset.strided(), tile_offset.contiguous()}, byte_offset);
  }

  /// Notify the iterator which k-group it is currently pointing to.
  ///
  /// This does not advance the iterator. Rather, it overrides its internal
  /// tracking with constant-valued k-group index to enable the compiler to
  /// fold constants and achieve more efficient code.
  ///
  /// This is used by some nontrivial permuted layouts.
  MCTLASS_DEVICE
  void set_kgroup_index(int k_group) {
    iterator_.set_kgroup_index(k_group);
  }
};

/// This tile iterator is specialized for 32-thread TensorOps. It uses LDSM to
/// load from shared memory and therefore must be initialized with a TensorRef
/// to shared memory.
///
/// Satisfies:
///   ReadableRandomAccessContiguousTileIteratorConcept
///
template <
    /// Size of the matrix to load (concept: MatrixShape)
    typename Shape_,
    /// Identifies A or B multiplicand
    Operand Operand_,
    /// Data type of elements
    typename Element_,
    /// Shape of one matrix product operation (concept: MatrixShape)
    typename InstructionShape_,
    /// Interval between adjacent *MMA instructions (in units of MMA
    /// instructions)
    int OpDelta_,
    /// Element number when the layout crosses (in units of elements)
    int Crosswise,
    /// Number of partitions along K dimension
    int PartitionsK_,
    /// Number of partitions along M dimension
    int PartitionsM_>
class MmaTensorOpMultiplicandTileIterator<
    Shape_, Operand_, Element_,
    mctlass::layout::MacaColumnMajorTensorOpMultiplicandMultipleLdgCrosswise<
        sizeof_bits<Element_>::value, Crosswise>,
    InstructionShape_, OpDelta_, 32, PartitionsK_, PartitionsM_> {
 public:
  /// Shape of tile to load (concept: PitchLinearShape)
  using Shape = Shape_;

  /// Operand tag
  static Operand const kOperand = Operand_;

  static_assert(kOperand == Operand::kB,
                "MmaTensorOpMultiplicandIterator for RowMajor Crosswise may "
                "only be instantiated for A operand to warp-level Mma.");

  /// Element type
  using Element = Element_;

  /// Element number when the layout crosses
  static int const kCrosswise = Crosswise;

  /// Layout of source tile
  using Layout = mctlass::layout::MacaColumnMajorTensorOpMultiplicandMultipleLdgCrosswise<
      sizeof_bits<Element_>::value, kCrosswise>;

  /// Shape of one matrix product operation (concept: MatrixShape)
  using InstructionShape = InstructionShape_;

  /// Delta between *MMA operations (in units of *MMA operations, concept:
  /// MatrixShape)
  static int const kOpDelta = OpDelta_;

  /// Number of participating threads
  static int const kThreads = 64;

  /// TensorRef type for loading element from a tensor
  using TensorRef = TensorRef<Element, Layout>;

  /// Index type
  using Index = typename TensorRef::Index;

  /// Long Index type
  using LongIndex = typename TensorRef::LongIndex;

  /// Coordinate for an element in the tensor
  using TensorCoord = typename TensorRef::TensorCoord;

  /// Underlying tile iterator implementation
  using Base = MmaTensorOpMultiplicandTilePrecomputeIterator<
      layout::PitchLinearShape<Shape::kRow, Shape::kColumn>, kOperand, Element,
      layout::MacaTensorOpMultiplicandConflictFreeCrosswise<sizeof_bits<Element_>::value,
                                            kCrosswise>,
      layout::PitchLinearShape<InstructionShape::kRow,
                               InstructionShape::kColumn>,
      kOpDelta, kThreads, PartitionsK_, PartitionsM_>;

 public:
  //
  // Derived quantities
  //

  /// Fragment object holding a thread's part of a tile
  using Fragment = typename Base::Fragment;

 private:
  /// Underlying tile iterator
  Base iterator_;

 public:
  /// Default ctor constructs null iterator
  MCTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator() {}

  /// Constructor from TensorRef
  MCTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator(TensorRef const &ref, int lane_id)
      : iterator_({ref.data(), ref.stride()}, lane_id) {}

  /// Adds a pointer offset to internal pointer(s) to advance through memory
  MCTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator &add_pointer_offset(LongIndex offset) {

    iterator_.add_pointer_offset(offset);

    return *this;
  }

  /// Advances an iterator along logical dimensions of matrix in units of whole
  /// tiles
  MCTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator &add_tile_offset(
      TensorCoord const &tile_offset) {
    iterator_.add_tile_offset({tile_offset.row(), tile_offset.column()});

    return *this;
  }

  /// Advances an iterator along logical dimensions of matrix in units of whole
  /// tiles
  MCTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator &add_tile_offset_negative(
      TensorCoord const &tile_offset) {
    iterator_.add_tile_offset_negative({tile_offset.row(), tile_offset.column()});

    return *this;
  }

  /// Advances the iterator along the advance dimension
  MCTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator &operator++() {
    ++iterator_;

    return *this;
  }

  /// Advances the iterator along the advance dimension
  MCTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator &operator--() {
    --iterator_;

    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of
  ///< the tensor
  MCTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator &operator+=(
      TensorCoord const &tile_offset) {
    add_tile_offset(PitchLinearCoord(tile_offset.row(), tile_offset.column()));
    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of
  ///< the tensor
  MCTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator &operator-=(
      TensorCoord const &tile_offset) {
    add_tile_offset(-PitchLinearCoord(tile_offset.row(), tile_offset.column()));
    return *this;
  }

  MCTLASS_DEVICE
  void pre_compute_pointer(int warp_idx_n = 0, int tile_k = 0){
    iterator_.pre_compute_pointer_col_major(warp_idx_n, tile_k);
  }
  /// Loads a fragment from memory at the location pointed to by the iterator.
  MCTLASS_HOST_DEVICE
  void load(Fragment &frag)  { iterator_.load(frag); }

  MCTLASS_HOST_DEVICE
  void load_with_index(Fragment &frag,int index) { iterator_.load_with_index_B(frag,index); }

  /// Loads a fragment from memory with additional logical offset
  MCTLASS_DEVICE
  void load_with_pointer_offset(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a linear offset
      Index pointer_offset) const {
    iterator_.load_with_pointer_offset(frag, pointer_offset);
  }

  /// Loads a fragment from memory with additional logical offset
  MCTLASS_DEVICE
  void load_with_byte_offset(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a linear offset
      Index byte_offset) const {
    iterator_.load_with_byte_offset(frag, byte_offset);
  }

  /// Loads a fragment from memory with logical offset in units of whole tiles.
  MCTLASS_DEVICE
  void load(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a logical offset in units of whole tiles
      TensorCoord const &tile_offset) const {
    // TODO
    assert(0);
  }

  /// Loads a fragment from memory with logical offset in units of whole tiles.
  MCTLASS_DEVICE
  void load(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a logical offset in units of whole tiles
      TensorCoord const &tile_offset,
      /// loads a tile with a logical offset AND a pointer offset
      Index pointer_offset) const {
    // TODO
    assert(0);
  }

  /// Loads a fragment from memory with logical offset in units of whole tiles.
  MCTLASS_DEVICE
  void load_with_byte_offset(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a logical offset in units of whole tiles
      TensorCoord const &tile_offset,
      /// loads a tile with a logical offset AND a pointer offset
      Index byte_offset) const {
    iterator_.load_with_byte_offset(
        frag, {tile_offset.contiguous(), tile_offset.strided()}, byte_offset);
  }

  /// Notify the iterator which k-group it is currently pointing to.
  ///
  /// This does not advance the iterator. Rather, it overrides its internal
  /// tracking with constant-valued k-group index to enable the compiler to
  /// fold constants and achieve more efficient code.
  ///
  /// This is used by some nontrivial permuted layouts.
  MCTLASS_DEVICE
  void set_kgroup_index(int k_group) {
    iterator_.set_kgroup_index(k_group);
  }
};

////////////////////////////////////////////////////////////////////////////////

/// This tile iterator is specialized for 64-thread TensorOps. It's used to load from shared
/// memory and therefore must be initialized with a TensorRef to shared memory.
///
/// Satisfies:
///   ReadableRandomAccessContiguousTileIteratorConcept
///
template <
    /// Size of the matrix to load (concept: PitchLinearShape)
    typename Shape_,
    /// Identifies A or B multiplicand
    Operand Operand_,
    /// Data type of elements
    typename Element_,
    /// Shape of one matrix product operation (concept: PitchLinearShape)
    typename InstructionShape_,
    /// Interval between adjacent *MMA instructions (in units of MMA
    /// instructions)
    int OpDelta_,
    /// Number of partitions along K dimension
    int PartitionsK_>
class MmaTensorOpMultiplicandTileIterator<
    Shape_, Operand_, Element_,
    mctlass::layout::MacaTensorOpMultiplicandCongruous4x4Perm<sizeof_bits<Element_>::value,
                                                       int(128 / sizeof(Element_))>,
    InstructionShape_, OpDelta_, 64, PartitionsK_> {
 public:

  /// Shape of tile to load (concept: PitchLinearShape)
  using Shape = Shape_;

  /// Operand tag
  static Operand const kOperand = Operand_;

  static_assert(kOperand == Operand::kA || kOperand== Operand::kB,
    "MmaTensorOpMultiplicandIterator may only be instantiated for A or B operands to warp-level Mma.");

  /// Element type
  using Element = Element_;

  /// Layout of source tile
  using Layout = mctlass::layout::MacaTensorOpMultiplicandCongruous4x4Perm<
      sizeof_bits<Element_>::value, int(128 / sizeof(Element_))>;

  /// Shape of one matrix product operation (concept: GemmShape)
  using InstructionShape = InstructionShape_;

  /// Delta between *MMA operations (in units of *MMA operations, concept: MatrixShape)
  static int const kOpDelta = OpDelta_;

  /// Number of participating threads
  static int const kThreads = 64;

  /// Number of partitions along K dimension
  static int const kPartitionsK = PartitionsK_;

  /// TensorRef type for loading element from a tensor
  using TensorRef = TensorRef<Element, Layout>;

  /// Index type
  using Index = typename TensorRef::Index;

  /// Long Index type
  using LongIndex = typename TensorRef::LongIndex;

  /// Long Index type
  using StrideIndex = typename TensorRef::Layout::Stride::Index;

  /// Coordinate for an element in the tensor
  using TensorCoord = typename TensorRef::TensorCoord;

  /// Internal structure of iterator - made public to enable introspection
  struct Policy {
    static_assert(
        !(Shape::kContiguous % InstructionShape::kContiguous),
        "Shape of warp-level Mma must be divisible by operator shape.");

    // Determine number of elements along outer dimension per individual LDSM op
    static int const kLdsmOpOuter = Layout::kElementsPerAccess;
    static int const kLdsmOpInner = (sizeof(Element_) == 1) ? 4 : 8;

    static_assert(!(Shape::kContiguous % kLdsmOpOuter),
      "Shape of warp-level mma must be divisible by LDSM's fundamental tile size.");

    static_assert(!(Shape::kStrided % kLdsmOpInner),
      "Shape of warp-level mma must be divisible by LDSM's fundamental tile size.");

    /// Shape of one individual LDSM instruction
    static int const LdsmShapeStrided =
        InstructionShape::kStrided / kLdsmOpInner;
    static int const LdsmShapeContiguous = 4 / LdsmShapeStrided;
    using LdsmShape =
        layout::PitchLinearShape<LdsmShapeContiguous, LdsmShapeStrided>;

    /// Number and arrangement of LDSM instructions
    using LdsmIterations = layout::PitchLinearShape<
        Shape::kContiguous / Layout::kElementsPerAccess / LdsmShapeContiguous,
        1>;

    /// Number of groups for each tile
    static int const kGroupsPerTile =
        Shape::kStrided / InstructionShape::kStrided;
  };

private:

  /// Not working on this feature at the moment.
  static_assert(kOpDelta == 1,
    "Alternative arrangements not supported at present.");

  /// Number of internal pointers needed to reference shared memory
  static int const kPointerCount =
      Policy::LdsmIterations::kStrided * Policy::LdsmIterations::kContiguous;

  /// Pointer type used for accesses
  using AccessType = Array<Element, Layout::kElementsPerAccess>;

  /// Internal counter used to jump to next K partition
  int k_group_idx_;

public:

  //
  // Derived quantities
  //

  /// Fragment object holding a thread's part of a tile
 using Fragment =
     Array<Element, Shape::kContiguous * InstructionShape::kStrided / kThreads>;
 using Fragment_int =
     Array<int, Shape::kContiguous * InstructionShape::kStrided / kThreads>;

private:

  /// Layout object storing stride values
  StrideIndex stride_;

  /// Shared memory base pointers - not advanced
  AccessType const *pointer_;

  /// Byte offset incremented as iterator advances
  Index byte_offset_;

  int bank1_;

public:

  /// Default ctor constructs null iterator
  MCTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator(): stride_(0), byte_offset_(0) { }

  /// Constructor from TensorRef
  MCTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator(
    TensorRef const &ref,
    int lane_id
  ):
    stride_(ref.stride(0)), byte_offset_(0),
    k_group_idx_(0) {

    pointer_ = reinterpret_cast<AccessType const *>(ref.data());
    Index col = (((lane_id >> 4) << 2) ^ (InstructionShape::kContiguous - 1)) - 3;
    Index row = lane_id & (InstructionShape::kStrided - 1);
    byte_offset_ = (col * stride_ + row * 4) * (sizeof_bits<Element>::value / 8);
  }

  /// Adds a pointer offset to internal pointer(s) to advance through memory
  MCTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator &add_pointer_offset(LongIndex offset) {

    byte_offset_ += offset * sizeof(Element);

    return *this;
  }

  /// Advances an iterator along logical dimensions of matrix in units of whole tiles
  MCTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator &add_tile_offset(TensorCoord const &tile_offset) {

    LongIndex offset = tile_offset.strided() * InstructionShape::kStrided * stride_
                        + tile_offset.contiguous() * Shape::kContiguous;

    byte_offset_ += offset * sizeof(Element);

    return *this;
  }

  /// Advances the iterator along the advance dimension
  MCTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator & operator++() {

    byte_offset_ += stride_ * InstructionShape::kStrided * sizeof(Element);
    return *this;
  }

  /// Advances the iterator along the opposite of the advance dimension
  MCTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator & operator--() {
    byte_offset_ -= stride_ * InstructionShape::kStrided * sizeof(Element);

    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of the tensor
  MCTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator & operator+=(TensorCoord const &tile_offset) {
    add_tile_offset(tile_offset);
    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of the tensor
  MCTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator & operator-=(TensorCoord const &tile_offset) {
    add_tile_offset(-tile_offset);
    return *this;
  }

  MCTLASS_DEVICE
  void fragment_int_conversion(const Fragment_int &frag_in, Fragment &frag) const {
    using LoadIntType = Array<int, Policy::LdsmShape::kCount>;
    using LoadType = Array<int, Policy::LdsmShape::kCount / 2>;

    const LoadIntType *fetch_int_element_ptr = reinterpret_cast<const LoadIntType *>(&frag_in);
    LoadType *fetch_element_ptr = reinterpret_cast<LoadType *>(&frag);

    int hi;
    int lo;

    MCTLASS_PRAGMA_UNROLL
    for (int s = 0; s < Policy::LdsmIterations::kStrided; ++s) {

      MCTLASS_PRAGMA_UNROLL
      for (int c = 0; c < Policy::LdsmIterations::kContiguous; ++c) {
        int access_idx = c + s * Policy::LdsmIterations::kContiguous;
        for (int i = 0; i < Policy::LdsmShape::kCount / 2; ++i) {
          lo = __builtin_mxc_byte_perm(fetch_int_element_ptr[access_idx][2 * i + 1],
                                       fetch_int_element_ptr[access_idx][2 * i], 0x05040100);
          hi = __builtin_mxc_byte_perm(fetch_int_element_ptr[access_idx][2 * i + 1],
                                       fetch_int_element_ptr[access_idx][2 * i], 0x07060302);
          fetch_element_ptr[access_idx][i] = (bank1_)? hi : lo;
        }
      }
    }
  }

  MCTLASS_HOST_DEVICE
  void load(Fragment_int &frag) const {

    load_with_byte_offset(frag, 0);
  }

  MCTLASS_DEVICE
  void load_with_byte_offset(
      /// fragment to load from the tensor
      Fragment_int &frag,
      /// loads a tile with a linear offset in units of bytes
      Index byte_offset) const {

    Array<int, Policy::LdsmShape::kCount> *fetch_element_ptr =
        reinterpret_cast<Array<int, Policy::LdsmShape::kCount> *>(&frag);

    MCTLASS_PRAGMA_UNROLL
    for (int s = 0; s < Policy::LdsmIterations::kStrided; ++s) {

      MCTLASS_PRAGMA_UNROLL
      for (int c = 0; c < Policy::LdsmIterations::kContiguous; ++c) {

        int access_idx = c + s * Policy::LdsmIterations::kContiguous;
        char const *source_ptr = reinterpret_cast<char const *>(pointer_[access_idx])
                                  + byte_offset + byte_offset_;
        // int const *ptr = reinterpret_cast<int const *>(source_ptr);
        Element const *ptr = reinterpret_cast<Element const *>(source_ptr) - bank1_;

        for (int i = 0; i < Policy::LdsmShape::kCount; ++i) {
          Index offset = i * stride_;
          fetch_element_ptr[access_idx][i] = *(int *)(&(ptr[offset]));
        }
      }
    }

  }

  /// Loads a fragment from memory at the location pointed to by the iterator.
  MCTLASS_HOST_DEVICE
  void load(Fragment &frag) const {

    load_with_byte_offset(frag, 0);
  }

  /// Loads a fragment from memory with additional logical offset
  MCTLASS_DEVICE
  void load_with_byte_offset(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a linear offset in units of bytes
      Index byte_offset) const {
    using LoadType =  Array<int, Policy::LdsmShape::kCount / 2>;
    LoadType fetch_element_ptr[4];

    char const *source_ptr = reinterpret_cast<char const *>(pointer_) + byte_offset_;
    Element const *ptr = reinterpret_cast<Element const *>(source_ptr);
    MCTLASS_PRAGMA_UNROLL
    for(int i = 0; i < 4; ++i) {
      fetch_element_ptr[i] = *reinterpret_cast<LoadType const *>(ptr + i * stride_);
    }

    Array<int, 2> *perm_dst = reinterpret_cast<Array<int, 2> *>(&frag);
    perm_dst[0][0] = __builtin_mxc_byte_perm(fetch_element_ptr[1][0], fetch_element_ptr[0][0],
                                             0x05040100);
    perm_dst[0][1] = __builtin_mxc_byte_perm(fetch_element_ptr[3][0], fetch_element_ptr[2][0],
                                             0x05040100);
    perm_dst[1][0] = __builtin_mxc_byte_perm(fetch_element_ptr[1][0], fetch_element_ptr[0][0],
                                             0x07060302);
    perm_dst[1][1] = __builtin_mxc_byte_perm(fetch_element_ptr[3][0], fetch_element_ptr[2][0],
                                             0x07060302);
    perm_dst[2][0] = __builtin_mxc_byte_perm(fetch_element_ptr[1][1], fetch_element_ptr[0][1],
                                             0x05040100);
    perm_dst[2][1] = __builtin_mxc_byte_perm(fetch_element_ptr[3][1], fetch_element_ptr[2][1],
                                             0x05040100);
    perm_dst[3][0] = __builtin_mxc_byte_perm(fetch_element_ptr[1][1], fetch_element_ptr[0][1],
                                             0x07060302);
    perm_dst[3][1] = __builtin_mxc_byte_perm(fetch_element_ptr[3][1], fetch_element_ptr[2][1],
                                             0x07060302);
  }

  /// Loads a fragment from memory with additional logical offset
  MCTLASS_DEVICE
  void load_with_pointer_offset(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a linear offset
      Index pointer_offset) const {
    load_with_byte_offset(frag, pointer_offset * sizeof(Element));
  }

  /// Loads a fragment from memory with logical offset in units of whole tiles.
  MCTLASS_DEVICE
  void load(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a logical offset in units of whole tiles
      TensorCoord const &tile_offset) const {
    load_with_byte_offset(frag, tile_offset, 0);
  }

  /// Loads a fragment from memory with logical offset in units of whole tiles.
  MCTLASS_DEVICE
  void load(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a logical offset in units of whole tiles
      TensorCoord const &tile_offset,
      /// loads a tile with a logical offset AND a pointer offset
      Index pointer_offset) const {
    load_with_byte_offset(frag, tile_offset, pointer_offset * sizeof(Element));
  }

  /// Loads a fragment from memory with logical offset in units of whole tiles.
  MCTLASS_DEVICE
  void load_with_byte_offset(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a logical offset in units of whole tiles
      TensorCoord const &tile_offset,
      /// loads a tile with a logical offset AND a pointer offset
      Index byte_offset) const {
    Index pointer_offset =
      tile_offset.contiguous() * Shape::kContiguous / Layout::kElementsPerAccess +
      tile_offset.strided() * InstructionShape::kStrided * stride_ / Layout::kElementsPerAccess;

    byte_offset += sizeof(AccessType) * pointer_offset;

    load_with_byte_offset(frag, byte_offset);
  }

  /// Notify the iterator which k-group it is currently pointing to.
  ///
  /// This does not advance the iterator. Rather, it overrides its internal
  /// tracking with constant-valued k-group index to enable the compiler to
  /// fold constants and achieve more efficient code.
  ///
  /// This is used by some nontrivial permuted layouts.
  MCTLASS_DEVICE
  void set_kgroup_index(int k_group) {
    // no op
  }
};

////////////////////////////////////////////////////////////////////////////////

/// This tile iterator is specialized for 64-thread TensorOps. It's used to load from shared
/// memory and therefore must be initialized with a TensorRef to shared memory.
///
/// Satisfies:
///   ReadableRandomAccessContiguousTileIteratorConcept
///
template <
    /// Size of the matrix to load (concept: MatrixShape)
    typename Shape_,
    /// Identifies A or B multiplicand
    Operand Operand_,
    /// Data type of elements
    typename Element_,
    /// Shape of one matrix product operation (concept: MatrixShape)
    typename InstructionShape_,
    /// Interval between adjacent *MMA instructions (in units of MMA
    /// instructions)
    int OpDelta_,
    /// Number of partitions along K dimension
    int PartitionsK_>
class MmaTensorOpMultiplicandTileIterator<
    Shape_, Operand_, Element_,
    mctlass::layout::MacaColumnMajorTensorOpMultiplicandCongruous4x4Perm<
        sizeof_bits<Element_>::value, int(128 / sizeof(Element_))>,
    InstructionShape_, OpDelta_, 32, PartitionsK_> {
 public:

  /// Shape of tile to load (concept: PitchLinearShape)
  using Shape = Shape_;

  /// Operand tag
  static Operand const kOperand = Operand_;

  static_assert(kOperand == Operand::kA,
                "MmaTensorOpMultiplicandIterator for ColumnMajor Congruous may "
                "only be instantiated for A operand to warp-level Mma.");

  /// Element type
  using Element = Element_;

  /// Layout of source tile
  using Layout = mctlass::layout::MacaColumnMajorTensorOpMultiplicandCongruous4x4Perm<
      sizeof_bits<Element_>::value, int(128 / sizeof(Element_))>;

  /// Shape of one matrix product operation (concept: MatrixShape)
  using InstructionShape = InstructionShape_;

  /// Delta between *MMA operations (in units of *MMA operations, concept: MatrixShape)
  static int const kOpDelta = OpDelta_;

  /// Number of participating threads
  static int const kThreads = 64;

  /// TensorRef type for loading element from a tensor
  using TensorRef = TensorRef<Element, Layout>;

  /// Index type
  using Index = typename TensorRef::Index;

  /// Long Index type
  using LongIndex = typename TensorRef::LongIndex;

  /// Long Index type
  using StrideIndex = typename TensorRef::Layout::Stride::Index;

  /// Coordinate for an element in the tensor
  using TensorCoord = typename TensorRef::TensorCoord;

  /// Underlying tile iterator implementation
  using Base = MmaTensorOpMultiplicandTilePrecomputeIterator<
      layout::PitchLinearShape<Shape::kRow, Shape::kColumn>, kOperand, Element,
      layout::MacaTensorOpMultiplicandCongruous4x4Perm<sizeof_bits<Element_>::value,
                                            int(128 / sizeof(Element_))>,
      layout::PitchLinearShape<InstructionShape::kRow,
                               InstructionShape::kColumn>,
      kOpDelta, kThreads, PartitionsK_>;

 public:

  //
  // Derived quantities
  //

  /// Fragment object holding a thread's part of a tile
  using Fragment = typename Base::Fragment;

private:

  /// Underlying tile iterator
  Base iterator_;

public:

  /// Default ctor constructs null iterator
  MCTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator() { }

  /// Constructor from TensorRef
  MCTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator(
    TensorRef const &ref,
    int lane_id
  ): iterator_({ref.data(), ref.stride()}, lane_id) {
  }

  /// Adds a pointer offset to internal pointer(s) to advance through memory
  MCTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator &add_pointer_offset(LongIndex offset) {
    iterator_.add_pointer_offset(offset);

    return *this;
  }

  /// Advances an iterator along logical dimensions of matrix in units of whole tiles
  MCTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator &add_tile_offset(TensorCoord const &tile_offset) {

    iterator_.add_tile_offset({tile_offset.row(), tile_offset.column()});

    return *this;
  }

  MCTLASS_DEVICE
  void pre_compute_pointer(int warp_idx_m = 0, int tile_k = 0) {
    iterator_.pre_compute_offset(warp_idx_m, tile_k);
  }

  MCTLASS_HOST_DEVICE
  void load_with_index(Fragment &frag, int index) {
    iterator_.load_with_index(frag, index);
  }

  /// Advances the iterator along the advance dimension
  MCTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator & operator++() {

    ++iterator_;

    return *this;
  }

  /// Advances the iterator along the advance dimension
  MCTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator & operator--() {

    --iterator_;

    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of the tensor
  MCTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator & operator+=(TensorCoord const &tile_offset) {
    add_tile_offset(PitchLinearCoord(tile_offset.row(), tile_offset.column()));
    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of the tensor
  MCTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator & operator-=(TensorCoord const &tile_offset) {
    add_tile_offset(-PitchLinearCoord(tile_offset.row(), tile_offset.column()));
    return *this;
  }

  /// Loads a fragment from memory at the location pointed to by the iterator.
  MCTLASS_HOST_DEVICE
  void load(Fragment &frag) const {

    iterator_.load(frag);
  }

  /// Loads a fragment from memory with additional logical offset
  MCTLASS_DEVICE
  void load_with_pointer_offset(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a linear offset
      Index pointer_offset) const {
    iterator_.load_with_pointer_offset(frag, pointer_offset);
  }

  /// Loads a fragment from memory with additional logical offset
  MCTLASS_DEVICE
  void load_with_byte_offset(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a linear offset
      Index byte_offset) const {
    iterator_.load_with_byte_offset(frag, byte_offset);
  }

  /// Loads a fragment from memory with logical offset in units of whole tiles.
  MCTLASS_DEVICE
  void load(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a logical offset in units of whole tiles
      TensorCoord const &tile_offset) const {
    // TODO
  }

  /// Loads a fragment from memory with logical offset in units of whole tiles.
  MCTLASS_DEVICE
  void load(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a logical offset in units of whole tiles
      TensorCoord const &tile_offset,
      /// loads a tile with a logical offset AND a pointer offset
      Index pointer_offset) const {
    // TODO
  }

  /// Loads a fragment from memory with logical offset in units of whole tiles.
  MCTLASS_DEVICE
  void load_with_byte_offset(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a logical offset in units of whole tiles
      TensorCoord const &tile_offset,
      /// loads a tile with a logical offset AND a pointer offset
      Index byte_offset) const {
    iterator_.load_with_byte_offset(
      frag,
      {tile_offset.contiguous(), tile_offset.strided()},
      byte_offset);
  }

  /// Notify the iterator which k-group it is currently pointing to.
  ///
  /// This does not advance the iterator. Rather, it overrides its internal
  /// tracking with constant-valued k-group index to enable the compiler to
  /// fold constants and achieve more efficient code.
  ///
  /// This is used by some nontrivial permuted layouts.
  MCTLASS_DEVICE
  void set_kgroup_index(int k_group) {
    iterator_.set_kgroup_index(k_group);
  }
};

////////////////////////////////////////////////////////////////////////////////

/// This tile iterator is specialized for 64-thread TensorOps. It's used to load from shared
/// memory and therefore must be initialized with a TensorRef to shared memory.
///
/// Satisfies:
///   ReadableRandomAccessContiguousTileIteratorConcept
///
template <
    /// Size of the matrix to load (concept: MatrixShape)
    typename Shape_,
    /// Identifies A or B multiplicand
    Operand Operand_,
    /// Data type of elements
    typename Element_,
    /// Shape of one matrix product operation (concept: MatrixShape)
    typename InstructionShape_,
    /// Interval between adjacent *MMA instructions (in units of MMA
    /// instructions)
    int OpDelta_,
    /// Number of partitions along K dimension
    int PartitionsK_>
class MmaTensorOpMultiplicandTileIterator<
    Shape_, Operand_, Element_,
    mctlass::layout::MacaRowMajorTensorOpMultiplicandCongruous4x4Perm<
        sizeof_bits<Element_>::value, int(128 / sizeof(Element_))>,
    InstructionShape_, OpDelta_, 32, PartitionsK_> {
 public:

  /// Shape of tile to load (concept: PitchLinearShape)
  using Shape = Shape_;

  /// Operand tag
  static Operand const kOperand = Operand_;

  static_assert(kOperand == Operand::kB,
                "MmaTensorOpMultiplicandIterator for RowMajor Congruous may "
                "only be instantiated for B operand to warp-level Mma.");

  /// Element type
  using Element = Element_;

  /// Layout of source tile
  using Layout = mctlass::layout::MacaRowMajorTensorOpMultiplicandCongruous4x4Perm<
      sizeof_bits<Element_>::value, int(128 / sizeof(Element_))>;

  /// Shape of one matrix product operation (concept: MatrixShape)
  using InstructionShape = InstructionShape_;

  /// Delta between *MMA operations (in units of *MMA operations, concept: MatrixShape)
  static int const kOpDelta = OpDelta_;

  /// Number of participating threads
  static int const kThreads = 64;

  /// TensorRef type for loading element from a tensor
  using TensorRef = TensorRef<Element, Layout>;

  /// Index type
  using Index = typename TensorRef::Index;

  /// Long Index type
  using LongIndex = typename TensorRef::LongIndex;

  /// Coordinate for an element in the tensor
  using TensorCoord = typename TensorRef::TensorCoord;

  /// Underlying tile iterator implementation
  using Base = MmaTensorOpMultiplicandTilePrecomputeIterator<
      layout::PitchLinearShape<Shape::kColumn, Shape::kRow>, kOperand, Element,
      layout::MacaTensorOpMultiplicandCongruous4x4Perm<sizeof_bits<Element_>::value,
                                            int(128 / sizeof(Element_))>,
      layout::PitchLinearShape<InstructionShape::kColumn,
                               InstructionShape::kRow>,
      kOpDelta, kThreads, PartitionsK_>;

 public:

  //
  // Derived quantities
  //

  /// Fragment object holding a thread's part of a tile
  using Fragment = typename Base::Fragment;
  using Fragment_int = typename Base::Fragment_int;

private:

  /// Underlying tile iterator
  Base iterator_;

public:

  /// Default ctor constructs null iterator
  MCTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator() {}

  /// Constructor from TensorRef
  MCTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator(
    TensorRef const &ref,
    int lane_id
  ): iterator_({ref.data(), ref.stride()}, lane_id) {
  }

  /// Adds a pointer offset to internal pointer(s) to advance through memory
  MCTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator &add_pointer_offset(LongIndex offset) {

    iterator_.add_pointer_offset(offset);

    return *this;
  }

  /// Advances an iterator along logical dimensions of matrix in units of whole tiles
  MCTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator &add_tile_offset(TensorCoord const &tile_offset) {

    iterator_.add_tile_offset({tile_offset.column(), tile_offset.row()});

    return *this;
  }

  /// Advances the iterator along the advance dimension
  MCTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator & operator++() {

    ++iterator_;

    return *this;
  }

  /// Advances the iterator along the advance dimension
  MCTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator & operator--() {

    --iterator_;

    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of the tensor
  MCTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator & operator+=(TensorCoord const &tile_offset) {
    add_tile_offset(PitchLinearCoord(tile_offset.column(), tile_offset.row()));
    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of the tensor
  MCTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator & operator-=(TensorCoord const &tile_offset) {
    add_tile_offset(-PitchLinearCoord(tile_offset.column(), tile_offset.row()));
    return *this;
  }

  MCTLASS_DEVICE
  void fragment_int_conversion(const Fragment_int &frag_in, Fragment &frag) const {
    iterator_.fragment_int_conversion(frag_in, frag);
  }

 /// Loads a fragment from memory at the location pointed to by the iterator.
  MCTLASS_HOST_DEVICE
  void load(Fragment_int &frag) const {

    iterator_.load(frag);
  }

  /// Loads a fragment from memory with additional logical offset
  MCTLASS_DEVICE
  void load_with_pointer_offset(
      /// fragment to load from the tensor
      Fragment_int &frag,
      /// loads a tile with a linear offset
      Index pointer_offset) const {
    iterator_.load_with_pointer_offset(frag, pointer_offset);
  }

  MCTLASS_DEVICE
  void pre_compute_pointer(int warp_idx_n = 0, int tile_k = 0) {
    iterator_.pre_compute_offset(warp_idx_n, tile_k);
  }

  MCTLASS_HOST_DEVICE
  void load_with_index(Fragment &frag, int index) {
    iterator_.load_with_index(frag, index);
  }

  /// Loads a fragment from memory at the location pointed to by the iterator.
  MCTLASS_HOST_DEVICE
  void load(Fragment &frag) const {

    iterator_.load(frag);
  }

  /// Loads a fragment from memory with additional logical offset
  MCTLASS_DEVICE
  void load_with_pointer_offset(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a linear offset
      Index pointer_offset) const {
    iterator_.load_with_pointer_offset(frag, pointer_offset);
  }

  /// Loads a fragment from memory with additional logical offset
  MCTLASS_DEVICE
  void load_with_byte_offset(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a linear offset
      Index byte_offset) const {
    iterator_.load_with_byte_offset(frag, byte_offset);
  }

  /// Loads a fragment from memory with logical offset in units of whole tiles.
  MCTLASS_DEVICE
  void load(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a logical offset in units of whole tiles
      TensorCoord const &tile_offset) const {
    // TODO
  }

  /// Loads a fragment from memory with logical offset in units of whole tiles.
  MCTLASS_DEVICE
  void load(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a logical offset in units of whole tiles
      TensorCoord const &tile_offset,
      /// loads a tile with a logical offset AND a pointer offset
      Index pointer_offset) const {
    // TODO
  }

  /// Loads a fragment from memory with logical offset in units of whole tiles.
  MCTLASS_DEVICE
  void load_with_byte_offset(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a logical offset in units of whole tiles
      TensorCoord const &tile_offset,
      /// loads a tile with a logical offset AND a pointer offset
      Index byte_offset) const {
    iterator_.load_with_byte_offset(
      frag,
      {tile_offset.strided(), tile_offset.contiguous()},
      byte_offset);
  }

  /// Notify the iterator which k-group it is currently pointing to.
  ///
  /// This does not advance the iterator. Rather, it overrides its internal
  /// tracking with constant-valued k-group index to enable the compiler to
  /// fold constants and achieve more efficient code.
  ///
  /// This is used by some nontrivial permuted layouts.
  MCTLASS_DEVICE
  void set_kgroup_index(int k_group) {
    iterator_.set_kgroup_index(k_group);
  }
};

////////////////////////////////////////////////////////////////////////////////

/// This tile iterator is specialized for 32-thread TensorOps. It uses LDSM to
/// load from shared memory and therefore must be initialized with a TensorRef
/// to shared memory.
///
/// Satisfies:
///   ReadableRandomAccessContiguousTileIteratorConcept
///
template <
    /// Size of the matrix to load (concept: MatrixShape)
    typename Shape_,
    /// Identifies A or B multiplicand
    Operand Operand_,
    /// Data type of elements
    typename Element_,
    /// Shape of one matrix product operation (concept: MatrixShape)
    typename InstructionShape_,
    /// Interval between adjacent *MMA instructions (in units of MMA
    /// instructions)
    int OpDelta_,
    /// Element number when the layout crosses (in units of elements)
    int Crosswise,
    /// Number of partitions along K dimension
    int PartitionsK_,
    /// Number of partitions along M dimension
    int PartitionsM_>
class MmaTensorOpMultiplicandTileIterator<
    Shape_, Operand_, Element_,
    mctlass::layout::MacaRowMajorTensorOpMultiplicandMultipleLdg64Crosswise<
        sizeof_bits<Element_>::value, Crosswise>,
    InstructionShape_, OpDelta_, 32, PartitionsK_, PartitionsM_> {
 public:
  /// Shape of tile to load (concept: PitchLinearShape)
  using Shape = Shape_;

  /// Operand tag
  static Operand const kOperand = Operand_;

  static_assert(kOperand == Operand::kA,
                "MmaTensorOpMultiplicandIterator for RowMajor Crosswise may "
                "only be instantiated for A operand to warp-level Mma.");

  /// Element type
  using Element = Element_;

  /// Element number when the layout crosses
  static int const kCrosswise = Crosswise;

  /// Layout of source tile
  using Layout = mctlass::layout::MacaRowMajorTensorOpMultiplicandMultipleLdg64Crosswise<
      sizeof_bits<Element_>::value, kCrosswise>;

  /// Shape of one matrix product operation (concept: MatrixShape)
  using InstructionShape = InstructionShape_;

  /// Delta between *MMA operations (in units of *MMA operations, concept:
  /// MatrixShape)
  static int const kOpDelta = OpDelta_;

  /// Number of participating threads
  static int const kThreads = 64;

  /// TensorRef type for loading element from a tensor
  using TensorRef = TensorRef<Element, Layout>;

  /// Index type
  using Index = typename TensorRef::Index;

  /// Long Index type
  using LongIndex = typename TensorRef::LongIndex;

  /// Coordinate for an element in the tensor
  using TensorCoord = typename TensorRef::TensorCoord;

  /// Underlying tile iterator implementation
  using Base = MmaTensorOpMultiplicandTilePrecomputeIterator<
      layout::PitchLinearShape<Shape::kColumn, Shape::kRow>, kOperand, Element,
      layout::MacaTensorOpMultiplicandMultipleLdg64Crosswise<sizeof_bits<Element_>::value,
                                            kCrosswise>,
      layout::PitchLinearShape<InstructionShape::kColumn,
                               InstructionShape::kRow>,
      kOpDelta, kThreads, PartitionsK_, PartitionsM_>;

 public:
  //
  // Derived quantities
  //

  /// Fragment object holding a thread's part of a tile
  using Fragment = typename Base::Fragment;

 private:
  /// Underlying tile iterator
  Base iterator_;

 public:
  /// Default ctor constructs null iterator
  MCTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator() {}

  /// Constructor from TensorRef
  MCTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator(TensorRef const &ref, int lane_id)
      : iterator_({ref.data(), ref.stride()}, lane_id) {}

  /// Adds a pointer offset to internal pointer(s) to advance through memory
  MCTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator &add_pointer_offset(LongIndex offset) {
    iterator_.add_pointer_offset(offset);

    return *this;
  }

  /// Advances an iterator along logical dimensions of matrix in units of whole
  /// tiles
  MCTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator &add_tile_offset(
      TensorCoord const &tile_offset) {
    iterator_.add_tile_offset({tile_offset.column(), tile_offset.row()});

    return *this;
  }

  /// Advances an iterator along logical dimensions of matrix in units of whole
  /// tiles
  MCTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator &add_tile_offset_negative(
      TensorCoord const &tile_offset) {
    iterator_.add_tile_offset_negative({tile_offset.column(), tile_offset.row()});

    return *this;
  }

  /// Advances the iterator along the advance dimension
  MCTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator &operator++() {
    ++iterator_;

    return *this;
  }

  /// Advances the iterator along the advance dimension
  MCTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator &operator--() {
    --iterator_;

    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of
  ///< the tensor
  MCTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator &operator+=(
      TensorCoord const &tile_offset) {
    add_tile_offset(PitchLinearCoord(tile_offset.column(), tile_offset.row()));
    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of
  ///< the tensor
  MCTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator &operator-=(
      TensorCoord const &tile_offset) {
    add_tile_offset(-PitchLinearCoord(tile_offset.column(), tile_offset.row()));
    return *this;
  }

  MCTLASS_DEVICE
  void pre_compute_pointer(){
    iterator_.pre_compute_pointer_row_major();
  }

  /// Loads a fragment from memory at the location pointed to by the iterator.
  MCTLASS_HOST_DEVICE
  void load(Fragment &frag) { iterator_.load(frag); }

  MCTLASS_HOST_DEVICE
  void load_with_index(Fragment &frag,int index,int offset = 0) { iterator_.load_with_index_A(frag,index,offset); }

  /// Loads a fragment from memory with additional logical offset
  MCTLASS_DEVICE
  void load_with_pointer_offset(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a linear offset
      Index pointer_offset) const {
    iterator_.load_with_pointer_offset(frag, pointer_offset);
  }

  /// Loads a fragment from memory with additional logical offset
  MCTLASS_DEVICE
  void load_with_byte_offset(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a linear offset
      Index byte_offset) const {
    iterator_.load_with_byte_offset(frag, byte_offset);
  }

  /// Loads a fragment from memory with logical offset in units of whole tiles.
  MCTLASS_DEVICE
  void load(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a logical offset in units of whole tiles
      TensorCoord const &tile_offset) const {
    // TODO
    assert(0);
  }

  /// Loads a fragment from memory with logical offset in units of whole tiles.
  MCTLASS_DEVICE
  void load(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a logical offset in units of whole tiles
      TensorCoord const &tile_offset,
      /// loads a tile with a logical offset AND a pointer offset
      Index pointer_offset) const {
    // TODO
    assert(0);
  }

  /// Loads a fragment from memory with logical offset in units of whole tiles.
  MCTLASS_DEVICE
  void load_with_byte_offset(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a logical offset in units of whole tiles
      TensorCoord const &tile_offset,
      /// loads a tile with a logical offset AND a pointer offset
      Index byte_offset) const {
    iterator_.load_with_byte_offset(
        frag, {tile_offset.strided(), tile_offset.contiguous()}, byte_offset);
  }

  /// Notify the iterator which k-group it is currently pointing to.
  ///
  /// This does not advance the iterator. Rather, it overrides its internal
  /// tracking with constant-valued k-group index to enable the compiler to
  /// fold constants and achieve more efficient code.
  ///
  /// This is used by some nontrivial permuted layouts.
  MCTLASS_DEVICE
  void set_kgroup_index(int k_group) {
    iterator_.set_kgroup_index(k_group);
  }
};

/// This tile iterator is specialized for 32-thread TensorOps. It uses LDSM to
/// load from shared memory and therefore must be initialized with a TensorRef
/// to shared memory.
///
/// Satisfies:
///   ReadableRandomAccessContiguousTileIteratorConcept
///
template <
    /// Size of the matrix to load (concept: MatrixShape)
    typename Shape_,
    /// Identifies A or B multiplicand
    Operand Operand_,
    /// Data type of elements
    typename Element_,
    /// Shape of one matrix product operation (concept: MatrixShape)
    typename InstructionShape_,
    /// Interval between adjacent *MMA instructions (in units of MMA
    /// instructions)
    int OpDelta_,
    /// Element number when the layout crosses (in units of elements)
    int Crosswise,
    /// Number of partitions along K dimension
    int PartitionsK_,
    /// Number of partitions along M dimension
    int PartitionsM_>
class MmaTensorOpMultiplicandTileIterator<
    Shape_, Operand_, Element_,
    mctlass::layout::MacaColumnMajorTensorOpMultiplicandMultipleLdg64Crosswise<
        sizeof_bits<Element_>::value, Crosswise>,
    InstructionShape_, OpDelta_, 32, PartitionsK_, PartitionsM_> {
 public:
  /// Shape of tile to load (concept: PitchLinearShape)
  using Shape = Shape_;

  /// Operand tag
  static Operand const kOperand = Operand_;

  static_assert(kOperand == Operand::kB,
                "MmaTensorOpMultiplicandIterator for RowMajor Crosswise may "
                "only be instantiated for A operand to warp-level Mma.");

  /// Element type
  using Element = Element_;

  /// Element number when the layout crosses
  static int const kCrosswise = Crosswise;

  /// Layout of source tile
  using Layout = mctlass::layout::MacaColumnMajorTensorOpMultiplicandMultipleLdg64Crosswise<
      sizeof_bits<Element_>::value, kCrosswise>;

  /// Shape of one matrix product operation (concept: MatrixShape)
  using InstructionShape = InstructionShape_;

  /// Delta between *MMA operations (in units of *MMA operations, concept:
  /// MatrixShape)
  static int const kOpDelta = OpDelta_;

  /// Number of participating threads
  static int const kThreads = 64;

  /// TensorRef type for loading element from a tensor
  using TensorRef = TensorRef<Element, Layout>;

  /// Index type
  using Index = typename TensorRef::Index;

  /// Long Index type
  using LongIndex = typename TensorRef::LongIndex;

  /// Coordinate for an element in the tensor
  using TensorCoord = typename TensorRef::TensorCoord;

  /// Underlying tile iterator implementation
  using Base = MmaTensorOpMultiplicandTilePrecomputeIterator<
      layout::PitchLinearShape<Shape::kRow, Shape::kColumn>, kOperand, Element,
      layout::MacaTensorOpMultiplicandMultipleLdg64Crosswise<sizeof_bits<Element_>::value,
                                            kCrosswise>,
      layout::PitchLinearShape<InstructionShape::kRow,
                               InstructionShape::kColumn>,
      kOpDelta, kThreads, PartitionsK_, PartitionsM_>;

 public:
  //
  // Derived quantities
  //

  /// Fragment object holding a thread's part of a tile
  using Fragment = typename Base::Fragment;

 private:
  /// Underlying tile iterator
  Base iterator_;

 public:
  /// Default ctor constructs null iterator
  MCTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator() {}

  /// Constructor from TensorRef
  MCTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator(TensorRef const &ref, int lane_id)
      : iterator_({ref.data(), ref.stride()}, lane_id) {}

  /// Adds a pointer offset to internal pointer(s) to advance through memory
  MCTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator &add_pointer_offset(LongIndex offset) {

    iterator_.add_pointer_offset(offset);

    return *this;
  }

  /// Advances an iterator along logical dimensions of matrix in units of whole
  /// tiles
  MCTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator &add_tile_offset(
      TensorCoord const &tile_offset) {
    iterator_.add_tile_offset({tile_offset.row(), tile_offset.column()});

    return *this;
  }

  /// Advances an iterator along logical dimensions of matrix in units of whole
  /// tiles
  MCTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator &add_tile_offset_negative(
      TensorCoord const &tile_offset) {
    iterator_.add_tile_offset_negative({tile_offset.row(), tile_offset.column()});

    return *this;
  }

  /// Advances the iterator along the advance dimension
  MCTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator &operator++() {
    ++iterator_;

    return *this;
  }

  /// Advances the iterator along the advance dimension
  MCTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator &operator--() {
    --iterator_;

    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of
  ///< the tensor
  MCTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator &operator+=(
      TensorCoord const &tile_offset) {
    add_tile_offset(PitchLinearCoord(tile_offset.row(), tile_offset.column()));
    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of
  ///< the tensor
  MCTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator &operator-=(
      TensorCoord const &tile_offset) {
    add_tile_offset(-PitchLinearCoord(tile_offset.row(), tile_offset.column()));
    return *this;
  }

  MCTLASS_DEVICE
  void pre_compute_pointer(){
    iterator_.pre_compute_pointer_col_major();
  }
  /// Loads a fragment from memory at the location pointed to by the iterator.
  MCTLASS_HOST_DEVICE
  void load(Fragment &frag)  { iterator_.load(frag); }

  MCTLASS_HOST_DEVICE
  void load_with_index(Fragment &frag,int index, int offset = 0) { iterator_.load_with_index_B(frag,index,offset); }

  /// Loads a fragment from memory with additional logical offset
  MCTLASS_DEVICE
  void load_with_pointer_offset(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a linear offset
      Index pointer_offset) const {
    iterator_.load_with_pointer_offset(frag, pointer_offset);
  }

  /// Loads a fragment from memory with additional logical offset
  MCTLASS_DEVICE
  void load_with_byte_offset(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a linear offset
      Index byte_offset) const {
    iterator_.load_with_byte_offset(frag, byte_offset);
  }

  /// Loads a fragment from memory with logical offset in units of whole tiles.
  MCTLASS_DEVICE
  void load(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a logical offset in units of whole tiles
      TensorCoord const &tile_offset) const {
    // TODO
    assert(0);
  }

  /// Loads a fragment from memory with logical offset in units of whole tiles.
  MCTLASS_DEVICE
  void load(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a logical offset in units of whole tiles
      TensorCoord const &tile_offset,
      /// loads a tile with a logical offset AND a pointer offset
      Index pointer_offset) const {
    // TODO
    assert(0);
  }

  /// Loads a fragment from memory with logical offset in units of whole tiles.
  MCTLASS_DEVICE
  void load_with_byte_offset(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a logical offset in units of whole tiles
      TensorCoord const &tile_offset,
      /// loads a tile with a logical offset AND a pointer offset
      Index byte_offset) const {
    iterator_.load_with_byte_offset(
        frag, {tile_offset.contiguous(), tile_offset.strided()}, byte_offset);
  }

  /// Notify the iterator which k-group it is currently pointing to.
  ///
  /// This does not advance the iterator. Rather, it overrides its internal
  /// tracking with constant-valued k-group index to enable the compiler to
  /// fold constants and achieve more efficient code.
  ///
  /// This is used by some nontrivial permuted layouts.
  MCTLASS_DEVICE
  void set_kgroup_index(int k_group) {
    iterator_.set_kgroup_index(k_group);
  }
};

} // namespace warp
} // namespace gemm
} // namespace mctlass

////////////////////////////////////////////////////////////////////////////////
