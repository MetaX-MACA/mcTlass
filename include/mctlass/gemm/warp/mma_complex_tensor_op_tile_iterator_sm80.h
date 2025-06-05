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
#include "mctlass/layout/tensor_op_multiplicand_sm80.h"

#include "mctlass/platform/platform.h"
#include "mctlass/fast_math.h"

#include "mctlass/gemm/warp/mma_tensor_op_tile_iterator.h"

////////////////////////////////////////////////////////////////////////////////

namespace mctlass {
namespace gemm {
namespace warp {

////////////////////////////////////////////////////////////////////////////////

/// This tile iterator is specialized for loading 128b vectors of 128b elements.
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
    mctlass::layout::TensorOpMultiplicandCongruous128b,
    InstructionShape_, OpDelta_, 32, PartitionsK_> {
 public:

  /// Shape of tile to load (concept: PitchLinearShape)
  using Shape = Shape_;

  /// Operand tag
  static Operand const kOperand = Operand_;

  static_assert(kOperand == Operand::kA || kOperand== Operand::kB,
    "MmaTensorOpMultiplicandIterator may only be instantiated for A or B operands to warp-level Mma.");

  static_assert(!(Shape::kContiguous % 8) && !(Shape::kStrided % 4), "Divisibility.");

  static_assert(sizeof_bits<Element_>::value == 128, "This is specialized for 128b accesses.");

  /// Element type
  using Element = Element_;

  /// Layout of source tile
  using Layout = mctlass::layout::TensorOpMultiplicandCongruous128b;

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

  /// Load two elements per access
  static int const kElementsPerAccess = 1;

  /// Policy defining internal details of tile iterator
  struct Policy {

    /// Shape of one access
    using Delta = layout::PitchLinearShape<8, 4>;

    /// Number of iterations to load
    using Iterations = layout::PitchLinearShape<
      Shape::kContiguous / Delta::kContiguous,
      InstructionShape::kStrided / Delta::kStrided
    >;
  };

private:

  /// Not working on this feature at the moment.
  static_assert(kOpDelta == 1,
    "Alternative arrangements not supported at present.");

  /// Pointer type used for accesses
  using AccessType = AlignedArray<Element, kElementsPerAccess, 16>;

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
  AccessType const *pointer_;

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
    stride_(ref.stride(0) / kElementsPerAccess), byte_offset_(0) {

    int quad_pair = lane_id / 8;
    int quad = lane_id / 4;
    int lane = lane_id % 4;

    int row = (quad & 1) * 4 + (lane ^ quad_pair);
    
    byte_offset_ = (row + quad_pair * stride_) * sizeof(AccessType);

    pointer_= reinterpret_cast<AccessType const *>(ref.data());

    //Special compute for mma shape with m8n8k4f64
    if (platform::is_same<Element, mctlass::complex<double>>::value == true) {
      
      const int quad_pair = lane_id & 0x7;
      const int quad = (lane_id >> 4);
      const int lane = quad_pair / 2;
      const int row = ((quad_pair & 1) * 4) | (quad ^ lane);
      byte_offset_ = (row + lane * stride_) * sizeof(AccessType);

    }

  }

  /// Adds a pointer offset to internal pointer(s) to advance through memory
  MCTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator &add_pointer_offset(LongIndex offset) {

    pointer_ += offset;

    return *this;
  }

  /// Advances an iterator along logical dimensions of matrix in units of whole tiles
  MCTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator &add_tile_offset(TensorCoord const &tile_offset) {

    int offset =
      (tile_offset.contiguous() * Shape::kContiguous) +
      (tile_offset.strided() * InstructionShape::kStrided * stride_);

    add_pointer_offset(offset);

    return *this;
  }

  /// Advances the iterator along the advance dimension
  MCTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator & operator++() {

    pointer_ += stride_ * InstructionShape::kStrided;

    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of the tensor
  MCTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator & operator+=(TensorCoord const &tile_offset) {
    add_tile_offset(tile_offset);
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

    AccessType *fetch_ptr = reinterpret_cast<AccessType *>(&frag);

    MCTLASS_PRAGMA_UNROLL
    for (int s = 0; s < Policy::Iterations::kStrided; ++s) {

      MCTLASS_PRAGMA_UNROLL
      for (int c = 0; c < Policy::Iterations::kContiguous; ++c) {

        int access_idx = c + s * Policy::Iterations::kContiguous;

        AccessType const *source_ptr = pointer_ +
            Policy::Delta::kContiguous * c +
            Policy::Delta::kStrided * s * stride_;

        char const *source_byte_ptr = reinterpret_cast<char const *>(source_ptr) + byte_offset + byte_offset_;

        AccessType const *source = reinterpret_cast<AccessType const *>(source_byte_ptr);

        fetch_ptr[access_idx] = *source;
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
        tile_offset.contiguous() * Shape::kContiguous +
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

  }
};

////////////////////////////////////////////////////////////////////////////////
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
    mctlass::layout::RowMajorTensorOpMultiplicandCongruous128b,
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
  using Layout = mctlass::layout::RowMajorTensorOpMultiplicandCongruous128b;

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
      layout::PitchLinearShape<Shape::kColumn, Shape::kRow>, kOperand, Element,
      layout::TensorOpMultiplicandCongruous128b,
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
    add_tile_offset(layout::PitchLinearCoord(tile_offset.column(), tile_offset.row()));
    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of the tensor
  MCTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator & operator-=(TensorCoord const &tile_offset) {
    add_tile_offset(layout::PitchLinearCoord(-tile_offset.column(), -tile_offset.row()));
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
    mctlass::layout::ColumnMajorTensorOpMultiplicandCongruous128b,
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
  using Layout = mctlass::layout::ColumnMajorTensorOpMultiplicandCongruous128b;

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
      layout::TensorOpMultiplicandCongruous128b,
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
    add_tile_offset(layout::PitchLinearCoord(tile_offset.row(), tile_offset.column()));
    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of the tensor
  MCTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator & operator-=(TensorCoord const &tile_offset) {
    add_tile_offset(layout::PitchLinearCoord(-tile_offset.row(), -tile_offset.column()));
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

/////////////////////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////////////////////
/// 
/// Partial specialization for complex<T>
///
template <
    /// Size of the matrix to load (concept: MatrixShape)
    typename Shape_,
    /// Data type of underlying field of reals.
    typename RealElement,
    /// Shape of one matrix product operation (concept: MatrixShape)
    typename InstructionShape_,
    /// Interval between adjacent *MMA instructions (in units of MMA
    /// instructions, concept: MatrixShape)
    typename OpDelta_>
class MmaTensorOpAccumulatorTileIterator<
    Shape_, complex<RealElement>, mctlass::layout::RowMajor, InstructionShape_, OpDelta_> {
 public:

  /// Shape of tile to load (concept: MatrixShape)
  using Shape = Shape_;

  /// Operand tag
  static Operand const kOperand = Operand::kC;

  /// Element type
  using Element = complex<RealElement>;

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

    static_assert(platform::is_same<TensorCoord, MatrixCoord>::value,
      "Layouts must be defined for logical MatrixCoord coordinate space.");

    /// Number of mma operations performed
    using MmaIterations = MatrixShape<Shape::kRow / InstructionShape::kM,
                                      Shape::kColumn / InstructionShape::kN>;
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

  /// Fragment object holding a thread's part of a tile. It is assumed that the accumulators
  /// are stored in a planar complex arrangement with the real parts as entirely contiguous
  /// followed by the imaginary parts.
  using Fragment = Array<RealElement, Shape::kCount / kThreads * 2>;

  static int const kRealIndex = 0;
  static int const kImaginaryIndex = Shape::kCount / kThreads;

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
    
    if (platform::is_same<Element, mctlass::complex<double>>::value == true) {
      //Special process for mma shape: m8n8k4f64
      if (InstructionShape::kM == 8 && InstructionShape::kN == 8 
          && InstructionShape::kK == 4) {
            
          const int lane_in_quad = lane_id & 0x7;
          const int quad = (lane_id >> 4) & 0x3;
          MatrixCoord lane_offset(quad, lane_in_quad);
          ref_.add_coord_offset(lane_offset);

      }
      else {
        //To Do
        printf("Not Impl for other InstructionShape...\n");
      }
    }
    else if (platform::is_same<Element, mctlass::complex<float>>::value == true) {
      const int quad = (lane_id >> 4) << 2;
      const int lane_in_quad = lane_id & 0x7;
      MatrixCoord lane_offset(quad, lane_in_quad);
      ref_.add_coord_offset(lane_offset);
    }
    else {

      const int quad = (lane_id >> 2);
      const int lane_in_quad = (lane_id & 3);
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

            Element z = offset_ref.at({accum_m, accum_n});

            frag[mma_accum_start + row * kElementsPerAccess + col + kRealIndex] = z.real();
            frag[mma_accum_start + row * kElementsPerAccess + col + kImaginaryIndex] = z.imag();
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
            int accum_m = 0;
            int accum_n = 0;
            int idx = 0;
            if (platform::is_same<Element, mctlass::complex<double>>::value == true) {
              //Special process for mma shape: m8n8k4f64
              if (InstructionShape::kM == 8 &&InstructionShape::kN == 8 
                  && InstructionShape::kK == 4) {
                accum_m = mma_m * InstructionShape::kM * OpDelta::kRow + row * kRowsPerTile +
                          col * (InstructionShape::kM / kElementsPerAccess);
                accum_n = mma_n * InstructionShape::kN * OpDelta::kColumn;
                idx =mma_accum_start + row * kElementsPerAccess + col;
              }
              else {
                //To Do
                printf("Not Impl now.\n");
              }
            }
            else if (platform::is_same<Element, mctlass::complex<float>>::value == true) {
              accum_m = mma_m * InstructionShape::kM;
              accum_n = mma_n * InstructionShape::kN * OpDelta::kColumn ;
              int t_idx = row * kElementsPerAccess + col;
              idx = mma_accum_start + t_idx;
              accum_m += t_idx;
            }
            else {
              accum_m = mma_m * InstructionShape::kM * OpDelta::kRow +
                          row * kRowsPerTile;
              accum_n = mma_n * InstructionShape::kN * OpDelta::kColumn + col;
              idx =mma_accum_start + row * kElementsPerAccess + col;
            }

            Element z(frag[kRealIndex + idx], frag[kImaginaryIndex + idx]);

            offset_ref.at({accum_m, accum_n}) = z;
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

/////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////////////

/// This tile iterator is specialized for loading 128b vectors of 128b elements.
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
    mctlass::layout::TensorOpMultiplicandCrosswise128x4,
    InstructionShape_, OpDelta_, 32, PartitionsK_> {
 public:

  /// Shape of tile to load (concept: PitchLinearShape)
  using Shape = Shape_;

  /// Operand tag
  static Operand const kOperand = Operand_;

  static_assert(kOperand == Operand::kA || kOperand== Operand::kB,
    "MmaTensorOpMultiplicandIterator may only be instantiated for A or B operands to warp-level Mma.");

  static_assert(!(Shape::kContiguous % 4) && !(Shape::kStrided % 8), "Divisibility.");

  static_assert(sizeof_bits<Element_>::value == 128, "This is specialized for 128b accesses.");

  /// Element type
  using Element = Element_;

  /// Layout of source tile
  using Layout = mctlass::layout::TensorOpMultiplicandCrosswise128x4;

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

  /// Load two elements per access
  static int const kElementsPerAccess = 1;

  /// Policy defining internal details of tile iterator
  struct Policy {

    /// Shape of one access
    using Delta = layout::PitchLinearShape<4, 8>;

    /// Number of iterations to load
    using Iterations = layout::PitchLinearShape<
      InstructionShape::kContiguous / Delta::kContiguous,
      Shape::kStrided / Delta::kStrided
    >;
  };

private:

  /// Not working on this feature at the moment.
  static_assert(kOpDelta == 1,
    "Alternative arrangements not supported at present.");

  /// Pointer type used for accesses
  using AccessType = AlignedArray<Element, kElementsPerAccess, 16>;

public:

  //
  // Derived quantities
  //

  /// Fragment object holding a thread's part of a tile
 using Fragment =
     Array<Element, Shape::kStrided * InstructionShape::kContiguous / kThreads>;

private:

  /// Layout object storing stride values
  StrideIndex stride_;

  /// Shared memory base pointers - not advanced
  AccessType const *pointer_;

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
    stride_(ref.stride(0) / kElementsPerAccess), byte_offset_(0) {

    int quad = lane_id / 4;
    int liq = lane_id % 4;

    int c = liq + (quad & 1) * 4;
    int s = (quad / 2);

    byte_offset_ = (c + s * stride_) * sizeof(AccessType);

    pointer_= reinterpret_cast<AccessType const *>(ref.data());

    //Special process for mma shape: m8n8k4f64
    if (platform::is_same<Element, mctlass::complex<double>>::value == true) {
     
      const int quad = lane_id & 0x7;
      const int liq = (lane_id >> 4) & 0x3;
      const int c = liq + (quad & 1) * 4;
      const int s = (quad / 4) * 2 + (quad & 2) / 2;
      byte_offset_ = (c + s * stride_) * sizeof(AccessType);

    }

  }

  /// Adds a pointer offset to internal pointer(s) to advance through memory
  MCTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator &add_pointer_offset(LongIndex offset) {

    pointer_ += offset;

    return *this;
  }

  /// Advances an iterator along logical dimensions of matrix in units of whole tiles
  MCTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator &add_tile_offset(TensorCoord const &tile_offset) {

    // Compute the offset in units of elements. Note, the external coordinate system is
    // approximately transposed with respect to the tiled internal structure
    int offset =
      (tile_offset.contiguous() * InstructionShape::kContiguous) * stride_ +
      (tile_offset.strided() * Shape::kStrided);

    add_pointer_offset(offset);

    byte_offset_ ^= (tile_offset.contiguous() & 1) * 4 * sizeof(AccessType);

    return *this;
  }

  /// Advances the iterator along the advance dimension
  MCTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator & operator++() {

    pointer_ += stride_ * InstructionShape::kContiguous;

    byte_offset_ ^= 4 * sizeof(AccessType);

    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of the tensor
  MCTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator & operator+=(TensorCoord const &tile_offset) {
    add_tile_offset(tile_offset);

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

    AccessType *fetch_ptr = reinterpret_cast<AccessType *>(&frag);

    MCTLASS_PRAGMA_UNROLL
    for (int c = 0; c < Policy::Iterations::kContiguous; ++c) {

      MCTLASS_PRAGMA_UNROLL
      for (int s = 0; s < Policy::Iterations::kStrided; ++s) {

        int access_idx = s + c * Policy::Iterations::kStrided;

        AccessType const *source_ptr = pointer_ +
            Policy::Delta::kContiguous * c * stride_ +
            Policy::Delta::kStrided * s;

        char const *source_byte_ptr = reinterpret_cast<char const *>(source_ptr) + byte_offset + byte_offset_;

        AccessType const *source = reinterpret_cast<AccessType const *>(source_byte_ptr);

        fetch_ptr[access_idx] = *source;
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
        tile_offset.contiguous() * InstructionShape::kContiguous * stride_ +
        tile_offset.strided() * Shape::kStrided;

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

  }
};


////////////////////////////////////////////////////////////////////////////////
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
    mctlass::layout::RowMajorTensorOpMultiplicandCrosswise128x4,
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
  using Layout = mctlass::layout::RowMajorTensorOpMultiplicandCrosswise128x4;

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
      layout::PitchLinearShape<Shape::kColumn, Shape::kRow>, kOperand, Element,
      layout::TensorOpMultiplicandCrosswise128x4,
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
    add_tile_offset(layout::PitchLinearCoord(tile_offset.column(), tile_offset.row()));
    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of the tensor
  MCTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator & operator-=(TensorCoord const &tile_offset) {
    add_tile_offset(layout::PitchLinearCoord(-tile_offset.column(), -tile_offset.row()));
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
    mctlass::layout::ColumnMajorTensorOpMultiplicandCrosswise128x4,
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
  using Layout = mctlass::layout::ColumnMajorTensorOpMultiplicandCrosswise128x4;

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
      layout::TensorOpMultiplicandCrosswise128x4,
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
    add_tile_offset(layout::PitchLinearCoord(tile_offset.row(), tile_offset.column()));
    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of the tensor
  MCTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator & operator-=(TensorCoord const &tile_offset) {
    add_tile_offset(layout::PitchLinearCoord(-tile_offset.row(), -tile_offset.column()));
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

/////////////////////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////////////////////
// Congruous shared memory layout
// Warp-level iterators for complex<float>*complex<float> + complex<float> => complex<float>
// The underlying iterators are similar to that for MMA f64*f64 + f64 = f64 
/////////////////////////////////////////////////////////////////////////////////////////////////

/// This tile iterator is specialized for loading 128b vectors of 64b elements.
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
    /// Number of partitions along K dimension
    int PartitionsK_>
class MmaTensorOpMultiplicandTileIterator<
    Shape_, Operand_, mctlass::complex<float>,
    mctlass::layout::TensorOpMultiplicandCongruous64b,
    InstructionShape_, OpDelta_, 32, PartitionsK_> {
 public:

  /// Shape of tile to load (concept: PitchLinearShape)
  using Shape = Shape_;

  /// Operand tag
  static Operand const kOperand = Operand_;

  static_assert(kOperand == Operand::kA || kOperand== Operand::kB,
    "MmaTensorOpMultiplicandIterator may only be instantiated for A or B operands to warp-level Mma.");

  static_assert(!(Shape::kContiguous % 16) && !(Shape::kStrided % 8), "Divisibility.");

  /// Element type
  using Element = mctlass::complex<float>;

  /// Layout of source tile
  using Layout = mctlass::layout::TensorOpMultiplicandCongruous64b;

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

  /// Load two elements per access
  static int const kElementsPerAccess = 2;

  /// Policy defining internal details of tile iterator
  struct Policy {

    /// Shape of one access
    using Delta = layout::PitchLinearShape<8, 4>;

    /// Number of iterations to load
    using Iterations = layout::PitchLinearShape<
      Shape::kContiguous / kElementsPerAccess / Delta::kContiguous,
      InstructionShape::kStrided / Delta::kStrided
    >;

  };

private:

  /// Not working on this feature at the moment.
  static_assert(kOpDelta == 1,
    "Alternative arrangements not supported at present.");

  /// Pointer type used for accesses
  using AccessType = AlignedArray<Element, kElementsPerAccess, 16>;

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
  AccessType const *pointer_[kElementsPerAccess];

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
    stride_(ref.stride(0) / kElementsPerAccess), byte_offset_(0),
    k_group_idx_(0) {

   /* int access_strided = lane_id / Policy::Delta::kContiguous;
    int access_contiguous = (lane_id % Policy::Delta::kContiguous) ^ access_strided;
    pointer_ = reinterpret_cast<AccessType const *>(ref.data()) +
        access_contiguous + access_strided * stride_;
    */
    int access_strided = lane_id / Policy::Delta::kContiguous;
    for (int i = 0; i < kElementsPerAccess; ++i) {
      int lane_id_offset = ((__lane_id() >> 4) & 0x1) - 2 * (__lane_id() >> 5) + i;
      int access_contiguous = ((lane_id + lane_id_offset) % Policy::Delta::kContiguous) ^ access_strided;
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

    int offset = 
      (tile_offset.strided() * InstructionShape::kStrided) * stride_ * kElementsPerAccess + 
      tile_offset.contiguous() * Shape::kContiguous;

    add_pointer_offset(offset);

    return *this;
  }

  /// Advances the iterator along the advance dimension
  MCTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator & operator++() {

    add_tile_offset({0, 1});

    return *this;
  }

  /// Advances the iterator along the opposite of the advance dimension
  MCTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator & operator--() {
    
    add_tile_offset({0, -1});

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

    AccessType *fetch_ptr = reinterpret_cast<AccessType *>(&frag);

    MCTLASS_PRAGMA_UNROLL
    for (int s = 0; s < Policy::Iterations::kStrided; ++s) {

      MCTLASS_PRAGMA_UNROLL
      for (int c = 0; c < Policy::Iterations::kContiguous; ++c) {

        int access_idx = c + s * Policy::Iterations::kContiguous;

       /* AccessType const *source_ptr = pointer_ +
            Policy::Delta::kContiguous * c +
            Policy::Delta::kStrided * s * stride_;

        char const *source_byte_ptr = reinterpret_cast<char const *>(source_ptr) + byte_offset + byte_offset_;

        AccessType const *source = reinterpret_cast<AccessType const *>(source_byte_ptr);

        fetch_ptr[access_idx] = *source;
       */
        if (kOperand == Operand::kA) {
          AccessType const *source_ptr[kElementsPerAccess];
          char const *source_byte_ptr[kElementsPerAccess];
          Element const *source_element[kElementsPerAccess];
          for (int i = 0; i < kElementsPerAccess; ++i) {
            source_ptr[i] = pointer_[i] + Policy::Delta::kStrided * (__lane_id() >> 5) * stride_;
            source_byte_ptr[i] = reinterpret_cast<char const *>(source_ptr[i]) + byte_offset + byte_offset_ + 16 * access_idx * Policy::Delta::kContiguous;
            source_element[i] = reinterpret_cast<Element const *>(source_byte_ptr[i] + ((__lane_id() >> 3) & 0x1) * Policy::Delta::kContiguous);
          }
          Element source_element_ptr[kElementsPerAccess] ={*source_element[0], *source_element[1]};
          char const *source_byte_temp_ptr = reinterpret_cast<char const *>(source_element_ptr);
          AccessType const *source = reinterpret_cast<AccessType const *>(source_byte_temp_ptr);
          fetch_ptr[access_idx] = *source;
        }
        else {
          int index = access_idx & 0x1;
          int offset = access_idx / kElementsPerAccess;
          AccessType const *source_ptr = pointer_[index] + Policy::Delta::kStrided * (__lane_id() >> 5) * stride_;
          char const *source_byte_ptr = reinterpret_cast<char const *>(source_ptr) + byte_offset + byte_offset_ 
                                        + 16 * offset * Policy::Delta::kContiguous;
          AccessType const *source = reinterpret_cast<AccessType const *>(source_byte_ptr);
          if(Policy::Iterations::kContiguous > 1) {
            fetch_ptr[index * 2 + offset] = *source;
          }
          else {
            fetch_ptr[index] = *source;
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

  }
};

////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////////////////////
// Crosswise shared memory layout
// Warp-level iterators for complex<float>*complex<float> + complex<float> => complex<float>
// The underlying iterators are similar to that for f64*f64 + f64 = f64 
/////////////////////////////////////////////////////////////////////////////////////////////////

/// This tile iterator is specialized for loading 128b vectors of 64b elements.
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
    /// Number of partitions along K dimension
    int PartitionsK_>
class MmaTensorOpMultiplicandTileIterator<
    Shape_, Operand_, complex<float>,
    mctlass::layout::TensorOpMultiplicand64bCrosswise,
    InstructionShape_, OpDelta_, 32, PartitionsK_> {
 public:

  /// Shape of tile to load (concept: PitchLinearShape)
  using Shape = Shape_;

  /// Operand tag
  static Operand const kOperand = Operand_;

  static_assert(kOperand == Operand::kA || kOperand== Operand::kB,
    "MmaTensorOpMultiplicandIterator may only be instantiated for A or B operands to warp-level Mma.");

  static_assert(!(Shape::kContiguous % 4) && !(Shape::kStrided % 16), "Divisibility.");

  static_assert(sizeof_bits<complex<float>>::value == 64, "This is specialized for 64b accesses.");

  /// Element type
  using Element = complex<float>;

  /// Layout of source tile
  using Layout = mctlass::layout::TensorOpMultiplicand64bCrosswise;

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

  /// Load two elements per access
  static int const kElementsPerAccess = 2;

  /// Policy defining internal details of tile iterator
  struct Policy {

    /// Shape of one access
    using Delta = layout::PitchLinearShape<4, 16>;

    /// Number of iterations to load
    using Iterations = layout::PitchLinearShape<
      InstructionShape::kContiguous / Delta::kContiguous,
      Shape::kStrided / Delta::kStrided
    >;

  };

private:

  /// Not working on this feature at the moment.
  static_assert(kOpDelta == 1,
    "Alternative arrangements not supported at present.");

  /// Pointer type used for accesses
  using AccessType = AlignedArray<Element, kElementsPerAccess, 16>;

public:

  //
  // Derived quantities
  //

  /// Fragment object holding a thread's part of a tile
 using Fragment =
     Array<Element, Shape::kStrided * InstructionShape::kContiguous / kThreads>;

private:

  /// Layout object storing stride values
  StrideIndex stride_;

  /// Shared memory base pointers - not advanced
  AccessType const *pointer_;

  /// Byte offset incremented as iterator advances
  Index byte_offset_;

  /// Internal counter for tracking K-group
  Index k_group_idx_;

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
    stride_(ref.stride(0) / kElementsPerAccess), byte_offset_(0),
    k_group_idx_(0) {
   /*
    int access_strided = lane_id / 8;
    int access_contiguous = (lane_id  % 8);

    byte_offset_ = (access_contiguous + access_strided * stride_) * sizeof(AccessType);

    pointer_= reinterpret_cast<AccessType const *>(ref.data());
    */
    lane_id = __lane_id();
    if (kOperand == Operand::kA) {
      int row = lane_id & 0xf;
      int col = (lane_id >> 4) << 1;
      TensorCoord coord(col, row);
      Layout lo(ref.stride(0));
      byte_offset_ =  lo(coord) * (sizeof_bits<Element>::value / 8);

    } 
    else {
      int row = (lane_id >> 4) << 1;
      int col = lane_id & 0x7;
      TensorCoord coord(row, col);
      Layout lo(ref.stride(0));
      byte_offset_ =  lo(coord) * (sizeof_bits<Element>::value / 8);
    }
    pointer_= reinterpret_cast<AccessType const *>(ref.data());
  }

  /// Adds a pointer offset to internal pointer(s) to advance through memory
  MCTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator &add_pointer_offset(LongIndex offset) {

    pointer_ += offset / kElementsPerAccess;

    return *this;
  }

  /// Advances an iterator along logical dimensions of matrix in units of whole tiles
  MCTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator &add_tile_offset(TensorCoord const &tile_offset) {
    int offset = (tile_offset.contiguous() * InstructionShape::kContiguous) *
                     stride_ * kElementsPerAccess +
                 tile_offset.strided() * Shape::kStrided;

    add_pointer_offset(offset);
    
    
    return *this;
  }

  /// Advances an iterator along logical dimensions of matrix in units of whole tiles
  MCTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator &add_tile_offset_negative(TensorCoord const &tile_offset) {

    add_tile_offset(tile_offset);

    if (k_group_idx_ & 1)
      byte_offset_ ^= 0x40;

    return *this;
  }

  /// Advances the iterator along the advance dimension
  MCTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator & operator++() {

    pointer_ += stride_ * InstructionShape::kContiguous;
    
    // xor ptr
    byte_offset_ ^= 0x40;

    ++k_group_idx_;

    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of the tensor
  MCTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator & operator+=(TensorCoord const &tile_offset) {
    add_tile_offset(tile_offset);
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

    AccessType *fetch_ptr = reinterpret_cast<AccessType *>(&frag);
    MCTLASS_PRAGMA_UNROLL
    for (int c = 0; c < Policy::Iterations::kContiguous; ++c) {

      MCTLASS_PRAGMA_UNROLL
      for (int s = 0; s < Policy::Iterations::kStrided; ++s) {

        int access_idx = c * Policy::Iterations::kStrided + s;

       /* AccessType const *source_ptr = pointer_ +
            Policy::Delta::kContiguous * c * stride_ +
            Policy::Delta::kStrided * s / kElementsPerAccess;

        char const *source_byte_ptr = reinterpret_cast<char const *>(source_ptr) + byte_offset + byte_offset_;

        AccessType const *source = reinterpret_cast<AccessType const *>(source_byte_ptr);

        fetch_ptr[access_idx] = *source;
      */
        /// For a 4x4 A-matrix: 1  1  2  3
        //                      4  5  6  7
        //                      8  9  10 11
        //                      12 13 14 15
        //  the reading order is (__lane_id==0)1 1  ; (__lane_id==1)4 5  ; (__lane_id==2)8 9    ; (__lane_id==3)12 13
        //                       (__lane_id==16)2 3 ; (__lane_id==17)6 7 ; (__lane_id==18)10 11 ; (__lane_id==19)14 15
	
        /// For a 4x4 B-matrix: 1  1  2  3
        //                      4  5  6  7
        //                      8  9  10 11
        //                      12 13 14 15
        //  the reading order is (__lane_id==0)1 4   ; (__lane_id==1)1 5   ; (__lane_id==2)2 6    ; (__lane_id==3)3 7
        //                       (__lane_id==16)8 12 ; (__lane_id==17)9 13 ; (__lane_id==18)10 14 ; (__lane_id==19)11 15

         if (kOperand == Operand::kA) {
            AccessType const *source_ptr = pointer_ +
            Policy::Delta::kContiguous * c * stride_ +
            Policy::Delta::kStrided * s / kElementsPerAccess;
            char const *source_byte_ptr = reinterpret_cast<char const *>(source_ptr) + byte_offset + byte_offset_ ;
            AccessType const *source = reinterpret_cast<AccessType const *>(source_byte_ptr);
            char const *source_byte_ptr1 = reinterpret_cast<char const *>(source_ptr) + byte_offset + byte_offset_ + sizeof(AccessType);
            AccessType const *source1 = reinterpret_cast<AccessType const *>(source_byte_ptr1);
            Element source_element[kElementsPerAccess];

            source_element[0] = *reinterpret_cast<Element const *>(source);
            source_element[1] = *reinterpret_cast<Element const *>(source1);

            fetch_ptr[access_idx] = *reinterpret_cast<AccessType const *>(source_element);
            const mctlass::complex<float>* ptr= fetch_ptr[access_idx].data();
          } 
          else {
            AccessType const *source_ptr = pointer_ +
               Policy::Delta::kStrided * s / kElementsPerAccess;
            char const *source_byte_ptr = reinterpret_cast<char const *>(source_ptr) + byte_offset + byte_offset_ + c * sizeof(AccessType);
            AccessType const *source = reinterpret_cast<AccessType const *>(source_byte_ptr);
            char const *source_byte_ptr1 = source_byte_ptr + sizeof(AccessType);
            AccessType const *source1 = reinterpret_cast<AccessType const *>(source_byte_ptr1);
            Element source_element[kElementsPerAccess];
            source_element[0] = *reinterpret_cast<Element const *>(source);
            if (__lane_id() >= 32) {
              source_element[1] = *(reinterpret_cast<Element const *>(source) - 1);
            }  
            else {
              source_element[1] = *(reinterpret_cast<Element const *>(source1) - 1);
            }
            fetch_ptr[access_idx] = *reinterpret_cast<AccessType const *>(source_element);
        }
      }
    }

   /* Element *exchange_ptr = reinterpret_cast<Element *>(&frag);

    // exchange on 64b granularity only for fragments held in k=8/2 to k=8 
    MCTLASS_PRAGMA_UNROLL
    for (int i = Fragment::kElements/2; i < Fragment::kElements; i += 2) {
      Element tmp = exchange_ptr[i];
      exchange_ptr[i] = exchange_ptr[i + 1];
      exchange_ptr[i + 1] = tmp;
    }
   */
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
    k_group_idx_ = k_group;
  }
};

} // namespace warp
} // namespace gemm
} // namespace mctlass

/////////////////////////////////////////////////////////////////////////////////////////////////
