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
    \brief 
*/

#pragma once

#include "mctlass/array.h"
#include "mctlass/layout/matrix.h"
#include "mctlass/layout/pitch_linear.h"

#include "mctlass/arch/memory_sm75.h"
#include "mctlass/epilogue/warp/tensor_op_policy.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

// This is an optimization available on CUDA 11.2 and beyond that eliminates branches in the epilogue.
//#define MCTLASS_EPILOGUE_WARP_TILE_ITERATOR_TENSOR_OP_MIXED_OPTIMIZATION_ENABLED ((__CUDACC_VER_MAJOR__ * 10 + __CUDACC_VER_MINOR__) >= 112)
#define MCTLASS_EPILOGUE_WARP_TILE_ITERATOR_TENSOR_OP_MIXED_OPTIMIZATION_ENABLED 0

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace mctlass {
namespace epilogue {
namespace warp {

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Template for reading and writing tiles of accumulators to shared memory. This is optimized
/// for mixed-precision epilogues in which the accumulators are 32b in width, but the output
/// data type is smaller. 
template <
  typename WarpShape_,            ///< shape of warp-level GEMM (concept: GemmShape)
  typename OperatorShape_,        ///< matrix multiply operation shape (concept: gemm::GemmShape)
  typename Element_,              ///< data type of accumulator element
  int ElementSizeBits,            ///< Size of accumulator element in bits
  int OutputSizeBits,             ///< Size of output element in bits
  int OutputElementCount,         ///< number of elements in output vector
  int ContiguousLanes             ///< Number of consecutive lanes writing to contiguous memory
>
class TileIteratorTensorOpMixed {
public:

  using WarpShape = WarpShape_;
  using OperatorShape = OperatorShape_;
  using Element = Element_;
  using Layout = layout::RowMajor;
  static int const kOutputElementCount = OutputElementCount;

  using TensorRef = TensorRef<Element, Layout>;         ///< Tensor Reference object
  using TensorCoord = MatrixCoord;                      ///< Logical coordinate in referenced tensor
  using Index = typename TensorRef::Index;
  using LongIndex = typename TensorRef::LongIndex;

  using Policy = TensorOpPolicy<WarpShape, OperatorShape, Layout>;

  /// Shape of the tile in memory
  using Shape = MatrixShape<
    Policy::kRowsPerIteration,
    WarpShape::kN
  >;

  /// This is the fragment size produced by one access of the iterator.
  using Fragment = Array<
    Element, 
    Policy::OperatorCount::kColumn * Policy::kElementsPerAccess>;

  /// This is the complete warp-level accumulator tile.
  //using AccumulatorTile = typename Operator::FragmentC;

  /// Number of times this iterator can be incremented
  static int const kIterations = Policy::kIterations;

  // Internal constants
  struct Detail {
    static int const kLanesInQuad = 4;

    /// Number of pointers needed to write accumulators
    static int const kPointerCount = 
      (OutputElementCount * sizeof_bits<Element>::value) / (const_min(128, OutputElementCount * sizeof_bits<Element>::value));

    static_assert(kPointerCount <= 4, "Can only accommodate four pointers at present.");
    static_assert(sizeof(Element) == 4, "This can only be used with 32b accumulator data types (f32, s32).");
  };

  /// Padding quantity
  using Padding = MatrixShape<
    0,
    Detail::kLanesInQuad * Policy::kElementsPerAccess>;

private:

  /// Storage type for accessing memory
  using AccessType = AlignedArray<Element, Policy::kElementsPerAccess>;

  //
  // Data members
  //

  /// Internal pointer to memory
  AccessType *pointers_[Detail::kPointerCount];

  /// Stride in units of AccessType
  int stride_;

  /// Logical column in which warp tile is aligned
  int warp_column_;

public:

  /// Default constructor
  MCTLASS_HOST_DEVICE
  TileIteratorTensorOpMixed() {
    MCTLASS_PRAGMA_UNROLL
    for (int64_t i = 0; i < Detail::kPointerCount; ++i) {
      pointers_[i] = nullptr;
    }
  }

  /// Constructor from TensorRef
  MCTLASS_HOST_DEVICE
  TileIteratorTensorOpMixed(
    TensorRef const &ref,
    unsigned lane_id
  ):
    stride_(ref.stride()[0] / Policy::kElementsPerAccess),
    warp_column_(0) { 

    int quad_id = (lane_id / Detail::kLanesInQuad); 
    int lane_in_quad = (lane_id % Detail::kLanesInQuad);

    if (OperatorShape::kM == 16 && OperatorShape::kN == 8 && OperatorShape::kK == 16 && platform::is_same<Element, float>::value == true) {
      quad_id = ((lane_id >> 4) << 2) + ((lane_id >> 5) << 3);
      lane_in_quad = lane_id & 0x7;
      if (WarpShape::kM == 64 && WarpShape::kN == 64 && WarpShape::kK == 32) { //Only for Epilgoue_with_broadcast
        quad_id = (lane_id / Detail::kLanesInQuad); 
        lane_in_quad = (lane_id % Detail::kLanesInQuad);
      }
    }
    else if (OperatorShape::kM == 16 && OperatorShape::kN == 8 && OperatorShape::kK ==8 && platform::is_same<Element, float>::value == true) {
      quad_id = ((lane_id >> 5) + (((lane_id >> 4) & 0x1) << 1)) << 1;
      lane_in_quad = lane_id & 0x7;
    }
    MCTLASS_PRAGMA_UNROLL
    for (int64_t i = 0; i < Detail::kPointerCount; ++i) {
      AccessType *ptr = reinterpret_cast<AccessType *>(ref.data()) + quad_id * stride_;
      int column_idx = (lane_in_quad % 2) + (((lane_in_quad / 2) + i) % Detail::kPointerCount) * 2;

      ptr += column_idx;

      if ((OperatorShape::kM == 16 && OperatorShape::kN == 8 && OperatorShape::kK == 16 && platform::is_same<Element, float>::value == true) ||
          (OperatorShape::kM == 16 && OperatorShape::kN == 8 && OperatorShape::kK == 8 && platform::is_same<Element, float>::value == true)) {
        int column_idx = (lane_in_quad ^ (4 * (i % Detail::kPointerCount)));
        Element *element_ptr = ref.data() + quad_id * stride_ * Policy::kElementsPerAccess + column_idx;
        ptr = reinterpret_cast<AccessType *>(element_ptr);
      }

      if (OperatorShape::kM == 16 && OperatorShape::kN == 8 && OperatorShape::kK == 16 && platform::is_same<Element, float>::value == true) {
        if (WarpShape::kM == 64 && WarpShape::kN == 64 && WarpShape::kK == 32) {//Only for Epilgoue_with_broadcast
          ptr = reinterpret_cast<AccessType *>(ref.data()) + quad_id * stride_;
          int column_idx = (lane_in_quad % 2) + (((lane_in_quad / 2) + i) % Detail::kPointerCount) * 2;
          ptr += column_idx;
        }
      }

      if (i == 0) {
        pointers_[0 % Detail::kPointerCount] = ptr;
      }
      else if (i == 1) {
        pointers_[1 % Detail::kPointerCount] = ptr;
      }
      else if (i == 2) {
        pointers_[2 % Detail::kPointerCount] = ptr;
      }
      else if (i == 3) {
        pointers_[3 % Detail::kPointerCount] = ptr;
      }
    }
  }

  /// Adds a pointer offset
  MCTLASS_HOST_DEVICE
  TileIteratorTensorOpMixed & add_pointer_offset(Index pointer_offset) {
    
    int offsetAccess = Policy::kElementsPerAccess;
    if (OperatorShape::kM == 16 && OperatorShape::kN == 8 && OperatorShape::kK == 16 && platform::is_same<Element, float>::value == true) {
      offsetAccess = 16;
      if (WarpShape::kM == 64 && WarpShape::kN == 64 && WarpShape::kK == 32) {
        offsetAccess = Policy::kElementsPerAccess;
      }
    }
    MCTLASS_PRAGMA_UNROLL
    for (int64_t i = 0; i < Detail::kPointerCount; ++i) {
        pointers_[i] += pointer_offset / offsetAccess;
    }

    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of the tensor
  MCTLASS_HOST_DEVICE
  TileIteratorTensorOpMixed & add_tile_offset(TensorCoord const &tile_offset) {
    
    MCTLASS_PRAGMA_UNROLL
    for (int64_t i = 0; i < Detail::kPointerCount; ++i) {
      pointers_[i] += tile_offset.row() * Shape::kRow * stride_ + 
        tile_offset.column() * Shape::kColumn / Policy::kElementsPerAccess;
    }

    warp_column_ += tile_offset.column() * Shape::kColumn;

    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of the tensor
  MCTLASS_HOST_DEVICE
  TileIteratorTensorOpMixed & operator+=(TensorCoord const &tile_offset) {
    return add_tile_offset(tile_offset);
  }

  /// Store
  MCTLASS_DEVICE
  void store_with_pointer_offset(Fragment const &frag, Index pointer_offset) {

    AccessType const *frag_ptr = reinterpret_cast<AccessType const *>(&frag);

    AccessType *ptr = pointers_[0];

#if MCTLASS_EPILOGUE_WARP_TILE_ITERATOR_TENSOR_OP_MIXED_OPTIMIZATION_ENABLED

    // When the optimization is enabled, small tiles require separate logic.
    bool kN32_optimization = (WarpShape::kN * Detail::kLanesInQuad * Policy::kElementsPerAccess * sizeof_bits<Element>::value) % 1024 == 0;
    if (kN32_optimization) {
      int ptr_idx = ((warp_column_ * sizeof_bits<Element>::value) / 1024) % Detail::kPointerCount;
      if (ptr_idx == 0) {
        ptr = pointers_[0];
      } else if (ptr_idx == 1) {
        ptr = pointers_[1];
      } else if (ptr_idx == 2) {
        ptr = pointers_[2];
      } else if (ptr_idx == 3) {
        ptr = pointers_[3];
      }
    }

#endif

    MCTLASS_PRAGMA_UNROLL
    for (int64_t n = 0; n < Policy::OperatorCount::kColumn; ++n) {
      
#if MCTLASS_EPILOGUE_WARP_TILE_ITERATOR_TENSOR_OP_MIXED_OPTIMIZATION_ENABLED

      //
      // When the optimization is enabled, this expression suffices to obtain the SMEM pointer.
      //
      if (WarpShape::kN == 64) {
        ptr = pointers_[n / 4];
      }
      else if (!kN32_optimization)
#endif
      {
        // This is the reference implementation
        int column_idx = warp_column_ + n * Detail::kLanesInQuad * Policy::kElementsPerAccess;
        int ptr_idx = ((column_idx * sizeof_bits<Element>::value) / 1024) % Detail::kPointerCount;
  
        if (ptr_idx == 0) {
          ptr = pointers_[0 % Detail::kPointerCount];
        }
        else if (ptr_idx == 1) {
          ptr = pointers_[1 % Detail::kPointerCount];
        }
        else if (ptr_idx == 2) {
          ptr = pointers_[2 % Detail::kPointerCount];
        }
        else if (ptr_idx == 3) {
          ptr = pointers_[3 % Detail::kPointerCount];
        }
      }

      int offset = n * Detail::kLanesInQuad + pointer_offset / Policy::kElementsPerAccess;
      if (OperatorShape::kM == 16 && OperatorShape::kN == 8 && OperatorShape::kK == 8 && platform::is_same<Element, float>::value == true) {
        ptr[offset][0] = frag_ptr[n][0];
        ptr[offset + stride_][0] = frag_ptr[n][1];
      }
      else if (OperatorShape::kM == 16 && OperatorShape::kN == 8 && OperatorShape::kK == 16 && platform::is_same<Element, float>::value == true) {
        if (WarpShape::kM == 64 && WarpShape::kN == 64 && WarpShape::kK == 32) {
          ptr[offset] = frag_ptr[n];
        }
        else {
          ptr[offset][0] = frag_ptr[n][0];
          ptr[offset + stride_][0] = frag_ptr[n][1];
        }
      }
      else {
        ptr[offset] = frag_ptr[n];
      }
    }
  }

  /// Store
  MCTLASS_HOST_DEVICE
  void store(Fragment const &frag) {
    if (OperatorShape::kM == 16 && OperatorShape::kN == 8 && OperatorShape::kK == 16 && platform::is_same<Element, float>::value == true) {
        if (WarpShape::kM == 64 && WarpShape::kN == 64 && WarpShape::kK == 32) {
          if(__lane_id()>=32)return;
        }
    }
    store_with_pointer_offset(frag, 0);
  }

  /// Load
  MCTLASS_HOST_DEVICE
  void load_with_pointer_offset(Fragment &frag, Index pointer_offset) const {

    AccessType *frag_ptr = reinterpret_cast<AccessType *>(&frag);

    MCTLASS_PRAGMA_UNROLL
    for (int64_t n = 0; n < Policy::OperatorCount::kColumn; ++n) {

      int column_idx = warp_column_ + n * Detail::kLanesInQuad * Policy::kElementsPerAccess;
      int ptr_idx = ((column_idx * sizeof_bits<Element>::value) / 1024) % Detail::kPointerCount;

      AccessType const *smem_ptr = pointers_[ptr_idx];
      frag_ptr[n] = smem_ptr[n * Detail::kLanesInQuad + pointer_offset / Policy::kElementsPerAccess];
    }
  }

  /// Load
  MCTLASS_HOST_DEVICE
  void load(Fragment &frag) const {
    load_with_pointer_offset(frag, 0);
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Partial specialization for int32_t x 16 => int8_t/int4b_t x 16
template <
  typename WarpShape_,            ///< shape of warp-level GEMM (concept: GemmShape)
  typename OperatorShape_,        ///< matrix multiply operation shape (concept: gemm::GemmShape),
  int OutputSizeBits              ///< Size of output element in bits
>
class TileIteratorTensorOpMixed<WarpShape_, OperatorShape_, int32_t, 32, OutputSizeBits, 16, 8> {
public:

  using WarpShape = WarpShape_;
  using OperatorShape = OperatorShape_;
  using Element = int32_t;
  using Layout = layout::RowMajor;
  static int const kOutputElementCount = 16;

  using TensorRef = TensorRef<Element, Layout>;         ///< Tensor Reference object
  using TensorCoord = MatrixCoord;                      ///< Logical coordinate in referenced tensor
  using Index = typename TensorRef::Index;
  using LongIndex = typename TensorRef::LongIndex;

  using Policy = TensorOpPolicy<WarpShape, OperatorShape, Layout>;

  /// Shape of the tile in memory
  using Shape = MatrixShape<
    Policy::kRowsPerIteration,
    WarpShape::kN
  >;

  /// This is the fragment size produced by one access of the iterator.
  using Fragment = Array<
    Element, 
    Policy::OperatorCount::kColumn * Policy::kElementsPerAccess>;

  /// This is the complete warp-level accumulator tile.
  //using AccumulatorTile = typename Operator::FragmentC;

  /// Number of times this iterator can be incremented
  static int const kIterations = Policy::kIterations;

  // Internal constants
  struct Detail {
    static int const kLanesInQuad = 4;

    /// Number of pointers needed to write accumulators
    static int const kPointerCount = 2;

    /// Offsets added 
    static int const kOffsetCount = 4;

    static_assert(sizeof(Element) == 4, "This can only be used with 32b accumulator data types (f32, s32).");
  };

  /// Padding quantity
  using Padding = MatrixShape<0, Detail::kLanesInQuad * 2>;

private:

  /// Storage type for accessing memory
  using AccessType = AlignedArray<Element, 2>;

  //
  // Data members
  //

  /// Internal pointer to memory
  AccessType *pointers_[Detail::kPointerCount];

  /// Stride in units of AccessType
  int stride_;

  /// Uniform offset in bytes added to warp tile iterator
  int uniform_offset_[Detail::kOffsetCount];

public:

  /// Default constructor
  MCTLASS_HOST_DEVICE
  TileIteratorTensorOpMixed() {
    MCTLASS_PRAGMA_UNROLL
    for (int64_t i = 0; i < Detail::kPointerCount; ++i) {
      pointers_[i] = nullptr;
    }
  }

  /// Constructor from TensorRef
  MCTLASS_HOST_DEVICE
  TileIteratorTensorOpMixed(
    TensorRef const &ref,
    unsigned lane_id
  ):
    stride_(ref.stride()[0] / AccessType::kElements) { 

    int quad_id = (lane_id / Detail::kLanesInQuad); 
    int lane_in_quad = (lane_id % Detail::kLanesInQuad);

    if (OperatorShape::kM == 16 && OperatorShape::kN == 8 && OperatorShape::kK == 32) {
      quad_id = ((lane_id >> 4) << 2) - (lane_id >> 5) * 6;
      lane_in_quad = (lane_id & 0x7);
    }
    MCTLASS_PRAGMA_UNROLL
    for (int i = 0; i < Detail::kPointerCount; ++i) {
      AccessType *ptr = reinterpret_cast<AccessType *>(ref.data()) + quad_id * stride_;
      int column_idx = lane_in_quad ^ (i * 2);

      ptr += column_idx;

      if (OperatorShape::kM == 16 && OperatorShape::kN == 8 && OperatorShape::kK == 32) {
        column_idx = lane_in_quad ^ (i * 4);
        Element *element_ptr = ref.data() + quad_id * stride_ * Policy::kElementsPerAccess + column_idx;
        ptr = reinterpret_cast<AccessType *>(element_ptr);
      }

      if (i == 0) {
        pointers_[0] = ptr;
      }
      else if (i == 1) {
        pointers_[1] = ptr;
      }
    }

    MCTLASS_PRAGMA_UNROLL
    for (int i = 0; i < Detail::kOffsetCount; ++i) {
      uniform_offset_[i] = (i ^ 0) * 4 * sizeof(AccessType);
    }
  }

  /// Adds a pointer offset
  MCTLASS_HOST_DEVICE
  TileIteratorTensorOpMixed & add_pointer_offset(Index pointer_offset) {

    MCTLASS_PRAGMA_UNROLL
    for (int64_t i = 0; i < Detail::kPointerCount; ++i) {
      pointers_[i] += pointer_offset / AccessType::kElements;
    }

    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of the tensor
  MCTLASS_HOST_DEVICE
  TileIteratorTensorOpMixed & add_tile_offset(TensorCoord const &tile_offset) {
    
    int ptr_offset = tile_offset.row() * Shape::kRow * stride_ + 
      tile_offset.column() * Shape::kColumn / AccessType::kElements;

    pointers_[0] += ptr_offset;
    pointers_[1] += ptr_offset;
    
    MCTLASS_PRAGMA_UNROLL
    for (int i = 0; i < Detail::kOffsetCount; ++i) {
      uniform_offset_[i] = (i ^ tile_offset.column()) * 4 * sizeof(AccessType);
    }

    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of the tensor
  MCTLASS_HOST_DEVICE
  TileIteratorTensorOpMixed & operator+=(TensorCoord const &tile_offset) {
    return add_tile_offset(tile_offset);
  }

  /// Store
  MCTLASS_DEVICE
  void store_with_pointer_offset(Fragment const &frag, Index pointer_offset) {

    AccessType const *frag_ptr = reinterpret_cast<AccessType const *>(&frag);

    MCTLASS_PRAGMA_UNROLL
    for (int n = 0; n < Policy::OperatorCount::kColumn; ++n) {

      int ptr_idx = (n / 4);
      int offset_idx = (n % 4);

      AccessType *ptr;
      if (ptr_idx == 0) {
        ptr = pointers_[0];
      }
      else if (ptr_idx == 1) {
        ptr = pointers_[1];
      }

      int offset = (n / 4) * 16 + pointer_offset / AccessType::kElements;

#if 0
      //
      // Using inline PTX to avoid generic memory
      //
      uint32_t smem_addr = arch::mctlass_get_smem_pointer(ptr);
      uint32_t const *data = reinterpret_cast<uint32_t const *>(frag_ptr + n);
      uint32_t offset_in_bytes = offset * sizeof(AccessType) + uniform_offset_[offset_idx];

      asm volatile(
        "{ .reg .u32 smem_ptr; add.u32 smem_ptr, %0, %1; st.shared.v2.u32 [smem_ptr], {%2, %3}; }\n"
        : : "r"(smem_addr), "r"(offset_in_bytes), "r"(data[0]), "r"(data[1])
      );
#else
      AccessType *smem_ptr = pointers_[ptr_idx];
      //
      //When inline PTX is not used, offset needs uniform_offset_(not multipiled by sizeof(AccessType))
      //
      int offset_without_bytes = offset + uniform_offset_[offset_idx] / sizeof(AccessType);
      if (OperatorShape::kM == 16 && OperatorShape::kN == 8 && OperatorShape::kK == 32) {  
        smem_ptr[offset_without_bytes][0] = frag_ptr[n][0];
        smem_ptr[offset_without_bytes + stride_][0] = frag_ptr[n][1];
      }
      else {
        smem_ptr[offset_without_bytes] = frag_ptr[n];
      }

#endif
    }
  }

  /// Store
  MCTLASS_HOST_DEVICE
  void store(Fragment const &frag) {
    store_with_pointer_offset(frag, 0);
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Partial specialization for int32_t x 8 => int8_t/int4b_t x 8
template <
  typename WarpShape_,            ///< shape of warp-level GEMM (concept: GemmShape)
  typename OperatorShape_,        ///< matrix multiply operation shape (concept: gemm::GemmShape)
  int OutputSizeBits              ///< Size of output element in bits
>
class TileIteratorTensorOpMixed<WarpShape_, OperatorShape_, int32_t, 32, OutputSizeBits, 8, 8> {
public:

  using WarpShape = WarpShape_;
  using OperatorShape = OperatorShape_;
  using Element = int32_t;
  using Layout = layout::RowMajor;
  static int const kOutputElementCount = 8;

  using TensorRef = TensorRef<Element, Layout>;         ///< Tensor Reference object
  using TensorCoord = MatrixCoord;                      ///< Logical coordinate in referenced tensor
  using Index = typename TensorRef::Index;
  using LongIndex = typename TensorRef::LongIndex;

  using Policy = TensorOpPolicy<WarpShape, OperatorShape, Layout>;

  /// Shape of the tile in memory
  using Shape = MatrixShape<
    Policy::kRowsPerIteration,
    WarpShape::kN
  >;

  /// This is the fragment size produced by one access of the iterator.
  using Fragment = Array<
    Element, 
    Policy::OperatorCount::kColumn * Policy::kElementsPerAccess>;

  /// This is the complete warp-level accumulator tile.
  //using AccumulatorTile = typename Operator::FragmentC;

  /// Number of times this iterator can be incremented
  static int const kIterations = Policy::kIterations;

  // Internal constants
  struct Detail {
    static int const kLanesInQuad = 4;

    /// Number of pointers needed to write accumulators
    static int const kPointerCount = 2;

    static_assert(sizeof(Element) == 4, "This can only be used with 32b accumulator data types (f32, s32).");
  };

  /// Padding quantity
  using Padding = MatrixShape<0, Detail::kLanesInQuad * 2>;

private:

  /// Storage type for accessing memory
  using AccessType = AlignedArray<Element, 2>;

  //
  // Data members
  //

  /// Internal pointer to memory
  AccessType *pointers_[Detail::kPointerCount];

  /// Stride in units of AccessType
  int stride_;

public:

  /// Default constructor
  MCTLASS_HOST_DEVICE
  TileIteratorTensorOpMixed() {
    MCTLASS_PRAGMA_UNROLL
    for (int64_t i = 0; i < Detail::kPointerCount; ++i) {
      pointers_[i] = nullptr;
    }
  }

  /// Constructor from TensorRef
  MCTLASS_HOST_DEVICE
  TileIteratorTensorOpMixed(
    TensorRef const &ref,
    unsigned lane_id
  ):
    stride_(ref.stride()[0] / AccessType::kElements) { 

    int quad_id = (lane_id / Detail::kLanesInQuad); 
    int lane_in_quad = (lane_id % Detail::kLanesInQuad);

    if (OperatorShape::kM == 16 && OperatorShape::kN == 8 && OperatorShape::kK == 32) {
      quad_id = ((lane_id >> 4) << 2) - (lane_id >> 5) * 6;
      lane_in_quad = (lane_id & 0x7);
    }

    MCTLASS_PRAGMA_UNROLL
    for (int i = 0; i < Detail::kPointerCount; ++i) {
      AccessType *ptr = reinterpret_cast<AccessType *>(ref.data()) + quad_id * stride_;
      int column_idx = lane_in_quad ^ (i * 2);

      ptr += column_idx;
    
      if (OperatorShape::kM == 16 && OperatorShape::kN == 8 && OperatorShape::kK == 32) {
        column_idx = lane_in_quad ^ (i * 4);
        Element *element_ptr = ref.data() + quad_id * stride_ * Policy::kElementsPerAccess + column_idx;
        ptr = reinterpret_cast<AccessType *>(element_ptr);
      }
      if (i == 0) {
        pointers_[0] = ptr;
      }
      else if (i == 1) {
        pointers_[1] = ptr;
      }
    }
  }

  /// Adds a pointer offset
  MCTLASS_HOST_DEVICE
  TileIteratorTensorOpMixed & add_pointer_offset(Index pointer_offset) {

    MCTLASS_PRAGMA_UNROLL
    for (int64_t i = 0; i < Detail::kPointerCount; ++i) {
      pointers_[i] += pointer_offset / AccessType::kElements;
    }

    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of the tensor
  MCTLASS_HOST_DEVICE
  TileIteratorTensorOpMixed & add_tile_offset(TensorCoord const &tile_offset) {
    
    int ptr_offset = tile_offset.row() * Shape::kRow * stride_ + 
      tile_offset.column() * Shape::kColumn / AccessType::kElements;

    pointers_[0] += ptr_offset;
    pointers_[1] += ptr_offset;
   
    if (tile_offset.column() % 2) {
      auto tmp = pointers_[0];
      pointers_[0] = pointers_[1];
      pointers_[1] = tmp;
    }
 
    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of the tensor
  MCTLASS_HOST_DEVICE
  TileIteratorTensorOpMixed & operator+=(TensorCoord const &tile_offset) {
    return add_tile_offset(tile_offset);
  }

  /// Store
  MCTLASS_DEVICE
  void store_with_pointer_offset(Fragment const &frag, Index pointer_offset) {

    AccessType const *frag_ptr = reinterpret_cast<AccessType const *>(&frag);

    MCTLASS_PRAGMA_UNROLL
    for (int n = 0; n < Policy::OperatorCount::kColumn; ++n) {

      int ptr_idx = (n / 4);

      AccessType *ptr;
      if (ptr_idx == 0) {
        ptr = pointers_[0];
      }
      else if (ptr_idx == 1) {
        ptr = pointers_[1];
      }

      int offset = (n / 4) * 16 + pointer_offset / AccessType::kElements + (n % 4) * 4;

#if 0
      //
      // Using inline PTX to avoid generic memory
      //
      uint32_t smem_addr = arch::mctlass_get_smem_pointer(ptr);
      uint32_t const *data = reinterpret_cast<uint32_t const *>(frag_ptr + n);
      uint32_t offset_in_bytes = offset * sizeof(AccessType);
      asm volatile(
        "{ .reg .u32 smem_ptr; add.u32 smem_ptr, %0, %1; st.shared.v2.u32 [smem_ptr], {%2, %3}; }\n"
        : : "r"(smem_addr), "r"(offset_in_bytes), "r"(data[0]), "r"(data[1])
      );
#else
      AccessType *smem_ptr = pointers_[ptr_idx];
      if (OperatorShape::kM == 16 && OperatorShape::kN == 8 && OperatorShape::kK == 32) {  
        smem_ptr[offset][0] = frag_ptr[n][0];
        smem_ptr[offset + stride_][0] = frag_ptr[n][1];
      }
      else {
        smem_ptr[offset] = frag_ptr[n];
      }
#endif      
    }
  }

  /// Store
  MCTLASS_HOST_DEVICE
  void store(Fragment const &frag) {
    store_with_pointer_offset(frag, 0);
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace warp
} // namespace epilogue
} // namespace mctlass

/////////////////////////////////////////////////////////////////////////////////////////////////

#undef MCTLASS_EPILOGUE_WARP_TILE_ITERATOR_TENSOR_OP_MIXED_OPTIMIZATION_ENABLED

/////////////////////////////////////////////////////////////////////////////////////////////////
