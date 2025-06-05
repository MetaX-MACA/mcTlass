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
class MmaTensorOpMultiplicandTilePrecomputeIterator;



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
class MmaTensorOpMultiplicandTilePrecomputeIterator<
    Shape_, Operand_, Element_,
    mctlass::layout::MacaTensorOpMultiplicandCrosswise<sizeof_bits<Element_>::value, Crosswise>,
    InstructionShape_, OpDelta_, 64, PartitionsK_, PartitionsM_> {
 public:
  /// Shape of tile to load (concept: PitchLinearShape)
  using Shape = Shape_;

  /// Operand tag
  static Operand const kOperand = Operand_;

  static_assert(kOperand == Operand::kA || kOperand == Operand::kB,
                "MmaTensorOpMultiplicandTilePrecomputeIterator may only be instantiated for "
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

  AccessType const *pre_pointers_[4];
  Index pre_byte_offsets_[4];

  /// Byte offset incremented as iterator advances
  Index byte_offset_;

  /// Internal counter used to determine when to increment byte offset and when
  /// to XOR it
  int k_group_idx_;

 public:
  /// Default ctor constructs null iterator
  MCTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTilePrecomputeIterator()
      : pointer_(nullptr),
        sections_(0),
        stride_(0),
        byte_offset_(0),
        k_group_idx_(0) {}

  /// Constructor from TensorRef
  MCTLASS_DEVICE
  MmaTensorOpMultiplicandTilePrecomputeIterator(TensorRef const &ref, int lane_id)
      : pointer_(reinterpret_cast<AccessType const *>(ref.data())),
        sections_(ref.stride(0) / kCrosswise),
        // stride_ = kCrosswise x sections_ x kFactor
        stride_(ref.stride(0) * Layout::kFactor / Layout::kElementsPerAccess),
        byte_offset_(0),
        k_group_idx_(0) {

    Index row = (((lane_id >> 4) << 2) ^ (InstructionShape::kContiguous - 1)) - 3;
    Index col = lane_id & (InstructionShape::kStrided - 1);

    pointer_ = reinterpret_cast<AccessType const *>(ref.data());
    byte_offset_ = ref.layout()(TensorCoord({row, col})) * (sizeof_bits<Element>::value / 8);

  }

  MCTLASS_DEVICE
  void pre_compute_pointer_row_major(int warp_idx_m = 0, int tile_k = 0){

    auto org_pointer = pointer_;
    auto org_byte_offset = byte_offset_;

    const int warp_id = threadIdx.x / 64;
    pre_pointers_[0] = pointer_ + (warp_id % 4) * 256; // 256x128x32 -> 64x64x32
    pre_pointers_[1] = pointer_ + (warp_id % 4) * 256;

    add_tile_offset({0, PartitionsM});
    pre_pointers_[2] = pointer_ + (warp_id % 4) * 256;
    pre_pointers_[3] = pointer_ + (warp_id % 4) * 256;

    byte_offset_ = org_byte_offset;

    pre_byte_offsets_[0] = byte_offset_;
    this->operator++();
    pre_byte_offsets_[1] = byte_offset_;

    this->operator++();
    pre_byte_offsets_[2] = byte_offset_;
    this->operator++();
    pre_byte_offsets_[3] = byte_offset_;

    k_group_idx_ = 0;
    byte_offset_ = org_byte_offset;
    pointer_ = org_pointer;

  }

  MCTLASS_DEVICE
  void pre_compute_pointer_col_major(int warp_idx_m = 0, int tile_k = 0){

    auto org_pointer = pointer_;
    auto org_byte_offset = byte_offset_;

    const int warp_id = threadIdx.x / 64;
    pre_pointers_[0] = pointer_ + (warp_id / 4) * 256; // 256x128x32 -> 64x64x32
    pre_pointers_[1] = pointer_ + (warp_id / 4) * 256;

    add_tile_offset({0, PartitionsM});
    pre_pointers_[2] = pointer_ + (warp_id / 4) * 256;
    pre_pointers_[3] = pointer_ + (warp_id / 4) * 256;

    byte_offset_ = org_byte_offset;

    pre_byte_offsets_[0] = byte_offset_;
    this->operator++();
    pre_byte_offsets_[1] = byte_offset_;

    this->operator++();
    pre_byte_offsets_[2] = byte_offset_;
    this->operator++();
    pre_byte_offsets_[3] = byte_offset_;

    k_group_idx_ = 0;
    byte_offset_ = org_byte_offset;
    pointer_ = org_pointer;
  }

  /// Adds a pointer offset to internal pointer(s) to advance through memory
  MCTLASS_DEVICE
  MmaTensorOpMultiplicandTilePrecomputeIterator &add_pointer_offset(LongIndex offset) {
    byte_offset_ += offset * sizeof_bits<Element>::value / 8;

    return *this;
  }

  /// Advances an iterator along logical dimensions of matrix in units of whole
  /// tiles
  MCTLASS_DEVICE
  MmaTensorOpMultiplicandTilePrecomputeIterator &add_tile_offset(
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
  MmaTensorOpMultiplicandTilePrecomputeIterator &add_tile_offset_negative(
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
  MmaTensorOpMultiplicandTilePrecomputeIterator &operator++() {

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
  MmaTensorOpMultiplicandTilePrecomputeIterator &operator--() { assert(0); }

  ///< advances in units of whole tiles along the logical coordinate space of
  ///< the tensor
  MCTLASS_DEVICE
  MmaTensorOpMultiplicandTilePrecomputeIterator &operator+=(
      TensorCoord const &tile_offset) {
    add_tile_offset(tile_offset);
    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of
  ///< the tensor
  MCTLASS_DEVICE
  MmaTensorOpMultiplicandTilePrecomputeIterator &operator-=(
      TensorCoord const &tile_offset) {
    add_tile_offset(-tile_offset);
    return *this;
  }

  /// Loads a fragment from memory at the location pointed to by the iterator.
  MCTLASS_HOST_DEVICE
  void load(Fragment &frag)  {
    load_with_byte_offset(frag, 0);
    }

    MCTLASS_HOST_DEVICE
    void load_with_index_A(Fragment &frag,int index)  {

        using LoadType = Array<float, Policy::LdsmShape::kCount / 2>;
        LoadType *fetch_ptr = reinterpret_cast<LoadType *>(&frag);

        char const *source_byte_ptr = reinterpret_cast<char const *>(pre_pointers_[index]) + pre_byte_offsets_[index];
        auto tmp = Policy::kLdsmOpInner / Layout::kFactor * Policy::LdsmShape::kStrided  * stride_ * sizeof_bits<Element_>::value;

        MCTLASS_PRAGMA_UNROLL
        for (int s = 0; s < Policy::LdsmIterations::kStrided; ++s) {
          fetch_ptr[s] = *reinterpret_cast<LoadType const *>(source_byte_ptr + tmp * s);
        }

    }

    MCTLASS_HOST_DEVICE
    void load_with_index_B(Fragment &frag,int index)  {

        using LoadType = Array<float, Policy::LdsmShape::kCount / 2>;
        LoadType *fetch_ptr = reinterpret_cast<LoadType *>(&frag);

        char const *source_byte_ptr = reinterpret_cast<char const *>(pre_pointers_[index]) + pre_byte_offsets_[index];
        auto tmp = Policy::kLdsmOpInner / Layout::kFactor * Policy::LdsmShape::kStrided  * stride_ * sizeof_bits<Element_>::value;

        MCTLASS_PRAGMA_UNROLL
        for (int s = 0; s < Policy::LdsmIterations::kStrided; ++s) {
          fetch_ptr[s] = *reinterpret_cast<LoadType const *>(source_byte_ptr + tmp * s);
        }
    }

  /// Loads a fragment from memory with additional logical offset
  MCTLASS_DEVICE
  void load_with_byte_offset(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a linear offset in units of bytes
      Index byte_offset) const {

    using LoadType = Array<float, Policy::LdsmShape::kCount / 2>;
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
class MmaTensorOpMultiplicandTilePrecomputeIterator<
    Shape_, Operand_, Element_,
    mctlass::layout::MacaTensorOpMultiplicandCrosswise<32, Crosswise>,
    InstructionShape_, OpDelta_, 64, PartitionsK_, PartitionsM_> {
 public:
  /// Shape of tile to load (concept: PitchLinearShape)
  using Shape = Shape_;

  /// Operand tag
  static Operand const kOperand = Operand_;

  static_assert(kOperand == Operand::kA || kOperand == Operand::kB,
                "MmaTensorOpMultiplicandTilePrecomputeIterator may only be instantiated for "
                "A or B operands to warp-level Mma.");

  static_assert(sizeof(Element_) == 4,
                "MacaTensorOpMultiplicandCrosswise only support 32bit now");

  /// Element type
  using Element = Element_;

  /// Element number when the layout crosses
  static int const kCrosswise = Crosswise;

  /// Layout of source tile
  using Layout = mctlass::layout::MacaTensorOpMultiplicandCrosswise<32, Crosswise>;

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

  AccessType const *pre_pointers_[4];
  Index pre_byte_offsets_[4];

  /// Byte offset incremented as iterator advances
  Index byte_offset_;

  /// Internal counter used to determine when to increment byte offset and when
  /// to XOR it
  int k_group_idx_;

 public:
  /// Default ctor constructs null iterator
  MCTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTilePrecomputeIterator()
      : pointer_(nullptr),
        sections_(0),
        stride_(0),
        byte_offset_(0),
        k_group_idx_(0) {}

  /// Constructor from TensorRef
  MCTLASS_DEVICE
  MmaTensorOpMultiplicandTilePrecomputeIterator(TensorRef const &ref, int lane_id)
      : pointer_(reinterpret_cast<AccessType const *>(ref.data())),
        sections_(ref.stride(0) / kCrosswise),
        // stride_ = kCrosswise x sections_ x kFactor
        stride_(ref.stride(0) * Layout::kFactor / Layout::kElementsPerAccess),
        byte_offset_(0),
        k_group_idx_(0) {

    Index row = (3 - (lane_id >> 4)) << 1;
    Index col = lane_id & (InstructionShape::kStrided - 1);

    pointer_ = reinterpret_cast<AccessType const *>(ref.data());
    byte_offset_ = ref.layout()(TensorCoord({row, col})) * (sizeof_bits<Element>::value / 8);
  }

  MCTLASS_DEVICE
  void pre_compute_pointer_row_major(int warp_idx_m = 0, int tile_k = 0){

    auto org_pointer = pointer_;
    auto org_byte_offset = byte_offset_;

    const int warp_id = threadIdx.x / 64;
    pre_pointers_[0] = pointer_ + (warp_id % 4) * 256; // 256x128x32 -> 64x64x32
    pre_pointers_[1] = pointer_ + (warp_id % 4) * 256;

    add_tile_offset({0, PartitionsM});
    pre_pointers_[2] = pointer_ + (warp_id % 4) * 256;
    pre_pointers_[3] = pointer_ + (warp_id % 4) * 256;

    byte_offset_ = org_byte_offset;

    pre_byte_offsets_[0] = byte_offset_;
    this->operator++();
    pre_byte_offsets_[1] = byte_offset_;

    this->operator++();
    pre_byte_offsets_[2] = byte_offset_;
    this->operator++();
    pre_byte_offsets_[3] = byte_offset_;

    k_group_idx_ = 0;
    byte_offset_ = org_byte_offset;
    pointer_ = org_pointer;

  }

  MCTLASS_DEVICE
  void pre_compute_pointer_col_major(int warp_idx_n = 0, int tile_k = 0){

    auto org_pointer = pointer_;
    auto org_byte_offset = byte_offset_;

    const int warp_id = threadIdx.x / 64;
    pre_pointers_[0] = pointer_ + (warp_id / 4) * 256; // 256x128x32 -> 64x64x32
    pre_pointers_[1] = pointer_ + (warp_id / 4) * 256;

    add_tile_offset({0, PartitionsM});
    pre_pointers_[2] = pointer_ + (warp_id / 4) * 256;
    pre_pointers_[3] = pointer_ + (warp_id / 4) * 256;

    byte_offset_ = org_byte_offset;

    pre_byte_offsets_[0] = byte_offset_;
    this->operator++();
    pre_byte_offsets_[1] = byte_offset_;

    this->operator++();
    pre_byte_offsets_[2] = byte_offset_;
    this->operator++();
    pre_byte_offsets_[3] = byte_offset_;

    k_group_idx_ = 0;
    byte_offset_ = org_byte_offset;
    pointer_ = org_pointer;
  }

  /// Adds a pointer offset to internal pointer(s) to advance through memory
  MCTLASS_DEVICE
  MmaTensorOpMultiplicandTilePrecomputeIterator &add_pointer_offset(LongIndex offset) {
    byte_offset_ += offset * sizeof_bits<Element>::value / 8;

    return *this;
  }

  /// Advances an iterator along logical dimensions of matrix in units of whole
  /// tiles
  MCTLASS_DEVICE
  MmaTensorOpMultiplicandTilePrecomputeIterator &add_tile_offset(
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
  MmaTensorOpMultiplicandTilePrecomputeIterator &add_tile_offset_negative(
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
  MmaTensorOpMultiplicandTilePrecomputeIterator &operator++() {

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
  MmaTensorOpMultiplicandTilePrecomputeIterator &operator--() { assert(0); }

  ///< advances in units of whole tiles along the logical coordinate space of
  ///< the tensor
  MCTLASS_DEVICE
  MmaTensorOpMultiplicandTilePrecomputeIterator &operator+=(
      TensorCoord const &tile_offset) {
    add_tile_offset(tile_offset);
    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of
  ///< the tensor
  MCTLASS_DEVICE
  MmaTensorOpMultiplicandTilePrecomputeIterator &operator-=(
      TensorCoord const &tile_offset) {
    add_tile_offset(-tile_offset);
    return *this;
  }

  /// Loads a fragment from memory at the location pointed to by the iterator.
  MCTLASS_HOST_DEVICE
  void load(Fragment &frag)  {
    load_with_byte_offset(frag, 0);
  }

  MCTLASS_HOST_DEVICE
  void load_with_index_A(Fragment &frag,int index)  {
    using LoadType = Array<float, Policy::LdsmShape::kCount / 2>;
    LoadType *fetch_ptr = reinterpret_cast<LoadType *>(&frag);

    char const *source_byte_ptr = reinterpret_cast<char const *>(pre_pointers_[index]) + pre_byte_offsets_[index];
    auto tmp = Policy::kLdsmOpInner / Layout::kFactor * Policy::LdsmShape::kStrided  * stride_ * sizeof_bits<float>::value;

    MCTLASS_PRAGMA_UNROLL
    for (int s = 0; s < Policy::LdsmIterations::kStrided; ++s) {
      fetch_ptr[s] = *reinterpret_cast<LoadType const *>(source_byte_ptr + tmp * s);
    }
  }

  MCTLASS_HOST_DEVICE
  void load_with_index_B(Fragment &frag,int index)  {
    using LoadType = Array<float, Policy::LdsmShape::kCount / 2>;
    LoadType *fetch_ptr = reinterpret_cast<LoadType *>(&frag);

    char const *source_byte_ptr = reinterpret_cast<char const *>(pre_pointers_[index]) + pre_byte_offsets_[index];
    auto tmp = Policy::kLdsmOpInner / Layout::kFactor * Policy::LdsmShape::kStrided  * stride_ * sizeof_bits<float>::value;

    MCTLASS_PRAGMA_UNROLL
    for (int s = 0; s < Policy::LdsmIterations::kStrided; ++s) {
      fetch_ptr[s] = *reinterpret_cast<LoadType const *>(source_byte_ptr + tmp * s);
    }
  }

  /// Loads a fragment from memory with additional logical offset
  MCTLASS_DEVICE
  void load_with_byte_offset(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a linear offset in units of bytes
      Index byte_offset) const {

    using LoadType = Array<float, Policy::LdsmShape::kCount / 2>;
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
class MmaTensorOpMultiplicandTilePrecomputeIterator<
    Shape_, Operand_, Element_,
    mctlass::layout::MacaTensorOpMultiplicandConflictFreeCrosswise<16, Crosswise>,
    InstructionShape_, OpDelta_, 64, PartitionsK_, PartitionsM_> {
 public:
  /// Shape of tile to load (concept: PitchLinearShape)
  using Shape = Shape_;

  /// Operand tag
  static Operand const kOperand = Operand_;

  static_assert(kOperand == Operand::kA || kOperand == Operand::kB,
                "MmaTensorOpMultiplicandTilePrecomputeIterator may only be instantiated for "
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

  AccessType const *pre_pointers_[4];
  Index pre_byte_offsets_[4];

  /// Byte offset incremented as iterator advances
  Index byte_offset_;

  Index register_offset_ = 0;

  /// Internal counter used to determine when to increment byte offset and when
  /// to XOR it
  int k_group_idx_;

 public:
  /// Default ctor constructs null iterator
  MCTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTilePrecomputeIterator()
      : pointer_(nullptr),
        sections_(0),
        stride_(0),
        byte_offset_(0),
        k_group_idx_(0) {}

  /// Constructor from TensorRef
  MCTLASS_DEVICE
  MmaTensorOpMultiplicandTilePrecomputeIterator(TensorRef const &ref, int lane_id)
      : pointer_(reinterpret_cast<AccessType const *>(ref.data())),
        sections_(ref.stride(0) / kCrosswise),
        // stride_ = kCrosswise x sections_ x kFactor
        stride_(ref.stride(0) * Layout::kFactor / Layout::kElementsPerAccess),
        byte_offset_(0),
        k_group_idx_(0) {

    Index row = (((lane_id >> 4) << 2) ^ (InstructionShape::kContiguous - 1)) - 3;
    Index col = lane_id & (InstructionShape::kStrided - 1);

    pointer_ = reinterpret_cast<AccessType const *>(ref.data());
    byte_offset_ = ref.layout()(TensorCoord({row, col})) * (sizeof_bits<Element>::value / 8);


    if ((lane_id / 8) % 2 == 0) {
      if ((lane_id / 2) % 2 != 0) {
        register_offset_ =  ((lane_id / 8) % 4 == 0) ? -8 : 8;
      }
    } else {
      if ((lane_id / 2) % 2 == 0) {
        register_offset_ =  ((lane_id / 8) % 4 == 1) ? -8 : 8;
      }
    }

  }

  MCTLASS_DEVICE
  void pre_compute_pointer_row_major(int warp_idx_m = 0, int tile_k = 0){

    auto org_pointer = pointer_;
    auto org_byte_offset = byte_offset_;

    pre_pointers_[0] = pointer_ + warp_idx_m * Shape::kStrided * Shape::kContiguous / Layout::kElementsPerAccess; // 256x128x32 -> 64x64x32
    pre_pointers_[1] = pointer_ + warp_idx_m * Shape::kStrided * Shape::kContiguous / Layout::kElementsPerAccess;

    add_tile_offset({0, PartitionsM});
    pre_pointers_[2] = pointer_ + warp_idx_m * Shape::kStrided * Shape::kContiguous / Layout::kElementsPerAccess;
    pre_pointers_[3] = pointer_ + warp_idx_m * Shape::kStrided * Shape::kContiguous / Layout::kElementsPerAccess;

    byte_offset_ = org_byte_offset;

    pre_byte_offsets_[0] = byte_offset_;
    this->operator++();
    pre_byte_offsets_[1] = byte_offset_;

    this->operator++();
    pre_byte_offsets_[2] = byte_offset_;
    this->operator++();
    pre_byte_offsets_[3] = byte_offset_;

    k_group_idx_ = 0;
    byte_offset_ = org_byte_offset;
    pointer_ = org_pointer;

  }

  MCTLASS_DEVICE
  void pre_compute_pointer_col_major(int warp_idx_n = 0, int tile_k = 0){

    auto org_pointer = pointer_;
    auto org_byte_offset = byte_offset_;

    pre_pointers_[0] = pointer_ + warp_idx_n * Shape::kStrided * Shape::kContiguous / Layout::kElementsPerAccess; // 256x128x32 -> 64x64x32
    pre_pointers_[1] = pointer_ + warp_idx_n * Shape::kStrided * Shape::kContiguous / Layout::kElementsPerAccess;

    add_tile_offset({0, PartitionsM});
    pre_pointers_[2] = pointer_ + warp_idx_n * Shape::kStrided * Shape::kContiguous / Layout::kElementsPerAccess;
    pre_pointers_[3] = pointer_ + warp_idx_n * Shape::kStrided * Shape::kContiguous / Layout::kElementsPerAccess;

    byte_offset_ = org_byte_offset;

    pre_byte_offsets_[0] = byte_offset_;
    this->operator++();
    pre_byte_offsets_[1] = byte_offset_;

    this->operator++();
    pre_byte_offsets_[2] = byte_offset_;
    this->operator++();
    pre_byte_offsets_[3] = byte_offset_;

    k_group_idx_ = 0;
    byte_offset_ = org_byte_offset;
    pointer_ = org_pointer;
  }

  /// Adds a pointer offset to internal pointer(s) to advance through memory
  MCTLASS_DEVICE
  MmaTensorOpMultiplicandTilePrecomputeIterator &add_pointer_offset(LongIndex offset) {
    byte_offset_ += offset * sizeof_bits<Element>::value / 8;

    return *this;
  }

  /// Advances an iterator along logical dimensions of matrix in units of whole
  /// tiles
  MCTLASS_DEVICE
  MmaTensorOpMultiplicandTilePrecomputeIterator &add_tile_offset(
      TensorCoord const &tile_offset) {
    int k_groups_delta = tile_offset.contiguous() % Policy::kGroupsPerTile;

    Index byte_offset = byte_offset_;

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
  MmaTensorOpMultiplicandTilePrecomputeIterator &add_tile_offset_negative(
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
  MmaTensorOpMultiplicandTilePrecomputeIterator &operator++() {

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
  MmaTensorOpMultiplicandTilePrecomputeIterator &operator--() { assert(0); }

  ///< advances in units of whole tiles along the logical coordinate space of
  ///< the tensor
  MCTLASS_DEVICE
  MmaTensorOpMultiplicandTilePrecomputeIterator &operator+=(
      TensorCoord const &tile_offset) {
    add_tile_offset(tile_offset);
    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of
  ///< the tensor
  MCTLASS_DEVICE
  MmaTensorOpMultiplicandTilePrecomputeIterator &operator-=(
      TensorCoord const &tile_offset) {
    add_tile_offset(-tile_offset);
    return *this;
  }

  /// Loads a fragment from memory at the location pointed to by the iterator.
  MCTLASS_HOST_DEVICE
  void load(Fragment &frag)  {
    load_with_byte_offset(frag, 0);
    }

    MCTLASS_HOST_DEVICE
    void load_with_index_A(Fragment &frag,int index)  {

        using LoadType = Array<float, Policy::LdsmShape::kCount / 2>;
        LoadType *fetch_ptr = reinterpret_cast<LoadType *>(&frag);

        char const *source_byte_ptr = reinterpret_cast<char const *>(pre_pointers_[index]) + pre_byte_offsets_[index] + register_offset_;
        auto tmp = Policy::kLdsmOpInner / Layout::kFactor * Policy::LdsmShape::kStrided  * stride_ * sizeof_bits<Element_>::value;

        MCTLASS_PRAGMA_UNROLL
        for (int s = 0; s < Policy::LdsmIterations::kStrided; ++s) {
          fetch_ptr[s] = *reinterpret_cast<LoadType const *>(source_byte_ptr + tmp * s);
        }

    }

    MCTLASS_HOST_DEVICE
    void load_with_index_B(Fragment &frag,int index)  {

        using LoadType = Array<float, Policy::LdsmShape::kCount / 2>;
        LoadType *fetch_ptr = reinterpret_cast<LoadType *>(&frag);

        char const *source_byte_ptr = reinterpret_cast<char const *>(pre_pointers_[index]) + pre_byte_offsets_[index] + register_offset_;
        auto tmp = Policy::kLdsmOpInner / Layout::kFactor * Policy::LdsmShape::kStrided  * stride_ * sizeof_bits<Element_>::value;

        MCTLASS_PRAGMA_UNROLL
        for (int s = 0; s < Policy::LdsmIterations::kStrided; ++s) {
          fetch_ptr[s] = *reinterpret_cast<LoadType const *>(source_byte_ptr + tmp * s);
        }
    }

  /// Loads a fragment from memory with additional logical offset
  MCTLASS_DEVICE
  void load_with_byte_offset(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a linear offset in units of bytes
      Index byte_offset) const {

    using LoadType = Array<float, Policy::LdsmShape::kCount / 2>;
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
            byte_offset_ + register_offset_;

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
    int PartitionsK_,
    /// Number of partitions along M dimension
    int PartitionsM_
    >
class MmaTensorOpMultiplicandTilePrecomputeIterator<
    Shape_, Operand_, int8_t,
    mctlass::layout::MacaTensorOpMultiplicandConflictFreeCrosswise<sizeof_bits<int8_t>::value, Crosswise>,
    InstructionShape_, OpDelta_, 64, PartitionsK_, PartitionsM_> {
 public:
  /// Shape of tile to load (concept: PitchLinearShape)
  using Shape = Shape_;

  /// Operand tag
  static Operand const kOperand = Operand_;

  static_assert(kOperand == Operand::kA || kOperand == Operand::kB,
                "MmaTensorOpMultiplicandTilePrecomputeIterator may only be instantiated for "
                "A or B operands to warp-level Mma.");

  /// Element type
  using Element = int8_t;

  /// Element number when the layout crosses
  static int const kCrosswise = Crosswise;

  /// Layout of source tile
  using Layout = mctlass::layout::MacaTensorOpMultiplicandCrosswise<sizeof_bits<int8_t>::value,
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
    static int const kLdsmOpInner = 8;

    static_assert(!(Shape::kContiguous % kLdsmOpOuter),
                  "Shape of warp-level mma must be divisible by LDSM's "
                  "fundamental tile size.");

    static_assert(!(Shape::kStrided % kLdsmOpInner),
                  "Shape of warp-level mma must be divisible by LDSM's "
                  "fundamental tile size.");

    /// Shape of one individual LDSM instruction
    // m16n16k32: LdsmShapeContiguous = 32 / 16 = 2
    // m16n16k16: LdsmShapeContiguous = 16 / 16 = 1
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

  AccessType const *pre_pointers_[4];
  Index pre_byte_offsets_[4];

  /// Byte offset incremented as iterator advances
  Index byte_offset_;

  Index register_offset_ = 0;
  /// Internal counter used to determine when to increment byte offset and when
  /// to XOR it
  int k_group_idx_;

 public:
  /// Default ctor constructs null iterator
  MCTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTilePrecomputeIterator()
      : pointer_(nullptr),
        sections_(0),
        stride_(0),
        byte_offset_(0),
        k_group_idx_(0) {}

  /// Constructor from TensorRef
  MCTLASS_DEVICE
  MmaTensorOpMultiplicandTilePrecomputeIterator(TensorRef const &ref, int lane_id)
      : pointer_(reinterpret_cast<AccessType const *>(ref.data())),
        sections_(ref.stride(0) / kCrosswise),
        // stride_ = kCrosswise x sections_ x kFactor
        stride_(ref.stride(0) * Layout::kFactor / Layout::kElementsPerAccess),
        byte_offset_(0),
        k_group_idx_(0) {

    static int const shl = (Policy::LdsmShape::kContiguous == 2)? 3 : 2;
    static int const vec_len = (Policy::LdsmShape::kContiguous == 2)? 7 : 3;
    Index row = (((lane_id >> 4) << shl) ^ (InstructionShape::kContiguous - 1)) - vec_len;
    Index col = lane_id & (InstructionShape::kStrided - 1);

    pointer_ = reinterpret_cast<AccessType const *>(ref.data());
    byte_offset_ = ref.layout()(TensorCoord({row, col})) * (sizeof_bits<Element>::value / 8);

    if ((lane_id / 8) % 2 == 0) {
      if ((lane_id / 2) % 2 != 0) {
        register_offset_ =  ((lane_id / 8) % 4 == 0) ? -8 : 8;
      }
    } else {
      if ((lane_id / 2) % 2 == 0) {
        register_offset_ =  ((lane_id / 8) % 4 == 1) ? -8 : 8;
      }
    }
  }

  MCTLASS_DEVICE
  void pre_compute_pointer_row_major(int warp_idx_m = 0, int tile_k = 0){

    auto org_pointer = pointer_;
    auto org_byte_offset = byte_offset_;

    pre_pointers_[0] = pointer_ + warp_idx_m * Shape::kStrided * Shape::kContiguous / Layout::kElementsPerAccess; // 256x128x32 -> 64x64x32
    pre_pointers_[1] = pointer_ + warp_idx_m * Shape::kStrided * Shape::kContiguous / Layout::kElementsPerAccess;

    add_tile_offset({0, PartitionsM});
    pre_pointers_[2] = pointer_ + warp_idx_m * Shape::kStrided * Shape::kContiguous / Layout::kElementsPerAccess;
    pre_pointers_[3] = pointer_ + warp_idx_m * Shape::kStrided * Shape::kContiguous / Layout::kElementsPerAccess;

    byte_offset_ = org_byte_offset;

    pre_byte_offsets_[0] = byte_offset_;
    this->operator++();
    pre_byte_offsets_[1] = byte_offset_;

    this->operator++();
    pre_byte_offsets_[2] = byte_offset_;
    this->operator++();
    pre_byte_offsets_[3] = byte_offset_;

    k_group_idx_ = 0;
    byte_offset_ = org_byte_offset;
    pointer_ = org_pointer;

  }

  MCTLASS_DEVICE
  void pre_compute_pointer_col_major(int warp_idx_n = 0, int tile_k = 0){

    auto org_pointer = pointer_;
    auto org_byte_offset = byte_offset_;

    pre_pointers_[0] = pointer_ + warp_idx_n * Shape::kStrided * Shape::kContiguous / Layout::kElementsPerAccess; // 256x128x32 -> 64x64x32
    pre_pointers_[1] = pointer_ + warp_idx_n * Shape::kStrided * Shape::kContiguous / Layout::kElementsPerAccess;

    add_tile_offset({0, PartitionsM});
    pre_pointers_[2] = pointer_ + warp_idx_n * Shape::kStrided * Shape::kContiguous / Layout::kElementsPerAccess;
    pre_pointers_[3] = pointer_ + warp_idx_n * Shape::kStrided * Shape::kContiguous / Layout::kElementsPerAccess;

    byte_offset_ = org_byte_offset;

    pre_byte_offsets_[0] = byte_offset_;
    this->operator++();
    pre_byte_offsets_[1] = byte_offset_;

    this->operator++();
    pre_byte_offsets_[2] = byte_offset_;
    this->operator++();
    pre_byte_offsets_[3] = byte_offset_;

    k_group_idx_ = 0;
    byte_offset_ = org_byte_offset;
    pointer_ = org_pointer;
  }

  /// Adds a pointer offset to internal pointer(s) to advance through memory
  MCTLASS_DEVICE
  MmaTensorOpMultiplicandTilePrecomputeIterator &add_pointer_offset(LongIndex offset) {
    byte_offset_ += offset * sizeof_bits<Element>::value / 8;

    return *this;
  }

  /// Advances an iterator along logical dimensions of matrix in units of whole
  /// tiles
  MCTLASS_DEVICE
  MmaTensorOpMultiplicandTilePrecomputeIterator &add_tile_offset(
      TensorCoord const &tile_offset) {
    int k_groups_delta = tile_offset.contiguous() % Policy::kGroupsPerTile;

    Index byte_offset = byte_offset_;

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
  MmaTensorOpMultiplicandTilePrecomputeIterator &add_tile_offset_negative(
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
  MmaTensorOpMultiplicandTilePrecomputeIterator &operator++() {

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
  MmaTensorOpMultiplicandTilePrecomputeIterator &operator--() { assert(0); }

  ///< advances in units of whole tiles along the logical coordinate space of
  ///< the tensor
  MCTLASS_DEVICE
  MmaTensorOpMultiplicandTilePrecomputeIterator &operator+=(
      TensorCoord const &tile_offset) {
    add_tile_offset(tile_offset);
    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of
  ///< the tensor
  MCTLASS_DEVICE
  MmaTensorOpMultiplicandTilePrecomputeIterator &operator-=(
      TensorCoord const &tile_offset) {
    add_tile_offset(-tile_offset);
    return *this;
  }

  /// Loads a fragment from memory at the location pointed to by the iterator.
  MCTLASS_HOST_DEVICE
  void load(Fragment &frag)  {
    load_with_byte_offset(frag, 0);
    }

  MCTLASS_HOST_DEVICE
  void load_with_index_A(Fragment &frag,int index)  {
    using LoadType = Array<int8_t, 4 * Policy::LdsmShape::kContiguous>;
    LoadType *fetch_ptr = reinterpret_cast<LoadType *>(&frag);

    char const *source_byte_ptr = reinterpret_cast<char const *>(pre_pointers_[index]) +
        pre_byte_offsets_[index] + register_offset_;
    auto tmp = Policy::kLdsmOpInner / Layout::kFactor * Policy::LdsmShape::kStrided *
        stride_ * sizeof_bits<Element>::value;

    int vpe = Policy::LdsmShape::kContiguous;
    MCTLASS_PRAGMA_UNROLL
    for (int s = 0; s < Policy::LdsmIterations::kStrided; ++s) {
      fetch_ptr[s] = *reinterpret_cast<LoadType const *>(source_byte_ptr + tmp * s * vpe);
    }
  }

  MCTLASS_HOST_DEVICE
  void load_with_index_B(Fragment &frag,int index)  {

    using LoadType = Array<int8_t, 4 * Policy::LdsmShape::kContiguous>;
    LoadType *fetch_ptr = reinterpret_cast<LoadType *>(&frag);

    char const *source_byte_ptr = reinterpret_cast<char const *>(pre_pointers_[index]) +
        pre_byte_offsets_[index] + register_offset_;
    auto tmp = Policy::kLdsmOpInner / Layout::kFactor * Policy::LdsmShape::kStrided *
        stride_ * sizeof_bits<Element>::value;

    int vpe = Policy::LdsmShape::kContiguous;
    MCTLASS_PRAGMA_UNROLL
    for (int s = 0; s < Policy::LdsmIterations::kStrided; ++s) {
      fetch_ptr[s] = *reinterpret_cast<LoadType const *>(source_byte_ptr + tmp * s * vpe);
    }
  }

  /// Loads a fragment from memory with additional logical offset
  MCTLASS_DEVICE
  void load_with_byte_offset(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a linear offset in units of bytes
      Index byte_offset) const {

    using LoadType = Array<int8_t, 4 * Policy::LdsmShape::kContiguous>;
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
            byte_offset_ + register_offset_;

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
class MmaTensorOpMultiplicandTilePrecomputeIterator<
    Shape_, Operand_, Element_,
    mctlass::layout::MacaTensorOpMultiplicandCongruous4x4Perm<16, 64>,
    InstructionShape_, OpDelta_, 64, PartitionsK_> {
 public:

  /// Shape of tile to load (concept: PitchLinearShape)
  using Shape = Shape_;

  /// Operand tag
  static Operand const kOperand = Operand_;

  static_assert(sizeof(Element_) == 2,
                "MacaTensorOpMultiplicandCongruous4x4Perm only support 16bit now");

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

  Index pre_byte_offsets_[4];

  int bank1_;

public:

  /// Default ctor constructs null iterator
  MCTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTilePrecomputeIterator(): stride_(0), byte_offset_(0) { }

  /// Constructor from TensorRef
  MCTLASS_DEVICE
  MmaTensorOpMultiplicandTilePrecomputeIterator(
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

  MCTLASS_DEVICE
  void pre_compute_offset(int warp_idx = 0, int tile_k = 0) {
    add_tile_offset({warp_idx, tile_k});
    pre_byte_offsets_[0] = byte_offset_;
    this->operator++();
    pre_byte_offsets_[1] = byte_offset_;

    this->operator++();
    pre_byte_offsets_[2] = byte_offset_;
    this->operator++();
    pre_byte_offsets_[3] = byte_offset_;

    k_group_idx_ = 0;
    byte_offset_ = pre_byte_offsets_[0];
  }

  /// Adds a pointer offset to internal pointer(s) to advance through memory
  MCTLASS_DEVICE
  MmaTensorOpMultiplicandTilePrecomputeIterator &add_pointer_offset(LongIndex offset) {

    byte_offset_ += offset * sizeof(Element);

    return *this;
  }

  /// Advances an iterator along logical dimensions of matrix in units of whole tiles
  MCTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTilePrecomputeIterator &add_tile_offset(TensorCoord const &tile_offset) {

    LongIndex offset = tile_offset.strided() * InstructionShape::kStrided * stride_
                        + tile_offset.contiguous() * Shape::kContiguous;

    byte_offset_ += offset * sizeof(Element);

    return *this;
  }

  /// Advances the iterator along the advance dimension
  MCTLASS_DEVICE
  MmaTensorOpMultiplicandTilePrecomputeIterator & operator++() {

    byte_offset_ += stride_ * InstructionShape::kStrided * sizeof(Element);
    return *this;
  }

  /// Advances the iterator along the opposite of the advance dimension
  MCTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTilePrecomputeIterator & operator--() {
    byte_offset_ -= stride_ * InstructionShape::kStrided * sizeof(Element);

    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of the tensor
  MCTLASS_DEVICE
  MmaTensorOpMultiplicandTilePrecomputeIterator & operator+=(TensorCoord const &tile_offset) {
    add_tile_offset(tile_offset);
    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of the tensor
  MCTLASS_DEVICE
  MmaTensorOpMultiplicandTilePrecomputeIterator & operator-=(TensorCoord const &tile_offset) {
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
    char const *source_ptr = reinterpret_cast<char const *>(pointer_) + byte_offset_ + byte_offset;
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

  MCTLASS_DEVICE
  void load_with_index(Fragment &frag, int index) {
    using LoadType =  Array<int, Policy::LdsmShape::kCount / 2>;
    LoadType fetch_element_ptr[4];
    char const *source_ptr = reinterpret_cast<char const *>(pointer_) + pre_byte_offsets_[index];
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
class MmaTensorOpMultiplicandTilePrecomputeIterator<
    Shape_, Operand_, Element_,
    mctlass::layout::MacaTensorOpMultiplicandConflictFreeCrosswise<32, Crosswise>,
    InstructionShape_, OpDelta_, 64, PartitionsK_, PartitionsM_> {
 public:
  /// Shape of tile to load (concept: PitchLinearShape)
  using Shape = Shape_;

  /// Operand tag
  static Operand const kOperand = Operand_;

  static_assert(kOperand == Operand::kA || kOperand == Operand::kB,
                "MmaTensorOpMultiplicandTilePrecomputeIterator may only be instantiated for "
                "A or B operands to warp-level Mma.");

  static_assert(sizeof(Element_) == 4,
                "MacaTensorOpMultiplicandConflictFreeCrosswise only support 32bit now");

  /// Element type
  using Element = Element_;

  /// Element number when the layout crosses
  static int const kCrosswise = Crosswise;

  /// Layout of source tile
  using Layout = mctlass::layout::MacaTensorOpMultiplicandCrosswise<sizeof_bits<float>::value,
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

  AccessType const *pre_pointers_[4];
  Index pre_byte_offsets_[4];

  /// Byte offset incremented as iterator advances
  Index byte_offset_;

  Index register_offset_ = 0;

  /// Internal counter used to determine when to increment byte offset and when
  /// to XOR it
  int k_group_idx_;

 public:
  /// Default ctor constructs null iterator
  MCTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTilePrecomputeIterator()
      : pointer_(nullptr),
        sections_(0),
        stride_(0),
        byte_offset_(0),
        k_group_idx_(0) {}

  /// Constructor from TensorRef
  MCTLASS_DEVICE
  MmaTensorOpMultiplicandTilePrecomputeIterator(TensorRef const &ref, int lane_id)
      : pointer_(reinterpret_cast<AccessType const *>(ref.data())),
        sections_(ref.stride(0) / kCrosswise),
        // stride_ = kCrosswise x sections_ x kFactor
        stride_(ref.stride(0) * Layout::kFactor / Layout::kElementsPerAccess),
        byte_offset_(0),
        k_group_idx_(0) {

    Index row = (3 - (lane_id >> 4)) << 1;
    Index col = lane_id & (InstructionShape::kStrided - 1);

    pointer_ = reinterpret_cast<AccessType const *>(ref.data());
    byte_offset_ = ref.layout()(TensorCoord({row, col})) * (sizeof_bits<Element>::value / 8);

    if ((lane_id / 8) % 2 == 0) {
      if ((lane_id / 2) % 2 != 0) {
        register_offset_ =  ((lane_id / 8) % 4 == 0) ? -8 : 8;
      }
    } else {
      if ((lane_id / 2) % 2 == 0) {
        register_offset_ =  ((lane_id / 8) % 4 == 1) ? -8 : 8;
      }
    }

  }

  MCTLASS_DEVICE
  void pre_compute_pointer_row_major(int warp_idx_m = 0, int tile_k = 0){

    auto org_pointer = pointer_;
    auto org_byte_offset = byte_offset_;

    pre_pointers_[0] = pointer_ + warp_idx_m * Shape::kStrided * Shape::kContiguous / Layout::kElementsPerAccess; // 256x128x32 -> 64x64x32
    pre_pointers_[1] = pointer_ + warp_idx_m * Shape::kStrided * Shape::kContiguous / Layout::kElementsPerAccess;

    add_tile_offset({0, PartitionsM});
    pre_pointers_[2] = pointer_ + warp_idx_m * Shape::kStrided * Shape::kContiguous / Layout::kElementsPerAccess;
    pre_pointers_[3] = pointer_ + warp_idx_m * Shape::kStrided * Shape::kContiguous / Layout::kElementsPerAccess;

    byte_offset_ = org_byte_offset;

    pre_byte_offsets_[0] = byte_offset_;
    this->operator++();
    pre_byte_offsets_[1] = byte_offset_;

    this->operator++();
    pre_byte_offsets_[2] = byte_offset_;
    this->operator++();
    pre_byte_offsets_[3] = byte_offset_;

    k_group_idx_ = 0;
    byte_offset_ = org_byte_offset;
    pointer_ = org_pointer;

  }

  MCTLASS_DEVICE
  void pre_compute_pointer_col_major(int warp_idx_n = 0, int tile_k = 0){

    auto org_pointer = pointer_;
    auto org_byte_offset = byte_offset_;

    pre_pointers_[0] = pointer_ + warp_idx_n * Shape::kStrided * Shape::kContiguous / Layout::kElementsPerAccess; // 256x128x32 -> 64x64x32
    pre_pointers_[1] = pointer_ + warp_idx_n * Shape::kStrided * Shape::kContiguous / Layout::kElementsPerAccess;

    add_tile_offset({0, PartitionsM});
    pre_pointers_[2] = pointer_ + warp_idx_n * Shape::kStrided * Shape::kContiguous / Layout::kElementsPerAccess;
    pre_pointers_[3] = pointer_ + warp_idx_n * Shape::kStrided * Shape::kContiguous / Layout::kElementsPerAccess;

    byte_offset_ = org_byte_offset;

    pre_byte_offsets_[0] = byte_offset_;
    this->operator++();
    pre_byte_offsets_[1] = byte_offset_;

    this->operator++();
    pre_byte_offsets_[2] = byte_offset_;
    this->operator++();
    pre_byte_offsets_[3] = byte_offset_;

    k_group_idx_ = 0;
    byte_offset_ = org_byte_offset;
    pointer_ = org_pointer;
  }

  /// Adds a pointer offset to internal pointer(s) to advance through memory
  MCTLASS_DEVICE
  MmaTensorOpMultiplicandTilePrecomputeIterator &add_pointer_offset(LongIndex offset) {
    byte_offset_ += offset * sizeof_bits<Element>::value / 8;

    return *this;
  }

  /// Advances an iterator along logical dimensions of matrix in units of whole
  /// tiles
  MCTLASS_DEVICE
  MmaTensorOpMultiplicandTilePrecomputeIterator &add_tile_offset(
      TensorCoord const &tile_offset) {
    int k_groups_delta = tile_offset.contiguous() % Policy::kGroupsPerTile;

    Index byte_offset = byte_offset_;

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
  MmaTensorOpMultiplicandTilePrecomputeIterator &add_tile_offset_negative(
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
  MmaTensorOpMultiplicandTilePrecomputeIterator &operator++() {

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
  MmaTensorOpMultiplicandTilePrecomputeIterator &operator--() { assert(0); }

  ///< advances in units of whole tiles along the logical coordinate space of
  ///< the tensor
  MCTLASS_DEVICE
  MmaTensorOpMultiplicandTilePrecomputeIterator &operator+=(
      TensorCoord const &tile_offset) {
    add_tile_offset(tile_offset);
    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of
  ///< the tensor
  MCTLASS_DEVICE
  MmaTensorOpMultiplicandTilePrecomputeIterator &operator-=(
      TensorCoord const &tile_offset) {
    add_tile_offset(-tile_offset);
    return *this;
  }

  /// Loads a fragment from memory at the location pointed to by the iterator.
  MCTLASS_HOST_DEVICE
  void load(Fragment &frag)  {
    load_with_byte_offset(frag, 0);
    }

  MCTLASS_HOST_DEVICE
  void load_with_index_A(Fragment &frag,int index)  {
    using LoadType = Array<float, Policy::LdsmShape::kCount / 2>;
    LoadType *fetch_ptr = reinterpret_cast<LoadType *>(&frag);

    char const *source_byte_ptr = reinterpret_cast<char const *>(pre_pointers_[index]) + pre_byte_offsets_[index] + register_offset_;
    auto tmp = Policy::kLdsmOpInner / Layout::kFactor * Policy::LdsmShape::kStrided  * stride_ * 16;

    MCTLASS_PRAGMA_UNROLL
    for (int s = 0; s < Policy::LdsmIterations::kStrided; ++s) {
      fetch_ptr[s] = *reinterpret_cast<LoadType const *>(source_byte_ptr + tmp * s);
    }
  }

  MCTLASS_HOST_DEVICE
  void load_with_index_B(Fragment &frag,int index)  {
    using LoadType = Array<float, Policy::LdsmShape::kCount / 2>;
    LoadType *fetch_ptr = reinterpret_cast<LoadType *>(&frag);

    char const *source_byte_ptr = reinterpret_cast<char const *>(pre_pointers_[index]) + pre_byte_offsets_[index] + register_offset_;
    auto tmp = Policy::kLdsmOpInner / Layout::kFactor * Policy::LdsmShape::kStrided  * stride_ * 16;

    MCTLASS_PRAGMA_UNROLL
    for (int s = 0; s < Policy::LdsmIterations::kStrided; ++s) {
      fetch_ptr[s] = *reinterpret_cast<LoadType const *>(source_byte_ptr + tmp * s);
    }
  }

  /// Loads a fragment from memory with additional logical offset
  MCTLASS_DEVICE
  void load_with_byte_offset(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a linear offset in units of bytes
      Index byte_offset) const {

    using LoadType = Array<float, Policy::LdsmShape::kCount / 2>;
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
            byte_offset_ + register_offset_;

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
class MmaTensorOpMultiplicandTilePrecomputeIterator<
    Shape_, Operand_, Element_,
    mctlass::layout::MacaTensorOpMultiplicandCongruous4x4Perm<32, 32>,
    InstructionShape_, OpDelta_, 64, PartitionsK_> {
 public:

  /// Shape of tile to load (concept: PitchLinearShape)
  using Shape = Shape_;

  /// Operand tag
  static Operand const kOperand = Operand_;

  static_assert(kOperand == Operand::kA || kOperand== Operand::kB,
    "MmaTensorOpMultiplicandIterator may only be instantiated for A or B operands to warp-level Mma.");

  static_assert(sizeof(Element_) == 4,
                "MacaTensorOpMultiplicandCongruous4x4Perm only support 32bit now");

  /// Element type
  using Element = Element_;

  /// Layout of source tile
  using Layout = mctlass::layout::MacaTensorOpMultiplicandCongruous4x4Perm<
      sizeof_bits<float>::value, int(128 / sizeof(float))>;

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
    static int const kLdsmOpInner = 4;

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

  Index pre_byte_offsets_[4];

  int bank1_;

public:

  /// Default ctor constructs null iterator
  MCTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTilePrecomputeIterator(): stride_(0), byte_offset_(0) { }

  /// Constructor from TensorRef
  MCTLASS_DEVICE
  MmaTensorOpMultiplicandTilePrecomputeIterator(
    TensorRef const &ref,
    int lane_id
  ):
    stride_(ref.stride(0)), byte_offset_(0),
    k_group_idx_(0) {

    pointer_ = reinterpret_cast<AccessType const *>(ref.data());

    Index col = 3 - (lane_id >> 4);
    Index row = lane_id & (InstructionShape::kContiguous - 1);
    byte_offset_ = (col * stride_ * 2 + row) * (sizeof_bits<Element>::value / 8);
  }

  MCTLASS_DEVICE
  void pre_compute_offset(int warp_idx = 0, int tile_k = 0) {
    add_tile_offset({warp_idx, tile_k});
    pre_byte_offsets_[0] = byte_offset_;
    this->operator++();
    pre_byte_offsets_[1] = byte_offset_;

    this->operator++();
    pre_byte_offsets_[2] = byte_offset_;
    this->operator++();
    pre_byte_offsets_[3] = byte_offset_;

    k_group_idx_ = 0;
    byte_offset_ = pre_byte_offsets_[0];
  }

  /// Adds a pointer offset to internal pointer(s) to advance through memory
  MCTLASS_DEVICE
  MmaTensorOpMultiplicandTilePrecomputeIterator &add_pointer_offset(LongIndex offset) {

    byte_offset_ += offset * sizeof(Element);

    return *this;
  }

  /// Advances an iterator along logical dimensions of matrix in units of whole tiles
  MCTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTilePrecomputeIterator &add_tile_offset(TensorCoord const &tile_offset) {

    LongIndex offset = tile_offset.strided() * InstructionShape::kStrided * stride_
                        + tile_offset.contiguous() * Shape::kContiguous;

    byte_offset_ += offset * sizeof(Element);
    return *this;
  }

  /// Advances the iterator along the advance dimension
  MCTLASS_DEVICE
  MmaTensorOpMultiplicandTilePrecomputeIterator & operator++() {

    byte_offset_ += stride_ * InstructionShape::kStrided * sizeof(Element);
    return *this;
  }

  /// Advances the iterator along the opposite of the advance dimension
  MCTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTilePrecomputeIterator & operator--() {
    byte_offset_ -= stride_ * InstructionShape::kStrided * sizeof(Element);

    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of the tensor
  MCTLASS_DEVICE
  MmaTensorOpMultiplicandTilePrecomputeIterator & operator+=(TensorCoord const &tile_offset) {
    add_tile_offset(tile_offset);
    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of the tensor
  MCTLASS_DEVICE
  MmaTensorOpMultiplicandTilePrecomputeIterator & operator-=(TensorCoord const &tile_offset) {
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
        char const *source_ptr = reinterpret_cast<char const *>(pointer_)
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
    using LoadType =  Array<float, Policy::LdsmShape::kCount / 2>;
    LoadType *perm_dst = reinterpret_cast<LoadType *>(&frag);
    char const *source_ptr = reinterpret_cast<char const *>(pointer_) + byte_offset_ + byte_offset;
    Element const *ptr = reinterpret_cast<Element const *>(source_ptr);
    MCTLASS_PRAGMA_UNROLL
    for(int i = 0; i < 4; ++i) {
      MCTLASS_PRAGMA_UNROLL
      for(int k = 0; k < 2; k++) {
        perm_dst[i][k] = *reinterpret_cast<float const *>(
            ptr + i * InstructionShape::kContiguous + k * stride_);
      }
    }
  }

  MCTLASS_DEVICE
  void load_with_index(Fragment &frag, int index) {
    using LoadType =  Array<float, Policy::LdsmShape::kCount / 2>;
    LoadType *perm_dst = reinterpret_cast<LoadType *>(&frag);
    char const *source_ptr = reinterpret_cast<char const *>(pointer_) + pre_byte_offsets_[index];
    Element const *ptr = reinterpret_cast<Element const *>(source_ptr);
    MCTLASS_PRAGMA_UNROLL
    for(int i = 0; i < 4; ++i) {
      MCTLASS_PRAGMA_UNROLL
      for(int k = 0; k < Policy::LdsmShape::kCount / 2; k++) {
        perm_dst[i][k] = *reinterpret_cast<float const *>(
            ptr + i * InstructionShape::kContiguous + k * stride_);
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
class MmaTensorOpMultiplicandTilePrecomputeIterator<
    Shape_, Operand_, Element_,
    mctlass::layout::MacaTensorOpMultiplicandMultipleLdg64Crosswise<sizeof_bits<Element_>::value, Crosswise>,
    InstructionShape_, OpDelta_, 64, PartitionsK_, PartitionsM_> {
 public:
  /// Shape of tile to load (concept: PitchLinearShape)
  using Shape = Shape_;

  /// Operand tag
  static Operand const kOperand = Operand_;

  static_assert(kOperand == Operand::kA || kOperand == Operand::kB,
                "MmaTensorOpMultiplicandTilePrecomputeIterator may only be instantiated for "
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

  AccessType const *pre_pointers_[2];
  Index pre_byte_offsets_[4];

  /// Byte offset incremented as iterator advances
  Index byte_offset_;

  Index register_offset_ = 0;

  /// Internal counter used to determine when to increment byte offset and when
  /// to XOR it
  int k_group_idx_;

 public:
  /// Default ctor constructs null iterator
  MCTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTilePrecomputeIterator()
      : pointer_(nullptr),
        sections_(0),
        stride_(0),
        byte_offset_(0),
        k_group_idx_(0) {}

  /// Constructor from TensorRef
  MCTLASS_DEVICE
  MmaTensorOpMultiplicandTilePrecomputeIterator(TensorRef const &ref, int lane_id)
      : pointer_(reinterpret_cast<AccessType const *>(ref.data())),
        sections_(ref.stride(0) / kCrosswise),
        // stride_ = kCrosswise x sections_ x kFactor
        stride_(ref.stride(0) * Layout::kFactor / Layout::kElementsPerAccess),
        byte_offset_(0),
        k_group_idx_(0) {

    Index row = (((lane_id >> 4) << 2) ^ (InstructionShape::kContiguous - 1)) - 3;
    Index col = lane_id & (InstructionShape::kStrided - 1);

    pointer_ = reinterpret_cast<AccessType const *>(ref.data());
    byte_offset_ = ref.layout()(TensorCoord({row, col})) * (sizeof_bits<Element>::value / 8);

    if ((lane_id / 8) % 2 == 0) {
      if (lane_id % 2 != 0) {
        register_offset_ =  ((lane_id / 8) % 4 == 0) ? -8 : 8;
      }
    } else {
      if (lane_id % 2 == 0) {
        register_offset_ =  ((lane_id / 8) % 4 == 1) ? -8 : 8;
      }
    }

  }

  MCTLASS_DEVICE
  void pre_compute_pointer_row_major(){

    auto org_pointer = pointer_;
    auto org_byte_offset = byte_offset_;

    const int warp_id = threadIdx.x / 64;
    pre_pointers_[0] = pointer_ + (warp_id % 4) * 256;

    add_tile_offset({0, PartitionsM});
    pre_pointers_[1] = pointer_ + (warp_id % 4) * 256;

    byte_offset_ = org_byte_offset;

    pre_byte_offsets_[0] = byte_offset_;
    this->operator++();
    pre_byte_offsets_[1] = byte_offset_;

    this->operator++();
    pre_byte_offsets_[2] = byte_offset_;
    this->operator++();
    pre_byte_offsets_[3] = byte_offset_;

    k_group_idx_ = 0;
    byte_offset_ = org_byte_offset;
    pointer_ = org_pointer;

  }

  MCTLASS_DEVICE
  void pre_compute_pointer_col_major(){

    auto org_pointer = pointer_;
    auto org_byte_offset = byte_offset_;

    const int warp_id = threadIdx.x / 64;
    pre_pointers_[0] = pointer_ + (warp_id / 4) * 2 * 256;

    add_tile_offset({0, PartitionsM});
    pre_pointers_[1] = pointer_ + (warp_id / 4) * 2 * 256;

    byte_offset_ = org_byte_offset;

    pre_byte_offsets_[0] = byte_offset_;
    this->operator++();
    pre_byte_offsets_[1] = byte_offset_;

    this->operator++();
    pre_byte_offsets_[2] = byte_offset_;
    this->operator++();
    pre_byte_offsets_[3] = byte_offset_;

    k_group_idx_ = 0;
    byte_offset_ = org_byte_offset;
    pointer_ = org_pointer;
  }

  /// Adds a pointer offset to internal pointer(s) to advance through memory
  MCTLASS_DEVICE
  MmaTensorOpMultiplicandTilePrecomputeIterator &add_pointer_offset(LongIndex offset) {
    byte_offset_ += offset * sizeof_bits<Element>::value / 8;

    return *this;
  }

  /// Advances an iterator along logical dimensions of matrix in units of whole
  /// tiles
  MCTLASS_DEVICE
  MmaTensorOpMultiplicandTilePrecomputeIterator &add_tile_offset(
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
  MmaTensorOpMultiplicandTilePrecomputeIterator &add_tile_offset_negative(
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
  MmaTensorOpMultiplicandTilePrecomputeIterator &operator++() {

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
  MmaTensorOpMultiplicandTilePrecomputeIterator &operator--() { assert(0); }

  ///< advances in units of whole tiles along the logical coordinate space of
  ///< the tensor
  MCTLASS_DEVICE
  MmaTensorOpMultiplicandTilePrecomputeIterator &operator+=(
      TensorCoord const &tile_offset) {
    add_tile_offset(tile_offset);
    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of
  ///< the tensor
  MCTLASS_DEVICE
  MmaTensorOpMultiplicandTilePrecomputeIterator &operator-=(
      TensorCoord const &tile_offset) {
    add_tile_offset(-tile_offset);
    return *this;
  }

  /// Loads a fragment from memory at the location pointed to by the iterator.
  MCTLASS_HOST_DEVICE
  void load(Fragment &frag)  {
    load_with_byte_offset(frag, 0);
    }

    MCTLASS_HOST_DEVICE
    void load_with_index_A(Fragment &frag,int index,int offset = 0)  {

        using LoadType = Array<float, Policy::LdsmShape::kCount / 2>;
        LoadType *fetch_ptr = reinterpret_cast<LoadType *>(&frag);

        char const *source_byte_ptr = reinterpret_cast<char const *>(pre_pointers_[index / 4]) + pre_byte_offsets_[index % 4] + register_offset_ + offset;
        auto tmp = Policy::kLdsmOpInner / Layout::kFactor * Policy::LdsmShape::kStrided  * stride_ * sizeof_bits<Element_>::value;

        MCTLASS_PRAGMA_UNROLL
        for (int s = 0; s < Policy::LdsmIterations::kStrided; ++s) {
          fetch_ptr[s] = *reinterpret_cast<LoadType const *>(source_byte_ptr + tmp * s);
        }

    }

    MCTLASS_HOST_DEVICE
    void load_with_index_B(Fragment &frag,int index, int offset = 0)  {

        using LoadType = Array<float, Policy::LdsmShape::kCount / 2>;
        LoadType *fetch_ptr = reinterpret_cast<LoadType *>(&frag);

        char const *source_byte_ptr = reinterpret_cast<char const *>(pre_pointers_[index / 4]) + pre_byte_offsets_[index % 4]  + register_offset_ + offset;
        auto tmp = Policy::kLdsmOpInner / Layout::kFactor * Policy::LdsmShape::kStrided  * stride_ * sizeof_bits<Element_>::value;

        MCTLASS_PRAGMA_UNROLL
        for (int s = 0; s < Policy::LdsmIterations::kStrided; ++s) {
          fetch_ptr[s] = *reinterpret_cast<LoadType const *>(source_byte_ptr + tmp * s);
        }
    }

  /// Loads a fragment from memory with additional logical offset
  MCTLASS_DEVICE
  void load_with_byte_offset(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a linear offset in units of bytes
      Index byte_offset) const {

    using LoadType = Array<float, Policy::LdsmShape::kCount / 2>;
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
            byte_offset_ + register_offset_;

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
    int PartitionsK_,
    /// Number of partitions along M dimension
    int PartitionsM_
    >
class MmaTensorOpMultiplicandTilePrecomputeIterator<
    Shape_, Operand_, int8_t,
    mctlass::layout::MacaTensorOpMultiplicandCrosswise<sizeof_bits<int8_t>::value, Crosswise>,
    InstructionShape_, OpDelta_, 64, PartitionsK_, PartitionsM_> {
 public:
  /// Shape of tile to load (concept: PitchLinearShape)
  using Shape = Shape_;

  /// Operand tag
  static Operand const kOperand = Operand_;

  static_assert(kOperand == Operand::kA || kOperand == Operand::kB,
                "MmaTensorOpMultiplicandTilePrecomputeIterator may only be instantiated for "
                "A or B operands to warp-level Mma.");

  /// Element type
  using Element = int8_t;

  /// Element number when the layout crosses
  static int const kCrosswise = Crosswise;

  /// Layout of source tile
  using Layout = mctlass::layout::MacaTensorOpMultiplicandCrosswise<sizeof_bits<int8_t>::value,
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

  AccessType const *pre_pointers_[4];
  Index pre_byte_offsets_[4];

  /// Byte offset incremented as iterator advances
  Index byte_offset_;

  /// Internal counter used to determine when to increment byte offset and when
  /// to XOR it
  int k_group_idx_;

 public:
  /// Default ctor constructs null iterator
  MCTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTilePrecomputeIterator()
      : pointer_(nullptr),
        sections_(0),
        stride_(0),
        byte_offset_(0),
        k_group_idx_(0) {}

  /// Constructor from TensorRef
  MCTLASS_DEVICE
  MmaTensorOpMultiplicandTilePrecomputeIterator(TensorRef const &ref, int lane_id)
      : pointer_(reinterpret_cast<AccessType const *>(ref.data())),
        sections_(ref.stride(0) / kCrosswise),
        // stride_ = kCrosswise x sections_ x kFactor
        stride_(ref.stride(0) * Layout::kFactor / Layout::kElementsPerAccess),
        byte_offset_(0),
        k_group_idx_(0) {
    static int const shl = (Policy::LdsmShape::kContiguous * sizeof(Element) == 2)? 3 : 2;
    static int const vec_len = (Policy::LdsmShape::kContiguous * sizeof(Element) == 2)? 7 : 3;
    Index row = (((lane_id >> 4) << shl) ^ (InstructionShape::kContiguous - 1)) - vec_len;
    Index col = lane_id & (InstructionShape::kStrided - 1);

    pointer_ = reinterpret_cast<AccessType const *>(ref.data());
    byte_offset_ = ref.layout()(TensorCoord({row, col})) * (sizeof_bits<Element>::value / 8);

  }

  MCTLASS_DEVICE
  void pre_compute_pointer_row_major(int warp_idx_m = 0){

    auto org_pointer = pointer_;
    auto org_byte_offset = byte_offset_;

    const int warp_id = threadIdx.x / 64;
    pre_pointers_[0] = pointer_ + (warp_id % 4) * 256; // 256x128x32 -> 64x64x32
    pre_pointers_[1] = pointer_ + (warp_id % 4) * 256;

    add_tile_offset({0, PartitionsM});
    pre_pointers_[2] = pointer_ + (warp_id % 4) * 256;
    pre_pointers_[3] = pointer_ + (warp_id % 4) * 256;

    byte_offset_ = org_byte_offset;

    pre_byte_offsets_[0] = byte_offset_;
    this->operator++();
    pre_byte_offsets_[1] = byte_offset_;

    this->operator++();
    pre_byte_offsets_[2] = byte_offset_;
    this->operator++();
    pre_byte_offsets_[3] = byte_offset_;

    k_group_idx_ = 0;
    byte_offset_ = org_byte_offset;
    pointer_ = org_pointer;

  }

  MCTLASS_DEVICE
  void pre_compute_pointer_col_major(int warp_idx_m = 0){

    auto org_pointer = pointer_;
    auto org_byte_offset = byte_offset_;

    const int warp_id = threadIdx.x / 64;
    pre_pointers_[0] = pointer_ + (warp_id / 4) * 256; // 256x128x32 -> 64x64x32
    pre_pointers_[1] = pointer_ + (warp_id / 4) * 256;

    add_tile_offset({0, PartitionsM});
    pre_pointers_[2] = pointer_ + (warp_id / 4) * 256;
    pre_pointers_[3] = pointer_ + (warp_id / 4) * 256;

    byte_offset_ = org_byte_offset;

    pre_byte_offsets_[0] = byte_offset_;
    this->operator++();
    pre_byte_offsets_[1] = byte_offset_;

    this->operator++();
    pre_byte_offsets_[2] = byte_offset_;
    this->operator++();
    pre_byte_offsets_[3] = byte_offset_;

    k_group_idx_ = 0;
    byte_offset_ = org_byte_offset;
    pointer_ = org_pointer;
  }

  /// Adds a pointer offset to internal pointer(s) to advance through memory
  MCTLASS_DEVICE
  MmaTensorOpMultiplicandTilePrecomputeIterator &add_pointer_offset(LongIndex offset) {
    byte_offset_ += offset * sizeof_bits<Element>::value / 8;

    return *this;
  }

  /// Advances an iterator along logical dimensions of matrix in units of whole
  /// tiles
  MCTLASS_DEVICE
  MmaTensorOpMultiplicandTilePrecomputeIterator &add_tile_offset(
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
  MmaTensorOpMultiplicandTilePrecomputeIterator &add_tile_offset_negative(
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
  MmaTensorOpMultiplicandTilePrecomputeIterator &operator++() {

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
  MmaTensorOpMultiplicandTilePrecomputeIterator &operator--() { assert(0); }

  ///< advances in units of whole tiles along the logical coordinate space of
  ///< the tensor
  MCTLASS_DEVICE
  MmaTensorOpMultiplicandTilePrecomputeIterator &operator+=(
      TensorCoord const &tile_offset) {
    add_tile_offset(tile_offset);
    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of
  ///< the tensor
  MCTLASS_DEVICE
  MmaTensorOpMultiplicandTilePrecomputeIterator &operator-=(
      TensorCoord const &tile_offset) {
    add_tile_offset(-tile_offset);
    return *this;
  }

  /// Loads a fragment from memory at the location pointed to by the iterator.
  MCTLASS_HOST_DEVICE
  void load(Fragment &frag)  {
    load_with_byte_offset(frag, 0);
    }

    MCTLASS_HOST_DEVICE
    void load_with_index_A(Fragment &frag,int index)  {

        using LoadType = Array<float, Policy::LdsmShape::kCount / 2>;
        LoadType *fetch_ptr = reinterpret_cast<LoadType *>(&frag);

        char const *source_byte_ptr = reinterpret_cast<char const *>(pre_pointers_[index]) + pre_byte_offsets_[index];
        auto tmp = Policy::kLdsmOpInner / Layout::kFactor * Policy::LdsmShape::kStrided  * stride_ * sizeof_bits<int8_t>::value;

        int vpe = Policy::LdsmShape::kContiguous;
        MCTLASS_PRAGMA_UNROLL
        for (int s = 0; s < Policy::LdsmIterations::kStrided; ++s) {
          fetch_ptr[s] = *reinterpret_cast<LoadType const *>(source_byte_ptr + tmp * s * vpe);
        }

    }

    MCTLASS_HOST_DEVICE
    void load_with_index_B(Fragment &frag,int index)  {

        using LoadType = Array<float, Policy::LdsmShape::kCount / 2>;
        LoadType *fetch_ptr = reinterpret_cast<LoadType *>(&frag);

        char const *source_byte_ptr = reinterpret_cast<char const *>(pre_pointers_[index]) + pre_byte_offsets_[index];
        auto tmp = Policy::kLdsmOpInner / Layout::kFactor * Policy::LdsmShape::kStrided  * stride_ * sizeof_bits<int8_t>::value;

        int vpe = Policy::LdsmShape::kContiguous;
        MCTLASS_PRAGMA_UNROLL
        for (int s = 0; s < Policy::LdsmIterations::kStrided; ++s) {
          fetch_ptr[s] = *reinterpret_cast<LoadType const *>(source_byte_ptr + tmp * s * vpe);
        }
    }

  /// Loads a fragment from memory with additional logical offset
  MCTLASS_DEVICE
  void load_with_byte_offset(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a linear offset in units of bytes
      Index byte_offset) const {

    using LoadType = Array<float, Policy::LdsmShape::kCount / 2>;
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
class MmaTensorOpMultiplicandTilePrecomputeIterator<
    Shape_, Operand_, int8_t,
    mctlass::layout::MacaTensorOpMultiplicandCongruous4x4Perm<sizeof_bits<int8_t>::value,
                                                       int(128 / sizeof(int8_t))>,
    InstructionShape_, OpDelta_, 64, PartitionsK_> {
 public:

  /// Shape of tile to load (concept: PitchLinearShape)
  using Shape = Shape_;

  /// Operand tag
  static Operand const kOperand = Operand_;

  static_assert(kOperand == Operand::kA || kOperand== Operand::kB,
    "MmaTensorOpMultiplicandIterator may only be instantiated for A or B operands to warp-level Mma.");

  /// Element type
  using Element = int8_t;

  /// Layout of source tile
  using Layout = mctlass::layout::MacaTensorOpMultiplicandCongruous4x4Perm<
      sizeof_bits<int8_t>::value, int(128 / sizeof(int8_t))>;

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

  Index pre_byte_offsets_[4];

  int bank1_;

public:

  /// Default ctor constructs null iterator
  MCTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTilePrecomputeIterator(): stride_(0), byte_offset_(0) { }

  /// Constructor from TensorRef
  MCTLASS_DEVICE
  MmaTensorOpMultiplicandTilePrecomputeIterator(
    TensorRef const &ref,
    int lane_id
  ):
    stride_(ref.stride(0)), byte_offset_(0),
    k_group_idx_(0) {
    pointer_ = reinterpret_cast<AccessType const *>(ref.data());
    Index col = (((lane_id >> 4) << 3) ^ (InstructionShape::kStrided - 1)) - 7;
    Index row = lane_id & (InstructionShape::kContiguous - 1);
    byte_offset_ = (col * stride_ + row * 4) * (sizeof_bits<Element>::value / 8);
  }

  MCTLASS_DEVICE
  void pre_compute_offset(int warp_idx = 0, int tile_k = 0) {
    pre_byte_offsets_[0] = byte_offset_;
    this->operator++();
    pre_byte_offsets_[1] = byte_offset_;

    this->operator++();
    pre_byte_offsets_[2] = byte_offset_;
    this->operator++();
    pre_byte_offsets_[3] = byte_offset_;

    k_group_idx_ = 0;
    byte_offset_ = pre_byte_offsets_[0];
  }

  /// Adds a pointer offset to internal pointer(s) to advance through memory
  MCTLASS_DEVICE
  MmaTensorOpMultiplicandTilePrecomputeIterator &add_pointer_offset(LongIndex offset) {

    byte_offset_ += offset * sizeof(Element);

    return *this;
  }

  /// Advances an iterator along logical dimensions of matrix in units of whole tiles
  MCTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTilePrecomputeIterator &add_tile_offset(TensorCoord const &tile_offset) {

    LongIndex offset = tile_offset.strided() * InstructionShape::kStrided * stride_
                        + tile_offset.contiguous() * Shape::kContiguous;

    byte_offset_ += offset * sizeof(Element);

    return *this;
  }

  /// Advances the iterator along the advance dimension
  MCTLASS_DEVICE
  MmaTensorOpMultiplicandTilePrecomputeIterator & operator++() {

    byte_offset_ += stride_ * InstructionShape::kStrided * sizeof(Element);
    return *this;
  }

  /// Advances the iterator along the opposite of the advance dimension
  MCTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTilePrecomputeIterator & operator--() {
    byte_offset_ -= stride_ * InstructionShape::kStrided * sizeof(Element);

    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of the tensor
  MCTLASS_DEVICE
  MmaTensorOpMultiplicandTilePrecomputeIterator & operator+=(TensorCoord const &tile_offset) {
    add_tile_offset(tile_offset);
    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of the tensor
  MCTLASS_DEVICE
  MmaTensorOpMultiplicandTilePrecomputeIterator & operator-=(TensorCoord const &tile_offset) {
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
        char const *source_ptr = reinterpret_cast<char const *>(pointer_ + access_idx)
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
    using LoadType =  Array<int, 2>;
    LoadType fetch_element_ptr[8];
    char const *source_ptr = reinterpret_cast<char const *>(pointer_) + byte_offset_ + byte_offset;
    Element const *ptr = reinterpret_cast<Element const *>(source_ptr);
    MCTLASS_PRAGMA_UNROLL
    for(int i = 0; i < 8; ++i) {
      fetch_element_ptr[i] = *reinterpret_cast<LoadType const *>(ptr + i * stride_);
    }

    Array<int, 2> *perm_dst = reinterpret_cast<Array<int, 2> *>(&frag);

    const unsigned int lo_2 = 0x05040100;
    const unsigned int hi_2 = 0x07060302;
    int tmp0 = fetch_element_ptr[0][0];
    int tmp2 = fetch_element_ptr[2][0];
    int tmp1 = fetch_element_ptr[1][0];
    int tmp3 = fetch_element_ptr[3][0];
    int tmp4 = fetch_element_ptr[4][0];
    int tmp5 = fetch_element_ptr[5][0];
    int tmp6 = fetch_element_ptr[6][0];
    int tmp7 = fetch_element_ptr[7][0];
    int tmp00 = __builtin_mxc_byte_perm(tmp2, tmp0, lo_2);
    int tmp02 = __builtin_mxc_byte_perm(tmp2, tmp0, hi_2);
    int tmp01 = __builtin_mxc_byte_perm(tmp3, tmp1, lo_2);
    int tmp03 = __builtin_mxc_byte_perm(tmp3, tmp1, hi_2);

    int lo_1 = 0x06020400;
    int tmp10 = __builtin_mxc_byte_perm(tmp6, tmp4, lo_2);
    int hi_1 = 0x07030501;
    int tmp12 = __builtin_mxc_byte_perm(tmp6, tmp4, hi_2);
    int tmp11 = __builtin_mxc_byte_perm(tmp7, tmp5, lo_2);
    int tmp13 = __builtin_mxc_byte_perm(tmp7, tmp5, hi_2);

    perm_dst[0][0] = __builtin_mxc_byte_perm(tmp01, tmp00, lo_1);
    perm_dst[1][0] = __builtin_mxc_byte_perm(tmp01, tmp00, hi_1);
    perm_dst[3][0] = __builtin_mxc_byte_perm(tmp03, tmp02, hi_1);
    perm_dst[2][0] = __builtin_mxc_byte_perm(tmp03, tmp02, lo_1);
    perm_dst[0][1] = __builtin_mxc_byte_perm(tmp11, tmp10, lo_1);
    perm_dst[1][1] = __builtin_mxc_byte_perm(tmp11, tmp10, hi_1);
    perm_dst[2][1] = __builtin_mxc_byte_perm(tmp13, tmp12, lo_1);
    perm_dst[3][1] = __builtin_mxc_byte_perm(tmp13, tmp12, hi_1);
  }

  /// Loads a fragment from memory with additional logical offset
  MCTLASS_DEVICE
  void load_tmp(
      /// tmp pointer
      int *fetch_element_ptr) const {
    char const *source_ptr = reinterpret_cast<char const *>(pointer_) + byte_offset_;
    Element const *ptr = reinterpret_cast<Element const *>(source_ptr);
    MCTLASS_PRAGMA_UNROLL
    for(int i = 0; i < 8; ++i) {
      fetch_element_ptr[i] = *reinterpret_cast<int const *>(ptr + i * stride_);
    }
  }

  /// Loads a fragment from memory with additional logical offset
  MCTLASS_DEVICE
  void int8_cvt_perm(
      int *fetch_element_ptr,
      /// fragment to load from the tensor
      Fragment &frag) const {
    Array<int, 2> *perm_dst = reinterpret_cast<Array<int, 2> *>(&frag);
    const unsigned int lo_2 = 0x05040100;
    const unsigned int hi_2 = 0x07060302;
    int tmp0 = fetch_element_ptr[0];
    int tmp2 = fetch_element_ptr[2];
    int tmp1 = fetch_element_ptr[1];
    int tmp3 = fetch_element_ptr[3];
    int tmp4 = fetch_element_ptr[4];
    int tmp5 = fetch_element_ptr[5];
    int tmp6 = fetch_element_ptr[6];
    int tmp7 = fetch_element_ptr[7];
    int tmp00 = __builtin_mxc_byte_perm(tmp2, tmp0, lo_2);
    int tmp02 = __builtin_mxc_byte_perm(tmp2, tmp0, hi_2);
    int tmp01 = __builtin_mxc_byte_perm(tmp3, tmp1, lo_2);
    int tmp03 = __builtin_mxc_byte_perm(tmp3, tmp1, hi_2);

    int lo_1 = 0x06020400;
    int tmp10 = __builtin_mxc_byte_perm(tmp6, tmp4, lo_2);
    int hi_1 = 0x07030501;
    int tmp12 = __builtin_mxc_byte_perm(tmp6, tmp4, hi_2);
    int tmp11 = __builtin_mxc_byte_perm(tmp7, tmp5, lo_2);
    int tmp13 = __builtin_mxc_byte_perm(tmp7, tmp5, hi_2);

    perm_dst[0][0] = __builtin_mxc_byte_perm(tmp01, tmp00, lo_1);
    perm_dst[1][0] = __builtin_mxc_byte_perm(tmp01, tmp00, hi_1);
    perm_dst[3][0] = __builtin_mxc_byte_perm(tmp03, tmp02, hi_1);
    perm_dst[2][0] = __builtin_mxc_byte_perm(tmp03, tmp02, lo_1);
    perm_dst[0][1] = __builtin_mxc_byte_perm(tmp11, tmp10, lo_1);
    perm_dst[1][1] = __builtin_mxc_byte_perm(tmp11, tmp10, hi_1);
    perm_dst[2][1] = __builtin_mxc_byte_perm(tmp13, tmp12, lo_1);
    perm_dst[3][1] = __builtin_mxc_byte_perm(tmp13, tmp12, hi_1);
  }

  MCTLASS_DEVICE
  void load_with_index(Fragment &frag, int index) {
    using LoadType =  Array<int, 2>;
    LoadType fetch_element_ptr[8];
    char const *source_ptr = reinterpret_cast<char const *>(pointer_) + pre_byte_offsets_[index];
    Element const *ptr = reinterpret_cast<Element const *>(source_ptr);
    MCTLASS_PRAGMA_UNROLL
    for(int i = 0; i < 8; ++i) {
      fetch_element_ptr[i] = *reinterpret_cast<LoadType const *>(ptr + i * stride_);
    }

    Array<int, 2> *perm_dst = reinterpret_cast<Array<int, 2> *>(&frag);
    const unsigned int lo_2 = 0x05040100;
    const unsigned int hi_2 = 0x07060302;
    int tmp0 = fetch_element_ptr[0][0];
    int tmp2 = fetch_element_ptr[2][0];
    int tmp1 = fetch_element_ptr[1][0];
    int tmp3 = fetch_element_ptr[3][0];
    int tmp4 = fetch_element_ptr[4][0];
    int tmp5 = fetch_element_ptr[5][0];
    int tmp6 = fetch_element_ptr[6][0];
    int tmp7 = fetch_element_ptr[7][0];
    int tmp00 = __builtin_mxc_byte_perm(tmp2, tmp0, lo_2);
    int tmp02 = __builtin_mxc_byte_perm(tmp2, tmp0, hi_2);
    int tmp01 = __builtin_mxc_byte_perm(tmp3, tmp1, lo_2);
    int tmp03 = __builtin_mxc_byte_perm(tmp3, tmp1, hi_2);

    int lo_1 = 0x06020400;
    int tmp10 = __builtin_mxc_byte_perm(tmp6, tmp4, lo_2);
    int hi_1 = 0x07030501;
    int tmp12 = __builtin_mxc_byte_perm(tmp6, tmp4, hi_2);
    int tmp11 = __builtin_mxc_byte_perm(tmp7, tmp5, lo_2);
    int tmp13 = __builtin_mxc_byte_perm(tmp7, tmp5, hi_2);

    perm_dst[0][0] = __builtin_mxc_byte_perm(tmp01, tmp00, lo_1);
    perm_dst[1][0] = __builtin_mxc_byte_perm(tmp01, tmp00, hi_1);
    perm_dst[3][0] = __builtin_mxc_byte_perm(tmp03, tmp02, hi_1);
    perm_dst[2][0] = __builtin_mxc_byte_perm(tmp03, tmp02, lo_1);
    perm_dst[0][1] = __builtin_mxc_byte_perm(tmp11, tmp10, lo_1);
    perm_dst[1][1] = __builtin_mxc_byte_perm(tmp11, tmp10, hi_1);
    perm_dst[2][1] = __builtin_mxc_byte_perm(tmp13, tmp12, lo_1);
    perm_dst[3][1] = __builtin_mxc_byte_perm(tmp13, tmp12, hi_1);
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

} // namespace warp
} // namespace gemm
} // namespace mctlass