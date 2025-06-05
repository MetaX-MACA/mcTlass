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
    \brief Template for a double-buffered threadblock-scoped GEMM kernel.
*/

#pragma once

#include "mctlass/aligned_buffer.h"
#include "mctlass/arch/memory.h"
#include "mctlass/array.h"
#include "mctlass/mctlass.h"
#include "mctlass/gemm/gemm.h"
#include "mctlass/matrix_shape.h"
#include "mctlass/numeric_types.h"

#include "mctlass/gemm/threadblock/mma_base.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace mctlass {
namespace gemm {
namespace threadblock {

/////////////////////////////////////////////////////////////////////////////////////////////////

// mcTlass: normal mmamultistage
template <
    /// Size of the Gemm problem - concept: gemm::GemmShape<>
    typename Shape_,
    /// Iterates over tiles of A operand in global memory
    //  (concept: ReadableTileIterator | ForwardTileIterator |
    //  MaskedTileIterator)
    typename IteratorA_,
    /// Iterates over tiles of A operand in shared memory
    /// (concept: WriteableTileIterator | RandomAccessTileIterator)
    typename SmemIteratorA_,
    /// Cache operation for operand A
    mctlass::arch::CacheOperation::Kind CacheOpA,
    /// Iterates over tiles of B operand in global memory
    //  (concept: ReadableTileIterator | ForwardTileIterator |
    //  MaskedTileIterator)
    typename IteratorB_,
    /// Iterates over tiles of B operand in shared memory
    /// (concept: WriteableTileIterator | RandomAccessTileIterator)
    typename SmemIteratorB_,
    /// Cache operation for operand B
    mctlass::arch::CacheOperation::Kind CacheOpB,
    /// Data type of accumulator matrix
    typename ElementC_,
    /// Data type of accumulator matrix
    typename LayoutC_,
    /// Policy describing tuning details (concept: MmaPolicy)
    typename Policy_,
    /// Number of stages,
    int Stages,
    /// Use zfill or predicate for out-of-bound cp.async
    SharedMemoryClearOption SharedMemoryClear = SharedMemoryClearOption::kNone,
    /// Flag tag for special process for m16n16k16 f16
    bool MacaSpecialTag_ = false,
    /// Used for partial specialization
    typename Enable = bool
    >
class MmaMultistage :
  public MmaBase<Shape_, Policy_, Stages> {
public:
  ///< Base class
  using Base = MmaBase<Shape_, Policy_, Stages>;
  ///< Size of the Gemm problem - concept: gemm::GemmShape<>
  using Shape = Shape_;
  ///< Iterates over tiles of A operand in global memory
  using IteratorA = IteratorA_;
  ///< Iterates over tiles of B operand in global memory
  using IteratorB = IteratorB_;
  ///< Data type of accumulator matrix
  using ElementC = ElementC_;
  ///< Layout of accumulator matrix
  using LayoutC = LayoutC_;
  ///< Policy describing tuning details
  using Policy = Policy_;

  using SmemIteratorA = SmemIteratorA_;
  using SmemIteratorB = SmemIteratorB_;

  static mctlass::arch::CacheOperation::Kind const kCacheOpA = CacheOpA;
  static mctlass::arch::CacheOperation::Kind const kCacheOpB = CacheOpB;
  // static constexpr bool MacaSpecialTag = MacaSpecialTag_;
  static constexpr bool MacaSpecialTag = false;

  static constexpr bool CrossOffset =
    platform::is_same<layout::MacaRowMajorTensorOpMultiplicandCpAsyncCrosswise<
      sizeof_bits<typename IteratorA::Element>::value, Shape::kK>,
    typename SmemIteratorA::Layout>::value;

  //
  // Dependent types
  //

  /// Fragment of accumulator tile
  using FragmentC = typename Policy::Operator::FragmentC;

  /// Warp-level Mma
  using Operator = typename Policy::Operator;

  /// Minimum architecture is Sm80 to support cp.async
  using ArchTag = arch::Sm80;

  /// Complex transform on A operand
  static ComplexTransform const kTransformA = Operator::kTransformA;

  /// Complex transform on B operand
  static ComplexTransform const kTransformB = Operator::kTransformB;

  static int const kSrcBytesB = sizeof_bits<typename IteratorB::Element>::value *
                                IteratorB::ThreadMap::kElementsPerAccess /
                                IteratorB::kAccessesPerVector / 8;
  static int const kSrcBytesA = sizeof_bits<typename IteratorA::Element>::value *
                                IteratorA::ThreadMap::kElementsPerAccess /
                                IteratorA::kAccessesPerVector / 8;
  using AccessTypeB  = Array<uint8_t, kSrcBytesB>;

  /// Internal structure exposed for introspection.
  struct Detail {

    /// Number of cp.async instructions to load one stage of operand A
    static int const AsyncCopyIterationsPerStageA =
        IteratorA::ThreadMap::Iterations::kCount;

    /// Number of cp.async instructions to load one stage of operand B
    static int const AsyncCopyIterationsPerStageB =
        IteratorB::ThreadMap::Iterations::kCount;

    /// Number of stages
    static int const kStages = Stages;

    /// Number of cp.async instructions to load on group of operand A
    static int const kAccessesPerGroupA =
        (AsyncCopyIterationsPerStageA + Base::kWarpGemmIterations - 1) / Base::kWarpGemmIterations;

    /// Number of cp.async instructions to load on group of operand B
    static int const kAccessesPerGroupB =
        (AsyncCopyIterationsPerStageB + Base::kWarpGemmIterations - 1) / Base::kWarpGemmIterations;
  };

 private:

  using WarpLoadedFragmentA = typename Operator::FragmentA;
  using WarpLoadedFragmentB = typename Operator::FragmentB;
  using WarpTransformedFragmentA = typename Operator::TransformedFragmentA;
  using WarpTransformedFragmentB = typename Operator::TransformedFragmentB;

 private:

  //
  // Data members
  //

  /// Iterator to write threadblock-scoped tile of A operand to shared memory
  SmemIteratorA smem_iterator_A_;

  /// Iterator to write threadblock-scoped tile of B operand to shared memory
  SmemIteratorB smem_iterator_B_;

  int cross_offset_;

public:

  /// Construct from tensor references
  MCTLASS_DEVICE
  MmaMultistage(
      ///< Shared storage needed for internal use by threadblock-scoped GEMM
      typename Base::SharedStorage &shared_storage,
      ///< ID within the threadblock
      int thread_idx,
      ///< ID of warp
      int warp_idx,
      ///< ID of each thread within a warp
      int lane_idx,
      int problem_size_k = 0,
      typename IteratorA::Element *ptr_A = nullptr,
      typename IteratorA::Element *ptr_B = nullptr
    ):
      Base(shared_storage, thread_idx, warp_idx, lane_idx),
      smem_iterator_A_(shared_storage.operand_A_ref(), thread_idx),
      smem_iterator_B_(shared_storage.operand_B_ref(), thread_idx)
  {
    if constexpr (CrossOffset) {
      int bid = lane_idx & 0x7;
      int group_id = (lane_idx >> 3) & 0x3;
      int cross_target = bid ^ group_id;
      cross_offset_ = cross_target - bid ;
    } else {
      cross_offset_ = 0;
    }

    // Compute warp location within threadblock tile by mapping the warp_id to
    // three coordinates:
    //   _m: the warp's position within the threadblock along the M dimension
    //   _n: the warp's position within the threadblock along the N dimension
    //   _k: the warp's position within the threadblock along the K dimension

    int warp_idx_mn = warp_idx % (Base::WarpCount::kM * Base::WarpCount::kN);
    int warp_idx_k = warp_idx / (Base::WarpCount::kM * Base::WarpCount::kN);

    int warp_idx_m = warp_idx_mn % Base::WarpCount::kM;
    int warp_idx_n = warp_idx_mn / Base::WarpCount::kM;

    // Add per-warp offsets in units of warp-level tiles
    this->warp_tile_iterator_A_.add_tile_offset(
        {warp_idx_m, Base::kWarpGemmIterations * warp_idx_k});
    this->warp_tile_iterator_B_.add_tile_offset(
        {Base::kWarpGemmIterations * warp_idx_k, warp_idx_n});
  }

  MCTLASS_DEVICE
  void copy_tiles_and_advance(IteratorA &iterator_A, IteratorB &iterator_B,
                              int group_start_A = 0, int group_start_B = 0) {
    iterator_A.set_iteration_index(group_start_A *
                                   IteratorA::kAccessesPerVector);
    this->smem_iterator_A_.set_iteration_index(group_start_A);

    // Async Copy for operand A
    MCTLASS_PRAGMA_UNROLL
    for (int j = 0; j < Detail::kAccessesPerGroupA; ++j) {
      if (group_start_A + j < Detail::AsyncCopyIterationsPerStageA) {
        typename IteratorA::AccessType *dst_ptr[IteratorA::kAccessesPerVector];
        typename IteratorA::AccessType *gmem_ptr[IteratorA::kAccessesPerVector];
        bool valid_tag[IteratorA::kAccessesPerVector];
        MCTLASS_PRAGMA_UNROLL
        for (int v = 0; v < IteratorA::kAccessesPerVector; ++v) {
          dst_ptr[v] = reinterpret_cast<typename IteratorA::AccessType *>(this->smem_iterator_A_.get()) + v - cross_offset_;
          gmem_ptr[v] = iterator_A.get() + cross_offset_;
          valid_tag[v] = iterator_A.valid();
          ++iterator_A;
        }

        MCTLASS_PRAGMA_UNROLL
        for (int v = 0; v < IteratorA::kAccessesPerVector; ++v) {
          if constexpr (SharedMemoryClear == SharedMemoryClearOption::kZfill) {
            mctlass::arch::cp_async_zfill<kSrcBytesA, kCacheOpA>(
                dst_ptr[v], gmem_ptr[v], valid_tag[v]);
          } else {
            mctlass::arch::cp_async<kSrcBytesA, kCacheOpA>(
                dst_ptr[v], gmem_ptr[v], valid_tag[v]);
          }
        }

        ++this->smem_iterator_A_;
      }
    }

    iterator_B.set_iteration_index(group_start_B *
                                   IteratorB::kAccessesPerVector);
    this->smem_iterator_B_.set_iteration_index(group_start_B);

    // Async Copy for operand B
    MCTLASS_PRAGMA_UNROLL
    for (int j = 0; j < Detail::kAccessesPerGroupB; ++j) {
      if (group_start_B + j < Detail::AsyncCopyIterationsPerStageB) {

        typename IteratorB::AccessType *dst_ptr[IteratorB::kAccessesPerVector];
        typename IteratorB::AccessType *gmem_ptr[IteratorB::kAccessesPerVector];
        bool valid_tag[IteratorB::kAccessesPerVector];

        MCTLASS_PRAGMA_UNROLL
        for (int v = 0; v < IteratorB::kAccessesPerVector; ++v) {
          dst_ptr[v] = reinterpret_cast<typename IteratorB::AccessType *>(this->smem_iterator_B_.get()) + v;
          gmem_ptr[v] = iterator_B.get();
          valid_tag[v] = iterator_B.valid();

          ++iterator_B;
        }

        MCTLASS_PRAGMA_UNROLL
        for (int v = 0; v < IteratorB::kAccessesPerVector; ++v) {
          if (SharedMemoryClear == SharedMemoryClearOption::kZfill) {
            mctlass::arch::cp_async_zfill<kSrcBytesB, kCacheOpB>(
                dst_ptr[v], gmem_ptr[v], valid_tag[v]);
          } else {
            mctlass::arch::cp_async<kSrcBytesB, kCacheOpB>(
                dst_ptr[v], gmem_ptr[v], valid_tag[v]);
          }
        }
        ++this->smem_iterator_B_;
      }
    }
  }

  /// Perform a threadblock-scoped matrix multiply-accumulate
  MCTLASS_DEVICE
  void operator()(
      ///< problem size of GEMM
      int gemm_k_iterations,
      ///< destination accumulator tile
      FragmentC &accum,
      ///< iterator over A operand in global memory
      IteratorA iterator_A,
      ///< iterator over B operand in global memory
      IteratorB iterator_B,
      ///< initial value of accumulator
      FragmentC const &src_accum) {
    //
    // Prologue
    //
    // Issue several complete stages
    MCTLASS_PRAGMA_UNROLL
    for (int stage = 0; stage < Base::kStages - 1;
         ++stage, --gemm_k_iterations) {

      iterator_A.clear_mask(gemm_k_iterations == 0);
      iterator_B.clear_mask(gemm_k_iterations == 0);

      iterator_A.set_iteration_index(0);
      this->smem_iterator_A_.set_iteration_index(0);

      // Async Copy for operand A
      MCTLASS_PRAGMA_UNROLL
      for (int j = 0; j < Detail::AsyncCopyIterationsPerStageA; ++j) {
        typename IteratorA::AccessType *dst_ptr[IteratorA::kAccessesPerVector];
        typename IteratorA::AccessType *gmem_ptr[IteratorA::kAccessesPerVector];
        bool valid_tag[IteratorA::kAccessesPerVector];
        MCTLASS_PRAGMA_UNROLL
        for (int v = 0; v < IteratorA::kAccessesPerVector; ++v) {
          dst_ptr[v] = reinterpret_cast<typename IteratorA::AccessType *>(this->smem_iterator_A_.get()) + v - cross_offset_;
          gmem_ptr[v] = iterator_A.get() + cross_offset_;
          valid_tag[v] = iterator_A.valid();
          ++iterator_A;
        }

        MCTLASS_PRAGMA_UNROLL
        for (int v = 0; v < IteratorA::kAccessesPerVector; ++v) {
          mctlass::arch::cp_async_zfill<kSrcBytesA, kCacheOpA>(
            dst_ptr[v], gmem_ptr[v], valid_tag[v]);
        }

        ++this->smem_iterator_A_;
      }

      iterator_B.set_iteration_index(0);
      this->smem_iterator_B_.set_iteration_index(0);

      // Async Copy for operand B
      MCTLASS_PRAGMA_UNROLL
      for (int j = 0; j < Detail::AsyncCopyIterationsPerStageB; ++j) {
        typename IteratorB::AccessType *dst_ptr[IteratorB::kAccessesPerVector];
        typename IteratorB::AccessType *gmem_ptr[IteratorB::kAccessesPerVector];
        bool valid_tag[IteratorB::kAccessesPerVector];

        MCTLASS_PRAGMA_UNROLL
        for (int v = 0; v < IteratorB::kAccessesPerVector; ++v) {
          dst_ptr[v] = reinterpret_cast<typename IteratorB::AccessType *>(this->smem_iterator_B_.get()) + v;
          gmem_ptr[v] = iterator_B.get();
          valid_tag[v] = iterator_B.valid();

          ++iterator_B;
        }

        MCTLASS_PRAGMA_UNROLL
        for (int v = 0; v < IteratorB::kAccessesPerVector; ++v) {
          // @TODO: it need to be filled with zeros while valid_tag[v] is false
          if constexpr (kCacheOpB == mctlass::arch::CacheOperation::Builtin) {
            mctlass::arch::cp_async<kSrcBytesB, kCacheOpB>(
              dst_ptr[v], gmem_ptr[v], valid_tag[v]);
          } else {
            mctlass::arch::cp_async_zfill<kSrcBytesB, kCacheOpB>(
              dst_ptr[v], gmem_ptr[v], valid_tag[v]);
          }
        }

        ++this->smem_iterator_B_;
      }

      // Move to the next stage
      iterator_A.add_tile_offset({0, 1});
      iterator_B.add_tile_offset({1, 0});

      this->smem_iterator_A_.add_tile_offset({0, 1});
      this->smem_iterator_B_.add_tile_offset({1, 0});

      // Defines the boundary of a stage of cp.async.
      mctlass::arch::cp_async_fence();
    }

    // Perform accumulation in the 'd' output operand
    accum = src_accum;

    //
    // Clear the remaining tiles of SMEM. This is a functional requirement for some kernels
    // so that all accumulator elements outside the GEMM footprint are zero.
    //

    if (SharedMemoryClear == SharedMemoryClearOption::kClearLastStage) {

      /// Iterator to write threadblock-scoped tile of A operand to shared memory
      SmemIteratorA last_smem_iterator_A(this->smem_iterator_A_);

      typename IteratorA::AccessType zero_A;
      zero_A.clear();

      last_smem_iterator_A.set_iteration_index(0);

      // Async Copy for operand A
      MCTLASS_PRAGMA_UNROLL
      for (int j = 0; j < Detail::AsyncCopyIterationsPerStageA; ++j) {

        typename IteratorA::AccessType *dst_ptr =
            reinterpret_cast<typename IteratorA::AccessType *>(
                last_smem_iterator_A.get());

        *dst_ptr = zero_A;

        ++last_smem_iterator_A;
      }

      /// Iterator to write threadblock-scoped tile of B operand to shared memory
      SmemIteratorB last_smem_iterator_B(this->smem_iterator_B_);
      typename IteratorB::AccessType zero_B;

      zero_B.clear();
      last_smem_iterator_B.set_iteration_index(0);

      // Async Copy for operand B
      MCTLASS_PRAGMA_UNROLL
      for (int j = 0; j < Detail::AsyncCopyIterationsPerStageB; ++j) {

        typename IteratorB::AccessType *dst_ptr =
            reinterpret_cast<typename IteratorB::AccessType *>(
                last_smem_iterator_B.get());

        *dst_ptr = zero_B;

        ++last_smem_iterator_B;
      }
    }

    // Waits until stages up to the previous (kStages-2)th stage have committed.
    // mctlass::arch::cp_async_wait<Base::kStages - 2>();
    // if constexpr (Syncthreads) __syncthreads();
    __builtin_mxc_arrive(64);
    __builtin_mxc_barrier();

    // Pair of fragments used to overlap shared memory loads and math
    // instructions
    WarpLoadedFragmentA warp_loaded_frag_A[2];
    WarpLoadedFragmentB warp_loaded_frag_B[2];
    WarpTransformedFragmentA warp_transformed_frag_A[2];
    WarpTransformedFragmentB warp_transformed_frag_B[2];

    Operator warp_mma;

    this->warp_tile_iterator_A_.set_kgroup_index(0);
    this->warp_tile_iterator_B_.set_kgroup_index(0);

    this->warp_tile_iterator_A_.load(warp_loaded_frag_A[0]);
    this->warp_tile_iterator_B_.load(warp_loaded_frag_B[0]);

    ++this->warp_tile_iterator_A_;
    ++this->warp_tile_iterator_B_;

    iterator_A.clear_mask(gemm_k_iterations == 0);
    iterator_B.clear_mask(gemm_k_iterations == 0);

    int smem_write_stage_idx = Base::kStages - 1;
    int smem_read_stage_idx = 0;

    warp_mma.transform(warp_transformed_frag_A[0], warp_transformed_frag_B[0],
                       warp_loaded_frag_A[0], warp_loaded_frag_B[0]);

    // tf32x3 kernels use staging accumulation. warp_mma uses a temporary
    // accumulator and this temporary accumulator is added to the final
    // accumulator once in every mainloop iteration.
    plus<FragmentC> plus_accum;

    FragmentC tmp_accum;

    if (platform::is_same<typename Operator::MathOperator,
                          arch::OpMultiplyAddFastF32>::value
      || platform::is_same<typename Operator::MathOperator,
                           arch::OpMultiplyAddComplexFastF32>::value) {

      tmp_accum.clear();
    }

    //
    // Mainloop
    //

    MCTLASS_GEMM_LOOP
    for (; gemm_k_iterations > (-Base::kStages + 1);) {
      //
      // Loop over GEMM K dimension
      //

      // Computes a warp-level GEMM on data held in shared memory
      // Each "warp_mma_k" refers to a warp-level matrix multiply-accumulate
      MCTLASS_PRAGMA_UNROLL
      for (int warp_mma_k = 0; warp_mma_k < Base::kWarpGemmIterations;
           ++warp_mma_k) {
        // Load warp-level tiles from shared memory, wrapping to k offset if
        // this is the last group as the case may be.

        this->warp_tile_iterator_A_.set_kgroup_index((warp_mma_k + 1) % Base::kWarpGemmIterations);
        this->warp_tile_iterator_B_.set_kgroup_index((warp_mma_k + 1) % Base::kWarpGemmIterations);

        this->warp_tile_iterator_A_.load(warp_loaded_frag_A[(warp_mma_k + 1) % 2]);
        this->warp_tile_iterator_B_.load(warp_loaded_frag_B[(warp_mma_k + 1) % 2]);

        ++this->warp_tile_iterator_A_;
        ++this->warp_tile_iterator_B_;

        if (warp_mma_k > 0)
          warp_mma.transform(warp_transformed_frag_A[warp_mma_k % 2],
                             warp_transformed_frag_B[warp_mma_k % 2],
                             warp_loaded_frag_A[warp_mma_k % 2],
                             warp_loaded_frag_B[warp_mma_k % 2]);

        if (platform::is_same<typename Operator::MathOperator,
                              arch::OpMultiplyAddFastF32>::value
          || platform::is_same<typename Operator::MathOperator,
                               arch::OpMultiplyAddComplexFastF32>::value) {

          warp_mma(
            tmp_accum,
            warp_transformed_frag_A[warp_mma_k % 2],
            warp_transformed_frag_B[warp_mma_k % 2],
            tmp_accum
          );
          // At present, PTX: "cp.async.commit_group", "cp.async.wait_group", "cp.async.wait_all" in memory_sm80.h is not implemented on maca
	  // gemm/device test: SM80_Device_Her2k_cf32n_cf32n_l_tensor_op_fast_f32.64x64x16_32x32x16 failed
	  // therefore, block the segment accumulation of fast branch.
	  // after PTX is implemented ,the following code should be opened again(2023.03.06).
         // if (warp_mma_k == 0) {
         //   accum = plus_accum(accum, tmp_accum);
         //   tmp_accum.clear();
         // }
        } else {
          warp_mma(
            accum,
            warp_transformed_frag_A[warp_mma_k % 2],
            warp_transformed_frag_B[warp_mma_k % 2],
            accum
          );
        }

        // Issue global->shared copies for the this stage
        if (warp_mma_k < Base::kWarpGemmIterations - 1) {
          int group_start_iteration_A, group_start_iteration_B;

          group_start_iteration_A = warp_mma_k * Detail::kAccessesPerGroupA;
          group_start_iteration_B = warp_mma_k * Detail::kAccessesPerGroupB;

          copy_tiles_and_advance(iterator_A, iterator_B, group_start_iteration_A,
                               group_start_iteration_B);
        }

        if (warp_mma_k + 2 == Base::kWarpGemmIterations) {
          int group_start_iteration_A, group_start_iteration_B;
          group_start_iteration_A =
              (warp_mma_k + 1) * Detail::kAccessesPerGroupA;
          group_start_iteration_B =
              (warp_mma_k + 1) * Detail::kAccessesPerGroupB;

          copy_tiles_and_advance(iterator_A, iterator_B, group_start_iteration_A,
                               group_start_iteration_B);


          // Inserts a memory fence between stages of cp.async instructions.
          // mctlass::arch::cp_async_fence();

          // // Waits until stages up to the previous (kStages-2)th stage have committed.
          // arch::cp_async_wait<Base::kStages - 2>();
          // if constexpr (Syncthreads) __syncthreads();
          if (mctlass::arch::CacheOperation::Builtin == kCacheOpA &&
              mctlass::arch::CacheOperation::Builtin == kCacheOpB) {
            __builtin_mxc_arrive(64*64 + 128 *10);
            __builtin_mxc_barrier();
          } else {
            if constexpr (!MacaSpecialTag) __syncthreads();
          }

          // Move to the next stage
          iterator_A.add_tile_offset({0, 1});
          iterator_B.add_tile_offset({1, 0});

          this->smem_iterator_A_.add_tile_offset({0, 1});
          this->smem_iterator_B_.add_tile_offset({1, 0});

          // Add negative offsets to return iterators to the 'start' of the
          // circular buffer in shared memory
          if (smem_write_stage_idx == (Base::kStages - 1)) {
            this->smem_iterator_A_.add_tile_offset({0, -Base::kStages});
            this->smem_iterator_B_.add_tile_offset({-Base::kStages, 0});
            smem_write_stage_idx = 0;
          } else {
            ++smem_write_stage_idx;
          }

          if (smem_read_stage_idx == (Base::kStages - 1)) {
            this->warp_tile_iterator_A_.add_tile_offset(
                {0, -Base::kStages * Policy::kPartitionsK *
                        Base::kWarpGemmIterations});
            this->warp_tile_iterator_B_.add_tile_offset(
                {-Base::kStages * Policy::kPartitionsK *
                     Base::kWarpGemmIterations,
                 0});
            smem_read_stage_idx = 0;
          } else {
            ++smem_read_stage_idx;
          }

          --gemm_k_iterations;
          iterator_A.clear_mask(gemm_k_iterations == 0);
          iterator_B.clear_mask(gemm_k_iterations == 0);
        }

        // Do any conversions feeding the first stage at the end of the loop so
        // we can start right away on mma instructions
        if (warp_mma_k + 1 == Base::kWarpGemmIterations)
          warp_mma.transform(warp_transformed_frag_A[(warp_mma_k + 1) % 2],
                             warp_transformed_frag_B[(warp_mma_k + 1) % 2],
                             warp_loaded_frag_A[(warp_mma_k + 1) % 2],
                             warp_loaded_frag_B[(warp_mma_k + 1) % 2]);
          __builtin_mxc_arrive(64);
      }

    }

    if (platform::is_same<typename Operator::MathOperator,
                          arch::OpMultiplyAddFastF32>::value
      || platform::is_same<typename Operator::MathOperator,
                           arch::OpMultiplyAddComplexFastF32>::value) {
      accum = plus_accum(accum, tmp_accum);
    }

    if (SharedMemoryClear == SharedMemoryClearOption::kZfill) {
      // commit and drain all pending and predicated LDGSTS pnz from the GEMM mainloop
      mctlass::arch::cp_async_fence();
      mctlass::arch::cp_async_wait<0>();
      if constexpr (!MacaSpecialTag) __syncthreads();
    }

  }
};

// mcTlass: problem: TN && stage!=2
//          (SmemIteratorA_::Layout == MacaRowMajorTensorOpMultiplicandCpAsyncCrosswise) &&
//          (SmemIteratorB_::Layout == MacaColumnMajorTensorOpMultiplicandCpAsyncCrosswise) &&
//          (stage != 2)
template <
    /// Size of the Gemm problem - concept: gemm::GemmShape<>
    typename Shape_,
    /// Iterates over tiles of A operand in global memory
    //  (concept: ReadableTileIterator | ForwardTileIterator |
    //  MaskedTileIterator)
    typename IteratorA_,
    /// Iterates over tiles of A operand in shared memory
    /// (concept: WriteableTileIterator | RandomAccessTileIterator)
    typename SmemIteratorA_,
    /// Cache operation for operand A
    mctlass::arch::CacheOperation::Kind CacheOpA,
    /// Iterates over tiles of B operand in global memory
    //  (concept: ReadableTileIterator | ForwardTileIterator |
    //  MaskedTileIterator)
    typename IteratorB_,
    /// Iterates over tiles of B operand in shared memory
    /// (concept: WriteableTileIterator | RandomAccessTileIterator)
    typename SmemIteratorB_,
    /// Cache operation for operand B
    mctlass::arch::CacheOperation::Kind CacheOpB,
    /// Data type of accumulator matrix
    typename ElementC_,
    /// Data type of accumulator matrix
    typename LayoutC_,
    /// Policy describing tuning details (concept: MmaPolicy)
    typename Policy_,
    /// Number of stages,
    int Stages,
    /// Use zfill or predicate for out-of-bound cp.async
    SharedMemoryClearOption SharedMemoryClear,
    /// Flag tag for special process for m16n16k16 f16
    bool MacaSpecialTag_
    >
class MmaMultistage <
  Shape_,
  IteratorA_,
  SmemIteratorA_,
  CacheOpA,
  IteratorB_,
  SmemIteratorB_,
  CacheOpB,
  ElementC_,
  LayoutC_,
  Policy_,
  Stages,
  SharedMemoryClear,
  MacaSpecialTag_,
  typename std::enable_if<
    ( (std::is_same<typename SmemIteratorA_::Layout,
        typename layout::MacaRowMajorTensorOpMultiplicandCpAsyncCrosswise<
          sizeof_bits<mctlass::half_t>::value, Shape_::kK>>::value &&
      std::is_same<typename SmemIteratorB_::Layout,
        typename layout::MacaColumnMajorTensorOpMultiplicandCpAsyncCrosswise<
          sizeof_bits<mctlass::half_t>::value, Shape_::kK>>::value) ||
      (std::is_same<typename SmemIteratorA_::Layout,
        typename layout::MacaRowMajorTensorOpMultiplicandCpAsyncCrosswise<
          sizeof_bits<int8_t>::value, Shape_::kK>>::value &&
      std::is_same<typename SmemIteratorB_::Layout,
        typename layout::MacaColumnMajorTensorOpMultiplicandCpAsyncCrosswise<
          sizeof_bits<int8_t>::value, Shape_::kK>>::value)||
      (std::is_same<typename SmemIteratorA_::Layout,
        typename layout::MacaRowMajorTensorOpMultiplicandCpAsyncCrosswise<
          sizeof_bits<float>::value, Shape_::kK>>::value &&
      std::is_same<typename SmemIteratorB_::Layout,
        typename layout::MacaColumnMajorTensorOpMultiplicandCpAsyncCrosswise<
          sizeof_bits<float>::value, Shape_::kK>>::value)),
    bool>::type
  >:public MmaBase<Shape_, Policy_, Stages> {
public:
  ///< Base class
  using Base = MmaBase<Shape_, Policy_, Stages>;
  ///< Size of the Gemm problem - concept: gemm::GemmShape<>
  using Shape = Shape_;
  ///< Iterates over tiles of A operand in global memory
  using IteratorA = IteratorA_;
  ///< Iterates over tiles of B operand in global memory
  using IteratorB = IteratorB_;
  ///< Data type of accumulator matrix
  using ElementC = ElementC_;
  ///< Layout of accumulator matrix
  using LayoutC = LayoutC_;
  ///< Policy describing tuning details
  using Policy = Policy_;

  using SmemIteratorA = SmemIteratorA_;
  using SmemIteratorB = SmemIteratorB_;

  static mctlass::arch::CacheOperation::Kind const kCacheOpA = CacheOpA;
  static mctlass::arch::CacheOperation::Kind const kCacheOpB = CacheOpB;
  // static constexpr bool MacaSpecialTag = MacaSpecialTag_;
  static constexpr bool MacaSpecialTag = false;

  static constexpr bool CrossOffsetA =
    platform::is_same<layout::MacaRowMajorTensorOpMultiplicandCpAsyncCrosswise<
      sizeof_bits<typename IteratorA::Element>::value, Shape::kK>,
    typename SmemIteratorA::Layout>::value;
  static constexpr bool CrossOffsetB =
    platform::is_same<layout::MacaColumnMajorTensorOpMultiplicandCpAsyncCrosswise<
      sizeof_bits<typename IteratorB::Element>::value, Shape::kK>,
    typename SmemIteratorB::Layout>::value;

  static_assert(Shape::kK % (512 / sizeof_bits<typename IteratorB::Element>::value) == 0,
      "Crosswise kContiguous must be 512bit aligned");
  //
  // Dependent types
  //

  /// Fragment of accumulator tile
  using FragmentC = typename Policy::Operator::FragmentC;

  /// Warp-level Mma
  using Operator = typename Policy::Operator;

  /// Minimum architecture is Sm80 to support cp.async
  using ArchTag = arch::Sm80;

  /// Complex transform on A operand
  static ComplexTransform const kTransformA = Operator::kTransformA;

  /// Complex transform on B operand
  static ComplexTransform const kTransformB = Operator::kTransformB;

  static int const kSrcBytesB = sizeof_bits<typename IteratorB::Element>::value *
                                IteratorB::ThreadMap::kElementsPerAccess /
                                IteratorB::kAccessesPerVector / 8;
  static int const kSrcBytesA = sizeof_bits<typename IteratorA::Element>::value *
                                IteratorA::ThreadMap::kElementsPerAccess /
                                IteratorA::kAccessesPerVector / 8;
  using AccessTypeB  = Array<uint8_t, kSrcBytesB>;

  /// Internal structure exposed for introspection.
  struct Detail {

    /// Number of cp.async instructions to load one stage of operand A
    static int const AsyncCopyIterationsPerStageA =
        IteratorA::ThreadMap::Iterations::kCount;

    /// Number of cp.async instructions to load one stage of operand B
    static int const AsyncCopyIterationsPerStageB =
        IteratorB::ThreadMap::Iterations::kCount;

    /// Number of stages
    static int const kStages = Stages;

    /// Number of cp.async instructions to load on group of operand A
    static int const kAccessesPerGroupA =
        (AsyncCopyIterationsPerStageA + Base::kWarpGemmIterations - 1) / Base::kWarpGemmIterations;

    /// Number of cp.async instructions to load on group of operand B
    static int const kAccessesPerGroupB =
        (AsyncCopyIterationsPerStageB + Base::kWarpGemmIterations - 1) / Base::kWarpGemmIterations;
  };

 private:

  using WarpLoadedFragmentA = typename Operator::FragmentA;
  using WarpLoadedFragmentB = typename Operator::FragmentB;
  using WarpTransformedFragmentA = typename Operator::TransformedFragmentA;
  using WarpTransformedFragmentB = typename Operator::TransformedFragmentB;

 private:

  //
  // Data members
  //

  /// Iterator to write threadblock-scoped tile of A operand to shared memory
  SmemIteratorA smem_iterator_A_;

  /// Iterator to write threadblock-scoped tile of B operand to shared memory
  SmemIteratorB smem_iterator_B_;

  int cross_offset_A_;
  int cross_offset_B_;

public:

  /// Construct from tensor references
  MCTLASS_DEVICE
  MmaMultistage(
      ///< Shared storage needed for internal use by threadblock-scoped GEMM
      typename Base::SharedStorage &shared_storage,
      ///< ID within the threadblock
      int thread_idx,
      ///< ID of warp
      int warp_idx,
      ///< ID of each thread within a warp
      int lane_idx,
      int problem_size_k = 0,
      typename IteratorA::Element *ptr_A = nullptr,
      typename IteratorA::Element *ptr_B = nullptr
    ):
      Base(shared_storage, thread_idx, warp_idx, lane_idx),
      smem_iterator_A_(shared_storage.operand_A_ref(), thread_idx),
      smem_iterator_B_(shared_storage.operand_B_ref(), thread_idx)
  {
    int bid = lane_idx & 0x7;
    int group_id = (lane_idx >> 3) & 0x3;
    int cross_target = bid ^ group_id;
    int cross_offset_ = cross_target - bid;
    cross_offset_A_ = CrossOffsetA ? cross_offset_ : 0;
    cross_offset_B_ = CrossOffsetB ? cross_offset_ : 0;

    // Compute warp location within threadblock tile by mapping the warp_id to
    // three coordinates:
    //   _m: the warp's position within the threadblock along the M dimension
    //   _n: the warp's position within the threadblock along the N dimension
    //   _k: the warp's position within the threadblock along the K dimension

    int warp_idx_mn = warp_idx % (Base::WarpCount::kM * Base::WarpCount::kN);
    int warp_idx_k = warp_idx / (Base::WarpCount::kM * Base::WarpCount::kN);

    int warp_idx_m = warp_idx_mn % Base::WarpCount::kM;
    int warp_idx_n = warp_idx_mn / Base::WarpCount::kM;

    // Add per-warp offsets in units of warp-level tiles
    this->warp_tile_iterator_A_.add_tile_offset(
        {warp_idx_m, Base::kWarpGemmIterations * warp_idx_k});
    this->warp_tile_iterator_B_.add_tile_offset(
        {Base::kWarpGemmIterations * warp_idx_k, warp_idx_n});
  }

  MCTLASS_DEVICE
  void copy_tiles_and_advance(IteratorA &iterator_A, IteratorB &iterator_B,
                              int group_start_A = 0, int group_start_B = 0) {
    iterator_A.set_iteration_index(group_start_A *
                                   IteratorA::kAccessesPerVector);
    this->smem_iterator_A_.set_iteration_index(group_start_A);

    // Async Copy for operand A
    MCTLASS_PRAGMA_UNROLL
    for (int j = 0; j < Detail::kAccessesPerGroupA; ++j) {
      if (group_start_A + j < Detail::AsyncCopyIterationsPerStageA) {
        typename IteratorA::AccessType *dst_ptr[IteratorA::kAccessesPerVector];
        typename IteratorA::AccessType *gmem_ptr[IteratorA::kAccessesPerVector];
        bool valid_tag[IteratorA::kAccessesPerVector];
        MCTLASS_PRAGMA_UNROLL
        for (int v = 0; v < IteratorA::kAccessesPerVector; ++v) {
          dst_ptr[v] = reinterpret_cast<typename IteratorA::AccessType *>(this->smem_iterator_A_.get()) + v - cross_offset_A_;
          gmem_ptr[v] = iterator_A.get() + cross_offset_A_;
          valid_tag[v] = iterator_A.valid();
          ++iterator_A;
        }

        MCTLASS_PRAGMA_UNROLL
        for (int v = 0; v < IteratorA::kAccessesPerVector; ++v) {
          if constexpr (SharedMemoryClear == SharedMemoryClearOption::kZfill) {
            mctlass::arch::cp_async_zfill<kSrcBytesA, kCacheOpA>(
                dst_ptr[v], gmem_ptr[v], valid_tag[v]);
          } else {
            mctlass::arch::cp_async<kSrcBytesA, kCacheOpA>(
                dst_ptr[v], gmem_ptr[v], valid_tag[v]);
          }
        }

        ++this->smem_iterator_A_;
      }
    }

    iterator_B.set_iteration_index(group_start_B *
                                   IteratorB::kAccessesPerVector);
    this->smem_iterator_B_.set_iteration_index(group_start_B);

    // Async Copy for operand B
    MCTLASS_PRAGMA_UNROLL
    for (int j = 0; j < Detail::kAccessesPerGroupB; ++j) {
      if (group_start_B + j < Detail::AsyncCopyIterationsPerStageB) {

        typename IteratorB::AccessType *dst_ptr[IteratorB::kAccessesPerVector];
        typename IteratorB::AccessType *gmem_ptr[IteratorB::kAccessesPerVector];
        bool valid_tag[IteratorB::kAccessesPerVector];

        MCTLASS_PRAGMA_UNROLL
        for (int v = 0; v < IteratorB::kAccessesPerVector; ++v) {
          dst_ptr[v] = reinterpret_cast<typename IteratorB::AccessType *>(this->smem_iterator_B_.get()) + v - cross_offset_B_;
          gmem_ptr[v] = iterator_B.get() + cross_offset_B_;
          valid_tag[v] = iterator_B.valid();

          ++iterator_B;
        }

        MCTLASS_PRAGMA_UNROLL
        for (int v = 0; v < IteratorB::kAccessesPerVector; ++v) {
          if (SharedMemoryClear == SharedMemoryClearOption::kZfill) {
            mctlass::arch::cp_async_zfill<kSrcBytesB, kCacheOpB>(
                dst_ptr[v], gmem_ptr[v], valid_tag[v]);
          } else {
            mctlass::arch::cp_async<kSrcBytesB, kCacheOpB>(
                dst_ptr[v], gmem_ptr[v], valid_tag[v]);
          }
        }
        ++this->smem_iterator_B_;
      }
    }
  }

  /// Perform a threadblock-scoped matrix multiply-accumulate
  MCTLASS_DEVICE
  void operator()(
      ///< problem size of GEMM
      int gemm_k_iterations,
      ///< destination accumulator tile
      FragmentC &accum,
      ///< iterator over A operand in global memory
      IteratorA iterator_A,
      ///< iterator over B operand in global memory
      IteratorB iterator_B,
      ///< initial value of accumulator
      FragmentC const &src_accum) {
    //
    // Prologue
    //
    static int const arrive_gvm_cnt = Detail::AsyncCopyIterationsPerStageA +
             Detail::AsyncCopyIterationsPerStageB;
    // Issue several complete stages
    MCTLASS_PRAGMA_UNROLL
    for (int stage = 0; stage < Base::kStages - 1;
         ++stage, --gemm_k_iterations) {

      iterator_A.clear_mask(gemm_k_iterations == 0);
      iterator_B.clear_mask(gemm_k_iterations == 0);

      iterator_A.set_iteration_index(0);
      this->smem_iterator_A_.set_iteration_index(0);

      // Async Copy for operand A
      MCTLASS_PRAGMA_UNROLL
      for (int j = 0; j < Detail::AsyncCopyIterationsPerStageA; ++j) {
        typename IteratorA::AccessType *dst_ptr[IteratorA::kAccessesPerVector];
        typename IteratorA::AccessType *gmem_ptr[IteratorA::kAccessesPerVector];
        bool valid_tag[IteratorA::kAccessesPerVector];
        MCTLASS_PRAGMA_UNROLL
        for (int v = 0; v < IteratorA::kAccessesPerVector; ++v) {
          dst_ptr[v] = reinterpret_cast<typename IteratorA::AccessType *>(this->smem_iterator_A_.get()) + v - cross_offset_A_;
          gmem_ptr[v] = iterator_A.get() + cross_offset_A_;
          valid_tag[v] = iterator_A.valid();
          ++iterator_A;
        }

        MCTLASS_PRAGMA_UNROLL
        for (int v = 0; v < IteratorA::kAccessesPerVector; ++v) {
          mctlass::arch::cp_async_zfill<kSrcBytesA, kCacheOpA>(
            dst_ptr[v], gmem_ptr[v], valid_tag[v]);
        }

        ++this->smem_iterator_A_;
      }

      iterator_B.set_iteration_index(0);
      this->smem_iterator_B_.set_iteration_index(0);

      // Async Copy for operand B
      MCTLASS_PRAGMA_UNROLL
      for (int j = 0; j < Detail::AsyncCopyIterationsPerStageB; ++j) {
        typename IteratorB::AccessType *dst_ptr[IteratorB::kAccessesPerVector];
        typename IteratorB::AccessType *gmem_ptr[IteratorB::kAccessesPerVector];
        bool valid_tag[IteratorB::kAccessesPerVector];

        MCTLASS_PRAGMA_UNROLL
        for (int v = 0; v < IteratorB::kAccessesPerVector; ++v) {
          dst_ptr[v] = reinterpret_cast<typename IteratorB::AccessType *>(this->smem_iterator_B_.get()) + v - cross_offset_B_;
          gmem_ptr[v] = iterator_B.get() + cross_offset_B_;
          valid_tag[v] = iterator_B.valid();

          ++iterator_B;
        }

        MCTLASS_PRAGMA_UNROLL
        for (int v = 0; v < IteratorB::kAccessesPerVector; ++v) {
          // @TODO: it need to be filled with zeros while valid_tag[v] is false
          if constexpr (kCacheOpB == mctlass::arch::CacheOperation::Builtin) {
            mctlass::arch::cp_async<kSrcBytesB, kCacheOpB>(
              dst_ptr[v], gmem_ptr[v], valid_tag[v]);
          } else {
            mctlass::arch::cp_async_zfill<kSrcBytesB, kCacheOpB>(
              dst_ptr[v], gmem_ptr[v], valid_tag[v]);
          }
        }

        ++this->smem_iterator_B_;
      }

      // Move to the next stage
      iterator_A.add_tile_offset({0, 1});
      iterator_B.add_tile_offset({1, 0});

      this->smem_iterator_A_.add_tile_offset({0, 1});
      this->smem_iterator_B_.add_tile_offset({1, 0});

      // Defines the boundary of a stage of cp.async.
      mctlass::arch::cp_async_fence();
    }

    // Perform accumulation in the 'd' output operand
    accum = src_accum;

    //
    // Clear the remaining tiles of SMEM. This is a functional requirement for some kernels
    // so that all accumulator elements outside the GEMM footprint are zero.
    //

    if (SharedMemoryClear == SharedMemoryClearOption::kClearLastStage) {

      /// Iterator to write threadblock-scoped tile of A operand to shared memory
      SmemIteratorA last_smem_iterator_A(this->smem_iterator_A_);

      typename IteratorA::AccessType zero_A;
      zero_A.clear();

      last_smem_iterator_A.set_iteration_index(0);

      // Async Copy for operand A
      MCTLASS_PRAGMA_UNROLL
      for (int j = 0; j < Detail::AsyncCopyIterationsPerStageA; ++j) {

        typename IteratorA::AccessType *dst_ptr =
            reinterpret_cast<typename IteratorA::AccessType *>(
                last_smem_iterator_A.get());

        *dst_ptr = zero_A;

        ++last_smem_iterator_A;
      }

      /// Iterator to write threadblock-scoped tile of B operand to shared memory
      SmemIteratorB last_smem_iterator_B(this->smem_iterator_B_);
      typename IteratorB::AccessType zero_B;

      zero_B.clear();
      last_smem_iterator_B.set_iteration_index(0);

      // Async Copy for operand B
      MCTLASS_PRAGMA_UNROLL
      for (int j = 0; j < Detail::AsyncCopyIterationsPerStageB; ++j) {

        typename IteratorB::AccessType *dst_ptr =
            reinterpret_cast<typename IteratorB::AccessType *>(
                last_smem_iterator_B.get());

        *dst_ptr = zero_B;

        ++last_smem_iterator_B;
      }
    }

    // Waits until stages up to the previous (kStages-2)th stage have committed.
    // mctlass::arch::cp_async_wait<Base::kStages - 2>();
    // if constexpr (Syncthreads) __syncthreads();
    __builtin_mxc_arrive(64);
    __builtin_mxc_barrier();

    // Pair of fragments used to overlap shared memory loads and math
    // instructions
    WarpLoadedFragmentA warp_loaded_frag_A[2];
    WarpLoadedFragmentB warp_loaded_frag_B[2];
    WarpTransformedFragmentA warp_transformed_frag_A[2];
    WarpTransformedFragmentB warp_transformed_frag_B[2];

    Operator warp_mma;

    this->warp_tile_iterator_A_.set_kgroup_index(0);
    this->warp_tile_iterator_B_.set_kgroup_index(0);

    // this->warp_tile_iterator_A_.load(warp_loaded_frag_A[0]);
    // this->warp_tile_iterator_B_.load(warp_loaded_frag_B[0]);
    this->warp_tile_iterator_B_.load(warp_loaded_frag_B[0]);
    this->warp_tile_iterator_A_.load(warp_loaded_frag_A[0]);

    ++this->warp_tile_iterator_A_;
    ++this->warp_tile_iterator_B_;

    iterator_A.clear_mask(gemm_k_iterations == 0);
    iterator_B.clear_mask(gemm_k_iterations == 0);

    int smem_write_stage_idx = Base::kStages - 1;
    int smem_read_stage_idx = 0;

    warp_mma.transform(warp_transformed_frag_A[0], warp_transformed_frag_B[0],
                       warp_loaded_frag_A[0], warp_loaded_frag_B[0]);

    // tf32x3 kernels use staging accumulation. warp_mma uses a temporary
    // accumulator and this temporary accumulator is added to the final
    // accumulator once in every mainloop iteration.
    plus<FragmentC> plus_accum;

    FragmentC tmp_accum;

    if (platform::is_same<typename Operator::MathOperator,
                          arch::OpMultiplyAddFastF32>::value
      || platform::is_same<typename Operator::MathOperator,
                          arch::OpMultiplyAddComplexFastF32>::value) {

      tmp_accum.clear();
    }

    //
    // Mainloop
    //

    MCTLASS_GEMM_LOOP
    for (; gemm_k_iterations > (-Base::kStages + 1);) {
      //
      // Loop over GEMM K dimension
      //

      // Computes a warp-level GEMM on data held in shared memory
      // Each "warp_mma_k" refers to a warp-level matrix multiply-accumulate
      MCTLASS_PRAGMA_UNROLL
      for (int warp_mma_k = 0; warp_mma_k < Base::kWarpGemmIterations;
           ++warp_mma_k) {

        // Issue global->shared copies for the this stage
        if (warp_mma_k < Base::kWarpGemmIterations - 1) {
          int group_start_iteration_A, group_start_iteration_B;

          group_start_iteration_A = warp_mma_k * Detail::kAccessesPerGroupA;
          group_start_iteration_B = warp_mma_k * Detail::kAccessesPerGroupB;
          if constexpr (Base::kStages == 2) {
            __builtin_mxc_arrive(64);
          }
          copy_tiles_and_advance(iterator_A, iterator_B, group_start_iteration_A,
                               group_start_iteration_B);
        }

        // Load warp-level tiles from shared memory, wrapping to k offset if
        // this is the last group as the case may be.

        this->warp_tile_iterator_A_.set_kgroup_index((warp_mma_k + 1) % Base::kWarpGemmIterations);
        this->warp_tile_iterator_B_.set_kgroup_index((warp_mma_k + 1) % Base::kWarpGemmIterations);

        // this->warp_tile_iterator_A_.load(warp_loaded_frag_A[(warp_mma_k + 1) % 2]);
        // this->warp_tile_iterator_B_.load(warp_loaded_frag_B[(warp_mma_k + 1) % 2]);

        if (warp_mma_k > 0)
          warp_mma.transform(warp_transformed_frag_A[warp_mma_k % 2],
                             warp_transformed_frag_B[warp_mma_k % 2],
                             warp_loaded_frag_A[warp_mma_k % 2],
                             warp_loaded_frag_B[warp_mma_k % 2]);

        if (platform::is_same<typename Operator::MathOperator,
                              arch::OpMultiplyAddFastF32>::value
          || platform::is_same<typename Operator::MathOperator,
                               arch::OpMultiplyAddComplexFastF32>::value) {

          warp_mma(
            tmp_accum,
            warp_transformed_frag_A[warp_mma_k % 2],
            warp_transformed_frag_B[warp_mma_k % 2],
            tmp_accum
          );
          // At present, PTX: "cp.async.commit_group", "cp.async.wait_group", "cp.async.wait_all" in memory_sm80.h is not implemented on maca
	  // gemm/device test: SM80_Device_Her2k_cf32n_cf32n_l_tensor_op_fast_f32.64x64x16_32x32x16 failed
	  // therefore, block the segment accumulation of fast branch.
	  // after PTX is implemented ,the following code should be opened again(2023.03.06).
         // if (warp_mma_k == 0) {
         //   accum = plus_accum(accum, tmp_accum);
         //   tmp_accum.clear();
         // }
        } else {
          warp_mma(
            accum,
            warp_transformed_frag_A[warp_mma_k % 2],
            warp_transformed_frag_B[warp_mma_k % 2],
            accum
          );
        }

        this->warp_tile_iterator_A_.load(warp_loaded_frag_A[(warp_mma_k + 1) % 2]);
        this->warp_tile_iterator_B_.load(warp_loaded_frag_B[(warp_mma_k + 1) % 2]);

        ++this->warp_tile_iterator_A_;
        ++this->warp_tile_iterator_B_;

        // // Issue global->shared copies for the this stage
        // if (warp_mma_k < Base::kWarpGemmIterations - 1) {
        //   int group_start_iteration_A, group_start_iteration_B;

        //   group_start_iteration_A = warp_mma_k * Detail::kAccessesPerGroupA;
        //   group_start_iteration_B = warp_mma_k * Detail::kAccessesPerGroupB;

        //   copy_tiles_and_advance(iterator_A, iterator_B, group_start_iteration_A,
        //                        group_start_iteration_B);
        // }

        if (warp_mma_k + 2 == Base::kWarpGemmIterations) {
          int group_start_iteration_A, group_start_iteration_B;
          group_start_iteration_A =
              (warp_mma_k + 1) * Detail::kAccessesPerGroupA;
          group_start_iteration_B =
              (warp_mma_k + 1) * Detail::kAccessesPerGroupB;
          if constexpr (Base::kStages == 2) {
            __builtin_mxc_arrive(64);
          }
          copy_tiles_and_advance(iterator_A, iterator_B, group_start_iteration_A,
                               group_start_iteration_B);


          // Inserts a memory fence between stages of cp.async instructions.
          // mctlass::arch::cp_async_fence();

          // // Waits until stages up to the previous (kStages-2)th stage have committed.
          // arch::cp_async_wait<Base::kStages - 2>();
          // if constexpr (Syncthreads) __syncthreads();
          __builtin_mxc_barrier();

          // Move to the next stage
          iterator_A.add_tile_offset({0, 1});
          iterator_B.add_tile_offset({1, 0});

          this->smem_iterator_A_.add_tile_offset({0, 1});
          this->smem_iterator_B_.add_tile_offset({1, 0});

          // Add negative offsets to return iterators to the 'start' of the
          // circular buffer in shared memory
          if (smem_write_stage_idx == (Base::kStages - 1)) {
            this->smem_iterator_A_.add_tile_offset({0, -Base::kStages});
            this->smem_iterator_B_.add_tile_offset({-Base::kStages, 0});
            smem_write_stage_idx = 0;
          } else {
            ++smem_write_stage_idx;
          }

          if (smem_read_stage_idx == (Base::kStages - 1)) {
            if (CrossOffsetA) {
              this->warp_tile_iterator_A_.add_tile_offset(
                {-Base::kStages * Base::WarpCount::kM, 0});
            } else {
              this->warp_tile_iterator_A_.add_tile_offset(
                {-Base::kStages * Base::WarpCount::kM, 0});
            }
            if (CrossOffsetB) {
              this->warp_tile_iterator_B_.add_tile_offset(
                {0, -Base::kStages * Base::WarpCount::kN});
            } else {
              this->warp_tile_iterator_B_.add_tile_offset(
                  {-Base::kStages * Policy::kPartitionsK *
                      Base::kWarpGemmIterations, 0});
            }
            smem_read_stage_idx = 0;
          } else {
            ++smem_read_stage_idx;
          }

          --gemm_k_iterations;
          iterator_A.clear_mask(gemm_k_iterations == 0);
          iterator_B.clear_mask(gemm_k_iterations == 0);
        }

        // Do any conversions feeding the first stage at the end of the loop so
        // we can start right away on mma instructions
        if (warp_mma_k + 1 == Base::kWarpGemmIterations)
          warp_mma.transform(warp_transformed_frag_A[(warp_mma_k + 1) % 2],
                             warp_transformed_frag_B[(warp_mma_k + 1) % 2],
                             warp_loaded_frag_A[(warp_mma_k + 1) % 2],
                             warp_loaded_frag_B[(warp_mma_k + 1) % 2]);

          // if (gemm_k_iterations > 0) {
          //   __builtin_mxc_arrive(64 + arrive_gvm_cnt);
          // } else {
        __builtin_mxc_arrive(64);
          // }
          // __builtin_mxc_barrier();
      }

    }

    if (platform::is_same<typename Operator::MathOperator,
                          arch::OpMultiplyAddFastF32>::value
      || platform::is_same<typename Operator::MathOperator,
                           arch::OpMultiplyAddComplexFastF32>::value) {
      accum = plus_accum(accum, tmp_accum);
    }

    if (SharedMemoryClear == SharedMemoryClearOption::kZfill) {
      // commit and drain all pending and predicated LDGSTS pnz from the GEMM mainloop
      mctlass::arch::cp_async_fence();
      mctlass::arch::cp_async_wait<0>();
      if constexpr (!MacaSpecialTag) __syncthreads();
    }

  }
};

// mcTlass: problem: NN && stage!=2
//          (SmemIteratorA_::Layout == MacaRowMajorTensorOpMultiplicandCpAsyncCrosswise) &&
//          (SmemIteratorB_::Layout == MacaRowMajorTensorOpMultiplicandCongruous) &&
//          (stage != 2)
template <
    /// Size of the Gemm problem - concept: gemm::GemmShape<>
    typename Shape_,
    /// Iterates over tiles of A operand in global memory
    //  (concept: ReadableTileIterator | ForwardTileIterator |
    //  MaskedTileIterator)
    typename IteratorA_,
    /// Iterates over tiles of A operand in shared memory
    /// (concept: WriteableTileIterator | RandomAccessTileIterator)
    typename SmemIteratorA_,
    /// Cache operation for operand A
    mctlass::arch::CacheOperation::Kind CacheOpA,
    /// Iterates over tiles of B operand in global memory
    //  (concept: ReadableTileIterator | ForwardTileIterator |
    //  MaskedTileIterator)
    typename IteratorB_,
    /// Iterates over tiles of B operand in shared memory
    /// (concept: WriteableTileIterator | RandomAccessTileIterator)
    typename SmemIteratorB_,
    /// Cache operation for operand B
    mctlass::arch::CacheOperation::Kind CacheOpB,
    /// Data type of accumulator matrix
    typename ElementC_,
    /// Data type of accumulator matrix
    typename LayoutC_,
    /// Policy describing tuning details (concept: MmaPolicy)
    typename Policy_,
    /// Number of stages,
    int Stages,
    /// Use zfill or predicate for out-of-bound cp.async
    SharedMemoryClearOption SharedMemoryClear,
    /// Flag tag for special process for m16n16k16 f16
    bool MacaSpecialTag_
    >
class MmaMultistage <
  Shape_,
  IteratorA_,
  SmemIteratorA_,
  CacheOpA,
  IteratorB_,
  SmemIteratorB_,
  CacheOpB,
  ElementC_,
  LayoutC_,
  Policy_,
  Stages,
  SharedMemoryClear,
  MacaSpecialTag_,
  typename std::enable_if<
    ( (std::is_same<typename SmemIteratorA_::Layout,
        typename layout::MacaRowMajorTensorOpMultiplicandCpAsyncCrosswise<
          sizeof_bits<mctlass::half_t>::value, Shape_::kK>>::value &&
      std::is_same<typename SmemIteratorB_::Layout,
        typename layout::MacaRowMajorTensorOpMultiplicandCongruous<
          sizeof_bits<mctlass::half_t>::value, int(128 / sizeof(mctlass::half_t))>>::value) ||
      (std::is_same<typename SmemIteratorA_::Layout,
        typename layout::MacaRowMajorTensorOpMultiplicandCpAsyncCrosswise<
          sizeof_bits<int8_t>::value, Shape_::kK>>::value &&
      std::is_same<typename SmemIteratorB_::Layout,
        typename layout::MacaRowMajorTensorOpMultiplicandCongruous<
          sizeof_bits<int8_t>::value, int(128 / sizeof(int8_t))>>::value)),
    bool>::type
  >:public MmaBase<Shape_, Policy_, Stages> {
public:
  ///< Base class
  using Base = MmaBase<Shape_, Policy_, Stages>;
  ///< Size of the Gemm problem - concept: gemm::GemmShape<>
  using Shape = Shape_;
  ///< Iterates over tiles of A operand in global memory
  using IteratorA = IteratorA_;
  ///< Iterates over tiles of B operand in global memory
  using IteratorB = IteratorB_;
  ///< Data type of accumulator matrix
  using ElementC = ElementC_;
  ///< Layout of accumulator matrix
  using LayoutC = LayoutC_;
  ///< Policy describing tuning details
  using Policy = Policy_;

  using SmemIteratorA = SmemIteratorA_;
  using SmemIteratorB = SmemIteratorB_;

  static mctlass::arch::CacheOperation::Kind const kCacheOpA = CacheOpA;
  static mctlass::arch::CacheOperation::Kind const kCacheOpB = CacheOpB;
  // static constexpr bool MacaSpecialTag = MacaSpecialTag_;
  static constexpr bool MacaSpecialTag = false;

  static constexpr bool CrossOffset =
    platform::is_same<layout::MacaRowMajorTensorOpMultiplicandCpAsyncCrosswise<
      sizeof_bits<typename IteratorA::Element>::value, Shape::kK>,
    typename SmemIteratorA::Layout>::value;

  static_assert(Shape::kK % (512 / sizeof_bits<typename IteratorA::Element>::value) == 0,
      "Crosswise Mat A kContiguous must be 512bit aligned");
  static_assert(Shape::kN % (512 / sizeof_bits<typename IteratorA::Element>::value) == 0,
      "Crosswise Mat B kContiguous must be 512bit aligned");
  //
  // Dependent types
  //

  /// Fragment of accumulator tile
  using FragmentC = typename Policy::Operator::FragmentC;

  /// Warp-level Mma
  using Operator = typename Policy::Operator;

  /// Minimum architecture is Sm80 to support cp.async
  using ArchTag = arch::Sm80;

  /// Complex transform on A operand
  static ComplexTransform const kTransformA = Operator::kTransformA;

  /// Complex transform on B operand
  static ComplexTransform const kTransformB = Operator::kTransformB;

  static int const kSrcBytesB = sizeof_bits<typename IteratorB::Element>::value *
                                IteratorB::ThreadMap::kElementsPerAccess /
                                IteratorB::kAccessesPerVector / 8;
  static int const kSrcBytesA = sizeof_bits<typename IteratorA::Element>::value *
                                IteratorA::ThreadMap::kElementsPerAccess /
                                IteratorA::kAccessesPerVector / 8;
  using AccessTypeB  = Array<uint8_t, kSrcBytesB>;

  /// Internal structure exposed for introspection.
  struct Detail {

    /// Number of cp.async instructions to load one stage of operand A
    static int const AsyncCopyIterationsPerStageA =
        IteratorA::ThreadMap::Iterations::kCount;

    /// Number of cp.async instructions to load one stage of operand B
    static int const AsyncCopyIterationsPerStageB =
        IteratorB::ThreadMap::Iterations::kCount;

    /// Number of stages
    static int const kStages = Stages;

    /// Number of cp.async instructions to load on group of operand A
    static int const kAccessesPerGroupA =
        (AsyncCopyIterationsPerStageA + Base::kWarpGemmIterations - 1) / Base::kWarpGemmIterations;

    /// Number of cp.async instructions to load on group of operand B
    static int const kAccessesPerGroupB =
        (AsyncCopyIterationsPerStageB + Base::kWarpGemmIterations - 1) / Base::kWarpGemmIterations;
  };

 private:

  using WarpLoadedFragmentA = typename Operator::FragmentA;
  using WarpLoadedFragmentB = typename Operator::FragmentB;
  using WarpTransformedFragmentA = typename Operator::TransformedFragmentA;
  using WarpTransformedFragmentB = typename Operator::TransformedFragmentB;

  using WarpLoadedFragmentB_int = typename Operator::IteratorB::Fragment_int;

 private:

  //
  // Data members
  //

  /// Iterator to write threadblock-scoped tile of A operand to shared memory
  SmemIteratorA smem_iterator_A_;

  /// Iterator to write threadblock-scoped tile of B operand to shared memory
  SmemIteratorB smem_iterator_B_;

  int cross_offset_;

public:

  /// Construct from tensor references
  MCTLASS_DEVICE
  MmaMultistage(
      ///< Shared storage needed for internal use by threadblock-scoped GEMM
      typename Base::SharedStorage &shared_storage,
      ///< ID within the threadblock
      int thread_idx,
      ///< ID of warp
      int warp_idx,
      ///< ID of each thread within a warp
      int lane_idx,
      int problem_size_k = 0,
      typename IteratorA::Element *ptr_A = nullptr,
      typename IteratorA::Element *ptr_B = nullptr
    ):
      Base(shared_storage, thread_idx, warp_idx, lane_idx),
      smem_iterator_A_(shared_storage.operand_A_ref(), thread_idx),
      smem_iterator_B_(shared_storage.operand_B_ref(), thread_idx)
  {
    if constexpr (CrossOffset) {
      int bid = lane_idx & 0x7;
      int group_id = (lane_idx >> 3) & 0x3;
      int cross_target = bid ^ group_id;
      cross_offset_ = cross_target - bid ;
    } else {
      cross_offset_ = 0;
    }

    // Compute warp location within threadblock tile by mapping the warp_id to
    // three coordinates:
    //   _m: the warp's position within the threadblock along the M dimension
    //   _n: the warp's position within the threadblock along the N dimension
    //   _k: the warp's position within the threadblock along the K dimension

    int warp_idx_mn = warp_idx % (Base::WarpCount::kM * Base::WarpCount::kN);
    int warp_idx_k = warp_idx / (Base::WarpCount::kM * Base::WarpCount::kN);

    int warp_idx_m = warp_idx_mn % Base::WarpCount::kM;
    int warp_idx_n = warp_idx_mn / Base::WarpCount::kM;

    // Add per-warp offsets in units of warp-level tiles
    this->warp_tile_iterator_A_.add_tile_offset(
        {warp_idx_m, Base::kWarpGemmIterations * warp_idx_k});
    this->warp_tile_iterator_B_.add_tile_offset(
        {Base::kWarpGemmIterations * warp_idx_k, warp_idx_n});
  }

  MCTLASS_DEVICE
  void copy_tiles_and_advance(IteratorA &iterator_A, IteratorB &iterator_B,
                              int group_start_A = 0, int group_start_B = 0) {
    iterator_A.set_iteration_index(group_start_A *
                                   IteratorA::kAccessesPerVector);
    this->smem_iterator_A_.set_iteration_index(group_start_A);

    // Async Copy for operand A
    MCTLASS_PRAGMA_UNROLL
    for (int j = 0; j < Detail::kAccessesPerGroupA; ++j) {
      if (group_start_A + j < Detail::AsyncCopyIterationsPerStageA) {
        typename IteratorA::AccessType *dst_ptr[IteratorA::kAccessesPerVector];
        typename IteratorA::AccessType *gmem_ptr[IteratorA::kAccessesPerVector];
        bool valid_tag[IteratorA::kAccessesPerVector];
        MCTLASS_PRAGMA_UNROLL
        for (int v = 0; v < IteratorA::kAccessesPerVector; ++v) {
          dst_ptr[v] = reinterpret_cast<typename IteratorA::AccessType *>(this->smem_iterator_A_.get()) + v - cross_offset_;
          gmem_ptr[v] = iterator_A.get() + cross_offset_;
          valid_tag[v] = iterator_A.valid();
          ++iterator_A;
        }

        MCTLASS_PRAGMA_UNROLL
        for (int v = 0; v < IteratorA::kAccessesPerVector; ++v) {
          if constexpr (SharedMemoryClear == SharedMemoryClearOption::kZfill) {
            mctlass::arch::cp_async_zfill<kSrcBytesA, kCacheOpA>(
                dst_ptr[v], gmem_ptr[v], valid_tag[v]);
          } else {
            mctlass::arch::cp_async<kSrcBytesA, kCacheOpA>(
                dst_ptr[v], gmem_ptr[v], valid_tag[v]);
          }
        }

        ++this->smem_iterator_A_;
      }
    }

    iterator_B.set_iteration_index(group_start_B *
                                   IteratorB::kAccessesPerVector);
    this->smem_iterator_B_.set_iteration_index(group_start_B);

    // Async Copy for operand B
    MCTLASS_PRAGMA_UNROLL
    for (int j = 0; j < Detail::kAccessesPerGroupB; ++j) {
      if (group_start_B + j < Detail::AsyncCopyIterationsPerStageB) {

        typename IteratorB::AccessType *dst_ptr[IteratorB::kAccessesPerVector];
        typename IteratorB::AccessType *gmem_ptr[IteratorB::kAccessesPerVector];
        bool valid_tag[IteratorB::kAccessesPerVector];

        MCTLASS_PRAGMA_UNROLL
        for (int v = 0; v < IteratorB::kAccessesPerVector; ++v) {
          dst_ptr[v] = reinterpret_cast<typename IteratorB::AccessType *>(this->smem_iterator_B_.get()) + v;
          gmem_ptr[v] = iterator_B.get();
          valid_tag[v] = iterator_B.valid();

          ++iterator_B;
        }

        MCTLASS_PRAGMA_UNROLL
        for (int v = 0; v < IteratorB::kAccessesPerVector; ++v) {
          if (SharedMemoryClear == SharedMemoryClearOption::kZfill) {
            mctlass::arch::cp_async_zfill<kSrcBytesB, kCacheOpB>(
                dst_ptr[v], gmem_ptr[v], valid_tag[v]);
          } else {
            mctlass::arch::cp_async<kSrcBytesB, kCacheOpB>(
                dst_ptr[v], gmem_ptr[v], valid_tag[v]);
          }
        }
        ++this->smem_iterator_B_;
      }
    }
  }

  /// Perform a threadblock-scoped matrix multiply-accumulate
  MCTLASS_DEVICE
  void operator()(
      ///< problem size of GEMM
      int gemm_k_iterations,
      ///< destination accumulator tile
      FragmentC &accum,
      ///< iterator over A operand in global memory
      IteratorA iterator_A,
      ///< iterator over B operand in global memory
      IteratorB iterator_B,
      ///< initial value of accumulator
      FragmentC const &src_accum) {
    //
    // Prologue
    //
    static int const arrive_gvm_cnt = Detail::AsyncCopyIterationsPerStageA +
             Detail::AsyncCopyIterationsPerStageB;
    // Issue several complete stages
    MCTLASS_PRAGMA_UNROLL
    for (int stage = 0; stage < Base::kStages - 1;
         ++stage, --gemm_k_iterations) {

      iterator_A.clear_mask(gemm_k_iterations == 0);
      iterator_B.clear_mask(gemm_k_iterations == 0);

      iterator_A.set_iteration_index(0);
      this->smem_iterator_A_.set_iteration_index(0);

      // Async Copy for operand A
      MCTLASS_PRAGMA_UNROLL
      for (int j = 0; j < Detail::AsyncCopyIterationsPerStageA; ++j) {
        typename IteratorA::AccessType *dst_ptr[IteratorA::kAccessesPerVector];
        typename IteratorA::AccessType *gmem_ptr[IteratorA::kAccessesPerVector];
        bool valid_tag[IteratorA::kAccessesPerVector];
        MCTLASS_PRAGMA_UNROLL
        for (int v = 0; v < IteratorA::kAccessesPerVector; ++v) {
          dst_ptr[v] = reinterpret_cast<typename IteratorA::AccessType *>(this->smem_iterator_A_.get()) + v - cross_offset_;
          gmem_ptr[v] = iterator_A.get() + cross_offset_;
          valid_tag[v] = iterator_A.valid();
          ++iterator_A;
        }

        MCTLASS_PRAGMA_UNROLL
        for (int v = 0; v < IteratorA::kAccessesPerVector; ++v) {
          mctlass::arch::cp_async_zfill<kSrcBytesA, kCacheOpA>(
            dst_ptr[v], gmem_ptr[v], valid_tag[v]);
        }

        ++this->smem_iterator_A_;
      }

      iterator_B.set_iteration_index(0);
      this->smem_iterator_B_.set_iteration_index(0);

      // Async Copy for operand B
      MCTLASS_PRAGMA_UNROLL
      for (int j = 0; j < Detail::AsyncCopyIterationsPerStageB; ++j) {
        typename IteratorB::AccessType *dst_ptr[IteratorB::kAccessesPerVector];
        typename IteratorB::AccessType *gmem_ptr[IteratorB::kAccessesPerVector];
        bool valid_tag[IteratorB::kAccessesPerVector];

        MCTLASS_PRAGMA_UNROLL
        for (int v = 0; v < IteratorB::kAccessesPerVector; ++v) {
          dst_ptr[v] = reinterpret_cast<typename IteratorB::AccessType *>(this->smem_iterator_B_.get()) + v;
          gmem_ptr[v] = iterator_B.get();
          valid_tag[v] = iterator_B.valid();

          ++iterator_B;
        }

        MCTLASS_PRAGMA_UNROLL
        for (int v = 0; v < IteratorB::kAccessesPerVector; ++v) {
          // @TODO: it need to be filled with zeros while valid_tag[v] is false
          if constexpr (kCacheOpB == mctlass::arch::CacheOperation::Builtin) {
            mctlass::arch::cp_async<kSrcBytesB, kCacheOpB>(
              dst_ptr[v], gmem_ptr[v], valid_tag[v]);
          } else {
            mctlass::arch::cp_async_zfill<kSrcBytesB, kCacheOpB>(
              dst_ptr[v], gmem_ptr[v], valid_tag[v]);
          }
        }

        ++this->smem_iterator_B_;
      }

      // Move to the next stage
      iterator_A.add_tile_offset({0, 1});
      iterator_B.add_tile_offset({1, 0});

      this->smem_iterator_A_.add_tile_offset({0, 1});
      this->smem_iterator_B_.add_tile_offset({1, 0});

      // Defines the boundary of a stage of cp.async.
      mctlass::arch::cp_async_fence();
    }

    // Perform accumulation in the 'd' output operand
    accum = src_accum;

    //
    // Clear the remaining tiles of SMEM. This is a functional requirement for some kernels
    // so that all accumulator elements outside the GEMM footprint are zero.
    //

    if (SharedMemoryClear == SharedMemoryClearOption::kClearLastStage) {

      /// Iterator to write threadblock-scoped tile of A operand to shared memory
      SmemIteratorA last_smem_iterator_A(this->smem_iterator_A_);

      typename IteratorA::AccessType zero_A;
      zero_A.clear();

      last_smem_iterator_A.set_iteration_index(0);

      // Async Copy for operand A
      MCTLASS_PRAGMA_UNROLL
      for (int j = 0; j < Detail::AsyncCopyIterationsPerStageA; ++j) {

        typename IteratorA::AccessType *dst_ptr =
            reinterpret_cast<typename IteratorA::AccessType *>(
                last_smem_iterator_A.get());

        *dst_ptr = zero_A;

        ++last_smem_iterator_A;
      }

      /// Iterator to write threadblock-scoped tile of B operand to shared memory
      SmemIteratorB last_smem_iterator_B(this->smem_iterator_B_);
      typename IteratorB::AccessType zero_B;

      zero_B.clear();
      last_smem_iterator_B.set_iteration_index(0);

      // Async Copy for operand B
      MCTLASS_PRAGMA_UNROLL
      for (int j = 0; j < Detail::AsyncCopyIterationsPerStageB; ++j) {

        typename IteratorB::AccessType *dst_ptr =
            reinterpret_cast<typename IteratorB::AccessType *>(
                last_smem_iterator_B.get());

        *dst_ptr = zero_B;

        ++last_smem_iterator_B;
      }
    }

    // Waits until stages up to the previous (kStages-2)th stage have committed.
    // mctlass::arch::cp_async_wait<Base::kStages - 2>();
    // if constexpr (Syncthreads) __syncthreads();
    __builtin_mxc_arrive(64);
    __builtin_mxc_barrier();

    // Pair of fragments used to overlap shared memory loads and math
    // instructions
    WarpLoadedFragmentA warp_loaded_frag_A[2];
    WarpLoadedFragmentB warp_loaded_frag_B[2];
    WarpTransformedFragmentA warp_transformed_frag_A[2];
    WarpTransformedFragmentB warp_transformed_frag_B[2];

    WarpLoadedFragmentB_int warp_loaded_frag_B_int[2];

    Operator warp_mma;

    this->warp_tile_iterator_A_.set_kgroup_index(0);
    this->warp_tile_iterator_B_.set_kgroup_index(0);

    // this->warp_tile_iterator_A_.load(warp_loaded_frag_A[0]);
    // this->warp_tile_iterator_B_.load(warp_loaded_frag_B[0]);
    this->warp_tile_iterator_B_.load(warp_loaded_frag_B_int[0]);
    this->warp_tile_iterator_A_.load(warp_loaded_frag_A[0]);
    this->warp_tile_iterator_B_.fragment_int_conversion(warp_loaded_frag_B_int[0], warp_loaded_frag_B[0]);

    ++this->warp_tile_iterator_A_;
    ++this->warp_tile_iterator_B_;

    iterator_A.clear_mask(gemm_k_iterations == 0);
    iterator_B.clear_mask(gemm_k_iterations == 0);

    int smem_write_stage_idx = Base::kStages - 1;
    int smem_read_stage_idx = 0;

    warp_mma.transform(warp_transformed_frag_A[0], warp_transformed_frag_B[0],
                       warp_loaded_frag_A[0], warp_loaded_frag_B[0]);

    // tf32x3 kernels use staging accumulation. warp_mma uses a temporary
    // accumulator and this temporary accumulator is added to the final
    // accumulator once in every mainloop iteration.
    plus<FragmentC> plus_accum;

    FragmentC tmp_accum;

    if (platform::is_same<typename Operator::MathOperator,
                          arch::OpMultiplyAddFastF32>::value
      || platform::is_same<typename Operator::MathOperator,
                          arch::OpMultiplyAddComplexFastF32>::value) {

      tmp_accum.clear();
    }

    //
    // Mainloop
    //

    MCTLASS_GEMM_LOOP
    for (; gemm_k_iterations > (-Base::kStages + 1);) {
      //
      // Loop over GEMM K dimension
      //

      // Computes a warp-level GEMM on data held in shared memory
      // Each "warp_mma_k" refers to a warp-level matrix multiply-accumulate
      MCTLASS_PRAGMA_UNROLL
      for (int warp_mma_k = 0; warp_mma_k < Base::kWarpGemmIterations;
           ++warp_mma_k) {

        // Issue global->shared copies for the this stage
        if (warp_mma_k < Base::kWarpGemmIterations - 1) {
          int group_start_iteration_A, group_start_iteration_B;

          group_start_iteration_A = warp_mma_k * Detail::kAccessesPerGroupA;
          group_start_iteration_B = warp_mma_k * Detail::kAccessesPerGroupB;
          if constexpr (Base::kStages == 2) {
            __builtin_mxc_arrive(64);
          }
          copy_tiles_and_advance(iterator_A, iterator_B, group_start_iteration_A,
                               group_start_iteration_B);
        }

        // Load warp-level tiles from shared memory, wrapping to k offset if
        // this is the last group as the case may be.

        this->warp_tile_iterator_A_.set_kgroup_index((warp_mma_k + 1) % Base::kWarpGemmIterations);
        this->warp_tile_iterator_B_.set_kgroup_index((warp_mma_k + 1) % Base::kWarpGemmIterations);

        // this->warp_tile_iterator_A_.load(warp_loaded_frag_A[(warp_mma_k + 1) % 2]);
        // this->warp_tile_iterator_B_.load(warp_loaded_frag_B[(warp_mma_k + 1) % 2]);
        this->warp_tile_iterator_B_.load(warp_loaded_frag_B_int[(warp_mma_k + 1) % 2]);

        if (warp_mma_k > 0)
          warp_mma.transform(warp_transformed_frag_A[warp_mma_k % 2],
                             warp_transformed_frag_B[warp_mma_k % 2],
                             warp_loaded_frag_A[warp_mma_k % 2],
                             warp_loaded_frag_B[warp_mma_k % 2]);

        if (platform::is_same<typename Operator::MathOperator,
                              arch::OpMultiplyAddFastF32>::value
          || platform::is_same<typename Operator::MathOperator,
                               arch::OpMultiplyAddComplexFastF32>::value) {

          warp_mma(
            tmp_accum,
            warp_transformed_frag_A[warp_mma_k % 2],
            warp_transformed_frag_B[warp_mma_k % 2],
            tmp_accum
          );
          // At present, PTX: "cp.async.commit_group", "cp.async.wait_group", "cp.async.wait_all" in memory_sm80.h is not implemented on maca
	  // gemm/device test: SM80_Device_Her2k_cf32n_cf32n_l_tensor_op_fast_f32.64x64x16_32x32x16 failed
	  // therefore, block the segment accumulation of fast branch.
	  // after PTX is implemented ,the following code should be opened again(2023.03.06).
         // if (warp_mma_k == 0) {
         //   accum = plus_accum(accum, tmp_accum);
         //   tmp_accum.clear();
         // }
        } else {
          warp_mma(
            accum,
            warp_transformed_frag_A[warp_mma_k % 2],
            warp_transformed_frag_B[warp_mma_k % 2],
            accum
          );
        }

        this->warp_tile_iterator_A_.load(warp_loaded_frag_A[(warp_mma_k + 1) % 2]);
        this->warp_tile_iterator_B_.fragment_int_conversion(warp_loaded_frag_B_int[(warp_mma_k + 1) % 2],
                warp_loaded_frag_B[(warp_mma_k +1) % 2]);

        ++this->warp_tile_iterator_A_;
        ++this->warp_tile_iterator_B_;

        // // Issue global->shared copies for the this stage
        // if (warp_mma_k < Base::kWarpGemmIterations - 1) {
        //   int group_start_iteration_A, group_start_iteration_B;

        //   group_start_iteration_A = warp_mma_k * Detail::kAccessesPerGroupA;
        //   group_start_iteration_B = warp_mma_k * Detail::kAccessesPerGroupB;

        //   copy_tiles_and_advance(iterator_A, iterator_B, group_start_iteration_A,
        //                        group_start_iteration_B);
        // }

        if (warp_mma_k + 2 == Base::kWarpGemmIterations) {
          int group_start_iteration_A, group_start_iteration_B;
          group_start_iteration_A =
              (warp_mma_k + 1) * Detail::kAccessesPerGroupA;
          group_start_iteration_B =
              (warp_mma_k + 1) * Detail::kAccessesPerGroupB;
          if constexpr (Base::kStages == 2) {
            __builtin_mxc_arrive(64);
          }
          copy_tiles_and_advance(iterator_A, iterator_B, group_start_iteration_A,
                               group_start_iteration_B);


          // Inserts a memory fence between stages of cp.async instructions.
          // mctlass::arch::cp_async_fence();

          // // Waits until stages up to the previous (kStages-2)th stage have committed.
          // arch::cp_async_wait<Base::kStages - 2>();
          // if constexpr (Syncthreads) __syncthreads();
          __builtin_mxc_barrier();

          // Move to the next stage
          iterator_A.add_tile_offset({0, 1});
          iterator_B.add_tile_offset({1, 0});

          this->smem_iterator_A_.add_tile_offset({0, 1});
          this->smem_iterator_B_.add_tile_offset({1, 0});

          // Add negative offsets to return iterators to the 'start' of the
          // circular buffer in shared memory
          if (smem_write_stage_idx == (Base::kStages - 1)) {
            this->smem_iterator_A_.add_tile_offset({0, -Base::kStages});
            this->smem_iterator_B_.add_tile_offset({-Base::kStages, 0});
            smem_write_stage_idx = 0;
          } else {
            ++smem_write_stage_idx;
          }

          if (smem_read_stage_idx == (Base::kStages - 1)) {
            this->warp_tile_iterator_A_.add_tile_offset(
                {-Base::kStages * Base::WarpCount::kM, 0});
            this->warp_tile_iterator_B_.add_tile_offset(
                {-Base::kStages * Policy::kPartitionsK *
                     Base::kWarpGemmIterations,
                 0});
            smem_read_stage_idx = 0;
          } else {
            ++smem_read_stage_idx;
          }

          --gemm_k_iterations;
          iterator_A.clear_mask(gemm_k_iterations == 0);
          iterator_B.clear_mask(gemm_k_iterations == 0);
        }

        // Do any conversions feeding the first stage at the end of the loop so
        // we can start right away on mma instructions
        if (warp_mma_k + 1 == Base::kWarpGemmIterations)
          warp_mma.transform(warp_transformed_frag_A[(warp_mma_k + 1) % 2],
                             warp_transformed_frag_B[(warp_mma_k + 1) % 2],
                             warp_loaded_frag_A[(warp_mma_k + 1) % 2],
                             warp_loaded_frag_B[(warp_mma_k + 1) % 2]);

          // if (gemm_k_iterations > 0) {
          //   __builtin_mxc_arrive(64 + arrive_gvm_cnt);
          // } else {
        __builtin_mxc_arrive(64);
          // }
          // __builtin_mxc_barrier();
      }

    }

    if (platform::is_same<typename Operator::MathOperator,
                          arch::OpMultiplyAddFastF32>::value
      || platform::is_same<typename Operator::MathOperator,
                           arch::OpMultiplyAddComplexFastF32>::value) {
      accum = plus_accum(accum, tmp_accum);
    }

    if (SharedMemoryClear == SharedMemoryClearOption::kZfill) {
      // commit and drain all pending and predicated LDGSTS pnz from the GEMM mainloop
      mctlass::arch::cp_async_fence();
      mctlass::arch::cp_async_wait<0>();
      if constexpr (!MacaSpecialTag) __syncthreads();
    }

  }
};

// mcTlass: problem: NN && stage==2
//          (SmemIteratorA_::Layout == MacaRowMajorTensorOpMultiplicandCpAsyncCrosswise) &&
//          (SmemIteratorB_::Layout == MacaRowMajorTensorOpMultiplicandCongruous) &&
//          (stage == 2)
template <
    /// Size of the Gemm problem - concept: gemm::GemmShape<>
    typename Shape_,
    /// Iterates over tiles of A operand in global memory
    //  (concept: ReadableTileIterator | ForwardTileIterator |
    //  MaskedTileIterator)
    typename IteratorA_,
    /// Iterates over tiles of A operand in shared memory
    /// (concept: WriteableTileIterator | RandomAccessTileIterator)
    typename SmemIteratorA_,
    /// Cache operation for operand A
    mctlass::arch::CacheOperation::Kind CacheOpA,
    /// Iterates over tiles of B operand in global memory
    //  (concept: ReadableTileIterator | ForwardTileIterator |
    //  MaskedTileIterator)
    typename IteratorB_,
    /// Iterates over tiles of B operand in shared memory
    /// (concept: WriteableTileIterator | RandomAccessTileIterator)
    typename SmemIteratorB_,
    /// Cache operation for operand B
    mctlass::arch::CacheOperation::Kind CacheOpB,
    /// Data type of accumulator matrix
    typename ElementC_,
    /// Data type of accumulator matrix
    typename LayoutC_,
    /// Policy describing tuning details (concept: MmaPolicy)
    typename Policy_,
    /// Number of stages,
    // int Stages,
    /// Use zfill or predicate for out-of-bound cp.async
    SharedMemoryClearOption SharedMemoryClear,
    /// Flag tag for special process for m16n16k16 f16
    bool MacaSpecialTag_
    >
class MmaMultistage <
  Shape_,
  IteratorA_,
  SmemIteratorA_,
  CacheOpA,
  IteratorB_,
  SmemIteratorB_,
  CacheOpB,
  ElementC_,
  LayoutC_,
  Policy_,
  2,
  SharedMemoryClear,
  MacaSpecialTag_,
  typename std::enable_if<
    ( (std::is_same<typename SmemIteratorA_::Layout,
        typename layout::MacaRowMajorTensorOpMultiplicandCpAsyncCrosswise<
          sizeof_bits<mctlass::half_t>::value, Shape_::kK>>::value &&
      (std::is_same<typename SmemIteratorB_::Layout,
          typename layout::MacaRowMajorTensorOpMultiplicandCongruous<
            sizeof_bits<half_t>::value, int(128 / sizeof(half_t))>>::value ||
         std::is_same<typename SmemIteratorB_::Layout,
          typename layout::MacaRowMajorTensorOpMultiplicandCongruous4x4Perm<
            sizeof_bits<half_t>::value, int(128 / sizeof(half_t))>>::value) ) ||
       (std::is_same<typename SmemIteratorA_::Layout,
        typename layout::MacaRowMajorTensorOpMultiplicandCpAsyncCrosswise<
          sizeof_bits<int8_t>::value, Shape_::kK>>::value &&
        std::is_same<typename SmemIteratorB_::Layout,
          typename layout::MacaRowMajorTensorOpMultiplicandCongruous<
            sizeof_bits<int8_t>::value, int(128 / sizeof(int8_t))>>::value) ),
    bool>::type
  >:public MmaBase<Shape_, Policy_, 2> {
public:
  ///< Base class
  using Base = MmaBase<Shape_, Policy_, 2>;
  ///< Size of the Gemm problem - concept: gemm::GemmShape<>
  using Shape = Shape_;
  ///< Iterates over tiles of A operand in global memory
  using IteratorA = IteratorA_;
  ///< Iterates over tiles of B operand in global memory
  using IteratorB = IteratorB_;
  ///< Data type of accumulator matrix
  using ElementC = ElementC_;
  ///< Layout of accumulator matrix
  using LayoutC = LayoutC_;
  ///< Policy describing tuning details
  using Policy = Policy_;

  using SmemIteratorA = SmemIteratorA_;
  using SmemIteratorB = SmemIteratorB_;

  static mctlass::arch::CacheOperation::Kind const kCacheOpA = CacheOpA;
  static mctlass::arch::CacheOperation::Kind const kCacheOpB = CacheOpB;
  // static constexpr bool MacaSpecialTag = MacaSpecialTag_;
  static constexpr bool MacaSpecialTag = false;

  static constexpr bool CrossOffsetA =
    platform::is_same<layout::MacaRowMajorTensorOpMultiplicandCpAsyncCrosswise<
      sizeof_bits<typename IteratorA::Element>::value, Shape::kK>,
    typename SmemIteratorA::Layout>::value;
  static constexpr bool CrossOffsetB =
    platform::is_same<layout::MacaColumnMajorTensorOpMultiplicandCpAsyncCrosswise<
      sizeof_bits<typename IteratorB::Element>::value, Shape::kK>,
    typename SmemIteratorB::Layout>::value;

  //
  // Dependent types
  //

  /// Fragment of accumulator tile
  using FragmentC = typename Policy::Operator::FragmentC;

  /// Warp-level Mma
  using Operator = typename Policy::Operator;

  /// Minimum architecture is Sm80 to support cp.async
  using ArchTag = arch::Sm80;

  /// Complex transform on A operand
  static ComplexTransform const kTransformA = Operator::kTransformA;

  /// Complex transform on B operand
  static ComplexTransform const kTransformB = Operator::kTransformB;

  static int const kSrcBytesB = sizeof_bits<typename IteratorB::Element>::value *
                                IteratorB::ThreadMap::kElementsPerAccess /
                                IteratorB::kAccessesPerVector / 8;
  static int const kSrcBytesA = sizeof_bits<typename IteratorA::Element>::value *
                                IteratorA::ThreadMap::kElementsPerAccess /
                                IteratorA::kAccessesPerVector / 8;
  using AccessTypeB  = Array<uint8_t, kSrcBytesB>;

  /// Internal structure exposed for introspection.
  struct Detail {

    /// Number of cp.async instructions to load one stage of operand A
    static int const AsyncCopyIterationsPerStageA =
        IteratorA::ThreadMap::Iterations::kCount;

    /// Number of cp.async instructions to load one stage of operand B
    static int const AsyncCopyIterationsPerStageB =
        IteratorB::ThreadMap::Iterations::kCount;

    /// Number of stages
    static int const kStages = 2;

    /// Number of cp.async instructions to load on group of operand A
    static int const kAccessesPerGroupA =
        (AsyncCopyIterationsPerStageA + Base::kWarpGemmIterations - 1) / Base::kWarpGemmIterations;

    /// Number of cp.async instructions to load on group of operand B
    static int const kAccessesPerGroupB =
        (AsyncCopyIterationsPerStageB + Base::kWarpGemmIterations - 1) / Base::kWarpGemmIterations;
  };

 private:

  using WarpLoadedFragmentA = typename Operator::FragmentA;
  using WarpLoadedFragmentB = typename Operator::FragmentB;
  using WarpTransformedFragmentA = typename Operator::TransformedFragmentA;
  using WarpTransformedFragmentB = typename Operator::TransformedFragmentB;

 private:

  //
  // Data members
  //

  /// Iterator to write threadblock-scoped tile of A operand to shared memory
  SmemIteratorA smem_iterator_A_;

  /// Iterator to write threadblock-scoped tile of B operand to shared memory
  SmemIteratorB smem_iterator_B_;

  int cross_offset_A_;
  int cross_offset_B_;

  typename IteratorA::AccessType* smem_ptr_A_[Base::kStages * Detail::AsyncCopyIterationsPerStageA * IteratorA::kAccessesPerVector];

public:

  /// Construct from tensor references
  MCTLASS_DEVICE
  MmaMultistage(
      ///< Shared storage needed for internal use by threadblock-scoped GEMM
      typename Base::SharedStorage &shared_storage,
      ///< ID within the threadblock
      int thread_idx,
      ///< ID of warp
      int warp_idx,
      ///< ID of each thread within a warp
      int lane_idx,
      int problem_size_k = 0,
      typename IteratorA::Element *ptr_A = nullptr,
      typename IteratorA::Element *ptr_B = nullptr
    ):
      Base(shared_storage, thread_idx, warp_idx, lane_idx),
      smem_iterator_A_(shared_storage.operand_A_ref(), thread_idx),
      smem_iterator_B_(shared_storage.operand_B_ref(), thread_idx)
  {
    int bid = lane_idx & 0x7;
    int group_id = (lane_idx >> 3) & 0x3;
    int cross_target = bid ^ group_id;
    int cross_offset_ = cross_target - bid;
    cross_offset_A_ = CrossOffsetA ? cross_offset_ : 0;
    cross_offset_B_ = CrossOffsetB ? cross_offset_ : 0;

    // Compute warp location within threadblock tile by mapping the warp_id to
    // three coordinates:
    //   _m: the warp's position within the threadblock along the M dimension
    //   _n: the warp's position within the threadblock along the N dimension
    //   _k: the warp's position within the threadblock along the K dimension

    int warp_idx_mn = warp_idx % (Base::WarpCount::kM * Base::WarpCount::kN);
    int warp_idx_k = warp_idx / (Base::WarpCount::kM * Base::WarpCount::kN);

    int warp_idx_m = warp_idx_mn % Base::WarpCount::kM;
    int warp_idx_n = warp_idx_mn / Base::WarpCount::kM;

    this->warp_tile_iterator_A_.pre_compute_pointer();

    // Add per-warp offsets in units of warp-level tiles
    this->warp_tile_iterator_A_.add_tile_offset(
        {warp_idx_m, Base::kWarpGemmIterations * warp_idx_k});
    this->warp_tile_iterator_B_.add_tile_offset(
        {Base::kWarpGemmIterations * warp_idx_k, warp_idx_n});

    // Compute A shared memory ptr
    this->smem_iterator_A_.set_iteration_index(0);

    for(int stage = 0; stage < Base::kStages; ++stage){
      for(int j = 0; j < Detail::AsyncCopyIterationsPerStageA; ++j){
        for(int v = 0; v < IteratorA::kAccessesPerVector; ++v ){

          smem_ptr_A_[stage * Detail::AsyncCopyIterationsPerStageA + j * IteratorA::kAccessesPerVector + v] =
            reinterpret_cast<typename IteratorA::AccessType *>(this->smem_iterator_A_.get()) + v - cross_offset_A_;;
          ++this->smem_iterator_A_;
        }
      }
      this->smem_iterator_A_.add_tile_offset({0,1});
    }
    // Reset tile offset to zero
    this->smem_iterator_A_.add_tile_offset({0,-Base::kStages});
  }


  MCTLASS_DEVICE
  void copy_tiles_and_advance_A(IteratorA &iterator_A, int group_start_A = 0) {
    iterator_A.set_iteration_index(group_start_A *
                                   IteratorA::kAccessesPerVector);
    this->smem_iterator_A_.set_iteration_index(group_start_A);

    // Async Copy for operand A
    MCTLASS_PRAGMA_UNROLL
    for (int j = 0; j < Detail::AsyncCopyIterationsPerStageA; ++j) {
        typename IteratorA::AccessType *dst_ptr[IteratorA::kAccessesPerVector];
        typename IteratorA::AccessType *gmem_ptr[IteratorA::kAccessesPerVector];
        bool valid_tag[IteratorA::kAccessesPerVector];
        MCTLASS_PRAGMA_UNROLL
        for (int v = 0; v < IteratorA::kAccessesPerVector; ++v) {
          dst_ptr[v] = reinterpret_cast<typename IteratorA::AccessType *>(this->smem_iterator_A_.get()) + v - cross_offset_A_;
          gmem_ptr[v] = iterator_A.get() + cross_offset_A_;
          valid_tag[v] = iterator_A.valid();
          ++iterator_A;
        }
        MCTLASS_PRAGMA_UNROLL
        for (int v = 0; v < IteratorA::kAccessesPerVector; ++v) {
          if constexpr (SharedMemoryClear == SharedMemoryClearOption::kZfill) {
            mctlass::arch::cp_async_zfill<kSrcBytesA, kCacheOpA>(
                dst_ptr[v], gmem_ptr[v], valid_tag[v]);
          } else {
            mctlass::arch::cp_async<kSrcBytesA, kCacheOpA>(
                dst_ptr[v], gmem_ptr[v], valid_tag[v]);
          }
        }
        this->smem_iterator_A_.set_iteration_index(++group_start_A);
    }
  }

  MCTLASS_DEVICE
  void copy_tiles_and_advance_B(IteratorB &iterator_B, int group_start_B = 0) {

    iterator_B.set_iteration_index(group_start_B *
                                   IteratorB::kAccessesPerVector);
    this->smem_iterator_B_.set_iteration_index(group_start_B);

    // Async Copy for operand B
    MCTLASS_PRAGMA_UNROLL
    for (int j = 0; j < Detail::AsyncCopyIterationsPerStageB; ++j) {
        typename IteratorB::AccessType *dst_ptr[IteratorB::kAccessesPerVector];
        typename IteratorB::AccessType *gmem_ptr[IteratorB::kAccessesPerVector];
        bool valid_tag[IteratorB::kAccessesPerVector];

        MCTLASS_PRAGMA_UNROLL
        for (int v = 0; v < IteratorB::kAccessesPerVector; ++v) {
          dst_ptr[v] = reinterpret_cast<typename IteratorB::AccessType *>(this->smem_iterator_B_.get()) + v - cross_offset_B_;
          gmem_ptr[v] = iterator_B.get() + cross_offset_B_;
          valid_tag[v] = iterator_B.valid();

          ++iterator_B;
        }

        MCTLASS_PRAGMA_UNROLL
        for (int v = 0; v < IteratorB::kAccessesPerVector; ++v) {
          if (SharedMemoryClear == SharedMemoryClearOption::kZfill) {
            mctlass::arch::cp_async_zfill<kSrcBytesB, kCacheOpB>(
                dst_ptr[v], gmem_ptr[v], valid_tag[v]);
          } else {
            mctlass::arch::cp_async<kSrcBytesB, kCacheOpB>(
                dst_ptr[v], gmem_ptr[v], valid_tag[v]);
          }
        }
        this->smem_iterator_B_.set_iteration_index(++group_start_B);
    }
  }

  MCTLASS_DEVICE
  void copy_tiles_and_advance_A_pre(IteratorA &iterator_A, int stage,int group_start_A = 0){

    iterator_A.set_iteration_index(group_start_A *
                                   IteratorA::kAccessesPerVector);

    MCTLASS_PRAGMA_UNROLL
    for (int j = 0; j < Detail::AsyncCopyIterationsPerStageA; ++j) {
        typename IteratorA::AccessType *gmem_ptr[IteratorA::kAccessesPerVector];

        MCTLASS_PRAGMA_UNROLL
        for (int v = 0; v < IteratorA::kAccessesPerVector; ++v) {
          gmem_ptr[v] = iterator_A.get() + cross_offset_A_;
          ++iterator_A;
        }
        MCTLASS_PRAGMA_UNROLL
        for (int v = 0; v < IteratorA::kAccessesPerVector; ++v) {
          if constexpr (SharedMemoryClear == SharedMemoryClearOption::kZfill) {
            mctlass::arch::cp_async_zfill<kSrcBytesA, kCacheOpA>(
                smem_ptr_A_[stage * Detail::AsyncCopyIterationsPerStageA + j * IteratorA::kAccessesPerVector + v], gmem_ptr[v], true);
          } else {
            mctlass::arch::cp_async<kSrcBytesA, kCacheOpA>(
                smem_ptr_A_[stage * Detail::AsyncCopyIterationsPerStageA + j * IteratorA::kAccessesPerVector + v], gmem_ptr[v], true);
          }
        }
    }
  }

  MCTLASS_DEVICE
  void copy_tiles_and_advance(IteratorA &iterator_A, IteratorB &iterator_B,
                              int group_start_A = 0, int group_start_B = 0) {
    copy_tiles_and_advance_A(iterator_A, group_start_A);
    copy_tiles_and_advance_B(iterator_B, group_start_B);
  }

  template<int N>
  struct Index{
    static constexpr int value = N;
  };

  template<int N>
  static constexpr int StageA = Index<N>::value;
  template<int N>
  static constexpr int StageB = Index<N>::value;

  template<int N>
  static constexpr int ComputeFragIdx = Index<N>::value;
  template<int N>
  static constexpr int LoadFragIdx = Index<N>::value;

  template<int N>
  static constexpr int LoadWarpTileIdxA = Index<N>::value;
  template<int N>
  static constexpr int LoadWarpTileIdxB = Index<N>::value;

  /// Perform a threadblock-scoped matrix multiply-accumulate
  MCTLASS_DEVICE
  void operator()(
      ///< problem size of GEMM
      int gemm_k_iterations,
      ///< destination accumulator tile
      FragmentC &accum,
      ///< iterator over A operand in global memory
      IteratorA iterator_A,
      ///< iterator over B operand in global memory
      IteratorB iterator_B,
      ///< initial value of accumulator
      FragmentC const &src_accum) {
    //
    // Prologue
    //
    static int const arrive_gvm_cnt = Detail::AsyncCopyIterationsPerStageA +
             Detail::AsyncCopyIterationsPerStageB;
    // Issue several complete stages
    copy_tiles_and_advance_A_pre(iterator_A,StageA<0>);
    copy_tiles_and_advance_B(iterator_B, 0);
    gemm_k_iterations -= 1;

    // Move to the next stage
    iterator_A.add_tile_offset({0, 1});
    iterator_B.add_tile_offset({1, 0});
    this->smem_iterator_B_.add_tile_offset({1, 0});

    // Perform accumulation in the 'd' output operand
    accum = src_accum;

    //
    // Clear the remaining tiles of SMEM. This is a functional requirement for some kernels
    // so that all accumulator elements outside the GEMM footprint are zero.
    //

    if (SharedMemoryClear == SharedMemoryClearOption::kClearLastStage) {

      /// Iterator to write threadblock-scoped tile of A operand to shared memory
      SmemIteratorA last_smem_iterator_A(this->smem_iterator_A_);

      typename IteratorA::AccessType zero_A;
      zero_A.clear();

      last_smem_iterator_A.set_iteration_index(0);

      // Async Copy for operand A
      MCTLASS_PRAGMA_UNROLL
      for (int j = 0; j < Detail::AsyncCopyIterationsPerStageA; ++j) {

        typename IteratorA::AccessType *dst_ptr =
            reinterpret_cast<typename IteratorA::AccessType *>(
                last_smem_iterator_A.get());

        *dst_ptr = zero_A;

        ++last_smem_iterator_A;
      }

      /// Iterator to write threadblock-scoped tile of B operand to shared memory
      SmemIteratorB last_smem_iterator_B(this->smem_iterator_B_);
      typename IteratorB::AccessType zero_B;

      zero_B.clear();
      last_smem_iterator_B.set_iteration_index(0);

      // Async Copy for operand B
      MCTLASS_PRAGMA_UNROLL
      for (int j = 0; j < Detail::AsyncCopyIterationsPerStageB; ++j) {

        typename IteratorB::AccessType *dst_ptr =
            reinterpret_cast<typename IteratorB::AccessType *>(
                last_smem_iterator_B.get());

        *dst_ptr = zero_B;

        ++last_smem_iterator_B;
      }
    }

    // Waits until stages up to the previous (kStages-2)th stage have committed.
    // mctlass::arch::cp_async_wait<Base::kStages - 2>();
    // if constexpr (Syncthreads) __syncthreads();
    __builtin_mxc_arrive(64);
    __builtin_mxc_barrier();

    // Pair of fragments used to overlap shared memory loads and math
    // instructions
    WarpLoadedFragmentA warp_loaded_frag_A[4];
    WarpLoadedFragmentB warp_loaded_frag_B[4];

    Operator warp_mma;

    this->warp_tile_iterator_B_.set_kgroup_index(0);
    this->warp_tile_iterator_B_.load(warp_loaded_frag_B[0]);
    ++this->warp_tile_iterator_B_;
    this->warp_tile_iterator_A_.load_with_index(warp_loaded_frag_A[LoadFragIdx<0>],LoadWarpTileIdxA<0>);

    iterator_B.clear_mask(gemm_k_iterations == 0);

    copy_tiles_and_advance_A_pre(iterator_A,StageB<1>);
    copy_tiles_and_advance_B(iterator_B, 0);

    this->warp_tile_iterator_B_.load(warp_loaded_frag_B[1]);
    ++this->warp_tile_iterator_B_;
    this->warp_tile_iterator_A_.load_with_index(warp_loaded_frag_A[LoadFragIdx<1>],LoadWarpTileIdxA<1>);

    // tf32x3 kernels use staging accumulation. warp_mma uses a temporary
    // accumulator and this temporary accumulator is added to the final
    // accumulator once in every mainloop iteration.
    plus<FragmentC> plus_accum;

    FragmentC tmp_accum;

    if (platform::is_same<typename Operator::MathOperator,
                          arch::OpMultiplyAddFastF32>::value
      || platform::is_same<typename Operator::MathOperator,
                          arch::OpMultiplyAddComplexFastF32>::value) {

      tmp_accum.clear();
    }

    //
    // Mainloop
    //
    if (threadIdx.x / 256 == 0) {
      MCTLASS_GEMM_LOOP
      for (; gemm_k_iterations > (-Base::kStages + 2);) {
        __builtin_mxc_arrive(64);
        __builtin_mxc_barrier();
        iterator_A.add_tile_offset({0, 1});
        iterator_B.add_tile_offset({1, 0});
        iterator_B.clear_mask(gemm_k_iterations <= 1);
        this->smem_iterator_B_.add_tile_offset({1, 0});
        this->smem_iterator_B_.add_tile_offset({-Base::kStages, 0});
        copy_tiles_and_advance_A_pre(iterator_A,StageA<0>);
        copy_tiles_and_advance_B(iterator_B, 0);
        asm(";------");
        this->warp_tile_iterator_B_.set_kgroup_index(0);
        this->warp_tile_iterator_B_.load(warp_loaded_frag_B[2]);
        ++this->warp_tile_iterator_B_;
        this->warp_tile_iterator_A_.load_with_index(warp_loaded_frag_A[LoadFragIdx<2>],LoadWarpTileIdxA<2>);
        asm(";------");
        warp_mma(accum, warp_loaded_frag_A[0], warp_loaded_frag_B[0], accum);
        asm(";------");
        this->warp_tile_iterator_B_.load(warp_loaded_frag_B[3]);
        ++this->warp_tile_iterator_B_;
        this->warp_tile_iterator_A_.load_with_index(warp_loaded_frag_A[LoadFragIdx<3>],LoadWarpTileIdxA<3>);
        asm(";------");
        warp_mma(accum, warp_loaded_frag_A[1], warp_loaded_frag_B[1], accum);

        __builtin_mxc_arrive(64);
        __builtin_mxc_barrier();
        iterator_A.add_tile_offset({0, 1});
        iterator_B.add_tile_offset({1, 0});
        iterator_B.clear_mask(gemm_k_iterations <= 1);
        this->smem_iterator_B_.add_tile_offset({1, 0});
        copy_tiles_and_advance_A_pre(iterator_A,StageA<1>);
        copy_tiles_and_advance_B(iterator_B, 0);
        asm(";------");
        this->warp_tile_iterator_B_.add_tile_offset(
                {-Base::kStages * Policy::kPartitionsK * Base::kWarpGemmIterations, 0});
        this->warp_tile_iterator_B_.set_kgroup_index(0);
        this->warp_tile_iterator_B_.load(warp_loaded_frag_B[0]);
        ++this->warp_tile_iterator_B_;
        this->warp_tile_iterator_A_.load_with_index(warp_loaded_frag_A[LoadFragIdx<0>],LoadWarpTileIdxA<0>);
        asm(";------");
        warp_mma(accum, warp_loaded_frag_A[2], warp_loaded_frag_B[2], accum);
        asm(";------");
        this->warp_tile_iterator_B_.load(warp_loaded_frag_B[1]);
        ++this->warp_tile_iterator_B_;
        this->warp_tile_iterator_A_.load_with_index(warp_loaded_frag_A[LoadFragIdx<1>],LoadWarpTileIdxA<1>);
        asm(";------");
        warp_mma(accum, warp_loaded_frag_A[3], warp_loaded_frag_B[3], accum);
        gemm_k_iterations -= 2;
      }
    } else {
      MCTLASS_GEMM_LOOP
      for (; gemm_k_iterations > (-Base::kStages + 2);) {
        __builtin_mxc_arrive(64);
        __builtin_mxc_barrier();
        iterator_A.add_tile_offset({0, 1});
        iterator_B.add_tile_offset({1, 0});
        iterator_B.clear_mask(gemm_k_iterations <= 1);
        this->smem_iterator_B_.add_tile_offset({1, 0});
        this->smem_iterator_B_.add_tile_offset({-Base::kStages, 0});
        copy_tiles_and_advance_A_pre(iterator_A,StageA<0>);
        copy_tiles_and_advance_B(iterator_B, 0);
        warp_mma.frontHalfMma(accum, warp_loaded_frag_A[0], warp_loaded_frag_B[0], accum);
        asm(";------");
        this->warp_tile_iterator_B_.set_kgroup_index(0);
        this->warp_tile_iterator_B_.load(warp_loaded_frag_B[2]);
        ++this->warp_tile_iterator_B_;
        this->warp_tile_iterator_A_.load_with_index(warp_loaded_frag_A[LoadFragIdx<2>],LoadWarpTileIdxA<2>);
        asm(";------");
        this->warp_tile_iterator_B_.load(warp_loaded_frag_B[3]);
        ++this->warp_tile_iterator_B_;
        warp_mma.laterHalfMma(accum, warp_loaded_frag_A[0], warp_loaded_frag_B[0], accum);
        warp_mma.frontHalfMma(accum, warp_loaded_frag_A[1], warp_loaded_frag_B[1], accum);
        asm(";------");
        this->warp_tile_iterator_A_.load_with_index(warp_loaded_frag_A[LoadFragIdx<3>],LoadWarpTileIdxA<3>);
        asm(";------");
        warp_mma.laterHalfMma(accum, warp_loaded_frag_A[1], warp_loaded_frag_B[1], accum);

        __builtin_mxc_arrive(64);
        __builtin_mxc_barrier();
        iterator_A.add_tile_offset({0, 1});
        iterator_B.add_tile_offset({1, 0});
        iterator_B.clear_mask(gemm_k_iterations <= 1);
        this->smem_iterator_B_.add_tile_offset({1, 0});
        copy_tiles_and_advance_A_pre(iterator_A,StageA<1>);
        copy_tiles_and_advance_B(iterator_B, 0);
        warp_mma.frontHalfMma(accum, warp_loaded_frag_A[2], warp_loaded_frag_B[2], accum);
        asm(";------");
        this->warp_tile_iterator_B_.add_tile_offset(
                {-Base::kStages * Policy::kPartitionsK * Base::kWarpGemmIterations, 0});
        this->warp_tile_iterator_B_.set_kgroup_index(0);
        this->warp_tile_iterator_B_.load(warp_loaded_frag_B[0]);
        ++this->warp_tile_iterator_B_;
        this->warp_tile_iterator_A_.load_with_index(warp_loaded_frag_A[LoadFragIdx<0>],LoadWarpTileIdxA<0>);
        asm(";------");
        this->warp_tile_iterator_B_.load(warp_loaded_frag_B[1]);
        ++this->warp_tile_iterator_B_;
        warp_mma.laterHalfMma(accum, warp_loaded_frag_A[2], warp_loaded_frag_B[2], accum);
        warp_mma.frontHalfMma(accum, warp_loaded_frag_A[3], warp_loaded_frag_B[3], accum);
        asm(";------");
        this->warp_tile_iterator_A_.load_with_index(warp_loaded_frag_A[LoadFragIdx<1>],LoadWarpTileIdxA<1>);
        asm(";------");
        warp_mma.laterHalfMma(accum, warp_loaded_frag_A[3], warp_loaded_frag_B[3], accum);
        gemm_k_iterations -= 2;
      }
    }
    if (gemm_k_iterations == 0) {
      warp_mma(accum, warp_loaded_frag_A[0],
              warp_loaded_frag_B[0], accum);
      warp_mma(accum, warp_loaded_frag_A[1],
              warp_loaded_frag_B[1], accum);
    }

    if (platform::is_same<typename Operator::MathOperator,
                          arch::OpMultiplyAddFastF32>::value
      || platform::is_same<typename Operator::MathOperator,
                           arch::OpMultiplyAddComplexFastF32>::value) {
      accum = plus_accum(accum, tmp_accum);
    }

    if (SharedMemoryClear == SharedMemoryClearOption::kZfill) {
      // commit and drain all pending and predicated LDGSTS pnz from the GEMM mainloop
      mctlass::arch::cp_async_fence();
      mctlass::arch::cp_async_wait<0>();
      if constexpr (!MacaSpecialTag) __syncthreads();
    }

  }
};

// mcTlass: problem: TN && stage==2
//          (SmemIteratorA_::Layout == MacaRowMajorTensorOpMultiplicandCpAsyncCrosswise) &&
//          (SmemIteratorB_::Layout == MacaColumnMajorTensorOpMultiplicandCpAsyncCrosswise) &&
//          (stage == 2)
template <
    /// Size of the Gemm problem - concept: gemm::GemmShape<>
    typename Shape_,
    /// Iterates over tiles of A operand in global memory
    //  (concept: ReadableTileIterator | ForwardTileIterator |
    //  MaskedTileIterator)
    typename IteratorA_,
    /// Iterates over tiles of A operand in shared memory
    /// (concept: WriteableTileIterator | RandomAccessTileIterator)
    typename SmemIteratorA_,
    /// Cache operation for operand A
    mctlass::arch::CacheOperation::Kind CacheOpA,
    /// Iterates over tiles of B operand in global memory
    //  (concept: ReadableTileIterator | ForwardTileIterator |
    //  MaskedTileIterator)
    typename IteratorB_,
    /// Iterates over tiles of B operand in shared memory
    /// (concept: WriteableTileIterator | RandomAccessTileIterator)
    typename SmemIteratorB_,
    /// Cache operation for operand B
    mctlass::arch::CacheOperation::Kind CacheOpB,
    /// Data type of accumulator matrix
    typename ElementC_,
    /// Data type of accumulator matrix
    typename LayoutC_,
    /// Policy describing tuning details (concept: MmaPolicy)
    typename Policy_,
    /// Number of stages,
    // int Stages,
    /// Use zfill or predicate for out-of-bound cp.async
    SharedMemoryClearOption SharedMemoryClear,
    /// Flag tag for special process for m16n16k16 f16
    bool MacaSpecialTag_
    >
class MmaMultistage <
  Shape_,
  IteratorA_,
  SmemIteratorA_,
  CacheOpA,
  IteratorB_,
  SmemIteratorB_,
  CacheOpB,
  ElementC_,
  LayoutC_,
  Policy_,
  2,
  SharedMemoryClear,
  MacaSpecialTag_,
  typename std::enable_if<
    ( (std::is_same<typename SmemIteratorA_::Layout,
        typename layout::MacaRowMajorTensorOpMultiplicandCpAsyncCrosswise<
          sizeof_bits<mctlass::half_t>::value, Shape_::kK>>::value &&
      std::is_same<typename SmemIteratorB_::Layout,
        typename layout::MacaColumnMajorTensorOpMultiplicandCpAsyncCrosswise<
          sizeof_bits<mctlass::half_t>::value, Shape_::kK>>::value) ||
      (std::is_same<typename SmemIteratorA_::Layout,
        typename layout::MacaRowMajorTensorOpMultiplicandCpAsyncCrosswise<
          sizeof_bits<int8_t>::value, Shape_::kK>>::value &&
      std::is_same<typename SmemIteratorB_::Layout,
        typename layout::MacaColumnMajorTensorOpMultiplicandCpAsyncCrosswise<
          sizeof_bits<int8_t>::value, Shape_::kK>>::value)),
    bool>::type
  >:public MmaBase<Shape_, Policy_, 2> {
public:
  ///< Base class
  using Base = MmaBase<Shape_, Policy_, 2>;
  ///< Size of the Gemm problem - concept: gemm::GemmShape<>
  using Shape = Shape_;
  ///< Iterates over tiles of A operand in global memory
  using IteratorA = IteratorA_;
  ///< Iterates over tiles of B operand in global memory
  using IteratorB = IteratorB_;
  ///< Data type of accumulator matrix
  using ElementC = ElementC_;
  ///< Layout of accumulator matrix
  using LayoutC = LayoutC_;
  ///< Policy describing tuning details
  using Policy = Policy_;

  using SmemIteratorA = SmemIteratorA_;
  using SmemIteratorB = SmemIteratorB_;

  static mctlass::arch::CacheOperation::Kind const kCacheOpA = CacheOpA;
  static mctlass::arch::CacheOperation::Kind const kCacheOpB = CacheOpB;
  // static constexpr bool MacaSpecialTag = MacaSpecialTag_;
  static constexpr bool MacaSpecialTag = false;

  static constexpr bool CrossOffsetA =
    platform::is_same<layout::MacaRowMajorTensorOpMultiplicandCpAsyncCrosswise<
      sizeof_bits<typename IteratorA::Element>::value, Shape::kK>,
    typename SmemIteratorA::Layout>::value;
  static constexpr bool CrossOffsetB =
    platform::is_same<layout::MacaColumnMajorTensorOpMultiplicandCpAsyncCrosswise<
      sizeof_bits<typename IteratorB::Element>::value, Shape::kK>,
    typename SmemIteratorB::Layout>::value;

  static_assert(Shape::kK % (512 / sizeof_bits<typename IteratorB::Element>::value) == 0,
      "Crosswise kContiguous must be 512bit aligned");
  //
  // Dependent types
  //

  /// Fragment of accumulator tile
  using FragmentC = typename Policy::Operator::FragmentC;

  /// Warp-level Mma
  using Operator = typename Policy::Operator;

  /// Minimum architecture is Sm80 to support cp.async
  using ArchTag = arch::Sm80;

  /// Complex transform on A operand
  static ComplexTransform const kTransformA = Operator::kTransformA;

  /// Complex transform on B operand
  static ComplexTransform const kTransformB = Operator::kTransformB;

  static int const kSrcBytesB = sizeof_bits<typename IteratorB::Element>::value *
                                IteratorB::ThreadMap::kElementsPerAccess /
                                IteratorB::kAccessesPerVector / 8;
  static int const kSrcBytesA = sizeof_bits<typename IteratorA::Element>::value *
                                IteratorA::ThreadMap::kElementsPerAccess /
                                IteratorA::kAccessesPerVector / 8;
  using AccessTypeB  = Array<uint8_t, kSrcBytesB>;

  /// Internal structure exposed for introspection.
  struct Detail {

    /// Number of cp.async instructions to load one stage of operand A
    static int const AsyncCopyIterationsPerStageA =
        IteratorA::ThreadMap::Iterations::kCount;

    /// Number of cp.async instructions to load one stage of operand B
    static int const AsyncCopyIterationsPerStageB =
        IteratorB::ThreadMap::Iterations::kCount;

    /// Number of stages
    static int const kStages = 2;

    /// Number of cp.async instructions to load on group of operand A
    static int const kAccessesPerGroupA =
        (AsyncCopyIterationsPerStageA + Base::kWarpGemmIterations - 1) / Base::kWarpGemmIterations;

    /// Number of cp.async instructions to load on group of operand B
    static int const kAccessesPerGroupB =
        (AsyncCopyIterationsPerStageB + Base::kWarpGemmIterations - 1) / Base::kWarpGemmIterations;
  };

 private:

  using WarpLoadedFragmentA = typename Operator::FragmentA;
  using WarpLoadedFragmentB = typename Operator::FragmentB;
  using WarpTransformedFragmentA = typename Operator::TransformedFragmentA;
  using WarpTransformedFragmentB = typename Operator::TransformedFragmentB;

 private:

  //
  // Data members
  //

  /// Iterator to write threadblock-scoped tile of A operand to shared memory
  SmemIteratorA smem_iterator_A_;

  /// Iterator to write threadblock-scoped tile of B operand to shared memory
  SmemIteratorB smem_iterator_B_;

  int cross_offset_A_;
  int cross_offset_B_;

  typename IteratorA::AccessType* smem_ptr_A_[Base::kStages * Detail::AsyncCopyIterationsPerStageA * IteratorA::kAccessesPerVector];
  typename IteratorB::AccessType* smem_ptr_B_[Base::kStages * Detail::AsyncCopyIterationsPerStageB * IteratorB::kAccessesPerVector];

public:

  /// Construct from tensor references
  MCTLASS_DEVICE
  MmaMultistage(
      ///< Shared storage needed for internal use by threadblock-scoped GEMM
      typename Base::SharedStorage &shared_storage,
      ///< ID within the threadblock
      int thread_idx,
      ///< ID of warp
      int warp_idx,
      ///< ID of each thread within a warp
      int lane_idx,
      int problem_size_k = 0,
      typename IteratorA::Element *ptr_A = nullptr,
      typename IteratorA::Element *ptr_B = nullptr
    ):
      Base(shared_storage, thread_idx, warp_idx, lane_idx),
      smem_iterator_A_(shared_storage.operand_A_ref(), thread_idx),
      smem_iterator_B_(shared_storage.operand_B_ref(), thread_idx)
  {
    int bid = lane_idx & 0x7;
    int group_id = (lane_idx >> 3) & 0x3;
    int cross_target = bid ^ group_id;
    int cross_offset_ = cross_target - bid;
    cross_offset_A_ = CrossOffsetA ? cross_offset_ : 0;
    cross_offset_B_ = CrossOffsetB ? cross_offset_ : 0;

    // Compute warp location within threadblock tile by mapping the warp_id to
    // three coordinates:
    //   _m: the warp's position within the threadblock along the M dimension
    //   _n: the warp's position within the threadblock along the N dimension
    //   _k: the warp's position within the threadblock along the K dimension

    int warp_idx_mn = warp_idx % (Base::WarpCount::kM * Base::WarpCount::kN);
    int warp_idx_k = warp_idx / (Base::WarpCount::kM * Base::WarpCount::kN);

    int warp_idx_m = warp_idx_mn % Base::WarpCount::kM;
    int warp_idx_n = warp_idx_mn / Base::WarpCount::kM;

    this->warp_tile_iterator_A_.pre_compute_pointer();
    this->warp_tile_iterator_B_.pre_compute_pointer();

    // Add per-warp offsets in units of warp-level tiles
    this->warp_tile_iterator_A_.add_tile_offset(
        {warp_idx_m, Base::kWarpGemmIterations * warp_idx_k});
    this->warp_tile_iterator_B_.add_tile_offset(
        {Base::kWarpGemmIterations * warp_idx_k, warp_idx_n});

    // Compute A shared memory ptr
    this->smem_iterator_A_.set_iteration_index(0);

    for(int stage = 0; stage < Base::kStages; ++stage){
      for(int j = 0; j < Detail::AsyncCopyIterationsPerStageA; ++j){
        for(int v = 0; v < IteratorA::kAccessesPerVector; ++v ){

          smem_ptr_A_[stage * Detail::AsyncCopyIterationsPerStageA + j * IteratorA::kAccessesPerVector + v] =
            reinterpret_cast<typename IteratorA::AccessType *>(this->smem_iterator_A_.get()) + v - cross_offset_A_;;
          ++this->smem_iterator_A_;
        }
      }
      this->smem_iterator_A_.add_tile_offset({0,1});
    }
    // Reset tile offset to zero
    this->smem_iterator_A_.add_tile_offset({0,-Base::kStages});

    // Compute B shared memory ptr
    this->smem_iterator_B_.set_iteration_index(0);
    MCTLASS_PRAGMA_UNROLL
    for(int stage = 0; stage < Base::kStages; ++stage){
      for(int j = 0; j < Detail::AsyncCopyIterationsPerStageB; ++j){
        for(int v = 0; v < IteratorB::kAccessesPerVector; ++v){
          smem_ptr_B_[stage * Detail::AsyncCopyIterationsPerStageB + j * IteratorB::kAccessesPerVector + v] =
            reinterpret_cast<typename IteratorA::AccessType *>(this->smem_iterator_B_.get())+ v - cross_offset_B_;
          ++this->smem_iterator_B_;
        }
      }
      this->smem_iterator_B_.add_tile_offset({1,0});
    }
    // Reset tile offset to zero
    this->smem_iterator_B_.add_tile_offset({-Base::kStages,0});
  }


  MCTLASS_DEVICE
  void copy_tiles_and_advance_A(IteratorA &iterator_A, int group_start_A = 0) {
    iterator_A.set_iteration_index(group_start_A *
                                   IteratorA::kAccessesPerVector);
    this->smem_iterator_A_.set_iteration_index(group_start_A);

    // Async Copy for operand A
    MCTLASS_PRAGMA_UNROLL
    for (int j = 0; j < Detail::AsyncCopyIterationsPerStageA; ++j) {
        typename IteratorA::AccessType *dst_ptr[IteratorA::kAccessesPerVector];
        typename IteratorA::AccessType *gmem_ptr[IteratorA::kAccessesPerVector];

        MCTLASS_PRAGMA_UNROLL
        for (int v = 0; v < IteratorA::kAccessesPerVector; ++v) {
          dst_ptr[v] = reinterpret_cast<typename IteratorA::AccessType *>(this->smem_iterator_A_.get()) + v - cross_offset_A_;
          gmem_ptr[v] = iterator_A.get() + cross_offset_A_;
          ++iterator_A;
        }
        MCTLASS_PRAGMA_UNROLL
        for (int v = 0; v < IteratorA::kAccessesPerVector; ++v) {
          if constexpr (SharedMemoryClear == SharedMemoryClearOption::kZfill) {
            mctlass::arch::cp_async_zfill<kSrcBytesA, kCacheOpA>(
                dst_ptr[v], gmem_ptr[v], true);
          } else {
            mctlass::arch::cp_async<kSrcBytesA, kCacheOpA>(
                dst_ptr[v], gmem_ptr[v], true);
          }
        }
        this->smem_iterator_A_.set_iteration_index(++group_start_A);
    }
  }

  MCTLASS_DEVICE
  void copy_tiles_and_advance_B(IteratorB &iterator_B, int group_start_B = 0) {

    iterator_B.set_iteration_index(group_start_B *
                                   IteratorB::kAccessesPerVector);
    this->smem_iterator_B_.set_iteration_index(group_start_B);

    // Async Copy for operand B
    MCTLASS_PRAGMA_UNROLL
    for (int j = 0; j < Detail::AsyncCopyIterationsPerStageB; ++j) {
        typename IteratorB::AccessType *dst_ptr[IteratorB::kAccessesPerVector];
        typename IteratorB::AccessType *gmem_ptr[IteratorB::kAccessesPerVector];

        MCTLASS_PRAGMA_UNROLL
        for (int v = 0; v < IteratorB::kAccessesPerVector; ++v) {
          dst_ptr[v] = reinterpret_cast<typename IteratorB::AccessType *>(this->smem_iterator_B_.get()) + v - cross_offset_B_;
          gmem_ptr[v] = iterator_B.get() + cross_offset_B_;
          ++iterator_B;
        }

        MCTLASS_PRAGMA_UNROLL
        for (int v = 0; v < IteratorB::kAccessesPerVector; ++v) {
          if (SharedMemoryClear == SharedMemoryClearOption::kZfill) {
            mctlass::arch::cp_async_zfill<kSrcBytesB, kCacheOpB>(
                dst_ptr[v], gmem_ptr[v], true);
          } else {
            mctlass::arch::cp_async<kSrcBytesB, kCacheOpB>(
                dst_ptr[v], gmem_ptr[v], true);
          }
        }
        this->smem_iterator_B_.set_iteration_index(++group_start_B);
    }
  }



  MCTLASS_DEVICE
  void copy_tiles_and_advance_A_pre(IteratorA &iterator_A, int stage,int group_start_A = 0){

    iterator_A.set_iteration_index(group_start_A *
                                   IteratorA::kAccessesPerVector);

    MCTLASS_PRAGMA_UNROLL
    for (int j = 0; j < Detail::AsyncCopyIterationsPerStageA; ++j) {
        typename IteratorA::AccessType *gmem_ptr[IteratorA::kAccessesPerVector];

        MCTLASS_PRAGMA_UNROLL
        for (int v = 0; v < IteratorA::kAccessesPerVector; ++v) {
          gmem_ptr[v] = iterator_A.get() + cross_offset_A_;
          ++iterator_A;
        }
        MCTLASS_PRAGMA_UNROLL
        for (int v = 0; v < IteratorA::kAccessesPerVector; ++v) {
          if constexpr (SharedMemoryClear == SharedMemoryClearOption::kZfill) {
            mctlass::arch::cp_async_zfill<kSrcBytesA, kCacheOpA>(
                smem_ptr_A_[stage * Detail::AsyncCopyIterationsPerStageA + j * IteratorA::kAccessesPerVector + v], gmem_ptr[v], true);
          } else {
            mctlass::arch::cp_async<kSrcBytesA, kCacheOpA>(
                smem_ptr_A_[stage * Detail::AsyncCopyIterationsPerStageA + j * IteratorA::kAccessesPerVector + v], gmem_ptr[v], true);
          }
        }
    }
  }

    MCTLASS_DEVICE
  void copy_tiles_and_advance_B_pre(IteratorB &iterator_B, int stage ,int group_start_B = 0) {

    iterator_B.set_iteration_index(group_start_B *
                                   IteratorB::kAccessesPerVector);

    // Async Copy for operand B
    MCTLASS_PRAGMA_UNROLL
    for (int j = 0; j < Detail::AsyncCopyIterationsPerStageB; ++j) {
        typename IteratorB::AccessType *gmem_ptr[IteratorB::kAccessesPerVector];

        MCTLASS_PRAGMA_UNROLL
        for (int v = 0; v < IteratorB::kAccessesPerVector; ++v) {
          gmem_ptr[v] = iterator_B.get() + cross_offset_B_;
          ++iterator_B;
        }

        MCTLASS_PRAGMA_UNROLL
        for (int v = 0; v < IteratorB::kAccessesPerVector; ++v) {
          if (SharedMemoryClear == SharedMemoryClearOption::kZfill) {
            mctlass::arch::cp_async_zfill<kSrcBytesB, kCacheOpB>(
                smem_ptr_B_[stage * Detail::AsyncCopyIterationsPerStageB + j * IteratorB::kAccessesPerVector + v], gmem_ptr[v], true);
          } else {
            mctlass::arch::cp_async<kSrcBytesB, kCacheOpB>(
                smem_ptr_B_[stage * Detail::AsyncCopyIterationsPerStageB + j * IteratorB::kAccessesPerVector + v], gmem_ptr[v], true);
          }
        }
    }
  }

  MCTLASS_DEVICE
  void copy_tiles_and_advance(IteratorA &iterator_A, IteratorB &iterator_B,
                              int group_start_A = 0, int group_start_B = 0) {
    copy_tiles_and_advance_A(iterator_A, group_start_A);
    copy_tiles_and_advance_B(iterator_B, group_start_B);
  }

#define GVM_ARRIVE_CNT(cnt) \
  __builtin_mxc_arrive(64 + cnt); \
  __builtin_mxc_barrier();

#define COMPUTE_FIRST_HALF_THREADS(launch_stage_a,launch_stage_b,load_a_idx,load_b_idx,compute_frag_idx,load_frag_idx) \
    iterator_A.add_tile_offset({0, 1}); \
    iterator_B.add_tile_offset({1, 0}); \
    copy_tiles_and_advance_A_pre(iterator_A,launch_stage_a); \
    copy_tiles_and_advance_B_pre(iterator_B,launch_stage_b); \
    asm(";------"); \
    this->warp_tile_iterator_B_.load_with_index(warp_loaded_frag_B[load_frag_idx],load_b_idx); \
    this->warp_tile_iterator_A_.load_with_index(warp_loaded_frag_A[load_frag_idx],load_a_idx); \
    asm(";------"); \
    warp_mma.frontHalfMma(accum,warp_loaded_frag_A[compute_frag_idx],warp_loaded_frag_B[compute_frag_idx],accum); \
    warp_mma.laterHalfMma(accum,warp_loaded_frag_A[compute_frag_idx],warp_loaded_frag_B[compute_frag_idx],accum); \
    asm(";------"); \
    this->warp_tile_iterator_B_.load_with_index(warp_loaded_frag_B[load_frag_idx + 1],load_b_idx + 1); \
    this->warp_tile_iterator_A_.load_with_index(warp_loaded_frag_A[load_frag_idx + 1],load_a_idx + 1); \
    asm(";------"); \
    warp_mma(accum,warp_loaded_frag_A[compute_frag_idx + 1],warp_loaded_frag_B[compute_frag_idx + 1],accum);

#define COMPUTE_SECOND_HALF_THREADS(launch_stage_a,launch_stage_b,load_a_idx,load_b_idx,compute_frag_idx,load_frag_idx) \
    iterator_A.add_tile_offset({0, 1}); \
    iterator_B.add_tile_offset({1, 0}); \
    copy_tiles_and_advance_A_pre(iterator_A, launch_stage_a); \
    copy_tiles_and_advance_B_pre(iterator_B, launch_stage_b); \
    warp_mma.frontHalfMma(accum, warp_loaded_frag_A[compute_frag_idx],warp_loaded_frag_B[compute_frag_idx], accum); \
    this->warp_tile_iterator_B_.load_with_index(warp_loaded_frag_B[load_frag_idx],load_b_idx); \
    this->warp_tile_iterator_A_.load_with_index(warp_loaded_frag_A[load_frag_idx],load_a_idx); \
    asm(";------"); \
    this->warp_tile_iterator_B_.load_with_index(warp_loaded_frag_B[load_frag_idx + 1],load_b_idx + 1); \
    warp_mma.laterHalfMma(accum, warp_loaded_frag_A[compute_frag_idx],warp_loaded_frag_B[compute_frag_idx], accum); \
    warp_mma.frontHalfMma(accum, warp_loaded_frag_A[compute_frag_idx + 1],warp_loaded_frag_B[compute_frag_idx + 1], accum); \
    asm(";------"); \
    this->warp_tile_iterator_A_.load_with_index(warp_loaded_frag_A[load_frag_idx + 1],load_a_idx + 1); \
    asm(";------"); \
    warp_mma.laterHalfMma(accum, warp_loaded_frag_A[compute_frag_idx + 1],warp_loaded_frag_B[compute_frag_idx + 1], accum);


  template<int N>
  struct Index{
    static constexpr int value = N;
  };

  template<int N>
  static constexpr int StageA = Index<N>::value;
  template<int N>
  static constexpr int StageB = Index<N>::value;

  template<int N>
  static constexpr int ComputeFragIdx = Index<N>::value;
  template<int N>
  static constexpr int LoadFragIdx = Index<N>::value;

  template<int N>
  static constexpr int LoadWarpTileIdxA = Index<N>::value;
  template<int N>
  static constexpr int LoadWarpTileIdxB = Index<N>::value;

  /// Perform a threadblock-scoped matrix multiply-accumulate
  MCTLASS_DEVICE
  void operator()(
      ///< problem size of GEMM
      int gemm_k_iterations,
      ///< destination accumulator tile
      FragmentC &accum,
      ///< iterator over A operand in global memory
      IteratorA iterator_A,
      ///< iterator over B operand in global memory
      IteratorB iterator_B,
      ///< initial value of accumulator
      FragmentC const &src_accum) {
    //
    // Prologue
    //
    static int const arrive_gvm_cnt = Detail::AsyncCopyIterationsPerStageA +
             Detail::AsyncCopyIterationsPerStageB;

    // Issue several complete stages
    copy_tiles_and_advance_A_pre(iterator_A,StageA<0>);
    copy_tiles_and_advance_B_pre(iterator_B,StageB<0>);

    // Defines the boundary of a stage of cp.async.
    // mctlass::arch::cp_async_fence();

    // Perform accumulation in the 'd' output operand
    accum = src_accum;

    //
    // Clear the remaining tiles of SMEM. This is a functional requirement for some kernels
    // so that all accumulator elements outside the GEMM footprint are zero.
    //

    if (SharedMemoryClear == SharedMemoryClearOption::kClearLastStage) {

      /// Iterator to write threadblock-scoped tile of A operand to shared memory
      SmemIteratorA last_smem_iterator_A(this->smem_iterator_A_);

      typename IteratorA::AccessType zero_A;
      zero_A.clear();

      last_smem_iterator_A.set_iteration_index(0);

      // Async Copy for operand A
      MCTLASS_PRAGMA_UNROLL
      for (int j = 0; j < Detail::AsyncCopyIterationsPerStageA; ++j) {

        typename IteratorA::AccessType *dst_ptr =
            reinterpret_cast<typename IteratorA::AccessType *>(
                last_smem_iterator_A.get());

        *dst_ptr = zero_A;

        ++last_smem_iterator_A;
      }

      /// Iterator to write threadblock-scoped tile of B operand to shared memory
      SmemIteratorB last_smem_iterator_B(this->smem_iterator_B_);
      typename IteratorB::AccessType zero_B;

      zero_B.clear();
      last_smem_iterator_B.set_iteration_index(0);

      // Async Copy for operand B
      MCTLASS_PRAGMA_UNROLL
      for (int j = 0; j < Detail::AsyncCopyIterationsPerStageB; ++j) {

        typename IteratorB::AccessType *dst_ptr =
            reinterpret_cast<typename IteratorB::AccessType *>(
                last_smem_iterator_B.get());

        *dst_ptr = zero_B;

        ++last_smem_iterator_B;
      }
    }

    // Waits until stages up to the previous (kStages-2)th stage have committed.
    // mctlass::arch::cp_async_wait<Base::kStages - 2>();
    // if constexpr (Syncthreads) __syncthreads();
    __builtin_mxc_arrive(64);
    __builtin_mxc_barrier();

    // Pair of fragments used to overlap shared memory loads and math
    // instructions
    WarpLoadedFragmentA warp_loaded_frag_A[4];
    WarpLoadedFragmentB warp_loaded_frag_B[4];

    Operator warp_mma;

    this->warp_tile_iterator_B_.load_with_index(warp_loaded_frag_B[LoadFragIdx<0>],LoadWarpTileIdxB<0>);
    this->warp_tile_iterator_A_.load_with_index(warp_loaded_frag_A[LoadFragIdx<0>],LoadWarpTileIdxA<0>);

    // Move to the next stage
    iterator_A.add_tile_offset({0, 1});
    iterator_B.add_tile_offset({1, 0});
    copy_tiles_and_advance_A_pre(iterator_A,StageB<1>);
    copy_tiles_and_advance_B_pre(iterator_B,StageB<1>);

    this->warp_tile_iterator_B_.load_with_index(warp_loaded_frag_B[LoadFragIdx<1>],LoadWarpTileIdxB<1>);
    this->warp_tile_iterator_A_.load_with_index(warp_loaded_frag_A[LoadFragIdx<1>],LoadWarpTileIdxA<1>);

    // tf32x3 kernels use staging accumulation. warp_mma uses a temporary
    // accumulator and this temporary accumulator is added to the final
    // accumulator once in every mainloop iteration.
    plus<FragmentC> plus_accum;

    FragmentC tmp_accum;

    if (platform::is_same<typename Operator::MathOperator,
                          arch::OpMultiplyAddFastF32>::value
      || platform::is_same<typename Operator::MathOperator,
                          arch::OpMultiplyAddComplexFastF32>::value) {

      tmp_accum.clear();
    }
    // Mainloop
    // first half threads
    if(threadIdx.x / 256 == 0){
      MCTLASS_GEMM_LOOP
      for (int k = 0; k < gemm_k_iterations / 2 - 1; ++k) {
        GVM_ARRIVE_CNT(0);
        COMPUTE_FIRST_HALF_THREADS(StageA<0>,StageB<0>,LoadWarpTileIdxA<2>,LoadWarpTileIdxB<2>,ComputeFragIdx<0>,LoadFragIdx<2>);
        GVM_ARRIVE_CNT(0);
        COMPUTE_FIRST_HALF_THREADS(StageA<1>,StageB<1>,LoadWarpTileIdxA<0>,LoadWarpTileIdxB<0>,ComputeFragIdx<2>,LoadFragIdx<0>);
      }
    }
    else // second half threads
    {
      MCTLASS_GEMM_LOOP
      for (int k = 0; k < gemm_k_iterations / 2 - 1; ++k) {
        GVM_ARRIVE_CNT(0);
        COMPUTE_SECOND_HALF_THREADS(StageA<0>,StageB<0>,LoadWarpTileIdxA<2>,LoadWarpTileIdxB<2>,ComputeFragIdx<0>,LoadFragIdx<2>);
        GVM_ARRIVE_CNT(0);
        COMPUTE_SECOND_HALF_THREADS(StageA<1>,StageB<1>,LoadWarpTileIdxA<0>,LoadWarpTileIdxB<0>,ComputeFragIdx<2>,LoadFragIdx<0>);
      }
    }

    GVM_ARRIVE_CNT(0);
    if (gemm_k_iterations % 2 == 1) {
      iterator_A.add_tile_offset({0, 1});
      iterator_B.add_tile_offset({1, 0});
      copy_tiles_and_advance_A_pre(iterator_A,StageA<0>);
      copy_tiles_and_advance_B_pre(iterator_B,StageB<0>);
    }
    this->warp_tile_iterator_B_.load_with_index(warp_loaded_frag_B[LoadFragIdx<2>],LoadWarpTileIdxB<2>);
    this->warp_tile_iterator_A_.load_with_index(warp_loaded_frag_A[LoadFragIdx<2>],LoadWarpTileIdxA<2>);
    this->warp_tile_iterator_B_.load_with_index(warp_loaded_frag_B[LoadFragIdx<3>],LoadWarpTileIdxB<3>);
    this->warp_tile_iterator_A_.load_with_index(warp_loaded_frag_A[LoadFragIdx<3>],LoadWarpTileIdxA<3>);
    warp_mma(accum, warp_loaded_frag_A[ComputeFragIdx<0>],warp_loaded_frag_B[ComputeFragIdx<0>], accum);
    warp_mma(accum, warp_loaded_frag_A[ComputeFragIdx<1>],warp_loaded_frag_B[ComputeFragIdx<1>], accum);
    warp_mma(accum, warp_loaded_frag_A[ComputeFragIdx<2>],warp_loaded_frag_B[ComputeFragIdx<2>], accum);
    warp_mma(accum, warp_loaded_frag_A[ComputeFragIdx<3>],warp_loaded_frag_B[ComputeFragIdx<3>], accum);

    if (gemm_k_iterations % 2 == 1) {
      GVM_ARRIVE_CNT(0);
      this->warp_tile_iterator_B_.load_with_index(warp_loaded_frag_B[LoadFragIdx<0>],LoadWarpTileIdxB<0>);
      this->warp_tile_iterator_A_.load_with_index(warp_loaded_frag_A[LoadFragIdx<0>],LoadWarpTileIdxA<0>);
      this->warp_tile_iterator_B_.load_with_index(warp_loaded_frag_B[LoadFragIdx<1>],LoadWarpTileIdxB<1>);
      this->warp_tile_iterator_A_.load_with_index(warp_loaded_frag_A[LoadFragIdx<1>],LoadWarpTileIdxA<1>);
      warp_mma(accum, warp_loaded_frag_A[ComputeFragIdx<0>],warp_loaded_frag_B[ComputeFragIdx<0>], accum);
      warp_mma(accum, warp_loaded_frag_A[ComputeFragIdx<1>],warp_loaded_frag_B[ComputeFragIdx<1>], accum);
    }

    if (platform::is_same<typename Operator::MathOperator,
                          arch::OpMultiplyAddFastF32>::value
      || platform::is_same<typename Operator::MathOperator,
                           arch::OpMultiplyAddComplexFastF32>::value) {
      accum = plus_accum(accum, tmp_accum);
    }

    if (SharedMemoryClear == SharedMemoryClearOption::kZfill) {
      // commit and drain all pending and predicated LDGSTS pnz from the GEMM mainloop
      mctlass::arch::cp_async_fence();
      mctlass::arch::cp_async_wait<0>();
      if constexpr (!MacaSpecialTag) __syncthreads();
    }

  }
#undef GVM_ARRIVE_CNT
#undef COMPUTE_FIRST_HALF_THREADS
#undef COMPUTE_SECOND_HALF_THREADS
};

// mcTlass: problem: NN && stage==2
//          (SmemIteratorA_::Layout == MacaRowMajorTensorOpMultiplicandMultipleLdgCrosswise) &&
//          (SmemIteratorB_::Layout == MacaRowMajorTensorOpMultiplicandCongruous4x4Perm) &&
//          (stage == 2)
template <
    /// Size of the Gemm problem - concept: gemm::GemmShape<>
    typename Shape_,
    /// Iterates over tiles of A operand in global memory
    //  (concept: ReadableTileIterator | ForwardTileIterator |
    //  MaskedTileIterator)
    typename IteratorA_,
    /// Iterates over tiles of A operand in shared memory
    /// (concept: WriteableTileIterator | RandomAccessTileIterator)
    typename SmemIteratorA_,
    /// Cache operation for operand A
    mctlass::arch::CacheOperation::Kind CacheOpA,
    /// Iterates over tiles of B operand in global memory
    //  (concept: ReadableTileIterator | ForwardTileIterator |
    //  MaskedTileIterator)
    typename IteratorB_,
    /// Iterates over tiles of B operand in shared memory
    /// (concept: WriteableTileIterator | RandomAccessTileIterator)
    typename SmemIteratorB_,
    /// Cache operation for operand B
    mctlass::arch::CacheOperation::Kind CacheOpB,
    /// Data type of accumulator matrix
    typename ElementC_,
    /// Data type of accumulator matrix
    typename LayoutC_,
    /// Policy describing tuning details (concept: MmaPolicy)
    typename Policy_,
    /// Number of stages,
    // int Stages,
    /// Use zfill or predicate for out-of-bound cp.async
    SharedMemoryClearOption SharedMemoryClear,
    /// Flag tag for special process for m16n16k16 f16
    bool MacaSpecialTag_
    >
class MmaMultistage <
  Shape_,
  IteratorA_,
  SmemIteratorA_,
  CacheOpA,
  IteratorB_,
  SmemIteratorB_,
  CacheOpB,
  ElementC_,
  LayoutC_,
  Policy_,
  2,
  SharedMemoryClear,
  MacaSpecialTag_,
  typename std::enable_if<
    ((std::is_same<typename SmemIteratorA_::Layout,
        typename layout::MacaRowMajorTensorOpMultiplicandMultipleLdgCrosswise<
          sizeof_bits<int8_t>::value, Shape_::kK>>::value &&
      std::is_same<typename SmemIteratorB_::Layout,
        typename layout::MacaRowMajorTensorOpMultiplicandCongruous4x4Perm<
          sizeof_bits<int8_t>::value, int(128 / sizeof(int8_t))>>::value)),
    bool>::type
  >:public MmaBase<Shape_, Policy_, 2> {
public:
  ///< Base class
  using Base = MmaBase<Shape_, Policy_, 2>;
  ///< Size of the Gemm problem - concept: gemm::GemmShape<>
  using Shape = Shape_;
  ///< Iterates over tiles of A operand in global memory
  using IteratorA = IteratorA_;
  ///< Iterates over tiles of B operand in global memory
  using IteratorB = IteratorB_;
  ///< Data type of accumulator matrix
  using ElementC = ElementC_;
  ///< Layout of accumulator matrix
  using LayoutC = LayoutC_;
  ///< Policy describing tuning details
  using Policy = Policy_;

  using SmemIteratorA = SmemIteratorA_;
  using SmemIteratorB = SmemIteratorB_;

  static mctlass::arch::CacheOperation::Kind const kCacheOpA = CacheOpA;
  static mctlass::arch::CacheOperation::Kind const kCacheOpB = CacheOpB;
  // static constexpr bool MacaSpecialTag = MacaSpecialTag_;
  static constexpr bool MacaSpecialTag = false;

  static constexpr bool CrossOffsetA =
    platform::is_same<layout::MacaRowMajorTensorOpMultiplicandMultipleLdgCrosswise<
      sizeof_bits<typename IteratorA::Element>::value, Shape::kK>,
    typename SmemIteratorA::Layout>::value;
  static constexpr bool CrossOffsetB =
    platform::is_same<layout::MacaColumnMajorTensorOpMultiplicandMultipleLdgCrosswise<
      sizeof_bits<typename IteratorB::Element>::value, Shape::kK>,
    typename SmemIteratorB::Layout>::value;

  static_assert(Shape::kK % (512 / sizeof_bits<typename IteratorB::Element>::value) == 0,
      "Crosswise kContiguous must be 512bit aligned");
  //
  // Dependent types
  //

  /// Fragment of accumulator tile
  using FragmentC = typename Policy::Operator::FragmentC;

  /// Warp-level Mma
  using Operator = typename Policy::Operator;

  /// Minimum architecture is Sm80 to support cp.async
  using ArchTag = arch::Sm80;

  /// Complex transform on A operand
  static ComplexTransform const kTransformA = Operator::kTransformA;

  /// Complex transform on B operand
  static ComplexTransform const kTransformB = Operator::kTransformB;

  /// Internal structure exposed for introspection.
  struct Detail {

    /// Number of cp.async instructions to load one stage of operand A
    static int const AsyncCopyIterationsPerStageA =
        IteratorA::ThreadMap::Iterations::kCount;

    /// Number of cp.async instructions to load one stage of operand B
    static int const AsyncCopyIterationsPerStageB =
        IteratorB::ThreadMap::Iterations::kCount;

    /// Number of stages
    static int const kStages = 2;
  };

 private:

  using WarpLoadedFragmentA = typename Operator::FragmentA;
  using WarpLoadedFragmentB = typename Operator::FragmentB;

 private:

  //
  // Data members
  //

  /// Iterator to write threadblock-scoped tile of A operand to shared memory
  SmemIteratorA smem_iterator_A_;

  /// Iterator to write threadblock-scoped tile of B operand to shared memory
  SmemIteratorB smem_iterator_B_;

  int cross_offset_A_;
  int cross_offset_B_;

  int register_offset_l = 0;
  int register_offset_r = 0;

  typename IteratorA::AccessType* smem_ptr_A_[Base::kStages * Detail::AsyncCopyIterationsPerStageA * IteratorA::kAccessesPerVector];
  typename IteratorB::AccessType* smem_ptr_B_[Base::kStages * Detail::AsyncCopyIterationsPerStageB * IteratorB::kAccessesPerVector];

public:

  /// Construct from tensor references
  MCTLASS_DEVICE
  MmaMultistage(
      ///< Shared storage needed for internal use by threadblock-scoped GEMM
      typename Base::SharedStorage &shared_storage,
      ///< ID within the threadblock
      int thread_idx,
      ///< ID of warp
      int warp_idx,
      ///< ID of each thread within a warp
      int lane_idx,
      int problem_size_k = 0,
      typename IteratorA::Element *ptr_A = nullptr,
      typename IteratorA::Element *ptr_B = nullptr
    ):
      Base(shared_storage, thread_idx, warp_idx, lane_idx),
      smem_iterator_A_(shared_storage.operand_A_ref(), thread_idx),
      smem_iterator_B_(shared_storage.operand_B_ref(), thread_idx)
  {
    int bid = lane_idx & 0x7;
    int group_id = (lane_idx >> 3) & 0x3;
    int cross_target = bid ^ group_id;
    int cross_offset_ = cross_target - bid;
    cross_offset_A_ = CrossOffsetA ? cross_offset_ : 0;
    cross_offset_B_ = CrossOffsetB ? cross_offset_ : 0;

    if ((threadIdx.x / 32) % 2 == 0) {
      if ((threadIdx.x / 8) % 2 != 0) {
        register_offset_l = 1;
        register_offset_r = -1;
      }
    } else {
      if ((threadIdx.x / 8) % 2 == 0) {
        register_offset_l = 1;
        register_offset_r = -1;
      }
    }

    // Compute warp location within threadblock tile by mapping the warp_id to
    // three coordinates:
    //   _m: the warp's position within the threadblock along the M dimension
    //   _n: the warp's position within the threadblock along the N dimension
    //   _k: the warp's position within the threadblock along the K dimension

    int warp_idx_mn = warp_idx % (Base::WarpCount::kM * Base::WarpCount::kN);
    int warp_idx_k = warp_idx / (Base::WarpCount::kM * Base::WarpCount::kN);

    int warp_idx_m = warp_idx_mn % Base::WarpCount::kM;
    int warp_idx_n = warp_idx_mn / Base::WarpCount::kM;

    this->warp_tile_iterator_B_.add_tile_offset(
        {Base::kWarpGemmIterations * warp_idx_k, warp_idx_n});

    // Add per-warp offsets in units of warp-level tiles
    this->warp_tile_iterator_A_.pre_compute_pointer(warp_idx_m);
    this->warp_tile_iterator_B_.pre_compute_pointer();

    // Compute A shared memory ptr
    this->smem_iterator_A_.set_iteration_index(0);

    for(int stage = 0; stage < Base::kStages; ++stage){
      for(int j = 0; j < Detail::AsyncCopyIterationsPerStageA; ++j){
        for(int v = 0; v < IteratorA::kAccessesPerVector; ++v ){

          smem_ptr_A_[stage * Detail::AsyncCopyIterationsPerStageA + j * IteratorA::kAccessesPerVector + v] =
            reinterpret_cast<typename IteratorA::AccessType *>(this->smem_iterator_A_.get()) + v - cross_offset_A_;;
          ++this->smem_iterator_A_;
        }
      }
      this->smem_iterator_A_.add_tile_offset({0,1});
    }
    // Reset tile offset to zero
    this->smem_iterator_A_.add_tile_offset({0,-Base::kStages});

    // Compute B shared memory ptr
    this->smem_iterator_B_.set_iteration_index(0);
    MCTLASS_PRAGMA_UNROLL
    for(int stage = 0; stage < Base::kStages; ++stage){
      for(int j = 0; j < Detail::AsyncCopyIterationsPerStageB; ++j){
        for(int v = 0; v < IteratorB::kAccessesPerVector; ++v){
          smem_ptr_B_[stage * Detail::AsyncCopyIterationsPerStageB + j * IteratorB::kAccessesPerVector + v] =
            reinterpret_cast<typename IteratorA::AccessType *>(this->smem_iterator_B_.get())+ v - cross_offset_B_;
          ++this->smem_iterator_B_;
        }
      }
      this->smem_iterator_B_.add_tile_offset({1,0});
    }
    // Reset tile offset to zero
    this->smem_iterator_B_.add_tile_offset({-Base::kStages,0});
  }

  MCTLASS_DEVICE
  void copy_tiles_and_advance_2register_A_(IteratorA &iterator_A, int group_start_A = 0, v4i *reg_ptr_A_ = nullptr, int32_t index_A = 0) {
    iterator_A.set_iteration_index(group_start_A * IteratorA::kAccessesPerVector);
    MCTLASS_PRAGMA_UNROLL
    for (int j = 0; j < Detail::AsyncCopyIterationsPerStageA; ++j) {
      MCTLASS_PRAGMA_UNROLL
      for (int v = 0; v < IteratorA::kAccessesPerVector; ++v) {
        reg_ptr_A_[index_A] = __builtin_mxc_load_global_async128((v4i *)(iterator_A.get() + cross_offset_A_));
        ++iterator_A;
        ++(index_A);
      }
    }
  }

  MCTLASS_DEVICE
  void copy_tiles_and_advance_2share_A_(int stage = 0, v4i *reg_ptr_A = nullptr, int32_t index_A = 0) {
    // Async Copy for operand A
    using HalfAccessType = Array<typename IteratorB::Element, IteratorB::ThreadMap::kElementsPerAccess / 2>;
    MCTLASS_PRAGMA_UNROLL
    for (int j = 0; j < Detail::AsyncCopyIterationsPerStageA; ++j) {
      MCTLASS_PRAGMA_UNROLL
      for (int v = 0; v < IteratorA::kAccessesPerVector; ++v) {
        HalfAccessType *reg_value = static_cast<HalfAccessType *>((void *)(&reg_ptr_A[index_A]));
        HalfAccessType *half_dst_ptr = reinterpret_cast<HalfAccessType *>(smem_ptr_A_[stage * Detail::AsyncCopyIterationsPerStageA + j * IteratorA::kAccessesPerVector + v]);

        *static_cast<HalfAccessType *>((void *)&half_dst_ptr[0  + register_offset_l]) = *static_cast<HalfAccessType *>((void *)(&reg_value[0]));
        *static_cast<HalfAccessType *>((void *)&half_dst_ptr[1  + register_offset_r]) = *static_cast<HalfAccessType *>((void *)(&reg_value[1]));

        ++(index_A);
      }
    }
  }

  MCTLASS_DEVICE
  void copy_tiles_and_advance_2register_B_(IteratorB &iterator_B, int group_start_B = 0,
                                        v4i *reg_ptr_B = nullptr, int32_t index_B = 0) {
    iterator_B.set_iteration_index(group_start_B * IteratorB::kAccessesPerVector);

    MCTLASS_PRAGMA_UNROLL
    for (int j = 0; j < Detail::AsyncCopyIterationsPerStageB; ++j) {
      MCTLASS_PRAGMA_UNROLL
      for (int v = 0; v < IteratorB::kAccessesPerVector; ++v) {
        reg_ptr_B[index_B]=__builtin_mxc_load_global_async128((v4i *)(iterator_B.get() + cross_offset_B_));
        ++iterator_B;
        ++(index_B);
      }
    }
  }

  MCTLASS_DEVICE
  void copy_tiles_and_advance_2share_B_(int stage = 0, v4i *reg_ptr_B = nullptr, int32_t index_B = 0) {
    // Async Copy for operand B
    using HalfAccessType = Array<typename IteratorB::Element, IteratorB::ThreadMap::kElementsPerAccess / 2>;
    MCTLASS_PRAGMA_UNROLL
    for (int j = 0; j < Detail::AsyncCopyIterationsPerStageB; ++j) {
      MCTLASS_PRAGMA_UNROLL
      for (int v = 0; v < IteratorB::kAccessesPerVector; ++v) {

        HalfAccessType *reg_value = static_cast<HalfAccessType *>((void *)(&reg_ptr_B[index_B]));
        HalfAccessType *half_dst_ptr = reinterpret_cast<HalfAccessType *>(smem_ptr_B_[stage * Detail::AsyncCopyIterationsPerStageB + j * IteratorB::kAccessesPerVector + v]);
        *static_cast<HalfAccessType *>((void *)&half_dst_ptr[0]) = *static_cast<HalfAccessType *>((void *)(&reg_value[0]));
        *static_cast<HalfAccessType *>((void *)&half_dst_ptr[1]) = *static_cast<HalfAccessType *>((void *)(&reg_value[1]));

        ++(index_B);
      }
    }
  }

  /// Perform a threadblock-scoped matrix multiply-accumulate
  MCTLASS_DEVICE
  void operator()(
      ///< problem size of GEMM
      int gemm_k_iterations,
      ///< destination accumulator tile
      FragmentC &accum,
      ///< iterator over A operand in global memory
      IteratorA iterator_A,
      ///< iterator over B operand in global memory
      IteratorB iterator_B,
      ///< initial value of accumulator
      FragmentC const &src_accum) {
    // Issue several complete stages
    v4i smem_register_A_[Detail::AsyncCopyIterationsPerStageA * IteratorA::kAccessesPerVector];
    v4i smem_register_B_[Detail::AsyncCopyIterationsPerStageB * IteratorB::kAccessesPerVector];

    copy_tiles_and_advance_2register_A_(iterator_A, 0, smem_register_A_);
    copy_tiles_and_advance_2register_B_(iterator_B, 0, smem_register_B_);

    // Perform accumulation in the 'd' output operand
    accum = src_accum;

    //
    // Clear the remaining tiles of SMEM. This is a functional requirement for some kernels
    // so that all accumulator elements outside the GEMM footprint are zero.
    //

    if (SharedMemoryClear == SharedMemoryClearOption::kClearLastStage) {

      /// Iterator to write threadblock-scoped tile of A operand to shared memory
      SmemIteratorA last_smem_iterator_A(this->smem_iterator_A_);

      typename IteratorA::AccessType zero_A;
      zero_A.clear();

      last_smem_iterator_A.set_iteration_index(0);

      // Async Copy for operand A
      MCTLASS_PRAGMA_UNROLL
      for (int j = 0; j < Detail::AsyncCopyIterationsPerStageA; ++j) {

        typename IteratorA::AccessType *dst_ptr =
            reinterpret_cast<typename IteratorA::AccessType *>(
                last_smem_iterator_A.get());

        *dst_ptr = zero_A;

        ++last_smem_iterator_A;
      }

      /// Iterator to write threadblock-scoped tile of B operand to shared memory
      SmemIteratorB last_smem_iterator_B(this->smem_iterator_B_);
      typename IteratorB::AccessType zero_B;

      zero_B.clear();
      last_smem_iterator_B.set_iteration_index(0);

      // Async Copy for operand B
      MCTLASS_PRAGMA_UNROLL
      for (int j = 0; j < Detail::AsyncCopyIterationsPerStageB; ++j) {

        typename IteratorB::AccessType *dst_ptr =
            reinterpret_cast<typename IteratorB::AccessType *>(
                last_smem_iterator_B.get());

        *dst_ptr = zero_B;

        ++last_smem_iterator_B;
      }
    }

    __builtin_mxc_arrive(64);
    copy_tiles_and_advance_2share_A_(0, smem_register_A_);
    copy_tiles_and_advance_2share_B_(0, smem_register_B_);

    // Move to the next stage
    iterator_A.add_tile_offset({0, 1});
    iterator_B.add_tile_offset({1, 0});

    copy_tiles_and_advance_2register_A_(iterator_A, 0, smem_register_A_);
    copy_tiles_and_advance_2register_B_(iterator_B, 0, smem_register_B_);

    __builtin_mxc_barrier();

    // Pair of fragments used to overlap shared memory loads and math
    // instructions
    WarpLoadedFragmentA warp_loaded_frag_A[4];
    WarpLoadedFragmentB warp_loaded_frag_B[4];

    Operator warp_mma;

    this->warp_tile_iterator_B_.load_with_index(warp_loaded_frag_B[0],0);
    this->warp_tile_iterator_A_.load_with_index(warp_loaded_frag_A[0],0);

    this->warp_tile_iterator_B_.load_with_index(warp_loaded_frag_B[1],1);
    this->warp_tile_iterator_A_.load_with_index(warp_loaded_frag_A[1],1);

    __builtin_mxc_arrive(64);

    copy_tiles_and_advance_2share_A_(1, smem_register_A_);
    copy_tiles_and_advance_2share_B_(1, smem_register_B_);

    // tf32x3 kernels use staging accumulation. warp_mma uses a temporary
    // accumulator and this temporary accumulator is added to the final
    // accumulator once in every mainloop iteration.
    plus<FragmentC> plus_accum;

    FragmentC tmp_accum;

    if (platform::is_same<typename Operator::MathOperator,
                          arch::OpMultiplyAddFastF32>::value
      || platform::is_same<typename Operator::MathOperator,
                          arch::OpMultiplyAddComplexFastF32>::value) {

      tmp_accum.clear();
    }

    //
    // Mainloop
    //
    v4i smem_register_A_1[Detail::AsyncCopyIterationsPerStageA * IteratorA::kAccessesPerVector];
    v4i smem_register_B_1[Detail::AsyncCopyIterationsPerStageB * IteratorB::kAccessesPerVector];

    v4i smem_register_A_2[Detail::AsyncCopyIterationsPerStageA * IteratorA::kAccessesPerVector];
    v4i smem_register_B_2[Detail::AsyncCopyIterationsPerStageB * IteratorB::kAccessesPerVector];

    v4i smem_register_A_3[Detail::AsyncCopyIterationsPerStageA * IteratorA::kAccessesPerVector];
    v4i smem_register_B_3[Detail::AsyncCopyIterationsPerStageB * IteratorB::kAccessesPerVector];

    v4i smem_register_A_4[Detail::AsyncCopyIterationsPerStageA * IteratorA::kAccessesPerVector];
    v4i smem_register_B_4[Detail::AsyncCopyIterationsPerStageB * IteratorB::kAccessesPerVector];

    iterator_A.add_tile_offset({0, 1});
    iterator_B.add_tile_offset({1, 0});
    asm(";------");
    copy_tiles_and_advance_2register_A_(iterator_A, 0, smem_register_A_1);
    copy_tiles_and_advance_2register_B_(iterator_B, 0, smem_register_B_1);

    iterator_A.add_tile_offset({0, 1});
    iterator_B.add_tile_offset({1, 0});
    asm(";------");
    copy_tiles_and_advance_2register_A_(iterator_A, 0, smem_register_A_2);
    copy_tiles_and_advance_2register_B_(iterator_B, 0, smem_register_B_2);

    __builtin_mxc_arrive(64 + 3);

    MCTLASS_GEMM_LOOP
    for (int k = 0; k < gemm_k_iterations / 4 - 1; ++k) {
      //
      // Loop over GEMM K dimension
      //

      if constexpr(2 == Base::kWarpGemmIterations){

        //k=0
        iterator_A.add_tile_offset({0, 1});
        iterator_B.add_tile_offset({1, 0});

        __builtin_mxc_barrier();
        copy_tiles_and_advance_2register_A_(iterator_A, 0, smem_register_A_3);
        copy_tiles_and_advance_2register_B_(iterator_B, 0, smem_register_B_3);
        asm(";------");
        this->warp_tile_iterator_A_.load_with_index(warp_loaded_frag_A[2],2);
        asm(";------");
        warp_mma.frontHalfFrontMma(accum,warp_loaded_frag_A[0],warp_loaded_frag_B[0],accum);
        asm(";------");
        this->warp_tile_iterator_B_.load_with_index(warp_loaded_frag_B[2],2);
        warp_mma.frontHalfLaterMma(accum,warp_loaded_frag_A[0],warp_loaded_frag_B[0],accum);
        asm(";------");
        warp_mma.laterHalfFrontMma(accum,warp_loaded_frag_A[0],warp_loaded_frag_B[0],accum);
        warp_mma.laterHalfLaterMma(accum,warp_loaded_frag_A[0],warp_loaded_frag_B[0],accum);
        this->warp_tile_iterator_B_.load_with_index(warp_loaded_frag_B[3],3);
        this->warp_tile_iterator_A_.load_with_index(warp_loaded_frag_A[3],3);

        warp_mma.frontHalfMma(accum,warp_loaded_frag_A[1],warp_loaded_frag_B[1],accum);
        asm(";------");
        copy_tiles_and_advance_2share_A_(0, smem_register_A_1);
        copy_tiles_and_advance_2share_B_(0, smem_register_B_1);
        warp_mma.laterHalfMma(accum,warp_loaded_frag_A[1],warp_loaded_frag_B[1],accum);

        // k = 1

        __builtin_mxc_barrier();
        iterator_A.add_tile_offset({0, 1});
        iterator_B.add_tile_offset({1, 0});
        copy_tiles_and_advance_2register_A_(iterator_A, 0, smem_register_A_4);
        copy_tiles_and_advance_2register_B_(iterator_B, 0, smem_register_B_4);

        asm(";------");
        this->warp_tile_iterator_A_.load_with_index(warp_loaded_frag_A[0],0);
        asm(";------");
        warp_mma.frontHalfFrontMma(accum,warp_loaded_frag_A[2],warp_loaded_frag_B[2],accum);
        asm(";------");
        this->warp_tile_iterator_B_.load_with_index(warp_loaded_frag_B[0],0);
        asm(";------");
        warp_mma.frontHalfLaterMma(accum,warp_loaded_frag_A[2],warp_loaded_frag_B[2],accum);
        asm(";------");
        warp_mma.laterHalfMma(accum, warp_loaded_frag_A[2],warp_loaded_frag_B[2], accum);
        asm(";------");
        this->warp_tile_iterator_B_.load_with_index(warp_loaded_frag_B[1],1);
        this->warp_tile_iterator_A_.load_with_index(warp_loaded_frag_A[1],1);
        asm(";------");
        __builtin_mxc_arrive(64 + 3);
        warp_mma.frontHalfMma(accum, warp_loaded_frag_A[3],warp_loaded_frag_B[3], accum);
        warp_mma.laterHalfMma(accum, warp_loaded_frag_A[3],warp_loaded_frag_B[3], accum);

        copy_tiles_and_advance_2share_A_(1, smem_register_A_2);
        copy_tiles_and_advance_2share_B_(1, smem_register_B_2);

        // k=2~3
        __builtin_mxc_barrier();
        iterator_A.add_tile_offset({0, 1});
        iterator_B.add_tile_offset({1, 0});
        copy_tiles_and_advance_2register_A_(iterator_A, 0, smem_register_A_1); // -> k = 1 -> ldg
        copy_tiles_and_advance_2register_B_(iterator_B, 0, smem_register_B_1);

        asm(";------");
        this->warp_tile_iterator_A_.load_with_index(warp_loaded_frag_A[2],2);
        asm(";------");
        warp_mma.frontHalfMma(accum,warp_loaded_frag_A[0],warp_loaded_frag_B[0],accum);
        asm(";------");
        this->warp_tile_iterator_B_.load_with_index(warp_loaded_frag_B[2],2);
        asm(";------");
        warp_mma.laterHalfMma(accum,warp_loaded_frag_A[0],warp_loaded_frag_B[0],accum);
        asm(";------");
        this->warp_tile_iterator_B_.load_with_index(warp_loaded_frag_B[3],3);
        this->warp_tile_iterator_A_.load_with_index(warp_loaded_frag_A[3],3);
        __builtin_mxc_arrive(64 + 3);
        asm(";------");
        warp_mma.frontHalfMma(accum,warp_loaded_frag_A[1],warp_loaded_frag_B[1],accum);
        asm(";------");
        warp_mma.laterHalfMma(accum,warp_loaded_frag_A[1],warp_loaded_frag_B[1],accum);

        copy_tiles_and_advance_2share_A_(0, smem_register_A_3);
        copy_tiles_and_advance_2share_B_(0, smem_register_B_3);


        __builtin_mxc_barrier();
        iterator_A.add_tile_offset({0, 1});
        iterator_B.add_tile_offset({1, 0});

        copy_tiles_and_advance_2register_A_(iterator_A, 0, smem_register_A_2);
        copy_tiles_and_advance_2register_B_(iterator_B, 0, smem_register_B_2);

        asm(";------");
        this->warp_tile_iterator_A_.load_with_index(warp_loaded_frag_A[0],0);
        asm(";------");
        warp_mma.frontHalfMma(accum, warp_loaded_frag_A[2],warp_loaded_frag_B[2], accum);
        asm(";------");
        this->warp_tile_iterator_B_.load_with_index(warp_loaded_frag_B[0],0);
        asm(";------");
        warp_mma.laterHalfMma(accum, warp_loaded_frag_A[2],warp_loaded_frag_B[2], accum);
        asm(";------");
        this->warp_tile_iterator_B_.load_with_index(warp_loaded_frag_B[1],1);
        this->warp_tile_iterator_A_.load_with_index(warp_loaded_frag_A[1],1);
        __builtin_mxc_arrive(64 + 3);
        asm(";------");
        warp_mma.frontHalfMma(accum, warp_loaded_frag_A[3],warp_loaded_frag_B[3], accum);
        asm(";------");
        warp_mma.laterHalfMma(accum, warp_loaded_frag_A[3],warp_loaded_frag_B[3], accum);

        copy_tiles_and_advance_2share_A_(1, smem_register_A_4);
        copy_tiles_and_advance_2share_B_(1, smem_register_B_4);
        asm(";------");

      }
    }

    int residue_gemm_k_iterations = gemm_k_iterations % 4;

    if (gemm_k_iterations >= 4) {
      //k=0
      if (residue_gemm_k_iterations >= 1) {
        iterator_A.add_tile_offset({0, 1});
        iterator_B.add_tile_offset({1, 0});
        asm(";------");
        copy_tiles_and_advance_2register_A_(iterator_A, 0, smem_register_A_3);
        copy_tiles_and_advance_2register_B_(iterator_B, 0, smem_register_B_3);
      }
      __builtin_mxc_barrier();
      this->warp_tile_iterator_A_.load_with_index(warp_loaded_frag_A[2],2);
      asm(";------");
      warp_mma.frontHalfFrontMma(accum,warp_loaded_frag_A[0],warp_loaded_frag_B[0],accum);
      asm(";------");
      this->warp_tile_iterator_B_.load_with_index(warp_loaded_frag_B[2],2);
      warp_mma.frontHalfLaterMma(accum,warp_loaded_frag_A[0],warp_loaded_frag_B[0],accum);
      asm(";------");
      warp_mma.laterHalfFrontMma(accum,warp_loaded_frag_A[0],warp_loaded_frag_B[0],accum);
      warp_mma.laterHalfLaterMma(accum,warp_loaded_frag_A[0],warp_loaded_frag_B[0],accum);
      this->warp_tile_iterator_B_.load_with_index(warp_loaded_frag_B[3],3);
      this->warp_tile_iterator_A_.load_with_index(warp_loaded_frag_A[3],3);


      warp_mma.frontHalfMma(accum,warp_loaded_frag_A[1],warp_loaded_frag_B[1],accum);
      asm(";------");
      copy_tiles_and_advance_2share_A_(0, smem_register_A_1);
      copy_tiles_and_advance_2share_B_(0, smem_register_B_1);
      warp_mma.laterHalfMma(accum,warp_loaded_frag_A[1],warp_loaded_frag_B[1],accum);

      // k = 1
      if (residue_gemm_k_iterations >= 2) {
        iterator_A.add_tile_offset({0, 1});
        iterator_B.add_tile_offset({1, 0});

        copy_tiles_and_advance_2register_A_(iterator_A, 0, smem_register_A_4);
        copy_tiles_and_advance_2register_B_(iterator_B, 0, smem_register_B_4);
      }
      __builtin_mxc_barrier();

      warp_mma.frontHalfFrontMma(accum,warp_loaded_frag_A[2],warp_loaded_frag_B[2],accum);
      asm(";------");
      this->warp_tile_iterator_B_.load_with_index(warp_loaded_frag_B[0],0);
      this->warp_tile_iterator_A_.load_with_index(warp_loaded_frag_A[0],0);
      asm(";------");
      warp_mma.frontHalfLaterMma(accum,warp_loaded_frag_A[2],warp_loaded_frag_B[2],accum);
      asm(";------");
      warp_mma.laterHalfMma(accum, warp_loaded_frag_A[2],warp_loaded_frag_B[2], accum);
      asm(";------");
      this->warp_tile_iterator_B_.load_with_index(warp_loaded_frag_B[1],1);
      this->warp_tile_iterator_A_.load_with_index(warp_loaded_frag_A[1],1);
      asm(";------");
      __builtin_mxc_arrive(64 + 3);
      warp_mma.frontHalfMma(accum, warp_loaded_frag_A[3],warp_loaded_frag_B[3], accum);
      warp_mma.laterHalfMma(accum, warp_loaded_frag_A[3],warp_loaded_frag_B[3], accum);

      copy_tiles_and_advance_2share_A_(1, smem_register_A_2);
      copy_tiles_and_advance_2share_B_(1, smem_register_B_2);

      // k=2
      if (residue_gemm_k_iterations >= 3) {
        iterator_A.add_tile_offset({0, 1});
        iterator_B.add_tile_offset({1, 0});
        copy_tiles_and_advance_2register_A_(iterator_A, 0, smem_register_A_1); // -> k = 1 -> ldg
        copy_tiles_and_advance_2register_B_(iterator_B, 0, smem_register_B_1);
      }


      __builtin_mxc_barrier();
      warp_mma.frontHalfMma(accum,warp_loaded_frag_A[0],warp_loaded_frag_B[0],accum);
      asm(";------");
      this->warp_tile_iterator_B_.load_with_index(warp_loaded_frag_B[2],2);
      this->warp_tile_iterator_A_.load_with_index(warp_loaded_frag_A[2],2);
      asm(";------");
      warp_mma.laterHalfMma(accum,warp_loaded_frag_A[0],warp_loaded_frag_B[0],accum);
      asm(";------");
      this->warp_tile_iterator_B_.load_with_index(warp_loaded_frag_B[3],3);
      this->warp_tile_iterator_A_.load_with_index(warp_loaded_frag_A[3],3);
      __builtin_mxc_arrive(64 + 3);
      asm(";------");
      warp_mma.frontHalfMma(accum,warp_loaded_frag_A[1],warp_loaded_frag_B[1],accum);
      asm(";------");
      warp_mma.laterHalfMma(accum,warp_loaded_frag_A[1],warp_loaded_frag_B[1],accum);

      copy_tiles_and_advance_2share_A_(0, smem_register_A_3);
      copy_tiles_and_advance_2share_B_(0, smem_register_B_3);

      // k = 3
      __builtin_mxc_barrier();

      warp_mma.frontHalfMma(accum, warp_loaded_frag_A[2],warp_loaded_frag_B[2], accum);
      asm(";------");
      this->warp_tile_iterator_B_.load_with_index(warp_loaded_frag_B[0],0);
      this->warp_tile_iterator_A_.load_with_index(warp_loaded_frag_A[0],0);
      asm(";------");
      warp_mma.laterHalfMma(accum, warp_loaded_frag_A[2],warp_loaded_frag_B[2], accum);
      asm(";------");
      this->warp_tile_iterator_B_.load_with_index(warp_loaded_frag_B[1],1);
      this->warp_tile_iterator_A_.load_with_index(warp_loaded_frag_A[1],1);
      __builtin_mxc_arrive(64 + 3);
      asm(";------");
      warp_mma.frontHalfMma(accum, warp_loaded_frag_A[3],warp_loaded_frag_B[3], accum);
      asm(";------");
      warp_mma.laterHalfMma(accum, warp_loaded_frag_A[3],warp_loaded_frag_B[3], accum);

      copy_tiles_and_advance_2share_A_(1, smem_register_A_4);
      copy_tiles_and_advance_2share_B_(1, smem_register_B_4);
      asm(";------");
    }


    if (residue_gemm_k_iterations != 0) {
      //k=0
      __builtin_mxc_barrier();
      this->warp_tile_iterator_A_.load_with_index(warp_loaded_frag_A[2],2);
      this->warp_tile_iterator_B_.load_with_index(warp_loaded_frag_B[2],2);

      if (residue_gemm_k_iterations >= 1) {
        warp_mma(accum, warp_loaded_frag_A[0],warp_loaded_frag_B[0], accum);
        warp_mma(accum, warp_loaded_frag_A[1],warp_loaded_frag_B[1], accum);
      }

      this->warp_tile_iterator_B_.load_with_index(warp_loaded_frag_B[3],3);
      this->warp_tile_iterator_A_.load_with_index(warp_loaded_frag_A[3],3);

      copy_tiles_and_advance_2share_A_(0, smem_register_A_1);
      copy_tiles_and_advance_2share_B_(0, smem_register_B_1);

      // k = 1
      __builtin_mxc_barrier();
      this->warp_tile_iterator_B_.load_with_index(warp_loaded_frag_B[0],0);
      this->warp_tile_iterator_A_.load_with_index(warp_loaded_frag_A[0],0);

      if (residue_gemm_k_iterations >= 2) {
        warp_mma(accum, warp_loaded_frag_A[2],warp_loaded_frag_B[2], accum);
        warp_mma(accum, warp_loaded_frag_A[3],warp_loaded_frag_B[3], accum);
      }

      this->warp_tile_iterator_B_.load_with_index(warp_loaded_frag_B[1],1);
      this->warp_tile_iterator_A_.load_with_index(warp_loaded_frag_A[1],1);

      // k=2
      if (residue_gemm_k_iterations == 3) {
        warp_mma(accum,warp_loaded_frag_A[0],warp_loaded_frag_B[0],accum);
        warp_mma(accum,warp_loaded_frag_A[1],warp_loaded_frag_B[1],accum);
      }
    }

    if (platform::is_same<typename Operator::MathOperator,
                          arch::OpMultiplyAddFastF32>::value
      || platform::is_same<typename Operator::MathOperator,
                           arch::OpMultiplyAddComplexFastF32>::value) {
      accum = plus_accum(accum, tmp_accum);
    }

    if (SharedMemoryClear == SharedMemoryClearOption::kZfill) {
      // commit and drain all pending and predicated LDGSTS pnz from the GEMM mainloop
      mctlass::arch::cp_async_fence();
      mctlass::arch::cp_async_wait<0>();
      if constexpr (!MacaSpecialTag) __syncthreads();
    }

  }
};

// mcTlass: problem: TN && stage==2
//          (SmemIteratorA_::Layout == MacaRowMajorTensorOpMultiplicandMultipleLdgCrosswise) &&
//          (SmemIteratorB_::Layout == MacaColumnMajorTensorOpMultiplicandMultipleLdgCrosswise)
// mcTlass: problem: NN && stage==2
//          (SmemIteratorA_::Layout == MacaRowMajorTensorOpMultiplicandMultipleLdgCrosswise) &&
//          (SmemIteratorB_::Layout == MacaRowMajorTensorOpMultiplicandCongruous4x4Perm)
// mcTlass: problem: TT && stage==2
//          (SmemIteratorA_::Layout == MacaColumnMajorTensorOpMultiplicandCongruous4x4Perm) &&
//          (SmemIteratorB_::Layout == MacaColumnMajorTensorOpMultiplicandMultipleLdgCrosswise)
// mcTlass: problem: NT && stage==2
//          (SmemIteratorA_::Layout == MacaColumnMajorTensorOpMultiplicandCongruous4x4Perm) &&
//          (SmemIteratorB_::Layout == MacaRowMajorTensorOpMultiplicandCongruous4x4Perm)
template <
    /// Size of the Gemm problem - concept: gemm::GemmShape<>
    typename Shape_,
    /// Iterates over tiles of A operand in global memory
    //  (concept: ReadableTileIterator | ForwardTileIterator |
    //  MaskedTileIterator)
    typename IteratorA_,
    /// Iterates over tiles of A operand in shared memory
    /// (concept: WriteableTileIterator | RandomAccessTileIterator)
    typename SmemIteratorA_,
    /// Cache operation for operand A
    mctlass::arch::CacheOperation::Kind CacheOpA,
    /// Iterates over tiles of B operand in global memory
    //  (concept: ReadableTileIterator | ForwardTileIterator |
    //  MaskedTileIterator)
    typename IteratorB_,
    /// Iterates over tiles of B operand in shared memory
    /// (concept: WriteableTileIterator | RandomAccessTileIterator)
    typename SmemIteratorB_,
    /// Cache operation for operand B
    mctlass::arch::CacheOperation::Kind CacheOpB,
    /// Data type of accumulator matrix
    typename ElementC_,
    /// Data type of accumulator matrix
    typename LayoutC_,
    /// Policy describing tuning details (concept: MmaPolicy)
    typename Policy_,
    /// Number of stages,
    // int Stages,
    /// Use zfill or predicate for out-of-bound cp.async
    SharedMemoryClearOption SharedMemoryClear,
    /// Flag tag for special process for m16n16k16 f16
    bool MacaSpecialTag_
    >
class MmaMultistage <
  Shape_,
  IteratorA_,
  SmemIteratorA_,
  CacheOpA,
  IteratorB_,
  SmemIteratorB_,
  CacheOpB,
  ElementC_,
  LayoutC_,
  Policy_,
  2,
  SharedMemoryClear,
  MacaSpecialTag_,
  typename std::enable_if<
    ((std::is_same<typename SmemIteratorA_::Layout,
        typename layout::MacaRowMajorTensorOpMultiplicandMultipleLdgCrosswise<
          sizeof_bits<mctlass::half_t>::value, Shape_::kK>>::value &&
      std::is_same<typename SmemIteratorB_::Layout,
        typename layout::MacaColumnMajorTensorOpMultiplicandMultipleLdgCrosswise<
          sizeof_bits<mctlass::half_t>::value, Shape_::kK>>::value) ||
     (std::is_same<typename SmemIteratorB_::Layout,
        typename layout::MacaRowMajorTensorOpMultiplicandCongruous4x4Perm<
          sizeof_bits<half_t>::value, int(128 / sizeof(half_t))>>::value &&
      std::is_same<typename SmemIteratorA_::Layout,
        typename layout::MacaColumnMajorTensorOpMultiplicandCongruous4x4Perm<
          sizeof_bits<half_t>::value, int(128 / sizeof(half_t))>>::value) ||
     (std::is_same<typename SmemIteratorA_::Layout,
        typename layout::MacaRowMajorTensorOpMultiplicandMultipleLdgCrosswise<
          sizeof_bits<mctlass::half_t>::value, Shape_::kK>>::value &&
      std::is_same<typename SmemIteratorB_::Layout,
        typename layout::MacaRowMajorTensorOpMultiplicandCongruous4x4Perm<
          sizeof_bits<half_t>::value, int(128 / sizeof(half_t))>>::value) ||
      (std::is_same<typename SmemIteratorB_::Layout,
        typename layout::MacaColumnMajorTensorOpMultiplicandMultipleLdgCrosswise<
          sizeof_bits<mctlass::half_t>::value, Shape_::kK>>::value &&
      std::is_same<typename SmemIteratorA_::Layout,
        typename layout::MacaColumnMajorTensorOpMultiplicandCongruous4x4Perm<
          sizeof_bits<half_t>::value, int(128 / sizeof(half_t))>>::value) ||
      (std::is_same<typename SmemIteratorA_::Layout,
        typename layout::MacaRowMajorTensorOpMultiplicandMultipleLdgCrosswise<
          sizeof_bits<float>::value, Shape_::kK>>::value &&
      std::is_same<typename SmemIteratorB_::Layout,
        typename layout::MacaRowMajorTensorOpMultiplicandCongruous4x4Perm<
          sizeof_bits<float>::value, int(128 / sizeof(float))>>::value) ||
      (std::is_same<typename SmemIteratorB_::Layout,
        typename layout::MacaColumnMajorTensorOpMultiplicandMultipleLdgCrosswise<
          sizeof_bits<float>::value, Shape_::kK>>::value &&
      std::is_same<typename SmemIteratorA_::Layout,
        typename layout::MacaColumnMajorTensorOpMultiplicandCongruous4x4Perm<
          sizeof_bits<float>::value, int(128 / sizeof(float))>>::value) ||
     (std::is_same<typename SmemIteratorB_::Layout,
        typename layout::MacaRowMajorTensorOpMultiplicandCongruous4x4Perm<
          sizeof_bits<float>::value, int(128 / sizeof(float))>>::value &&
      std::is_same<typename SmemIteratorA_::Layout,
        typename layout::MacaColumnMajorTensorOpMultiplicandCongruous4x4Perm<
          sizeof_bits<float>::value, int(128 / sizeof(float))>>::value)),
    bool>::type
  >:public MmaBase<Shape_, Policy_, 2> {
public:
  ///< Base class
  using Base = MmaBase<Shape_, Policy_, 2>;
  ///< Size of the Gemm problem - concept: gemm::GemmShape<>
  using Shape = Shape_;
  ///< Iterates over tiles of A operand in global memory
  using IteratorA = IteratorA_;
  ///< Iterates over tiles of B operand in global memory
  using IteratorB = IteratorB_;
  ///< Data type of accumulator matrix
  using ElementC = ElementC_;
  ///< Layout of accumulator matrix
  using LayoutC = LayoutC_;
  ///< Policy describing tuning details
  using Policy = Policy_;

  using SmemIteratorA = SmemIteratorA_;
  using SmemIteratorB = SmemIteratorB_;

  static mctlass::arch::CacheOperation::Kind const kCacheOpA = CacheOpA;
  static mctlass::arch::CacheOperation::Kind const kCacheOpB = CacheOpB;

  static constexpr bool kRegisterShiftA =
    platform::is_same<layout::MacaRowMajorTensorOpMultiplicandMultipleLdgCrosswise<
      sizeof_bits<typename IteratorA::Element>::value, Shape::kK>,
    typename SmemIteratorA::Layout>::value;
  static constexpr bool kRegisterShiftB =
    platform::is_same<layout::MacaColumnMajorTensorOpMultiplicandMultipleLdgCrosswise<
      sizeof_bits<typename IteratorA::Element>::value, Shape::kK>,
    typename SmemIteratorB::Layout>::value;

  static_assert(Shape::kK % (512 / sizeof_bits<typename IteratorB::Element>::value) == 0,
      "Crosswise kContiguous must be 512bit aligned");
  //
  // Dependent types
  //

  /// Fragment of accumulator tile
  using FragmentC = typename Policy::Operator::FragmentC;

  /// Warp-level Mma
  using Operator = typename Policy::Operator;

  /// Minimum architecture is Sm80 to support cp.async
  using ArchTag = arch::Sm80;

  /// Complex transform on A operand
  static ComplexTransform const kTransformA = Operator::kTransformA;

  /// Complex transform on B operand
  static ComplexTransform const kTransformB = Operator::kTransformB;

  /// Internal structure exposed for introspection.
  struct Detail {

    /// Number of cp.async instructions to load one stage of operand A
    static int const AsyncCopyIterationsPerStageA =
        IteratorA::ThreadMap::Iterations::kCount;

    /// Number of cp.async instructions to load one stage of operand B
    static int const AsyncCopyIterationsPerStageB =
        IteratorB::ThreadMap::Iterations::kCount;

    /// Number of stages
    static int const kStages = 2;
  };

 private:

  using WarpLoadedFragmentA = typename Operator::FragmentA;
  using WarpLoadedFragmentB = typename Operator::FragmentB;

 private:

  //
  // Data members
  //

  /// Iterator to write threadblock-scoped tile of A operand to shared memory
  SmemIteratorA smem_iterator_A_;

  /// Iterator to write threadblock-scoped tile of B operand to shared memory
  SmemIteratorB smem_iterator_B_;

  int register_offset_A_l = 0;
  int register_offset_A_r = 0;
  int register_offset_B_l = 0;
  int register_offset_B_r = 0;

  typename IteratorA::AccessType* smem_ptr_A_[Base::kStages * Detail::AsyncCopyIterationsPerStageA * IteratorA::kAccessesPerVector];
  typename IteratorB::AccessType* smem_ptr_B_[Base::kStages * Detail::AsyncCopyIterationsPerStageB * IteratorB::kAccessesPerVector];

public:

  /// Construct from tensor references
  MCTLASS_DEVICE
  MmaMultistage(
      ///< Shared storage needed for internal use by threadblock-scoped GEMM
      typename Base::SharedStorage &shared_storage,
      ///< ID within the threadblock
      int thread_idx,
      ///< ID of warp
      int warp_idx,
      ///< ID of each thread within a warp
      int lane_idx,
      int problem_size_k = 0,
      typename IteratorA::Element *ptr_A = nullptr,
      typename IteratorA::Element *ptr_B = nullptr
    ):
      Base(shared_storage, thread_idx, warp_idx, lane_idx),
      smem_iterator_A_(shared_storage.operand_A_ref(), thread_idx),
      smem_iterator_B_(shared_storage.operand_B_ref(), thread_idx)
  {
    int register_offset_l = 0;
    int register_offset_r = 0;
    if ((threadIdx.x / 32) % 2 == 0) {
      if ((threadIdx.x / 8) % 2 != 0) {
        register_offset_l = 1;
        register_offset_r = -1;
      }
    } else {
      if ((threadIdx.x / 8) % 2 == 0) {
        register_offset_l = 1;
        register_offset_r = -1;
      }
    }
    register_offset_A_l = kRegisterShiftA ? register_offset_l : 0;
    register_offset_A_r = kRegisterShiftA ? register_offset_r : 0;
    register_offset_B_l = kRegisterShiftB ? register_offset_l : 0;
    register_offset_B_r = kRegisterShiftB ? register_offset_r : 0;

    // Compute warp location within threadblock tile by mapping the warp_id to
    // three coordinates:
    //   _m: the warp's position within the threadblock along the M dimension
    //   _n: the warp's position within the threadblock along the N dimension
    //   _k: the warp's position within the threadblock along the K dimension
    int warp_idx_mn = warp_idx % (Base::WarpCount::kM * Base::WarpCount::kN);
    int warp_idx_k = warp_idx / (Base::WarpCount::kM * Base::WarpCount::kN);

    int warp_idx_m = warp_idx_mn % Base::WarpCount::kM;
    int warp_idx_n = warp_idx_mn / Base::WarpCount::kM;

    // Add per-warp offsets in units of warp-level tiles
    this->warp_tile_iterator_A_.pre_compute_pointer(warp_idx_m, Base::kWarpGemmIterations * warp_idx_k);
    this->warp_tile_iterator_B_.pre_compute_pointer(warp_idx_n, Base::kWarpGemmIterations * warp_idx_k);

    // Compute A shared memory ptr
    this->smem_iterator_A_.set_iteration_index(0);

    for(int stage = 0; stage < Base::kStages; ++stage){
      for(int j = 0; j < Detail::AsyncCopyIterationsPerStageA; ++j){
        for(int v = 0; v < IteratorA::kAccessesPerVector; ++v ){
          smem_ptr_A_[stage * Detail::AsyncCopyIterationsPerStageA + j * IteratorA::kAccessesPerVector + v] =
            reinterpret_cast<typename IteratorA::AccessType *>(this->smem_iterator_A_.get()) + v;
          ++this->smem_iterator_A_;
        }
      }
      this->smem_iterator_A_.add_tile_offset({0,1});
    }
    // Reset tile offset to zero
    this->smem_iterator_A_.add_tile_offset({0,-Base::kStages});

    // Compute B shared memory ptr
    this->smem_iterator_B_.set_iteration_index(0);
    MCTLASS_PRAGMA_UNROLL
    for(int stage = 0; stage < Base::kStages; ++stage){
      for(int j = 0; j < Detail::AsyncCopyIterationsPerStageB; ++j){
        for(int v = 0; v < IteratorB::kAccessesPerVector; ++v){
          smem_ptr_B_[stage * Detail::AsyncCopyIterationsPerStageB + j * IteratorB::kAccessesPerVector + v] =
            reinterpret_cast<typename IteratorA::AccessType *>(this->smem_iterator_B_.get())+ v;
          ++this->smem_iterator_B_;
        }
      }
      this->smem_iterator_B_.add_tile_offset({1,0});
    }
    // Reset tile offset to zero
    this->smem_iterator_B_.add_tile_offset({-Base::kStages,0});
  }

  MCTLASS_DEVICE
  void copy_tiles_and_advance_2register_A_(IteratorA &iterator_A, int group_start_A = 0, v4i *reg_ptr_A_ = nullptr, int32_t index_A = 0) {
    iterator_A.set_iteration_index(group_start_A * IteratorA::kAccessesPerVector);
    MCTLASS_PRAGMA_UNROLL
    for (int j = 0; j < Detail::AsyncCopyIterationsPerStageA; ++j) {
      MCTLASS_PRAGMA_UNROLL
      for (int v = 0; v < IteratorA::kAccessesPerVector; ++v) {
        reg_ptr_A_[index_A] = __builtin_mxc_load_global_async128((v4i *)(iterator_A.get()));
        ++iterator_A;
        ++(index_A);
      }
    }
  }

  MCTLASS_DEVICE
  void copy_tiles_and_advance_2share_A_(int stage = 0, v4i *reg_ptr_A = nullptr, int32_t index_A = 0) {
    // Async Copy for operand A
    using HalfAccessType = Array<typename IteratorB::Element, IteratorB::ThreadMap::kElementsPerAccess / 2>;
    MCTLASS_PRAGMA_UNROLL
    for (int j = 0; j < Detail::AsyncCopyIterationsPerStageA; ++j) {
      MCTLASS_PRAGMA_UNROLL
      for (int v = 0; v < IteratorA::kAccessesPerVector; ++v) {
        HalfAccessType *reg_value = static_cast<HalfAccessType *>((void *)(&reg_ptr_A[index_A]));
        HalfAccessType *half_dst_ptr = reinterpret_cast<HalfAccessType *>(smem_ptr_A_[stage * Detail::AsyncCopyIterationsPerStageA + j * IteratorA::kAccessesPerVector + v]);

        *static_cast<HalfAccessType *>((void *)&half_dst_ptr[0  + register_offset_A_l]) = *static_cast<HalfAccessType *>((void *)(&reg_value[0]));
        *static_cast<HalfAccessType *>((void *)&half_dst_ptr[1  + register_offset_A_r]) = *static_cast<HalfAccessType *>((void *)(&reg_value[1]));

        ++(index_A);
      }
    }
  }

  MCTLASS_DEVICE
  void copy_tiles_and_advance_2register_B_(IteratorB &iterator_B, int group_start_B = 0,
                                        v4i *reg_ptr_B = nullptr, int32_t index_B = 0) {
    iterator_B.set_iteration_index(group_start_B * IteratorB::kAccessesPerVector);

    MCTLASS_PRAGMA_UNROLL
    for (int j = 0; j < Detail::AsyncCopyIterationsPerStageB; ++j) {
      MCTLASS_PRAGMA_UNROLL
      for (int v = 0; v < IteratorB::kAccessesPerVector; ++v) {
        reg_ptr_B[index_B]=__builtin_mxc_load_global_async128((v4i *)(iterator_B.get()));
        ++iterator_B;
        ++(index_B);
      }
    }
  }

  MCTLASS_DEVICE
  void copy_tiles_and_advance_2share_B_(int stage = 0, v4i *reg_ptr_B = nullptr, int32_t index_B = 0) {
    // Async Copy for operand B
    using HalfAccessType = Array<typename IteratorB::Element, IteratorB::ThreadMap::kElementsPerAccess / 2>;
    MCTLASS_PRAGMA_UNROLL
    for (int j = 0; j < Detail::AsyncCopyIterationsPerStageB; ++j) {
      MCTLASS_PRAGMA_UNROLL
      for (int v = 0; v < IteratorB::kAccessesPerVector; ++v) {

        HalfAccessType *reg_value = static_cast<HalfAccessType *>((void *)(&reg_ptr_B[index_B]));
        HalfAccessType *half_dst_ptr = reinterpret_cast<HalfAccessType *>(smem_ptr_B_[stage * Detail::AsyncCopyIterationsPerStageB + j * IteratorB::kAccessesPerVector + v]);
        *static_cast<HalfAccessType *>((void *)&half_dst_ptr[0  + register_offset_B_l]) = *static_cast<HalfAccessType *>((void *)(&reg_value[0]));
        *static_cast<HalfAccessType *>((void *)&half_dst_ptr[1  + register_offset_B_r]) = *static_cast<HalfAccessType *>((void *)(&reg_value[1]));

        ++(index_B);
      }
    }
  }

  /// Perform a threadblock-scoped matrix multiply-accumulate
  MCTLASS_DEVICE
  void operator()(
      ///< problem size of GEMM
      int gemm_k_iterations,
      ///< destination accumulator tile
      FragmentC &accum,
      ///< iterator over A operand in global memory
      IteratorA iterator_A,
      ///< iterator over B operand in global memory
      IteratorB iterator_B,
      ///< initial value of accumulator
      FragmentC const &src_accum) {
    // Issue several complete stages
    v4i smem_register_A_[Detail::AsyncCopyIterationsPerStageA * IteratorA::kAccessesPerVector];
    v4i smem_register_B_[Detail::AsyncCopyIterationsPerStageB * IteratorB::kAccessesPerVector];

    copy_tiles_and_advance_2register_A_(iterator_A, 0, smem_register_A_);
    copy_tiles_and_advance_2register_B_(iterator_B, 0, smem_register_B_);

    // Perform accumulation in the 'd' output operand
    accum = src_accum;

    __builtin_mxc_arrive(64);
    copy_tiles_and_advance_2share_A_(0, smem_register_A_);
    copy_tiles_and_advance_2share_B_(0, smem_register_B_);

    // Move to the next stage
    iterator_A.add_tile_offset({0, 1});
    iterator_B.add_tile_offset({1, 0});

    copy_tiles_and_advance_2register_A_(iterator_A, 0, smem_register_A_);
    copy_tiles_and_advance_2register_B_(iterator_B, 0, smem_register_B_);

    __builtin_mxc_barrier();

    // Pair of fragments used to overlap shared memory loads and math
    // instructions
    WarpLoadedFragmentA warp_loaded_frag_A[4];
    WarpLoadedFragmentB warp_loaded_frag_B[4];

    Operator warp_mma;

    this->warp_tile_iterator_B_.load_with_index(warp_loaded_frag_B[0],0);
    this->warp_tile_iterator_A_.load_with_index(warp_loaded_frag_A[0],0);

    this->warp_tile_iterator_B_.load_with_index(warp_loaded_frag_B[1],1);
    this->warp_tile_iterator_A_.load_with_index(warp_loaded_frag_A[1],1);

    __builtin_mxc_arrive(64);

    copy_tiles_and_advance_2share_A_(1, smem_register_A_);
    copy_tiles_and_advance_2share_B_(1, smem_register_B_);

    //
    // Mainloop
    //
    v4i smem_register_A_1[Detail::AsyncCopyIterationsPerStageA * IteratorA::kAccessesPerVector];
    v4i smem_register_B_1[Detail::AsyncCopyIterationsPerStageB * IteratorB::kAccessesPerVector];

    v4i smem_register_A_2[Detail::AsyncCopyIterationsPerStageA * IteratorA::kAccessesPerVector];
    v4i smem_register_B_2[Detail::AsyncCopyIterationsPerStageB * IteratorB::kAccessesPerVector];

    v4i smem_register_A_3[Detail::AsyncCopyIterationsPerStageA * IteratorA::kAccessesPerVector];
    v4i smem_register_B_3[Detail::AsyncCopyIterationsPerStageB * IteratorB::kAccessesPerVector];

    v4i smem_register_A_4[Detail::AsyncCopyIterationsPerStageA * IteratorA::kAccessesPerVector];
    v4i smem_register_B_4[Detail::AsyncCopyIterationsPerStageB * IteratorB::kAccessesPerVector];

    iterator_A.add_tile_offset({0, 1});
    iterator_B.add_tile_offset({1, 0});
    asm(";------");
    copy_tiles_and_advance_2register_A_(iterator_A, 0, smem_register_A_1);
    copy_tiles_and_advance_2register_B_(iterator_B, 0, smem_register_B_1);

    iterator_A.add_tile_offset({0, 1});
    iterator_B.add_tile_offset({1, 0});
    asm(";------");
    copy_tiles_and_advance_2register_A_(iterator_A, 0, smem_register_A_2);
    copy_tiles_and_advance_2register_B_(iterator_B, 0, smem_register_B_2);

    __builtin_mxc_arrive(64 + 3);

    MCTLASS_GEMM_LOOP
    for (int k = 0; k < gemm_k_iterations / 4 - 1; ++k) {
      //
      // Loop over GEMM K dimension
      //

      if constexpr(2 == Base::kWarpGemmIterations){

        //k=0
        iterator_A.add_tile_offset({0, 1});
        iterator_B.add_tile_offset({1, 0});
        asm(";------");
        copy_tiles_and_advance_2register_A_(iterator_A, 0, smem_register_A_3);
        copy_tiles_and_advance_2register_B_(iterator_B, 0, smem_register_B_3);


        __builtin_mxc_barrier();
        this->warp_tile_iterator_A_.load_with_index(warp_loaded_frag_A[2],2);
        asm(";------");
        warp_mma.frontHalfFrontMma(accum,warp_loaded_frag_A[0],warp_loaded_frag_B[0],accum);
        asm(";------");
        this->warp_tile_iterator_B_.load_with_index(warp_loaded_frag_B[2],2);
        warp_mma.frontHalfLaterMma(accum,warp_loaded_frag_A[0],warp_loaded_frag_B[0],accum);
        asm(";------");
        warp_mma.laterHalfFrontMma(accum,warp_loaded_frag_A[0],warp_loaded_frag_B[0],accum);
        warp_mma.laterHalfLaterMma(accum,warp_loaded_frag_A[0],warp_loaded_frag_B[0],accum);
        this->warp_tile_iterator_B_.load_with_index(warp_loaded_frag_B[3],3);
        this->warp_tile_iterator_A_.load_with_index(warp_loaded_frag_A[3],3);

        warp_mma.frontHalfMma(accum,warp_loaded_frag_A[1],warp_loaded_frag_B[1],accum);
        asm(";------");
        copy_tiles_and_advance_2share_A_(0, smem_register_A_1);
        copy_tiles_and_advance_2share_B_(0, smem_register_B_1);
        warp_mma.laterHalfMma(accum,warp_loaded_frag_A[1],warp_loaded_frag_B[1],accum);

        // k = 1
        iterator_A.add_tile_offset({0, 1});
        iterator_B.add_tile_offset({1, 0});
        copy_tiles_and_advance_2register_A_(iterator_A, 0, smem_register_A_4);
        copy_tiles_and_advance_2register_B_(iterator_B, 0, smem_register_B_4);

        __builtin_mxc_barrier();

        warp_mma.frontHalfFrontMma(accum,warp_loaded_frag_A[2],warp_loaded_frag_B[2],accum);
        asm(";------");
        this->warp_tile_iterator_B_.load_with_index(warp_loaded_frag_B[0],0);
        this->warp_tile_iterator_A_.load_with_index(warp_loaded_frag_A[0],0);
        asm(";------");
        warp_mma.frontHalfLaterMma(accum,warp_loaded_frag_A[2],warp_loaded_frag_B[2],accum);
        asm(";------");
        warp_mma.laterHalfMma(accum, warp_loaded_frag_A[2],warp_loaded_frag_B[2], accum);
        asm(";------");
        this->warp_tile_iterator_B_.load_with_index(warp_loaded_frag_B[1],1);
        this->warp_tile_iterator_A_.load_with_index(warp_loaded_frag_A[1],1);
        asm(";------");
        __builtin_mxc_arrive(64 + 3);
        warp_mma.frontHalfMma(accum, warp_loaded_frag_A[3],warp_loaded_frag_B[3], accum);
        warp_mma.laterHalfMma(accum, warp_loaded_frag_A[3],warp_loaded_frag_B[3], accum);

        copy_tiles_and_advance_2share_A_(1, smem_register_A_2);
        copy_tiles_and_advance_2share_B_(1, smem_register_B_2);

        // k=2~3
        iterator_A.add_tile_offset({0, 1});
        iterator_B.add_tile_offset({1, 0});
        copy_tiles_and_advance_2register_A_(iterator_A, 0, smem_register_A_1); // -> k = 1 -> ldg
        copy_tiles_and_advance_2register_B_(iterator_B, 0, smem_register_B_1);


        __builtin_mxc_barrier();
        warp_mma.frontHalfMma(accum,warp_loaded_frag_A[0],warp_loaded_frag_B[0],accum);
        asm(";------");
        this->warp_tile_iterator_B_.load_with_index(warp_loaded_frag_B[2],2);
        this->warp_tile_iterator_A_.load_with_index(warp_loaded_frag_A[2],2);
        asm(";------");
        warp_mma.laterHalfMma(accum,warp_loaded_frag_A[0],warp_loaded_frag_B[0],accum);
        asm(";------");
        this->warp_tile_iterator_B_.load_with_index(warp_loaded_frag_B[3],3);
        this->warp_tile_iterator_A_.load_with_index(warp_loaded_frag_A[3],3);
        __builtin_mxc_arrive(64 + 3);
        asm(";------");
        warp_mma.frontHalfMma(accum,warp_loaded_frag_A[1],warp_loaded_frag_B[1],accum);
        asm(";------");
        warp_mma.laterHalfMma(accum,warp_loaded_frag_A[1],warp_loaded_frag_B[1],accum);

        copy_tiles_and_advance_2share_A_(0, smem_register_A_3);
        copy_tiles_and_advance_2share_B_(0, smem_register_B_3);

        iterator_A.add_tile_offset({0, 1});
        iterator_B.add_tile_offset({1, 0});

        copy_tiles_and_advance_2register_A_(iterator_A, 0, smem_register_A_2);
        copy_tiles_and_advance_2register_B_(iterator_B, 0, smem_register_B_2);

        __builtin_mxc_barrier();

        warp_mma.frontHalfMma(accum, warp_loaded_frag_A[2],warp_loaded_frag_B[2], accum);
        asm(";------");
        this->warp_tile_iterator_B_.load_with_index(warp_loaded_frag_B[0],0);
        this->warp_tile_iterator_A_.load_with_index(warp_loaded_frag_A[0],0);
        asm(";------");
        warp_mma.laterHalfMma(accum, warp_loaded_frag_A[2],warp_loaded_frag_B[2], accum);
        asm(";------");
        this->warp_tile_iterator_B_.load_with_index(warp_loaded_frag_B[1],1);
        this->warp_tile_iterator_A_.load_with_index(warp_loaded_frag_A[1],1);
        __builtin_mxc_arrive(64 + 3);
        asm(";------");
        warp_mma.frontHalfMma(accum, warp_loaded_frag_A[3],warp_loaded_frag_B[3], accum);
        asm(";------");
        warp_mma.laterHalfMma(accum, warp_loaded_frag_A[3],warp_loaded_frag_B[3], accum);

        copy_tiles_and_advance_2share_A_(1, smem_register_A_4);
        copy_tiles_and_advance_2share_B_(1, smem_register_B_4);
        asm(";------");

      }
    }

    int residue_gemm_k_iterations = gemm_k_iterations % 4;

    if (gemm_k_iterations >= 4) {
      //k=0
      if (residue_gemm_k_iterations >= 1) {
        iterator_A.add_tile_offset({0, 1});
        iterator_B.add_tile_offset({1, 0});
        asm(";------");
        copy_tiles_and_advance_2register_A_(iterator_A, 0, smem_register_A_3);
        copy_tiles_and_advance_2register_B_(iterator_B, 0, smem_register_B_3);
      }
      __builtin_mxc_barrier();
      this->warp_tile_iterator_A_.load_with_index(warp_loaded_frag_A[2],2);
      asm(";------");
      warp_mma.frontHalfFrontMma(accum,warp_loaded_frag_A[0],warp_loaded_frag_B[0],accum);
      asm(";------");
      this->warp_tile_iterator_B_.load_with_index(warp_loaded_frag_B[2],2);
      warp_mma.frontHalfLaterMma(accum,warp_loaded_frag_A[0],warp_loaded_frag_B[0],accum);
      asm(";------");
      warp_mma.laterHalfFrontMma(accum,warp_loaded_frag_A[0],warp_loaded_frag_B[0],accum);
      warp_mma.laterHalfLaterMma(accum,warp_loaded_frag_A[0],warp_loaded_frag_B[0],accum);
      this->warp_tile_iterator_B_.load_with_index(warp_loaded_frag_B[3],3);
      this->warp_tile_iterator_A_.load_with_index(warp_loaded_frag_A[3],3);


      warp_mma.frontHalfMma(accum,warp_loaded_frag_A[1],warp_loaded_frag_B[1],accum);
      asm(";------");
      copy_tiles_and_advance_2share_A_(0, smem_register_A_1);
      copy_tiles_and_advance_2share_B_(0, smem_register_B_1);
      warp_mma.laterHalfMma(accum,warp_loaded_frag_A[1],warp_loaded_frag_B[1],accum);

      // k = 1
      if (residue_gemm_k_iterations >= 2) {
        iterator_A.add_tile_offset({0, 1});
        iterator_B.add_tile_offset({1, 0});

        copy_tiles_and_advance_2register_A_(iterator_A, 0, smem_register_A_4);
        copy_tiles_and_advance_2register_B_(iterator_B, 0, smem_register_B_4);
      }
      __builtin_mxc_barrier();

      warp_mma.frontHalfFrontMma(accum,warp_loaded_frag_A[2],warp_loaded_frag_B[2],accum);
      asm(";------");
      this->warp_tile_iterator_B_.load_with_index(warp_loaded_frag_B[0],0);
      this->warp_tile_iterator_A_.load_with_index(warp_loaded_frag_A[0],0);
      asm(";------");
      warp_mma.frontHalfLaterMma(accum,warp_loaded_frag_A[2],warp_loaded_frag_B[2],accum);
      asm(";------");
      warp_mma.laterHalfMma(accum, warp_loaded_frag_A[2],warp_loaded_frag_B[2], accum);
      asm(";------");
      this->warp_tile_iterator_B_.load_with_index(warp_loaded_frag_B[1],1);
      this->warp_tile_iterator_A_.load_with_index(warp_loaded_frag_A[1],1);
      asm(";------");
      __builtin_mxc_arrive(64 + 3);
      warp_mma.frontHalfMma(accum, warp_loaded_frag_A[3],warp_loaded_frag_B[3], accum);
      warp_mma.laterHalfMma(accum, warp_loaded_frag_A[3],warp_loaded_frag_B[3], accum);

      copy_tiles_and_advance_2share_A_(1, smem_register_A_2);
      copy_tiles_and_advance_2share_B_(1, smem_register_B_2);

      // k=2
      if (residue_gemm_k_iterations >= 3) {
        iterator_A.add_tile_offset({0, 1});
        iterator_B.add_tile_offset({1, 0});
        copy_tiles_and_advance_2register_A_(iterator_A, 0, smem_register_A_1); // -> k = 1 -> ldg
        copy_tiles_and_advance_2register_B_(iterator_B, 0, smem_register_B_1);
      }

      __builtin_mxc_barrier();
      warp_mma.frontHalfMma(accum,warp_loaded_frag_A[0],warp_loaded_frag_B[0],accum);
      asm(";------");
      this->warp_tile_iterator_B_.load_with_index(warp_loaded_frag_B[2],2);
      this->warp_tile_iterator_A_.load_with_index(warp_loaded_frag_A[2],2);
      asm(";------");
      warp_mma.laterHalfMma(accum,warp_loaded_frag_A[0],warp_loaded_frag_B[0],accum);
      asm(";------");
      this->warp_tile_iterator_B_.load_with_index(warp_loaded_frag_B[3],3);
      this->warp_tile_iterator_A_.load_with_index(warp_loaded_frag_A[3],3);
      __builtin_mxc_arrive(64 + 3);
      asm(";------");
      warp_mma.frontHalfMma(accum,warp_loaded_frag_A[1],warp_loaded_frag_B[1],accum);
      asm(";------");
      warp_mma.laterHalfMma(accum,warp_loaded_frag_A[1],warp_loaded_frag_B[1],accum);

      copy_tiles_and_advance_2share_A_(0, smem_register_A_3);
      copy_tiles_and_advance_2share_B_(0, smem_register_B_3);

      // k = 3
      __builtin_mxc_barrier();

      warp_mma.frontHalfMma(accum, warp_loaded_frag_A[2],warp_loaded_frag_B[2], accum);
      asm(";------");
      this->warp_tile_iterator_B_.load_with_index(warp_loaded_frag_B[0],0);
      this->warp_tile_iterator_A_.load_with_index(warp_loaded_frag_A[0],0);
      asm(";------");
      warp_mma.laterHalfMma(accum, warp_loaded_frag_A[2],warp_loaded_frag_B[2], accum);
      asm(";------");
      this->warp_tile_iterator_B_.load_with_index(warp_loaded_frag_B[1],1);
      this->warp_tile_iterator_A_.load_with_index(warp_loaded_frag_A[1],1);
      __builtin_mxc_arrive(64 + 3);
      asm(";------");
      warp_mma.frontHalfMma(accum, warp_loaded_frag_A[3],warp_loaded_frag_B[3], accum);
      asm(";------");
      warp_mma.laterHalfMma(accum, warp_loaded_frag_A[3],warp_loaded_frag_B[3], accum);

      copy_tiles_and_advance_2share_A_(1, smem_register_A_4);
      copy_tiles_and_advance_2share_B_(1, smem_register_B_4);
      asm(";------");
    }

    if (residue_gemm_k_iterations != 0) {
      //k=0
      __builtin_mxc_barrier();
      this->warp_tile_iterator_A_.load_with_index(warp_loaded_frag_A[2],2);
      this->warp_tile_iterator_B_.load_with_index(warp_loaded_frag_B[2],2);

      if (residue_gemm_k_iterations >= 1) {
        warp_mma(accum, warp_loaded_frag_A[0],warp_loaded_frag_B[0], accum);
        warp_mma(accum, warp_loaded_frag_A[1],warp_loaded_frag_B[1], accum);
      }

      this->warp_tile_iterator_B_.load_with_index(warp_loaded_frag_B[3],3);
      this->warp_tile_iterator_A_.load_with_index(warp_loaded_frag_A[3],3);

      copy_tiles_and_advance_2share_A_(0, smem_register_A_1);
      copy_tiles_and_advance_2share_B_(0, smem_register_B_1);

      // k = 1
      __builtin_mxc_barrier();
      this->warp_tile_iterator_B_.load_with_index(warp_loaded_frag_B[0],0);
      this->warp_tile_iterator_A_.load_with_index(warp_loaded_frag_A[0],0);

      if (residue_gemm_k_iterations >= 2) {
        warp_mma(accum, warp_loaded_frag_A[2],warp_loaded_frag_B[2], accum);
        warp_mma(accum, warp_loaded_frag_A[3],warp_loaded_frag_B[3], accum);
      }

      this->warp_tile_iterator_B_.load_with_index(warp_loaded_frag_B[1],1);
      this->warp_tile_iterator_A_.load_with_index(warp_loaded_frag_A[1],1);

      // k=2
      if (residue_gemm_k_iterations == 3) {
        warp_mma(accum,warp_loaded_frag_A[0],warp_loaded_frag_B[0],accum);
        warp_mma(accum,warp_loaded_frag_A[1],warp_loaded_frag_B[1],accum);
      }
    }
  }
};

// mcTlass: problem: TN && stage==2
//          (SmemIteratorA_::Layout == MacaRowMajorTensorOpMultiplicandMultipleLdg64Crosswise) &&
//          (SmemIteratorB_::Layout == MacaColumnMajorTensorOpMultiplicandMultipleLdg64Crosswise) &&
//          (stage == 2)
//          (<128, 128, 64> & <32, 64, 64>)
template <
    /// Size of the Gemm problem - concept: gemm::GemmShape<>
    typename Shape_,
    /// Iterates over tiles of A operand in global memory
    //  (concept: ReadableTileIterator | ForwardTileIterator |
    //  MaskedTileIterator)
    typename IteratorA_,
    /// Iterates over tiles of A operand in shared memory
    /// (concept: WriteableTileIterator | RandomAccessTileIterator)
    typename SmemIteratorA_,
    /// Cache operation for operand A
    mctlass::arch::CacheOperation::Kind CacheOpA,
    /// Iterates over tiles of B operand in global memory
    //  (concept: ReadableTileIterator | ForwardTileIterator |
    //  MaskedTileIterator)
    typename IteratorB_,
    /// Iterates over tiles of B operand in shared memory
    /// (concept: WriteableTileIterator | RandomAccessTileIterator)
    typename SmemIteratorB_,
    /// Cache operation for operand B
    mctlass::arch::CacheOperation::Kind CacheOpB,
    /// Data type of accumulator matrix
    typename ElementC_,
    /// Data type of accumulator matrix
    typename LayoutC_,
    /// Policy describing tuning details (concept: MmaPolicy)
    typename Policy_,
    /// Number of stages,
    // int Stages,
    /// Use zfill or predicate for out-of-bound cp.async
    SharedMemoryClearOption SharedMemoryClear,
    /// Flag tag for special process for m16n16k16 f16
    bool MacaSpecialTag_
    >
class MmaMultistage <
  Shape_,
  IteratorA_,
  SmemIteratorA_,
  CacheOpA,
  IteratorB_,
  SmemIteratorB_,
  CacheOpB,
  ElementC_,
  LayoutC_,
  Policy_,
  2,
  SharedMemoryClear,
  MacaSpecialTag_,
  typename std::enable_if<
    ((std::is_same<typename SmemIteratorA_::Layout,
        typename layout::MacaRowMajorTensorOpMultiplicandMultipleLdg64Crosswise<
          sizeof_bits<mctlass::half_t>::value, Shape_::kK>>::value &&
      std::is_same<typename SmemIteratorB_::Layout,
        typename layout::MacaColumnMajorTensorOpMultiplicandMultipleLdg64Crosswise<
          sizeof_bits<mctlass::half_t>::value, Shape_::kK>>::value)),
    bool>::type
  >:public MmaBase<Shape_, Policy_, 2> {
public:
  ///< Base class
  using Base = MmaBase<Shape_, Policy_, 2>;
  ///< Size of the Gemm problem - concept: gemm::GemmShape<>
  using Shape = Shape_;
  ///< Iterates over tiles of A operand in global memory
  using IteratorA = IteratorA_;
  ///< Iterates over tiles of B operand in global memory
  using IteratorB = IteratorB_;
  ///< Data type of accumulator matrix
  using ElementC = ElementC_;
  ///< Layout of accumulator matrix
  using LayoutC = LayoutC_;
  ///< Policy describing tuning details
  using Policy = Policy_;

  using SmemIteratorA = SmemIteratorA_;
  using SmemIteratorB = SmemIteratorB_;

  static mctlass::arch::CacheOperation::Kind const kCacheOpA = CacheOpA;
  static mctlass::arch::CacheOperation::Kind const kCacheOpB = CacheOpB;
  // static constexpr bool MacaSpecialTag = MacaSpecialTag_;
  static constexpr bool MacaSpecialTag = false;

  static constexpr bool CrossOffsetA =
    platform::is_same<layout::MacaRowMajorTensorOpMultiplicandCpAsyncCrosswise<
      sizeof_bits<typename IteratorA::Element>::value, Shape::kK>,
    typename SmemIteratorA::Layout>::value;
  static constexpr bool CrossOffsetB =
    platform::is_same<layout::MacaColumnMajorTensorOpMultiplicandCpAsyncCrosswise<
      sizeof_bits<typename IteratorB::Element>::value, Shape::kK>,
    typename SmemIteratorB::Layout>::value;

  static_assert(Shape::kK % (512 / sizeof_bits<typename IteratorB::Element>::value) == 0,
      "Crosswise kContiguous must be 512bit aligned");
  //
  // Dependent types
  //

  /// Fragment of accumulator tile
  using FragmentC = typename Policy::Operator::FragmentC;

  /// Warp-level Mma
  using Operator = typename Policy::Operator;

  /// Minimum architecture is Sm80 to support cp.async
  using ArchTag = arch::Sm80;

  /// Complex transform on A operand
  static ComplexTransform const kTransformA = Operator::kTransformA;

  /// Complex transform on B operand
  static ComplexTransform const kTransformB = Operator::kTransformB;

  static int const kSrcBytesB = sizeof_bits<typename IteratorB::Element>::value *
                                IteratorB::ThreadMap::kElementsPerAccess /
                                IteratorB::kAccessesPerVector / 8;
  static int const kSrcBytesA = sizeof_bits<typename IteratorA::Element>::value *
                                IteratorA::ThreadMap::kElementsPerAccess /
                                IteratorA::kAccessesPerVector / 8;
  using AccessTypeB  = Array<uint8_t, kSrcBytesB>;

  /// Internal structure exposed for introspection.
  struct Detail {

    /// Number of cp.async instructions to load one stage of operand A
    static int const AsyncCopyIterationsPerStageA =
        IteratorA::ThreadMap::Iterations::kCount;

    /// Number of cp.async instructions to load one stage of operand B
    static int const AsyncCopyIterationsPerStageB =
        IteratorB::ThreadMap::Iterations::kCount;

    /// Number of stages
    static int const kStages = 2;

    /// Number of cp.async instructions to load on group of operand A
    static int const kAccessesPerGroupA =
        (AsyncCopyIterationsPerStageA + Base::kWarpGemmIterations - 1) / Base::kWarpGemmIterations;

    /// Number of cp.async instructions to load on group of operand B
    static int const kAccessesPerGroupB =
        (AsyncCopyIterationsPerStageB + Base::kWarpGemmIterations - 1) / Base::kWarpGemmIterations;
  };

 private:

  using WarpLoadedFragmentA = typename Operator::FragmentA;
  using WarpLoadedFragmentB = typename Operator::FragmentB;
  using WarpTransformedFragmentA = typename Operator::TransformedFragmentA;
  using WarpTransformedFragmentB = typename Operator::TransformedFragmentB;

 private:

  //
  // Data members
  //

  /// Iterator to write threadblock-scoped tile of A operand to shared memory
  SmemIteratorA smem_iterator_A_;

  /// Iterator to write threadblock-scoped tile of B operand to shared memory
  SmemIteratorB smem_iterator_B_;

  int cross_offset_A_;
  int cross_offset_B_;

  int share_offset_64 = 0;
  int cross_offset_A_64 = 0;
  int cross_offset_B_64 = 0;

  int register_offset_l = 0;
  int register_offset_r = 0;

  int register_offset_l_64 = 0;
  int register_offset_r_64 = 0;

  typename IteratorA::AccessType* smem_ptr_A_[Base::kStages * Detail::AsyncCopyIterationsPerStageA * IteratorA::kAccessesPerVector];
  typename IteratorB::AccessType* smem_ptr_B_[Base::kStages * Detail::AsyncCopyIterationsPerStageB * IteratorB::kAccessesPerVector];

public:

  /// Construct from tensor references
  MCTLASS_DEVICE
  MmaMultistage(
      ///< Shared storage needed for internal use by threadblock-scoped GEMM
      typename Base::SharedStorage &shared_storage,
      ///< ID within the threadblock
      int thread_idx,
      ///< ID of warp
      int warp_idx,
      ///< ID of each thread within a warp
      int lane_idx
    ):
      Base(shared_storage, thread_idx, warp_idx, lane_idx),
      smem_iterator_A_(shared_storage.operand_A_ref(), thread_idx),
      smem_iterator_B_(shared_storage.operand_B_ref(), thread_idx)
  {
    int bid = lane_idx & 0x7;
    int group_id = (lane_idx >> 3) & 0x3;
    int cross_target = bid ^ group_id;
    int cross_offset_ = cross_target - bid;
    // cross_offset_A_ = CrossOffsetA ? cross_offset_ : 0;
    // cross_offset_B_ = CrossOffsetB ? cross_offset_ : 0;

    cross_offset_A_ = cross_offset_;
    cross_offset_B_ = cross_offset_;

      if ((threadIdx.x / 8) % 2 != 0) {
        register_offset_l = 1;
        register_offset_r = -1;
      }
      if ((threadIdx.x / 8) % 2 == 0) {
        register_offset_l_64 = 1;
        register_offset_r_64 = -1;
      }
    // Compute warp location within threadblock tile by mapping the warp_id to
    // three coordinates:
    //   _m: the warp's position within the threadblock along the M dimension
    //   _n: the warp's position within the threadblock along the N dimension
    //   _k: the warp's position within the threadblock along the K dimension

    int warp_idx_mn = warp_idx % (Base::WarpCount::kM * Base::WarpCount::kN);
    int warp_idx_k = warp_idx / (Base::WarpCount::kM * Base::WarpCount::kN);

    int warp_idx_m = warp_idx_mn % Base::WarpCount::kM;
    int warp_idx_n = warp_idx_mn / Base::WarpCount::kM;

    // Add per-warp offsets in units of warp-level tiles
    this->warp_tile_iterator_A_.pre_compute_pointer();
    this->warp_tile_iterator_B_.pre_compute_pointer();

    // Compute A shared memory ptr
    this->smem_iterator_A_.set_iteration_index(0);

    for(int stage = 0; stage < Base::kStages; ++stage){
      for(int j = 0; j < Detail::AsyncCopyIterationsPerStageA; ++j){
        for(int v = 0; v < IteratorA::kAccessesPerVector; ++v ){

          smem_ptr_A_[stage * Detail::AsyncCopyIterationsPerStageA + j * IteratorA::kAccessesPerVector + v] =
            reinterpret_cast<typename IteratorA::AccessType *>(this->smem_iterator_A_.get()) + v - cross_offset_A_;
          ++this->smem_iterator_A_;
        }
      }
      this->smem_iterator_A_.add_tile_offset({0,1});
    }
    // Reset tile offset to zero
    this->smem_iterator_A_.add_tile_offset({0,-Base::kStages});

    // Compute B shared memory ptr
    this->smem_iterator_B_.set_iteration_index(0);
    MCTLASS_PRAGMA_UNROLL
    for(int stage = 0; stage < Base::kStages; ++stage){
      for(int j = 0; j < Detail::AsyncCopyIterationsPerStageB; ++j){
        for(int v = 0; v < IteratorB::kAccessesPerVector; ++v){
          smem_ptr_B_[stage * Detail::AsyncCopyIterationsPerStageB + j * IteratorB::kAccessesPerVector + v] =
            reinterpret_cast<typename IteratorA::AccessType *>(this->smem_iterator_B_.get())+ v - cross_offset_B_;
          ++this->smem_iterator_B_;
        }
      }
      this->smem_iterator_B_.add_tile_offset({1,0});
    }
    // Reset tile offset to zero
    this->smem_iterator_B_.add_tile_offset({-Base::kStages,0});

    if (lane_idx > 31) {
      cross_offset_A_64 = ((lane_idx / 4) % 2 == 0) ? 4 : -4;
      cross_offset_B_64 = ((lane_idx / 4) % 2 == 0) ? 4 : -4;
    }
    cross_offset_A_ += cross_offset_A_64;
    cross_offset_B_ += cross_offset_B_64;

    share_offset_64 = ((lane_idx / 4) % 2 == 0) ? 0 : 4 * 8 * 2; // 8 -> 128B; 2 -> stage 2

  }


  MCTLASS_DEVICE
  void copy_tiles_and_advance_A(IteratorA &iterator_A, int group_start_A = 0) {
    iterator_A.set_iteration_index(group_start_A *
                                   IteratorA::kAccessesPerVector);
    this->smem_iterator_A_.set_iteration_index(group_start_A);

    // Async Copy for operand A
    MCTLASS_PRAGMA_UNROLL
    for (int j = 0; j < Detail::AsyncCopyIterationsPerStageA; ++j) {
        typename IteratorA::AccessType *dst_ptr[IteratorA::kAccessesPerVector];
        typename IteratorA::AccessType *gmem_ptr[IteratorA::kAccessesPerVector];

        MCTLASS_PRAGMA_UNROLL
        for (int v = 0; v < IteratorA::kAccessesPerVector; ++v) {
          dst_ptr[v] = reinterpret_cast<typename IteratorA::AccessType *>(this->smem_iterator_A_.get()) + v - cross_offset_A_;
          gmem_ptr[v] = iterator_A.get() + cross_offset_A_;
          ++iterator_A;
        }
        MCTLASS_PRAGMA_UNROLL
        for (int v = 0; v < IteratorA::kAccessesPerVector; ++v) {
          if constexpr (SharedMemoryClear == SharedMemoryClearOption::kZfill) {
            mctlass::arch::cp_async_zfill<kSrcBytesA, kCacheOpA>(
                dst_ptr[v], gmem_ptr[v], true);
          } else {
            mctlass::arch::cp_async<kSrcBytesA, kCacheOpA>(
                dst_ptr[v], gmem_ptr[v], true);
          }
        }
        this->smem_iterator_A_.set_iteration_index(++group_start_A);
    }
  }

  MCTLASS_DEVICE
  void copy_tiles_and_advance_B(IteratorB &iterator_B, int group_start_B = 0) {

    iterator_B.set_iteration_index(group_start_B *
                                   IteratorB::kAccessesPerVector);
    this->smem_iterator_B_.set_iteration_index(group_start_B);

    // Async Copy for operand B
    MCTLASS_PRAGMA_UNROLL
    for (int j = 0; j < Detail::AsyncCopyIterationsPerStageB; ++j) {
        typename IteratorB::AccessType *dst_ptr[IteratorB::kAccessesPerVector];
        typename IteratorB::AccessType *gmem_ptr[IteratorB::kAccessesPerVector];

        MCTLASS_PRAGMA_UNROLL
        for (int v = 0; v < IteratorB::kAccessesPerVector; ++v) {
          dst_ptr[v] = reinterpret_cast<typename IteratorB::AccessType *>(this->smem_iterator_B_.get()) + v - cross_offset_B_;
          gmem_ptr[v] = iterator_B.get() + cross_offset_B_;
          ++iterator_B;
        }

        MCTLASS_PRAGMA_UNROLL
        for (int v = 0; v < IteratorB::kAccessesPerVector; ++v) {
          if (SharedMemoryClear == SharedMemoryClearOption::kZfill) {
            mctlass::arch::cp_async_zfill<kSrcBytesB, kCacheOpB>(
                dst_ptr[v], gmem_ptr[v], true);
          } else {
            mctlass::arch::cp_async<kSrcBytesB, kCacheOpB>(
                dst_ptr[v], gmem_ptr[v], true);
          }
        }
        this->smem_iterator_B_.set_iteration_index(++group_start_B);
    }
  }

  MCTLASS_DEVICE
  void copy_tiles_and_advance_A_pre(IteratorA &iterator_A, int stage,int group_start_A = 0){

    iterator_A.set_iteration_index(group_start_A *
                                   IteratorA::kAccessesPerVector);

    MCTLASS_PRAGMA_UNROLL
    for (int j = 0; j < Detail::AsyncCopyIterationsPerStageA; ++j) {
        typename IteratorA::AccessType *gmem_ptr[IteratorA::kAccessesPerVector];

        MCTLASS_PRAGMA_UNROLL
        for (int v = 0; v < IteratorA::kAccessesPerVector; ++v) {
          gmem_ptr[v] = iterator_A.get() + cross_offset_A_;
          ++iterator_A;
        }
        MCTLASS_PRAGMA_UNROLL
        for (int v = 0; v < IteratorA::kAccessesPerVector; ++v) {
          if constexpr (SharedMemoryClear == SharedMemoryClearOption::kZfill) {
            mctlass::arch::cp_async_zfill<kSrcBytesA, kCacheOpA>(
                smem_ptr_A_[stage * Detail::AsyncCopyIterationsPerStageA + j * IteratorA::kAccessesPerVector + v], gmem_ptr[v], true);
          } else {
            mctlass::arch::cp_async<kSrcBytesA, kCacheOpA>(
                smem_ptr_A_[stage * Detail::AsyncCopyIterationsPerStageA + j * IteratorA::kAccessesPerVector + v], gmem_ptr[v], true);
          }
        }
    }
  }

    MCTLASS_DEVICE
  void copy_tiles_and_advance_B_pre(IteratorB &iterator_B, int stage ,int group_start_B = 0) {

    iterator_B.set_iteration_index(group_start_B *
                                   IteratorB::kAccessesPerVector);

    // Async Copy for operand B
    MCTLASS_PRAGMA_UNROLL
    for (int j = 0; j < Detail::AsyncCopyIterationsPerStageB; ++j) {
        typename IteratorB::AccessType *gmem_ptr[IteratorB::kAccessesPerVector];

        MCTLASS_PRAGMA_UNROLL
        for (int v = 0; v < IteratorB::kAccessesPerVector; ++v) {
          gmem_ptr[v] = iterator_B.get() + cross_offset_B_;
          ++iterator_B;
        }

        MCTLASS_PRAGMA_UNROLL
        for (int v = 0; v < IteratorB::kAccessesPerVector; ++v) {
          if (SharedMemoryClear == SharedMemoryClearOption::kZfill) {
            mctlass::arch::cp_async_zfill<kSrcBytesB, kCacheOpB>(
                smem_ptr_B_[stage * Detail::AsyncCopyIterationsPerStageB + j * IteratorB::kAccessesPerVector + v], gmem_ptr[v], true);
          } else {
            mctlass::arch::cp_async<kSrcBytesB, kCacheOpB>(
                smem_ptr_B_[stage * Detail::AsyncCopyIterationsPerStageB + j * IteratorB::kAccessesPerVector + v], gmem_ptr[v], true);
          }
        }
    }
  }

  MCTLASS_DEVICE
  void copy_tiles_and_advance(IteratorA &iterator_A, IteratorB &iterator_B,
                              int group_start_A = 0, int group_start_B = 0) {
    copy_tiles_and_advance_A(iterator_A, group_start_A);
    copy_tiles_and_advance_B(iterator_B, group_start_B);
  }

  MCTLASS_DEVICE
  void copy_tiles_and_advance_2register_A_(IteratorA &iterator_A, int group_start_A = 0, v4i *reg_ptr_A_ = nullptr, int32_t index_A = 0) {
    iterator_A.set_iteration_index(group_start_A * IteratorA::kAccessesPerVector);
    MCTLASS_PRAGMA_UNROLL
    for (int j = 0; j < Detail::AsyncCopyIterationsPerStageA; ++j) {
      MCTLASS_PRAGMA_UNROLL
      for (int v = 0; v < IteratorA::kAccessesPerVector; ++v) {
        // if (iterator_A.valid()) {
          reg_ptr_A_[index_A] = __builtin_mxc_load_global_async128((v4i *)(iterator_A.get() + cross_offset_A_));
        // }
        ++iterator_A;
        ++(index_A);
      }
    }
  }

  MCTLASS_DEVICE
  void copy_tiles_and_advance_2share_A_(int stage = 0, v4i *reg_ptr_A = nullptr, int32_t index_A = 0) {
    // Async Copy for operand A
    using HalfAccessType = Array<typename IteratorB::Element, IteratorB::ThreadMap::kElementsPerAccess / 2>;

    HalfAccessType *reg_value = static_cast<HalfAccessType *>((void *)(&reg_ptr_A[0]));
    HalfAccessType *half_dst_ptr = reinterpret_cast<HalfAccessType *>(smem_ptr_A_[stage * Detail::AsyncCopyIterationsPerStageA + 0 * IteratorA::kAccessesPerVector]);
    *static_cast<HalfAccessType *>((void *)&half_dst_ptr[0  + register_offset_l]) = *static_cast<HalfAccessType *>((void *)(&reg_value[0]));
    *static_cast<HalfAccessType *>((void *)&half_dst_ptr[1  + register_offset_r]) = *static_cast<HalfAccessType *>((void *)(&reg_value[1]));

    reg_value = static_cast<HalfAccessType *>((void *)(&reg_ptr_A[1]));
    half_dst_ptr = reinterpret_cast<HalfAccessType *>(smem_ptr_A_[stage * Detail::AsyncCopyIterationsPerStageA + 1 * IteratorB::kAccessesPerVector]);
    *static_cast<HalfAccessType *>((void *)&half_dst_ptr[0  + register_offset_l_64]) = *static_cast<HalfAccessType *>((void *)(&reg_value[0]));
    *static_cast<HalfAccessType *>((void *)&half_dst_ptr[1  + register_offset_r_64]) = *static_cast<HalfAccessType *>((void *)(&reg_value[1]));
  }

  MCTLASS_DEVICE
  void copy_tiles_and_advance_2register_B_(IteratorB &iterator_B, int group_start_B = 0,
                                        v4i *reg_ptr_B = nullptr, int32_t index_B = 0) {
    iterator_B.set_iteration_index(group_start_B * IteratorB::kAccessesPerVector);

    MCTLASS_PRAGMA_UNROLL
    for (int j = 0; j < Detail::AsyncCopyIterationsPerStageB; ++j) {
      MCTLASS_PRAGMA_UNROLL
      for (int v = 0; v < IteratorB::kAccessesPerVector; ++v) {
        // if (iterator_B.valid()) {
          reg_ptr_B[index_B]=__builtin_mxc_load_global_async128((v4i *)(iterator_B.get() + cross_offset_B_));
        // }
        ++iterator_B;
        ++(index_B);
      }
    }
  }

  MCTLASS_DEVICE
  void copy_tiles_and_advance_2share_B_(int stage = 0, v4i *reg_ptr_B = nullptr, int32_t index_B = 0) {
    // Async Copy for operand B
    using HalfAccessType = Array<typename IteratorB::Element, IteratorB::ThreadMap::kElementsPerAccess / 2>;

    HalfAccessType *reg_value = static_cast<HalfAccessType *>((void *)(&reg_ptr_B[0]));
    HalfAccessType *half_dst_ptr = reinterpret_cast<HalfAccessType *>(smem_ptr_B_[stage * Detail::AsyncCopyIterationsPerStageB + 0 * IteratorB::kAccessesPerVector]);
    *static_cast<HalfAccessType *>((void *)&half_dst_ptr[0  + register_offset_l]) = *static_cast<HalfAccessType *>((void *)(&reg_value[0]));
    *static_cast<HalfAccessType *>((void *)&half_dst_ptr[1  + register_offset_r]) = *static_cast<HalfAccessType *>((void *)(&reg_value[1]));

    reg_value = static_cast<HalfAccessType *>((void *)(&reg_ptr_B[1]));
    half_dst_ptr = reinterpret_cast<HalfAccessType *>(smem_ptr_B_[stage * Detail::AsyncCopyIterationsPerStageB + 1 * IteratorB::kAccessesPerVector]);
    *static_cast<HalfAccessType *>((void *)&half_dst_ptr[0  + register_offset_l_64]) = *static_cast<HalfAccessType *>((void *)(&reg_value[0]));
    *static_cast<HalfAccessType *>((void *)&half_dst_ptr[1  + register_offset_r_64]) = *static_cast<HalfAccessType *>((void *)(&reg_value[1]));
  }

  /// Perform a threadblock-scoped matrix multiply-accumulate
  MCTLASS_DEVICE
  void operator()(
      ///< problem size of GEMM
      int gemm_k_iterations,
      ///< destination accumulator tile
      FragmentC &accum,
      ///< iterator over A operand in global memory
      IteratorA iterator_A,
      ///< iterator over B operand in global memory
      IteratorB iterator_B,
      ///< initial value of accumulator
      FragmentC const &src_accum) {
    //
    // Prologue
    //
    static int const arrive_gvm_cnt = Detail::AsyncCopyIterationsPerStageA +
             Detail::AsyncCopyIterationsPerStageB;

    // Issue several complete stages
    v4i smem_register_A_[Detail::AsyncCopyIterationsPerStageA * IteratorA::kAccessesPerVector];
    v4i smem_register_B_[Detail::AsyncCopyIterationsPerStageB * IteratorB::kAccessesPerVector];

    copy_tiles_and_advance_2register_A_(iterator_A, 0, smem_register_A_);
    copy_tiles_and_advance_2register_B_(iterator_B, 0, smem_register_B_);

    // Defines the boundary of a stage of cp.async.
    // mctlass::arch::cp_async_fence();

    // Perform accumulation in the 'd' output operand
    accum = src_accum;

    //
    // Clear the remaining tiles of SMEM. This is a functional requirement for some kernels
    // so that all accumulator elements outside the GEMM footprint are zero.
    //

    if (SharedMemoryClear == SharedMemoryClearOption::kClearLastStage) {

      /// Iterator to write threadblock-scoped tile of A operand to shared memory
      SmemIteratorA last_smem_iterator_A(this->smem_iterator_A_);

      typename IteratorA::AccessType zero_A;
      zero_A.clear();

      last_smem_iterator_A.set_iteration_index(0);

      // Async Copy for operand A
      MCTLASS_PRAGMA_UNROLL
      for (int j = 0; j < Detail::AsyncCopyIterationsPerStageA; ++j) {

        typename IteratorA::AccessType *dst_ptr =
            reinterpret_cast<typename IteratorA::AccessType *>(
                last_smem_iterator_A.get());

        *dst_ptr = zero_A;

        ++last_smem_iterator_A;
      }

      /// Iterator to write threadblock-scoped tile of B operand to shared memory
      SmemIteratorB last_smem_iterator_B(this->smem_iterator_B_);
      typename IteratorB::AccessType zero_B;

      zero_B.clear();
      last_smem_iterator_B.set_iteration_index(0);

      // Async Copy for operand B
      MCTLASS_PRAGMA_UNROLL
      for (int j = 0; j < Detail::AsyncCopyIterationsPerStageB; ++j) {

        typename IteratorB::AccessType *dst_ptr =
            reinterpret_cast<typename IteratorB::AccessType *>(
                last_smem_iterator_B.get());

        *dst_ptr = zero_B;

        ++last_smem_iterator_B;
      }
    }

    // Waits until stages up to the previous (kStages-2)th stage have committed.
    // mctlass::arch::cp_async_wait<Base::kStages - 2>();
    // if constexpr (Syncthreads) __syncthreads();
    __builtin_mxc_arrive(64);
    copy_tiles_and_advance_2share_A_(0, smem_register_A_);
    copy_tiles_and_advance_2share_B_(0, smem_register_B_);

    // Move to the next stage
    iterator_A.add_tile_offset({0, 1});
    iterator_B.add_tile_offset({1, 0});

    copy_tiles_and_advance_2register_A_(iterator_A, 0, smem_register_A_);
    copy_tiles_and_advance_2register_B_(iterator_B, 0, smem_register_B_);

    __builtin_mxc_barrier();

    // Pair of fragments used to overlap shared memory loads and math
    // instructions
    WarpLoadedFragmentA warp_loaded_frag_A[8];
    WarpLoadedFragmentB warp_loaded_frag_B[8];

    Operator warp_mma;

    this->warp_tile_iterator_B_.load_with_index(warp_loaded_frag_B[0],0, share_offset_64);
    this->warp_tile_iterator_A_.load_with_index(warp_loaded_frag_A[0],0, share_offset_64);

    this->warp_tile_iterator_B_.load_with_index(warp_loaded_frag_B[1],1, share_offset_64);
    this->warp_tile_iterator_A_.load_with_index(warp_loaded_frag_A[1],1, share_offset_64);

    this->warp_tile_iterator_B_.load_with_index(warp_loaded_frag_B[2],2, -1 * share_offset_64);
    this->warp_tile_iterator_A_.load_with_index(warp_loaded_frag_A[2],2, -1 * share_offset_64);

    this->warp_tile_iterator_B_.load_with_index(warp_loaded_frag_B[3],3, -1 * share_offset_64);
    this->warp_tile_iterator_A_.load_with_index(warp_loaded_frag_A[3],3, -1 * share_offset_64);

    __builtin_mxc_arrive(64);

    copy_tiles_and_advance_2share_A_(1, smem_register_A_);
    copy_tiles_and_advance_2share_B_(1, smem_register_B_);

    // tf32x3 kernels use staging accumulation. warp_mma uses a temporary
    // accumulator and this temporary accumulator is added to the final
    // accumulator once in every mainloop iteration.
    plus<FragmentC> plus_accum;

    FragmentC tmp_accum;

    if (platform::is_same<typename Operator::MathOperator,
                          arch::OpMultiplyAddFastF32>::value
      || platform::is_same<typename Operator::MathOperator,
                          arch::OpMultiplyAddComplexFastF32>::value) {

      tmp_accum.clear();
    }
    //
    // Mainloop
    //
    MCTLASS_GEMM_LOOP
    for (int k = 0; k < gemm_k_iterations / 2; ++k) {
      //
      // Loop over GEMM K dimension
      //
      iterator_A.add_tile_offset({0, 1});
      iterator_B.add_tile_offset({1, 0});
      asm(";------");
      copy_tiles_and_advance_2register_A_(iterator_A, 0, smem_register_A_);
      copy_tiles_and_advance_2register_B_(iterator_B, 0, smem_register_B_);
      __builtin_mxc_barrier();
      //warm_k = 0
      asm(";------");
      this->warp_tile_iterator_B_.load_with_index(warp_loaded_frag_B[4],4, share_offset_64);
      this->warp_tile_iterator_A_.load_with_index(warp_loaded_frag_A[4],4, share_offset_64);
      asm(";------");
      warp_mma.frontHalfMma(accum,warp_loaded_frag_A[0],warp_loaded_frag_B[0],accum);
      warp_mma.laterHalfMma(accum,warp_loaded_frag_A[0],warp_loaded_frag_B[0],accum);
      // warp_mma(accum,warp_loaded_frag_A[0],warp_loaded_frag_B[0],accum);

      //warm_k = 1
      asm(";------");
      this->warp_tile_iterator_B_.load_with_index(warp_loaded_frag_B[5],5, share_offset_64);
      this->warp_tile_iterator_A_.load_with_index(warp_loaded_frag_A[5],5, share_offset_64);
      asm(";------");
      warp_mma(accum,warp_loaded_frag_A[1],warp_loaded_frag_B[1],accum);

      //warm_k = 2
      asm(";------");
      this->warp_tile_iterator_B_.load_with_index(warp_loaded_frag_B[6],6, -1 * share_offset_64);
      this->warp_tile_iterator_A_.load_with_index(warp_loaded_frag_A[6],6, -1 * share_offset_64);
      asm(";------");
      warp_mma(accum,warp_loaded_frag_A[2],warp_loaded_frag_B[2],accum);

      //warm_k = 3
      asm(";------");
      this->warp_tile_iterator_B_.load_with_index(warp_loaded_frag_B[7],7, -1 * share_offset_64);
      this->warp_tile_iterator_A_.load_with_index(warp_loaded_frag_A[7],7, -1 * share_offset_64);
      asm(";------");
      warp_mma(accum,warp_loaded_frag_A[3],warp_loaded_frag_B[3],accum);

      __builtin_mxc_arrive(64);
      copy_tiles_and_advance_2share_A_(0, smem_register_A_);
      copy_tiles_and_advance_2share_B_(0, smem_register_B_);

      iterator_A.add_tile_offset({0, 1});
      iterator_B.add_tile_offset({1, 0});

      copy_tiles_and_advance_2register_A_(iterator_A, 0, smem_register_A_);
      copy_tiles_and_advance_2register_B_(iterator_B, 0, smem_register_B_);

      __builtin_mxc_barrier();

      // warm_k = 0
      asm(";------");
      this->warp_tile_iterator_B_.load_with_index(warp_loaded_frag_B[0],0, share_offset_64);
      this->warp_tile_iterator_A_.load_with_index(warp_loaded_frag_A[0],0, share_offset_64);
      asm(";------");
      warp_mma(accum,warp_loaded_frag_A[4],warp_loaded_frag_B[4],accum);

      // warm_k = 1
      asm(";------");
      this->warp_tile_iterator_B_.load_with_index(warp_loaded_frag_B[1],1, share_offset_64);
      this->warp_tile_iterator_A_.load_with_index(warp_loaded_frag_A[1],1, share_offset_64);
      asm(";------");
      warp_mma(accum, warp_loaded_frag_A[5],warp_loaded_frag_B[5], accum);

      // warm_k = 2
      asm(";------");
      this->warp_tile_iterator_B_.load_with_index(warp_loaded_frag_B[2],2, -1 * share_offset_64);
      this->warp_tile_iterator_A_.load_with_index(warp_loaded_frag_A[2],2, -1 * share_offset_64);
      asm(";------");
      warp_mma(accum, warp_loaded_frag_A[6],warp_loaded_frag_B[6], accum);

      // warm_k = 3
      asm(";------");
      this->warp_tile_iterator_B_.load_with_index(warp_loaded_frag_B[3],3, -1 * share_offset_64);
      this->warp_tile_iterator_A_.load_with_index(warp_loaded_frag_A[3],3, -1 * share_offset_64);
      asm(";------");
      warp_mma(accum, warp_loaded_frag_A[7],warp_loaded_frag_B[7], accum);

      __builtin_mxc_arrive(64);
      copy_tiles_and_advance_2share_A_(1, smem_register_A_);
      copy_tiles_and_advance_2share_B_(1, smem_register_B_);
      asm(";------");

    }

    if (gemm_k_iterations % 2 == 1) {
      warp_mma(accum, warp_loaded_frag_A[0],warp_loaded_frag_B[0], accum);
      warp_mma(accum, warp_loaded_frag_A[1],warp_loaded_frag_B[1], accum);
    }

    if (platform::is_same<typename Operator::MathOperator,
                          arch::OpMultiplyAddFastF32>::value
      || platform::is_same<typename Operator::MathOperator,
                           arch::OpMultiplyAddComplexFastF32>::value) {
      accum = plus_accum(accum, tmp_accum);
    }

    if (SharedMemoryClear == SharedMemoryClearOption::kZfill) {
      // commit and drain all pending and predicated LDGSTS pnz from the GEMM mainloop
      mctlass::arch::cp_async_fence();
      mctlass::arch::cp_async_wait<0>();
      if constexpr (!MacaSpecialTag) __syncthreads();
    }

  }
};

// mcTlass: problem: TN && stage==2
//          (SmemIteratorA_::Layout == MacaRowMajorTensorOpMultiplicandConflictFreeCrosswise) &&
//          (SmemIteratorB_::Layout == MacaColumnMajorTensorOpMultiplicandConflictFreeCrosswise) &&
//          (stage == 2)
template <
    /// Size of the Gemm problem - concept: gemm::GemmShape<>
    typename Shape_,
    /// Iterates over tiles of A operand in global memory
    //  (concept: ReadableTileIterator | ForwardTileIterator |
    //  MaskedTileIterator)
    typename IteratorA_,
    /// Iterates over tiles of A operand in shared memory
    /// (concept: WriteableTileIterator | RandomAccessTileIterator)
    typename SmemIteratorA_,
    /// Cache operation for operand A
    mctlass::arch::CacheOperation::Kind CacheOpA,
    /// Iterates over tiles of B operand in global memory
    //  (concept: ReadableTileIterator | ForwardTileIterator |
    //  MaskedTileIterator)
    typename IteratorB_,
    /// Iterates over tiles of B operand in shared memory
    /// (concept: WriteableTileIterator | RandomAccessTileIterator)
    typename SmemIteratorB_,
    /// Cache operation for operand B
    mctlass::arch::CacheOperation::Kind CacheOpB,
    /// Data type of accumulator matrix
    typename ElementC_,
    /// Data type of accumulator matrix
    typename LayoutC_,
    /// Policy describing tuning details (concept: MmaPolicy)
    typename Policy_,
    /// Number of stages,
    // int Stages,
    /// Use zfill or predicate for out-of-bound cp.async
    SharedMemoryClearOption SharedMemoryClear,
    /// Flag tag for special process for m16n16k16 f16
    bool MacaSpecialTag_
    >
class MmaMultistage <
  Shape_,
  IteratorA_,
  SmemIteratorA_,
  CacheOpA,
  IteratorB_,
  SmemIteratorB_,
  CacheOpB,
  ElementC_,
  LayoutC_,
  Policy_,
  2,
  SharedMemoryClear,
  MacaSpecialTag_,
  typename std::enable_if<
    ((std::is_same<typename SmemIteratorA_::Layout,
        typename layout::MacaRowMajorTensorOpMultiplicandConflictFreeCrosswise<
          sizeof_bits<mctlass::half_t>::value, Shape_::kK>>::value &&
      std::is_same<typename SmemIteratorB_::Layout,
        typename layout::MacaColumnMajorTensorOpMultiplicandConflictFreeCrosswise<
          sizeof_bits<mctlass::half_t>::value, Shape_::kK>>::value) ||
    (std::is_same<typename SmemIteratorA_::Layout,
        typename layout::MacaRowMajorTensorOpMultiplicandConflictFreeCrosswise<
          sizeof_bits<int8_t>::value, Shape_::kK>>::value &&
      std::is_same<typename SmemIteratorB_::Layout,
        typename layout::MacaColumnMajorTensorOpMultiplicandConflictFreeCrosswise<
          sizeof_bits<int8_t>::value, Shape_::kK>>::value) ||
    (std::is_same<typename SmemIteratorA_::Layout,
        typename layout::MacaRowMajorTensorOpMultiplicandConflictFreeCrosswise<
          sizeof_bits<float>::value, Shape_::kK>>::value &&
      std::is_same<typename SmemIteratorB_::Layout,
        typename layout::MacaColumnMajorTensorOpMultiplicandConflictFreeCrosswise<
          sizeof_bits<float>::value, Shape_::kK>>::value)),
    bool>::type
  >:public MmaBase<Shape_, Policy_, 2> {
public:
  ///< Base class
  using Base = MmaBase<Shape_, Policy_, 2>;
  ///< Size of the Gemm problem - concept: gemm::GemmShape<>
  using Shape = Shape_;
  ///< Iterates over tiles of A operand in global memory
  using IteratorA = IteratorA_;
  ///< Iterates over tiles of B operand in global memory
  using IteratorB = IteratorB_;
  ///< Data type of accumulator matrix
  using ElementC = ElementC_;
  ///< Layout of accumulator matrix
  using LayoutC = LayoutC_;
  ///< Policy describing tuning details
  using Policy = Policy_;

  using SmemIteratorA = SmemIteratorA_;
  using SmemIteratorB = SmemIteratorB_;

  static mctlass::arch::CacheOperation::Kind const kCacheOpA = CacheOpA;
  static mctlass::arch::CacheOperation::Kind const kCacheOpB = CacheOpB;
  // static constexpr bool MacaSpecialTag = MacaSpecialTag_;
  static constexpr bool MacaSpecialTag = false;

  static constexpr bool CrossOffsetA =
    platform::is_same<layout::MacaRowMajorTensorOpMultiplicandCpAsyncCrosswise<
      sizeof_bits<typename IteratorA::Element>::value, Shape::kK>,
    typename SmemIteratorA::Layout>::value;
  static constexpr bool CrossOffsetB =
    platform::is_same<layout::MacaColumnMajorTensorOpMultiplicandCpAsyncCrosswise<
      sizeof_bits<typename IteratorB::Element>::value, Shape::kK>,
    typename SmemIteratorB::Layout>::value;

  static_assert(Shape::kK % (512 / sizeof_bits<typename IteratorB::Element>::value) == 0,
      "Crosswise kContiguous must be 512bit aligned");
  //
  // Dependent types
  //

  /// Fragment of accumulator tile
  using FragmentC = typename Policy::Operator::FragmentC;

  /// Warp-level Mma
  using Operator = typename Policy::Operator;

  /// Minimum architecture is Sm80 to support cp.async
  using ArchTag = arch::Sm80;

  /// Complex transform on A operand
  static ComplexTransform const kTransformA = Operator::kTransformA;

  /// Complex transform on B operand
  static ComplexTransform const kTransformB = Operator::kTransformB;

  static int const kSrcBytesB = sizeof_bits<typename IteratorB::Element>::value *
                                IteratorB::ThreadMap::kElementsPerAccess /
                                IteratorB::kAccessesPerVector / 8;
  static int const kSrcBytesA = sizeof_bits<typename IteratorA::Element>::value *
                                IteratorA::ThreadMap::kElementsPerAccess /
                                IteratorA::kAccessesPerVector / 8;
  using AccessTypeB  = Array<uint8_t, kSrcBytesB>;

  /// Internal structure exposed for introspection.
  struct Detail {

    /// Number of cp.async instructions to load one stage of operand A
    static int const AsyncCopyIterationsPerStageA =
        IteratorA::ThreadMap::Iterations::kCount;

    /// Number of cp.async instructions to load one stage of operand B
    static int const AsyncCopyIterationsPerStageB =
        IteratorB::ThreadMap::Iterations::kCount;

    /// Number of stages
    static int const kStages = 2;

    /// Number of cp.async instructions to load on group of operand A
    static int const kAccessesPerGroupA =
        (AsyncCopyIterationsPerStageA + Base::kWarpGemmIterations - 1) / Base::kWarpGemmIterations;

    /// Number of cp.async instructions to load on group of operand B
    static int const kAccessesPerGroupB =
        (AsyncCopyIterationsPerStageB + Base::kWarpGemmIterations - 1) / Base::kWarpGemmIterations;
  };

 private:

  using WarpLoadedFragmentA = typename Operator::FragmentA;
  using WarpLoadedFragmentB = typename Operator::FragmentB;
  using WarpTransformedFragmentA = typename Operator::TransformedFragmentA;
  using WarpTransformedFragmentB = typename Operator::TransformedFragmentB;

 private:

  //
  // Data members
  //

  /// Iterator to write threadblock-scoped tile of A operand to shared memory
  SmemIteratorA smem_iterator_A_;

  /// Iterator to write threadblock-scoped tile of B operand to shared memory
  SmemIteratorB smem_iterator_B_;

  int cross_offset_A_;
  int cross_offset_B_;

  int register_offset_l = 0;
  int register_offset_r = 0;

  typename IteratorA::AccessType* smem_ptr_A_[Base::kStages * Detail::AsyncCopyIterationsPerStageA * IteratorA::kAccessesPerVector];
  typename IteratorB::AccessType* smem_ptr_B_[Base::kStages * Detail::AsyncCopyIterationsPerStageB * IteratorB::kAccessesPerVector];

  int problem_size_k_;
  int remain_k_;
  int group_k_;

  typename IteratorA::Element *ptr_A_;
  typename IteratorA::Element *ptr_B_;

public:

  /// Construct from tensor references
  MCTLASS_DEVICE
  MmaMultistage(
      ///< Shared storage needed for internal use by threadblock-scoped GEMM
      typename Base::SharedStorage &shared_storage,
      ///< ID within the threadblock
      int thread_idx,
      ///< ID of warp
      int warp_idx,
      ///< ID of each thread within a warp
      int lane_idx,

      int problem_size_k = 0,
      typename IteratorA::Element *ptr_A = nullptr,
      typename IteratorA::Element *ptr_B = nullptr
    ):
      Base(shared_storage, thread_idx, warp_idx, lane_idx),
      smem_iterator_A_(shared_storage.operand_A_ref(), thread_idx),
      smem_iterator_B_(shared_storage.operand_B_ref(), thread_idx)
  {
    int bid = lane_idx & 0x7;
    int group_id = (lane_idx >> 3) & 0x3;
    int cross_target = bid ^ group_id;
    int cross_offset_ = cross_target - bid;
    cross_offset_A_ = CrossOffsetA ? cross_offset_ : 0;
    cross_offset_B_ = CrossOffsetB ? cross_offset_ : 0;

    if ((threadIdx.x / 32) % 2 == 0) {
      if ((threadIdx.x / 8) % 2 != 0) {
        register_offset_l = 1;
        register_offset_r = -1;
      }
    } else {
      if ((threadIdx.x / 8) % 2 == 0) {
        register_offset_l = 1;
        register_offset_r = -1;
      }
    }

    // Compute warp location within threadblock tile by mapping the warp_id to
    // three coordinates:
    //   _m: the warp's position within the threadblock along the M dimension
    //   _n: the warp's position within the threadblock along the N dimension
    //   _k: the warp's position within the threadblock along the K dimension

    int warp_idx_mn = warp_idx % (Base::WarpCount::kM * Base::WarpCount::kN);
    int warp_idx_k = warp_idx / (Base::WarpCount::kM * Base::WarpCount::kN);

    int warp_idx_m = warp_idx_mn % Base::WarpCount::kM;
    int warp_idx_n = warp_idx_mn / Base::WarpCount::kM;

    // Add per-warp offsets in units of warp-level tiles
    this->warp_tile_iterator_A_.pre_compute_pointer(warp_idx_m);
    this->warp_tile_iterator_B_.pre_compute_pointer(warp_idx_n);

    // Compute A shared memory ptr
    this->smem_iterator_A_.set_iteration_index(0);
    for(int stage = 0; stage < Base::kStages; ++stage){
      for(int j = 0; j < Detail::AsyncCopyIterationsPerStageA; ++j){
        for(int v = 0; v < IteratorA::kAccessesPerVector; ++v ){

          smem_ptr_A_[stage * Detail::AsyncCopyIterationsPerStageA + j * IteratorA::kAccessesPerVector + v] =
            reinterpret_cast<typename IteratorA::AccessType *>(this->smem_iterator_A_.get()) + v - cross_offset_A_;
          ++this->smem_iterator_A_;
        }
      }
      this->smem_iterator_A_.add_tile_offset({0,1});
    }
    // Reset tile offset to zero
    this->smem_iterator_A_.add_tile_offset({0,-Base::kStages});

    // Compute B shared memory ptr
    this->smem_iterator_B_.set_iteration_index(0);
    for(int stage = 0; stage < Base::kStages; ++stage){
      for(int j = 0; j < Detail::AsyncCopyIterationsPerStageB; ++j){
        for(int v = 0; v < IteratorB::kAccessesPerVector; ++v){
          smem_ptr_B_[stage * Detail::AsyncCopyIterationsPerStageB + j * IteratorB::kAccessesPerVector + v] =
            reinterpret_cast<typename IteratorA::AccessType *>(this->smem_iterator_B_.get())+ v - cross_offset_B_;
          ++this->smem_iterator_B_;
        }
      }
      this->smem_iterator_B_.add_tile_offset({1,0});
    }
    // Reset tile offset to zero
    this->smem_iterator_B_.add_tile_offset({-Base::kStages,0});
    problem_size_k_ = problem_size_k;
    remain_k_ = problem_size_k_ % Shape_::kK;
    group_k_ = remain_k_ / IteratorA::ThreadMap::kElementsPerAccess * IteratorA::ThreadMap::kElementsPerAccess;
    remain_k_ = remain_k_ % IteratorA::ThreadMap::kElementsPerAccess;
    ptr_A_ = ptr_A;
    ptr_B_ = ptr_B;
  }

  MCTLASS_DEVICE
  void copy_tiles_and_advance_2register_A_(IteratorA &iterator_A, int group_start_A = 0, v4i *reg_ptr_A_ = nullptr, int32_t index_A = 0) {
    iterator_A.set_iteration_index(group_start_A * IteratorA::kAccessesPerVector);
    MCTLASS_PRAGMA_UNROLL
    for (int j = 0; j < Detail::AsyncCopyIterationsPerStageA; ++j) {
      MCTLASS_PRAGMA_UNROLL
      for (int v = 0; v < IteratorA::kAccessesPerVector; ++v) {
        // if (iterator_A.valid()) {
          reg_ptr_A_[index_A] = __builtin_mxc_load_global_async128((v4i *)(iterator_A.get() + cross_offset_A_));
        // }

        ++iterator_A;
        ++(index_A);
      }
    }
  }

  MCTLASS_DEVICE
  void remain_copy_tiles_and_advance_2register_A_(IteratorA &iterator_A, int group_start_A = 0, v4i *reg_ptr_A_ = nullptr, int32_t index_A = 0) {
    using AccessType = Array<typename IteratorA::Element, IteratorA::ThreadMap::kElementsPerAccess>;
    using DoubleElementAccessType = Array<typename IteratorB::Element, 2>;

    iterator_A.set_iteration_index(group_start_A * IteratorA::kAccessesPerVector);
    MCTLASS_PRAGMA_UNROLL
    for (int j = 0; j < Detail::AsyncCopyIterationsPerStageA; ++j) {
      MCTLASS_PRAGMA_UNROLL
      for (int v = 0; v < IteratorA::kAccessesPerVector; ++v) {
        reg_ptr_A_[index_A] = 0;
        int index_k = (typename IteratorA::Element *)(iterator_A.get() + cross_offset_A_) - ptr_A_;
        index_k = index_k % problem_size_k_;

        if (index_k <= (group_k_)) {
          *static_cast<AccessType *>((void *)&reg_ptr_A_[index_A]) = *static_cast<AccessType const *>(iterator_A.get() + cross_offset_A_);
          if (index_k == (group_k_)) {

            DoubleElementAccessType *reg_ptr_A_k = (DoubleElementAccessType *)(&reg_ptr_A_[index_A]);
            DoubleElementAccessType zeros;
            zeros.clear();

            if (remain_k_ / 2 == 0) {
              MCTLASS_PRAGMA_UNROLL
              for(int idx = 0; idx <  IteratorA::ThreadMap::kElementsPerAccess / 2; ++idx) {
                reg_ptr_A_k[idx] = zeros;
              }
            }

            if (remain_k_ / 2 == 1) {
              MCTLASS_PRAGMA_UNROLL
              for(int idx = 1; idx <  IteratorA::ThreadMap::kElementsPerAccess / 2; ++idx) {
                reg_ptr_A_k[idx] = zeros;
              }
            }

            if (remain_k_ / 2 == 2) {
              MCTLASS_PRAGMA_UNROLL
              for(int idx = 2; idx <  IteratorA::ThreadMap::kElementsPerAccess / 2; ++idx) {
                reg_ptr_A_k[idx] = zeros;
              }
            }

            if (remain_k_ / 2 == 3) {
              MCTLASS_PRAGMA_UNROLL
              for(int idx = 3; idx <  IteratorA::ThreadMap::kElementsPerAccess / 2; ++idx) {
                reg_ptr_A_k[idx] = zeros;
              }
            }

          }
        }
        ++iterator_A;
        ++(index_A);
      }
    }
  }

  MCTLASS_DEVICE
  void copy_tiles_and_advance_2share_A_(int stage = 0, v4i *reg_ptr_A = nullptr) {
    int32_t index_A = 0;
    // Async Copy for operand A
    using HalfAccessType = Array<typename IteratorB::Element, IteratorB::ThreadMap::kElementsPerAccess / 2>;
    MCTLASS_PRAGMA_UNROLL
    for (int j = 0; j < Detail::AsyncCopyIterationsPerStageA; ++j) {
      MCTLASS_PRAGMA_UNROLL
      for (int v = 0; v < IteratorA::kAccessesPerVector; ++v) {
        HalfAccessType *reg_value = static_cast<HalfAccessType *>((void *)(&reg_ptr_A[index_A]));
        HalfAccessType *half_dst_ptr = reinterpret_cast<HalfAccessType *>(smem_ptr_A_[stage * Detail::AsyncCopyIterationsPerStageA + j * IteratorA::kAccessesPerVector + v]);

        *static_cast<HalfAccessType *>((void *)&half_dst_ptr[0  + register_offset_l]) = *static_cast<HalfAccessType *>((void *)(&reg_value[0]));
        *static_cast<HalfAccessType *>((void *)&half_dst_ptr[1  + register_offset_r]) = *static_cast<HalfAccessType *>((void *)(&reg_value[1]));

        ++(index_A);
      }
    }
  }

  MCTLASS_DEVICE
  void copy_tiles_and_advance_2register_B_(IteratorB &iterator_B, int group_start_B = 0,
                                        v4i *reg_ptr_B = nullptr) {
    int32_t index_B = 0;
    iterator_B.set_iteration_index(group_start_B * IteratorB::kAccessesPerVector);

    MCTLASS_PRAGMA_UNROLL
    for (int j = 0; j < Detail::AsyncCopyIterationsPerStageB; ++j) {
      MCTLASS_PRAGMA_UNROLL
      for (int v = 0; v < IteratorB::kAccessesPerVector; ++v) {
        // if (iterator_B.valid()) {
          reg_ptr_B[index_B]=__builtin_mxc_load_global_async128((v4i *)(iterator_B.get() + cross_offset_B_));
        // }
        ++iterator_B;
        ++(index_B);
      }
    }
  }

  MCTLASS_DEVICE
  void remain_copy_tiles_and_advance_2register_B_(IteratorB &iterator_B, int group_start_B = 0,
                                        v4i *reg_ptr_B = nullptr) {
    using AccessType = Array<typename IteratorB::Element, IteratorB::ThreadMap::kElementsPerAccess>;
    using DoubleElementAccessType = Array<typename IteratorB::Element, 2>;
    int32_t index_B = 0;
    iterator_B.set_iteration_index(group_start_B * IteratorB::kAccessesPerVector);

    MCTLASS_PRAGMA_UNROLL
    for (int j = 0; j < Detail::AsyncCopyIterationsPerStageB; ++j) {
      MCTLASS_PRAGMA_UNROLL
      for (int v = 0; v < IteratorB::kAccessesPerVector; ++v) {
        reg_ptr_B[index_B] = 0;
        int index_k = (typename IteratorB::Element *)(iterator_B.get() + cross_offset_B_) - ptr_B_;
        index_k = index_k % problem_size_k_;

        if (index_k <= group_k_) {
          *static_cast<AccessType *>((void *)&reg_ptr_B[index_B]) = *static_cast<AccessType const *>(iterator_B.get() + cross_offset_B_);
          if (index_k == group_k_) {
            DoubleElementAccessType *reg_ptr_B_k = (DoubleElementAccessType *)((void *)&reg_ptr_B[index_B]);
            DoubleElementAccessType zeros;
            zeros.clear();

            if (remain_k_ / 2 == 0) {
              MCTLASS_PRAGMA_UNROLL
              for(int idx = 0; idx < IteratorB::ThreadMap::kElementsPerAccess / 2; ++idx) {
                reg_ptr_B_k[idx] = zeros;
              }
            }

            if (remain_k_ / 2 == 1) {
              MCTLASS_PRAGMA_UNROLL
              for(int idx = 1; idx < IteratorB::ThreadMap::kElementsPerAccess / 2; ++idx) {
                reg_ptr_B_k[idx] = zeros;
              }
            }

            if (remain_k_ / 2 == 2) {
              MCTLASS_PRAGMA_UNROLL
              for(int idx = 2; idx < IteratorB::ThreadMap::kElementsPerAccess / 2; ++idx) {
                reg_ptr_B_k[idx] = zeros;
              }
            }

            if (remain_k_ / 2 == 3) {
              MCTLASS_PRAGMA_UNROLL
              for(int idx = 3; idx < IteratorB::ThreadMap::kElementsPerAccess / 2; ++idx) {
                reg_ptr_B_k[idx] = zeros;
              }
            }

          }
        }
        ++iterator_B;
        ++(index_B);
      }
    }
  }

  MCTLASS_DEVICE
  void copy_tiles_and_advance_2share_B_(int stage = 0, v4i *reg_ptr_B = nullptr) {
    // Async Copy for operand B
    int32_t index_B = 0;
    using HalfAccessType = Array<typename IteratorB::Element, IteratorB::ThreadMap::kElementsPerAccess / 2>;
    MCTLASS_PRAGMA_UNROLL
    for (int j = 0; j < Detail::AsyncCopyIterationsPerStageB; ++j) {
      MCTLASS_PRAGMA_UNROLL
      for (int v = 0; v < IteratorB::kAccessesPerVector; ++v) {

        HalfAccessType *reg_value = static_cast<HalfAccessType *>((void *)(&reg_ptr_B[index_B]));
        HalfAccessType *half_dst_ptr = reinterpret_cast<HalfAccessType *>(smem_ptr_B_[stage * Detail::AsyncCopyIterationsPerStageB + j * IteratorB::kAccessesPerVector + v]);
        *static_cast<HalfAccessType *>((void *)&half_dst_ptr[0  + register_offset_l]) = *static_cast<HalfAccessType *>((void *)(&reg_value[0]));
        *static_cast<HalfAccessType *>((void *)&half_dst_ptr[1  + register_offset_r]) = *static_cast<HalfAccessType *>((void *)(&reg_value[1]));

        ++(index_B);
      }
    }
  }

  /// Perform a threadblock-scoped matrix multiply-accumulate
  MCTLASS_DEVICE
  void operator()(
      ///< problem size of GEMM
      int gemm_k_iterations,
      ///< destination accumulator tile
      FragmentC &accum,
      ///< iterator over A operand in global memory
      IteratorA iterator_A,
      ///< iterator over B operand in global memory
      IteratorB iterator_B,
      ///< initial value of accumulator
      FragmentC const &src_accum) {
    //
    // Prologue
    //

    // Issue several complete stages
    v4i smem_register_A_[Detail::AsyncCopyIterationsPerStageA * IteratorA::kAccessesPerVector];
    v4i smem_register_B_[Detail::AsyncCopyIterationsPerStageB * IteratorB::kAccessesPerVector];

    if (group_k_ + remain_k_ == 0) {
      copy_tiles_and_advance_2register_A_(iterator_A, 0, smem_register_A_);
      copy_tiles_and_advance_2register_B_(iterator_B, 0, smem_register_B_);
      __builtin_mxc_arrive(64);

    } else {
      remain_copy_tiles_and_advance_2register_A_(iterator_A, 0, smem_register_A_);
      remain_copy_tiles_and_advance_2register_B_(iterator_B, 0, smem_register_B_);
    }

    copy_tiles_and_advance_2share_A_(0, smem_register_A_);
    copy_tiles_and_advance_2share_B_(0, smem_register_B_);

    // Move to the next stage
    iterator_A.add_tile_offset({0, 1});
    iterator_B.add_tile_offset({1, 0});

    copy_tiles_and_advance_2register_A_(iterator_A, 0, smem_register_A_);
    copy_tiles_and_advance_2register_B_(iterator_B, 0, smem_register_B_);

    __builtin_mxc_barrier();

    // Pair of fragments used to overlap shared memory loads and math
    // instructions
    WarpLoadedFragmentA warp_loaded_frag_A[2];
    WarpLoadedFragmentB warp_loaded_frag_B[2];

    Operator warp_mma;

    this->warp_tile_iterator_B_.load_with_index(warp_loaded_frag_B[0],0);
    this->warp_tile_iterator_A_.load_with_index(warp_loaded_frag_A[0],0);

    this->warp_tile_iterator_B_.load_with_index(warp_loaded_frag_B[1],1);
    this->warp_tile_iterator_A_.load_with_index(warp_loaded_frag_A[1],1);

    __builtin_mxc_arrive(64);

    copy_tiles_and_advance_2share_A_(1, smem_register_A_);
    copy_tiles_and_advance_2share_B_(1, smem_register_B_);

    // tf32x3 kernels use staging accumulation. warp_mma uses a temporary
    // accumulator and this temporary accumulator is added to the final
    // accumulator once in every mainloop iteration.
    plus<FragmentC> plus_accum;

    FragmentC tmp_accum;

    if (platform::is_same<typename Operator::MathOperator,
                          arch::OpMultiplyAddFastF32>::value
      || platform::is_same<typename Operator::MathOperator,
                          arch::OpMultiplyAddComplexFastF32>::value) {

      tmp_accum.clear();
    }

    //
    // Mainloop
    //
    MCTLASS_GEMM_LOOP
    for (int k = 0; k < gemm_k_iterations / 2; ++k) {
      //
      // Loop over GEMM K dimension
      //
      v4i smem_register_A_[Detail::AsyncCopyIterationsPerStageA * IteratorA::kAccessesPerVector];
      v4i smem_register_B_[Detail::AsyncCopyIterationsPerStageB * IteratorB::kAccessesPerVector];

      if constexpr(2 == Base::kWarpGemmIterations){
        iterator_A.add_tile_offset({0, 1});
        iterator_B.add_tile_offset({1, 0});
        asm(";------");
        copy_tiles_and_advance_2register_A_(iterator_A, 0, smem_register_A_);
        copy_tiles_and_advance_2register_B_(iterator_B, 0, smem_register_B_);

        __builtin_mxc_barrier();
        asm(";------");
        warp_mma.frontHalfMma(accum,warp_loaded_frag_A[0],warp_loaded_frag_B[0],accum);
        warp_mma.laterHalfMma(accum,warp_loaded_frag_A[0],warp_loaded_frag_B[0],accum);
        asm(";------");
        this->warp_tile_iterator_B_.load_with_index(warp_loaded_frag_B[0],2);
        this->warp_tile_iterator_A_.load_with_index(warp_loaded_frag_A[0],2);
        asm(";------");
        warp_mma(accum,warp_loaded_frag_A[1],warp_loaded_frag_B[1],accum);
        this->warp_tile_iterator_B_.load_with_index(warp_loaded_frag_B[1],3);
        this->warp_tile_iterator_A_.load_with_index(warp_loaded_frag_A[1],3);

        __builtin_mxc_arrive(64);
        copy_tiles_and_advance_2share_A_(0, smem_register_A_);
        copy_tiles_and_advance_2share_B_(0, smem_register_B_);

        iterator_A.add_tile_offset({0, 1});
        iterator_B.add_tile_offset({1, 0});

        copy_tiles_and_advance_2register_A_(iterator_A, 0, smem_register_A_);
        copy_tiles_and_advance_2register_B_(iterator_B, 0, smem_register_B_);

        __builtin_mxc_barrier();

        asm(";------");
        warp_mma.frontHalfMma(accum, warp_loaded_frag_A[0],warp_loaded_frag_B[0], accum);
        warp_mma.laterHalfMma(accum, warp_loaded_frag_A[0],warp_loaded_frag_B[0], accum);
        asm(";------");
        this->warp_tile_iterator_B_.load_with_index(warp_loaded_frag_B[0],0);
        this->warp_tile_iterator_A_.load_with_index(warp_loaded_frag_A[0],0);
        asm(";------");
        warp_mma(accum, warp_loaded_frag_A[1],warp_loaded_frag_B[1], accum);
          asm(";------");
        this->warp_tile_iterator_B_.load_with_index(warp_loaded_frag_B[1],1);
        this->warp_tile_iterator_A_.load_with_index(warp_loaded_frag_A[1],1);

        __builtin_mxc_arrive(64);
        copy_tiles_and_advance_2share_A_(1, smem_register_A_);
        copy_tiles_and_advance_2share_B_(1, smem_register_B_);
        asm(";------");

      }
    }

    if (gemm_k_iterations % 2 == 1) {
      warp_mma(accum, warp_loaded_frag_A[0],warp_loaded_frag_B[0], accum);
      warp_mma(accum, warp_loaded_frag_A[1],warp_loaded_frag_B[1], accum);
    }

    if (platform::is_same<typename Operator::MathOperator,
                          arch::OpMultiplyAddFastF32>::value
      || platform::is_same<typename Operator::MathOperator,
                           arch::OpMultiplyAddComplexFastF32>::value) {
      accum = plus_accum(accum, tmp_accum);
    }

  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace threadblock
}  // namespace gemm
}  // namespace mctlass

/////////////////////////////////////////////////////////////////////////////////////////////////
