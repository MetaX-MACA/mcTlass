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
/*!
  \file
  \brief The universal GEMM accommodates serial reductions, parallel reductions, batched strided, and
    batched array variants.
*/

#pragma once

//#include <limits>

#include "mctlass/mctlass.h"
#include "mctlass/numeric_types.h"
#include "mctlass/arch/arch.h"
#include "mctlass/device_kernel.h"

#include "mctlass/gemm/gemm.h"
#include "mctlass/gemm/threadblock/threadblock_swizzle.h"
#include "mctlass/gemm/kernel/gemm_universal.h"

#include "mctlass/gemm/kernel/default_gemm_universal.h"
#include "mctlass/gemm/device/default_gemm_configuration.h"

#include "mctlass/trace.h"

////////////////////////////////////////////////////////////////////////////////

namespace mctlass {
namespace gemm {
namespace device {

/////////////////////////////////////////////////////////////////////////////////////////////////


template <typename GemmKernel_>
class GemmUniversalBase {
public:

  using GemmKernel = GemmKernel_;
  using ThreadblockShape = typename GemmKernel::Mma::Shape;

  using ElementA = typename GemmKernel::ElementA;
  using LayoutA = typename GemmKernel::LayoutA;
  using TensorRefA = TensorRef<ElementA const, LayoutA>;
  static ComplexTransform const kTransformA = GemmKernel::kTransformA;

  using ElementB = typename GemmKernel::ElementB;
  using LayoutB = typename GemmKernel::LayoutB;
  using TensorRefB = TensorRef<ElementB const, LayoutB>;
  static ComplexTransform const kTransformB = GemmKernel::kTransformB;

  using ElementC = typename GemmKernel::ElementC;
  using LayoutC = typename GemmKernel::LayoutC;
  using TensorRefC = TensorRef<ElementC const, LayoutC>;
  using TensorRefD = TensorRef<ElementC, LayoutC>;

  using ElementAccumulator = typename GemmKernel::Mma::Policy::Operator::ElementC;

  using EpilogueOutputOp = typename GemmKernel::EpilogueOutputOp;
  using ThreadblockSwizzle = typename GemmKernel::ThreadblockSwizzle;
  using Operator = typename GemmKernel::Operator;

  /// Argument structure
  using Arguments = typename GemmKernel::Arguments;

protected:

  /// Kernel parameters object
  typename GemmKernel::Params params_;

protected:

  /// Private helper to obtain the grid dimensions with fix-up for split-K
  static void get_grid_shape_(gemm::GemmCoord &grid_tiled_shape, int &gemm_k_size, Arguments const &args) {

    // Determine grid shape
    ThreadblockSwizzle threadblock_swizzle;

    grid_tiled_shape = threadblock_swizzle.get_tiled_shape(
      args.problem_size,
      {ThreadblockShape::kM, ThreadblockShape::kN, ThreadblockShape::kK},
      args.batch_count);

    gemm_k_size = args.problem_size.k();

    if (args.mode == GemmUniversalMode::kGemm || args.mode == GemmUniversalMode::kGemmSplitKParallel) {

      int const kAlignK = const_max(const_max(128 / sizeof_bits<ElementA>::value, 128 / sizeof_bits<ElementB>::value), 1);

      gemm_k_size = round_up(ceil_div(args.problem_size.k(), args.batch_count), kAlignK);

      if (gemm_k_size) {
        grid_tiled_shape.k() = ceil_div(args.problem_size.k(), gemm_k_size);
      }
    }
  }

public:

  /// Constructs the GEMM.
  GemmUniversalBase() { }

  /// Determines whether the GEMM can execute the given problem.
  static Status can_implement(Arguments const &args) {

    // Determine grid shape
    mctlass::gemm::GemmCoord grid_tiled_shape;
    int gemm_k_size = 0;

    get_grid_shape_(grid_tiled_shape, gemm_k_size, args);

    ThreadblockSwizzle threadblock_swizzle;
    dim3 grid = threadblock_swizzle.get_grid_shape(grid_tiled_shape);

    uint32_t const kGridYZMax = ((1 << (sizeof(uint16_t) * 8)) - 1);

    if (!(grid.y <= kGridYZMax && grid.z <= kGridYZMax)) {

      return Status::kErrorInvalidProblem;
    }

    return GemmKernel::can_implement(args);
  }

  /// Gets the workspace size
  static size_t get_workspace_size(Arguments const &args) {

    MCTLASS_TRACE_HOST("GemmUniversalBase::get_workspace_size()");

    size_t workspace_bytes = 0;

    // Determine grid shape
    mctlass::gemm::GemmCoord grid_tiled_shape;
    int gemm_k_size = 0;

    get_grid_shape_(grid_tiled_shape, gemm_k_size, args);

    if (args.mode == GemmUniversalMode::kGemmSplitKParallel) {

      // Split-K parallel always requires a temporary workspace
      workspace_bytes =
        sizeof(ElementC) *
        size_t(args.batch_stride_D) *
        size_t(grid_tiled_shape.k());
    }
    else if (args.mode == GemmUniversalMode::kGemm && grid_tiled_shape.k() > 1) {

      // Serial split-K only requires a temporary workspace if the number of partitions along the
      // GEMM K dimension is greater than one.
      workspace_bytes = sizeof(int) * size_t(grid_tiled_shape.m()) * size_t(grid_tiled_shape.n());
    }

    MCTLASS_TRACE_HOST("  workspace_bytes: " << workspace_bytes);

    workspace_bytes += GemmKernel::get_extra_workspace_size(args, grid_tiled_shape);

    return workspace_bytes;
  }

  /// Computes the grid shape
  static dim3 get_grid_shape(Arguments const &args) {

    MCTLASS_TRACE_HOST("GemmUniversalBase::get_grid_shape()");

    ThreadblockSwizzle threadblock_swizzle;

    mctlass::gemm::GemmCoord grid_tiled_shape;
    int gemm_k_size = 0;

    get_grid_shape_(grid_tiled_shape, gemm_k_size, args);
    dim3 result = threadblock_swizzle.get_grid_shape(grid_tiled_shape);

    MCTLASS_TRACE_HOST(
         "  grid_tiled_shape: " << grid_tiled_shape  << "\n"
      << "  result = {" << result << "}");

    return result;
  }

  /// Computes the maximum number of active blocks per multiprocessor
  static int maximum_active_blocks(int smem_capacity = -1) {

    MCTLASS_TRACE_HOST("GemmUniversalBase::maximum_active_blocks()");

    int max_active_blocks = -1;
    int smem_size = int(sizeof(typename GemmKernel::SharedStorage));

    MCTLASS_TRACE_HOST("  smem_size: " << smem_size << " bytes");

    if (smem_size <= (64 << 10)) {

      mcError_t result = mcOccupancyMaxActiveBlocksPerMultiprocessor(
        &max_active_blocks,
        Kernel<GemmKernel>,
        GemmKernel::kThreadCount,
        smem_size);

      if (result == mcSuccess) {
        MCTLASS_TRACE_HOST("  max_active_blocks: " << max_active_blocks);
        return max_active_blocks;
      }
    }
    else {

      // Query assuming zero shared memory then compute occupancy limit based on SMEM
      mcError_t result = mcOccupancyMaxActiveBlocksPerMultiprocessor(
        &max_active_blocks,
        Kernel<GemmKernel>,
        GemmKernel::kThreadCount,
        0);

      if (result != mcSuccess) {

        MCTLASS_TRACE_HOST(
          "  mcOccupancyMaxActiveBlocksPerMultiprocessor() returned error "
          << mcGetErrorString(result));

        return -1;
      }

      if (smem_capacity < 0) {
        int device_idx = 0;
        result = mcGetDevice(&device_idx);

        if (result != mcSuccess) {
          return -1;
        }

        mcDeviceProp_t properties;
        result = mcGetDeviceProperties(&properties, device_idx);

        if (result != mcSuccess) {
          return -1;
        }

        smem_capacity = static_cast<int>(properties.sharedMemPerMultiprocessor);
      }

      int occupancy = std::min(max_active_blocks, smem_capacity / smem_size);

      MCTLASS_TRACE_HOST("  occupancy: " << occupancy);

      return occupancy;
    }

    MCTLASS_TRACE_HOST("  returning internal error");

    return -1;
  }

  /// Initializes GEMM state from arguments.
  Status initialize(Arguments const &args, void *workspace = nullptr, mcStream_t stream = nullptr) {

    MCTLASS_TRACE_HOST("GemmUniversalBase::initialize() - workspace "
      << workspace << ", stream: " << (stream ? "non-null" : "null"));

    size_t workspace_bytes = get_workspace_size(args);

    MCTLASS_TRACE_HOST("  workspace_bytes: " << workspace_bytes);

    if (workspace_bytes) {

      if (!workspace) {
        MCTLASS_TRACE_HOST("  error: device workspace must not be null");

        return Status::kErrorWorkspaceNull;
      }

      if (args.mode == GemmUniversalMode::kGemm) {
        MCTLASS_TRACE_HOST("  clearing device workspace");
        mcError_t result = mcMemsetAsync(workspace, 0, workspace_bytes, stream);

        if (result != mcSuccess) {
          MCTLASS_TRACE_HOST("  mcMemsetAsync() returned error " << mcGetErrorString(result));

          return Status::kErrorInternal;
        }
      }
    }

    // Get CUDA grid shape
    mctlass::gemm::GemmCoord grid_tiled_shape;
    int gemm_k_size = 0;

    get_grid_shape_(grid_tiled_shape, gemm_k_size, args);

    // Initialize the Params structure
    params_ = typename GemmKernel::Params(
      args,
      grid_tiled_shape,
      gemm_k_size,
      static_cast<int *>(workspace)
    );

    // Specify shared memory capacity for kernel.
    int smem_size = int(sizeof(typename GemmKernel::SharedStorage));

    if (smem_size >= (64 << 10)) {
      mcError_t result = mcFuncSetAttribute(Kernel<GemmKernel>,
                                    mcFuncAttributeMaxDynamicSharedMemorySize,
                                    smem_size);

      if (result != mcSuccess) {
        return Status::kErrorInternal;
      }
    }

    return Status::kSuccess;
  }

  /// Lightweight update given a subset of arguments
  Status update(Arguments const &args, void *workspace = nullptr) {

    MCTLASS_TRACE_HOST("GemmUniversalBase()::update() - workspace: " << workspace);

    size_t workspace_bytes = get_workspace_size(args);

    if (workspace_bytes && !workspace) {
      return Status::kErrorWorkspaceNull;
    }

    params_.update(args, workspace);

    return Status::kSuccess;
  }

  /// Runs the kernel using initialized state.
  Status run(mcStream_t stream = nullptr) {
    MCTLASS_TRACE_HOST("GemmUniversalBase::run()");

    //
    // Configure grid and block dimensions
    //

    ThreadblockSwizzle threadblock_swizzle;

    dim3 grid = threadblock_swizzle.get_grid_shape(params_.grid_tiled_shape);
    dim3 block(GemmKernel::kThreadCount, 1, 1);

    int smem_size = int(sizeof(typename GemmKernel::SharedStorage));

    //
    // Launch kernel
    //

    MCTLASS_TRACE_HOST("  grid: (" << grid << "),  block: (" << block
      << "),  SMEM: " << smem_size << " bytes");

    // Launch
    mctlass::Kernel<GemmKernel><<<grid, block, smem_size, stream>>>(params_);

    //
    // Query for errors
    //
    mcError_t result = mcGetLastError();

    if (result != mcSuccess) {
      MCTLASS_TRACE_HOST("  grid launch failed with error " << mcGetErrorString(result));
      return Status::kErrorInternal;
    }

    return Status::kSuccess;
  }

  /// Runs the kernel using initialized state.
  Status operator()(mcStream_t stream = nullptr) {
    return run(stream);
  }

  /// Runs the kernel using initialized state.
  Status operator()(
    Arguments const &args,
    void *workspace = nullptr,
    mcStream_t stream = nullptr) {

    Status status = initialize(args, workspace, stream);

    if (status == Status::kSuccess) {
      status = run(stream);
    }

    return status;
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace device
} // namespace gemm
} // namespace mctlass

/////////////////////////////////////////////////////////////////////////////////////////////////
