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
    \brief Implementation of a CTA-wide semaphore for inter-CTA synchronization.
*/

#pragma once

#include "mctlass/mctlass.h"

#include "mctlass/aligned_buffer.h"
#include "mctlass/array.h"

#include "mctlass/numeric_types.h"
#include "mctlass/matrix_shape.h"

#include "mctlass/gemm/gemm.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace mctlass {

/////////////////////////////////////////////////////////////////////////////////////////////////

/// CTA-wide semaphore for inter-CTA synchronization.
class Semaphore { 
public:

  int *lock;
  bool wait_thread;
  int state;

public:

  /// Implements a semaphore to wait for a flag to reach a given value
  MCTLASS_HOST_DEVICE
  Semaphore(int *lock_, int thread_id): 
    lock(lock_), 
    wait_thread(thread_id < 0 || thread_id == 0),
    state(-1) {

  }

  /// Permit fetching the synchronization mechanism early
  MCTLASS_DEVICE
  void fetch() {
    if (wait_thread) {
      // #if defined(__MACA_ARCH__)
      //   asm volatile ("ld.global.acquire.gpu.b32 %0, [%1];\n" : "=r"(state) : "l"(lock));  
      // #else
      //   asm volatile ("ld.global.cg.b32 %0, [%1];\n" : "=r"(state) : "l"(lock)); 
      // #endif
#if defined(__MACA_ARCH__)
      state = __ldcg(lock);
#endif
    }
  }

  /// Gets the internal state
  MCTLASS_DEVICE
  int get_state() const {
    return state;
  }

  /// Waits until the semaphore is equal to the given value
  MCTLASS_DEVICE
  void wait(int status = 0) {
#if defined(__MXCC__) || (defined(__clang__) && defined(__MACA__)) || defined(__MACACC_RTC__)
    while( __syncthreads_and(state != status) ) {
      fetch();
    }

    __syncthreads();
#endif
  }

  /// Updates the lock with the given result
  MCTLASS_DEVICE
  void release(int status = 0) {
#if defined(__MXCC__) || (defined(__clang__) && defined(__MACA__)) || defined(__MACACC_RTC__)
    __syncthreads();

    if (wait_thread) {
      #if 0
      #if defined(__MACA_ARCH__)
      asm volatile ("st.global.release.gpu.b32 [%0], %1;\n" : : "l"(lock), "r"(status));
      #else
      asm volatile ("st.global.cg.b32 [%0], %1;\n" : : "l"(lock), "r"(status));
      #endif
      #else
      __stcg(lock, status);
      #endif
    }
#endif
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace mctlass

/////////////////////////////////////////////////////////////////////////////////////////////////
