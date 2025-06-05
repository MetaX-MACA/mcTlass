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
  \brief Functor performing linear combination operations used by epilogues.
*/

#pragma once

#include <maca_fp16.h>

#include "mctlass/mctlass.h"
#include "mctlass/numeric_types.h"
#include "mctlass/array.h"
#include "mctlass/functional.h"
#include "mctlass/numeric_conversion.h"
#include "mctlass/epilogue/thread/activation.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace mctlass {
namespace epilogue {
namespace thread {


/////////////////////////////////////////////////////////////////////////////////////////////////

namespace detail {

template <typename Element, int ElementsPerAccess>
struct ArrayMaximum {

  MCTLASS_HOST_DEVICE
  Array<Element, ElementsPerAccess> operator()(
    Array<Element, ElementsPerAccess>  const &lhs,
    Array<Element, ElementsPerAccess>  const &rhs) const {

    Array<Element, ElementsPerAccess> result;

    MCTLASS_PRAGMA_UNROLL
    for (int i = 0; i < ElementsPerAccess; ++i) {
      result[i] = fmax(lhs[i], rhs[i]);
    }

    return result;
  }
};

template <int ElementsPerAccess>
struct ArrayMaximum<half_t, ElementsPerAccess> {

  MCTLASS_DEVICE
  Array<half_t, ElementsPerAccess> operator()(
    Array<half_t, ElementsPerAccess>  const &lhs,
    Array<half_t, ElementsPerAccess>  const &rhs) const {

    Array<half_t, ElementsPerAccess> result;

    #if defined(USE_BUILTIN) && defined(__MACA_ARCH__)
    int const kVectorCount = ElementsPerAccess / 2;


    __half2 const *lhs_ptr = reinterpret_cast<__half2 const *>(lhs.raw_data());
    __half2 const *rhs_ptr = reinterpret_cast<__half2 const *>(rhs.raw_data());
    __half2       *res_ptr = reinterpret_cast<__half2 *>(result.raw_data());

    MCTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kVectorCount; ++i) {
      res_ptr[i] = __hmax2(lhs_ptr[i], rhs_ptr[i]);
    }

    #else
    __half const *lhs_ptr = reinterpret_cast<__half const *>(lhs.raw_data());
    __half const *rhs_ptr = reinterpret_cast<__half const *>(rhs.raw_data());
    __half       *res_ptr = reinterpret_cast<__half       *>(result.raw_data());

    MCTLASS_PRAGMA_UNROLL
    for (int i = 0; i < ElementsPerAccess; ++i) {
      res_ptr[i] = ((lhs_ptr[i] < rhs_ptr[i]) ? rhs_ptr[i] : lhs_ptr[i]);
    }

    #endif

    return result;
  }

  MCTLASS_DEVICE
  Array<half_t, ElementsPerAccess> operator()(
    Array<half_t, ElementsPerAccess>  const &lhs,
    half_t const &rhs) const {

    Array<half_t, ElementsPerAccess> result;

    #if defined(USE_BUILTIN) && defined(__MACA_ARCH__)
    int const kVectorCount = ElementsPerAccess / 2;


    __half rhs_raw = reinterpret_cast<__half const &>(rhs);
    __half2 rhs_pair = __half2half2(rhs_raw);

    __half2 const *lhs_ptr = reinterpret_cast<__half2 const *>(lhs.raw_data());
    __half2       *res_ptr = reinterpret_cast<__half2 *>(result.raw_data());

    MCTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kVectorCount; ++i) {
      res_ptr[i] = __hmax2(lhs_ptr[i], rhs_pair);
    }

    #else

    __half const *lhs_ptr = reinterpret_cast<__half const *>(lhs.raw_data());
    __half const  rhs_raw = reinterpret_cast<__half const &>(rhs);
    __half       *res_ptr = reinterpret_cast<__half       *>(result.raw_data());

    MCTLASS_PRAGMA_UNROLL
    for (int i = 0; i < ElementsPerAccess; ++i) {
      res_ptr[i] = ((lhs_ptr[i] < rhs_raw) ? rhs_raw : lhs_ptr[i]);
    }

    #endif

    return result;
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Element, int ElementsPerAccess>
struct ReluConditional {

  MCTLASS_HOST_DEVICE
  void operator()(
    bool conditional[],
    Array<Element, ElementsPerAccess> const &fragment, 
    Element threshold) const {

    MCTLASS_PRAGMA_UNROLL
    for (int i = 0; i < ElementsPerAccess; ++i) {
      conditional[i] = !(fragment[i] < threshold);
    }
  }
};

template <int ElementsPerAccess>
struct ReluConditional<half_t, ElementsPerAccess> {

  MCTLASS_DEVICE
  void operator()(
    bool conditional[],
    Array<half_t, ElementsPerAccess> const &fragment, 
    half_t threshold) const {

    __half y = reinterpret_cast<__half const &>(threshold);
    __half const *x = reinterpret_cast<__half const *>(fragment.raw_data());
    
    #if defined(USE_BUILTIN) && defined(__MACA_ARCH__)
    MCTLASS_PRAGMA_UNROLL
    for (int i = 0; i < ElementsPerAccess; ++i) {
      conditional[i] = !__hlt(x[i], y);
    }
    #else
    MCTLASS_PRAGMA_UNROLL
    for (int i = 0; i < ElementsPerAccess; ++i) {
      conditional[i] = float(x[i]) < float(y) ? 0 : 1;
    }
    #endif
  }
};

} // namespace detail

/////////////////////////////////////////////////////////////////////////////////////////////////

/// This is a partial specialization for fused Bias and ReLU. It supports the option of packing
/// ReLU conditionals in a bit vector that may be used by backwards passes as an optimization.
///
/// This class can only be used with mctlass::epilogue::threadblock::EpilogueWithBroadcast<>.
///
/// This base class is meant to define the concept required of the
/// EpilogueWithBroadcast::OutputOp
template <
  typename ElementC_,
  typename ElementAccumulator_,
  typename ElementCompute_,
  typename ElementZ_,
  int ElementsPerAccess,
  bool StoreT = true
>
class LinearCombinationBiasRelu {
public:

  using ElementOutput = ElementC_;
  using ElementC = ElementC_;
  using ElementAccumulator = ElementAccumulator_;
  using ElementCompute = ElementCompute_;
  using ElementZ = ElementZ_;

  using ElementT = uint1b_t;

  static int const kElementsPerAccess = ElementsPerAccess;
  static int const kCount = kElementsPerAccess;

  using ElementwiseOp = ReLu<ElementCompute>;
  using BinaryOp = plus<ElementCompute>;

  using FragmentAccumulator = Array<ElementAccumulator, kElementsPerAccess>;
  using FragmentCompute = Array<ElementCompute, kElementsPerAccess>;
  using FragmentC = Array<ElementOutput, kElementsPerAccess>;
  using FragmentZ = Array<ElementZ, kElementsPerAccess>;
  using FragmentT = Array<ElementT, kElementsPerAccess>;

  /// If true, the 'Z' tensor is stored
  static bool const kStoreZ = true;

  /// If true, the 'T' tensor is stored
  static bool const kStoreT = StoreT;

  /// Host-constructable parameters structure
  struct Params {

    ElementCompute alpha;                  ///< scales accumulators
    ElementCompute beta;                   ///< scales source tensor
    ElementCompute const *alpha_ptr;       ///< pointer to accumulator scalar - if not null, loads it from memory
    ElementCompute const *beta_ptr;        ///< pointer to source scalar - if not null, loads it from memory
    ElementZ threshold;                    ///< ReLu threshold

    //
    // Methods
    //
    //
    // Methods
    //

    MCTLASS_HOST_DEVICE
    Params(): 
      alpha(ElementCompute(1)), 
      beta(ElementCompute()), 
      alpha_ptr(nullptr), 
      beta_ptr(nullptr),
      threshold(ElementCompute()) { }

    MCTLASS_HOST_DEVICE
    Params(
      ElementCompute alpha,
      ElementCompute beta,
      ElementCompute threshold_ = ElementCompute()
    ): 
      alpha(alpha), beta(beta), alpha_ptr(nullptr), beta_ptr(nullptr) {

      NumericConverter<ElementZ, ElementCompute> convert_threshold;

      threshold = convert_threshold(threshold_);
    }

    MCTLASS_HOST_DEVICE
    Params(
      ElementCompute alpha
    ): alpha(alpha), beta(0), alpha_ptr(nullptr), beta_ptr(nullptr), threshold(ElementZ()) {

    }

    MCTLASS_HOST_DEVICE
    Params(
      ElementCompute const *alpha_ptr,
      ElementCompute const *beta_ptr,
      ElementCompute threshold_ = ElementCompute()
    ): alpha(0), beta(0), alpha_ptr(alpha_ptr), beta_ptr(beta_ptr) {

      NumericConverter<ElementZ, ElementCompute> convert_threshold;

      threshold = convert_threshold(threshold_);
    }

    MCTLASS_HOST_DEVICE
    Params(
      ElementCompute const *alpha_ptr
    ): alpha(0), beta(0), alpha_ptr(alpha_ptr), beta_ptr(nullptr), threshold(ElementZ()) {
    }

  };

private:

  //
  // Data members
  //

  ElementCompute alpha_;
  ElementCompute beta_;
  ElementZ threshold_;

public:

  //
  // Methods
  //

  /// Constructor from Params
  MCTLASS_HOST_DEVICE
  LinearCombinationBiasRelu(Params const &params) {

    alpha_ = (params.alpha_ptr ? *params.alpha_ptr : params.alpha);
    beta_ = (params.beta_ptr ? *params.beta_ptr : params.beta);
    threshold_ = params.threshold;
  }

  /// Returns true if source is needed
  MCTLASS_HOST_DEVICE
  bool is_source_needed() const {
    return beta_ != ElementCompute(0);
  }

  /// Functionally required for serial reduction in the epilogue
  MCTLASS_HOST_DEVICE
  void set_k_partition(int k_partition, int k_partition_count) {
    if (k_partition) {
      beta_ = ElementCompute(1);
    }

    if (k_partition != k_partition_count - 1) {
      // set to NaN to make ReLU no-op for all except last k partitions
      int64_t allones = -1;
      threshold_ = reinterpret_cast<ElementZ const &>(allones);
    }
  }

  /// Applies the operation when is_source_needed() is true
  MCTLASS_HOST_DEVICE
  void operator()(
    FragmentZ &frag_Z, 
    FragmentT &frag_T, 
    FragmentAccumulator const &AB,
    FragmentC const &frag_C,
    FragmentCompute const &V) const {

    BinaryOp binary_op;

    FragmentCompute tmp_Accum = NumericArrayConverter<ElementCompute, ElementAccumulator, kElementsPerAccess>()(AB);
    FragmentCompute tmp_C = NumericArrayConverter<ElementCompute, ElementC, kElementsPerAccess>()(frag_C);
    FragmentCompute result_Z;

    bool conditions[kElementsPerAccess];

    MCTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kElementsPerAccess; ++i) {

      ElementCompute z = alpha_ * tmp_Accum[i];
      z += beta_ * tmp_C[i];

      z = binary_op(z, V[i]);
      result_Z[i] = z;
      
      //For mxmaca f16, we cannot got same precision with cuda, so we got zero which is near with it.
      if (z < ElementCompute(1e-5f) && z > ElementCompute(-1e-5f)) {
        result_Z[i] = ElementCompute(0);
      }
      
    }

    NumericArrayConverter<ElementZ, ElementCompute, kElementsPerAccess> convert_z;
    frag_Z = convert_z(result_Z);

    //
    // Compute condition
    //

    detail::ReluConditional<ElementZ, kElementsPerAccess> relu_conditional;
    relu_conditional(conditions, frag_Z, threshold_);

    detail::ArrayMaximum<ElementZ, kElementsPerAccess> maximum_op;
    frag_Z = maximum_op(frag_Z, threshold_);

    if (kStoreT) {
      PackPredicates<kElementsPerAccess> pack_predicates;
      frag_T = pack_predicates(conditions); 
    }
  }

  /// Applies the operation when is_source_needed() is false
  MCTLASS_HOST_DEVICE
  void operator()(
    FragmentZ &frag_Z, 
    FragmentT &frag_T, 
    FragmentAccumulator const &AB,
    FragmentCompute const &V) const {

    BinaryOp binary_op;

    FragmentCompute tmp_Accum = NumericArrayConverter<ElementCompute, ElementAccumulator, kElementsPerAccess>()(AB);
    FragmentCompute result_Z;

    bool conditions[kElementsPerAccess];

    MCTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kElementsPerAccess; ++i) {
      ElementCompute z = binary_op(alpha_ * tmp_Accum[i], V[i]);
      result_Z[i] = z;

      //For mxmaca f16, we cannot got same precision with cuda, so we got zero which is near with it.
      if (z < ElementCompute(1e-5f) && z > ElementCompute(-1e-5f)) {
        result_Z[i] = ElementCompute(0);
      }

    }

    NumericArrayConverter<ElementZ, ElementCompute, kElementsPerAccess> convert_z;
    frag_Z = convert_z(result_Z);

    //
    // Compute condition
    //

    detail::ReluConditional<ElementZ, kElementsPerAccess> relu_conditional;
    relu_conditional(conditions, frag_Z, threshold_);

    detail::ArrayMaximum<ElementZ, kElementsPerAccess> maximum_op;
    frag_Z = maximum_op(frag_Z, threshold_);

    // 
    // Compute conditions
    //

    //
    // Store
    //
    if (kStoreT) {
      PackPredicates<kElementsPerAccess> pack_predicates;
      frag_T = pack_predicates(conditions);
    }
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace thread
} // namespace epilogue
} // namespace mctlass

/////////////////////////////////////////////////////////////////////////////////////////////////
