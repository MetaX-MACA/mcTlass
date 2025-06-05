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
    \brief Defines a proxy class for storing non-standard 16-bit floating point values with
          8 bits of exponent and 7 bit of mantissa.
*/
#pragma once

#if !defined(__MACACC_RTC__)
#include <cmath>
#include <limits>
#include <cstdint>
#include <cstring>
#endif

#include <maca_bfloat16.h>
#include "mctlass/mctlass.h"

namespace mctlass {

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Floating-point type with 8 bits of exponent and 7 bits of mantissa.
struct alignas(2) bfloat16_t {

  //
  // Data members
  //

  /// Storage type
  uint16_t storage;

  //
  // Methods
  //

  /// Constructs from an unsigned short
  MCTLASS_HOST_DEVICE
  static bfloat16_t bitcast(uint16_t x) {
    bfloat16_t h;
    h.storage = x;
    return h;
  }

  /// FP32 -> BF16 conversion - rounds to nearest even
  //#if defined(__MACA_ARCH__) && (__TLASS_ARCH__ < 530)
  MCTLASS_HOST_DEVICE
  static bfloat16_t convert(float const& flt) {
  #if defined(__MACA_ARCH__)
    return bfloat16_t(__float2bfloat16(flt));
  #else

    unsigned s;

    #if defined(__MACA_ARCH__)
    s = reinterpret_cast<unsigned const &>(flt);
    #else
    std::memcpy(&s, &flt, sizeof(s));
    #endif

    // float val
    uint16_t sign = uint16_t((s >> 16) & 0x8000);
    int16_t exp = uint16_t(((s >> 23) & 0xff) - 127);
    int mantissa = s & 0x7fffff;
    uint16_t u = 0;

    if ((s & 0x7fffffff) == 0) {
      // sign-preserving zero
      return bitcast(sign);
    }

    // nan and inf
    if (exp > 15) {
      if (exp == 128 && mantissa) {
        // not a number
        u = 0x7fff;
      } else {
        // overflow to infinity
        u = sign | 0x7f80;
      }
      return bitcast(u);
    }

    int sticky_bit = 0;

    if (exp >= -126) {
      // normal fp32 to normal fp16
      exp = uint16_t(exp + uint16_t(127));
      u = uint16_t(((exp & 0xff) << 7));
      u = uint16_t(u | (mantissa >> 16));
    } else {
      // normal single-precision to subnormal half_t-precision representation
      int rshift = (-126 - exp);
      if (rshift < 256) {
        mantissa |= (1 << 23);

        sticky_bit = ((mantissa & ((1 << rshift) - 1)) != 0);

        mantissa = (mantissa >> rshift);
        u = (uint16_t(mantissa >> 16) & 0x7f);
      } else {
        mantissa = 0;
        u = 0;
      }
    }

    // round to nearest even
    // int round_bit = ((mantissa >> 12) & 1);
    // sticky_bit |= ((mantissa & ((1 << 12) - 1)) != 0);
    int round_bit = ((mantissa >> 15) & 1);
    sticky_bit |= ((mantissa & ((1 << 15) - 1)) != 0);

    if ((round_bit && sticky_bit) || (round_bit && (u & 1))) {
      u = uint16_t(u + 1);
    }

    u |= sign;

    return bitcast(u);
  #endif
  }

  /// FP32 -> BF16 conversion - rounds to nearest even
  MCTLASS_HOST_DEVICE
  static bfloat16_t convert(int const& n) {
  #if defined(USE_BUILTIN) && defined(__MACA_ARCH__)
    return bfloat16_t(__1int2bfloat16_rn(n));
  #else
    return convert(float(n));
  #endif
  }

  /// FP32 -> BF16 conversion - rounds to nearest even
  MCTLASS_HOST_DEVICE
  static bfloat16_t convert(unsigned const& n) {
  #if defined(USE_BUILTIN) && defined(__MACA_ARCH__)
    return bfloat16_t(__uint2bfloat16_rn(n));
  #else
    return convert(float(n));
  #endif
  }

  /// Converts a __maca_bfloat16-precision value stored as a uint16_t to a float
  //#if defined(__MACA_ARCH__) && (__TLASS_ARCH__ < 530)
  MCTLASS_HOST_DEVICE
  static float convert(bfloat16_t const& x) {
  #if defined(__MACA_ARCH__)
    return __bfloat162float(x.to_bfloat());
  #else

    uint16_t const &h = x.storage;
    int sign = ((h >> 15) & 1);
    int exp = ((h >> 7) & 0xff);
    int mantissa = (h & 0x7f);
    unsigned f = 0;

    if (exp > 0 && exp < 31) {
      // normal
      // exp += 112;
      f = (sign << 31) | (exp << 23) | (mantissa << 16);
    } else if (exp == 0) {
      if (mantissa) {
        // subnormal
        // exp += 113;
        // while ((mantissa & (1 << 10)) == 0) {
        //   mantissa <<= 1;
        //   exp--;
        // }
        // mantissa &= 0x3ff;
        // f = (sign << 31) | (exp << 23) | (mantissa << 13);
        // subnormal
        while ((mantissa & (1 << 7)) == 0) {
          mantissa <<= 1;
        }
        mantissa &= 0x7f;
        f = (sign << 31) | (exp << 23) | (mantissa << 16);
      } else {
        // sign-preserving zero
        f = (sign << 31);
      }
    } else if (exp == 31) {
      if (mantissa) {
        f = 0x7fffffff;  // not a number
      } else {
        f = (0xff << 23) | (sign << 31);  //  inf
      }
    }
    #if defined(__MACA_ARCH__)
    return reinterpret_cast<float const&>(f);
    #else
    float flt;
    std::memcpy(&flt, &f, sizeof(flt));
    return flt;
    #endif
  #endif
  }

  //
  // Methods
  //

  /// Default constructor
  MCTLASS_HOST_DEVICE
  bfloat16_t() : storage(0) { }

  /// Reinterpret cast from __maca_bfloat16 type
  MCTLASS_HOST_DEVICE
  explicit bfloat16_t(__maca_bfloat16 const & x) {
    #if defined(__MACA_ARCH__)
    storage = reinterpret_cast<uint16_t const &>(x);
    #else
    __maca_bfloat16_raw raw(x);
    std::memcpy(&storage, &raw.x, sizeof(storage));
    #endif
  }

  /// Floating-point conversion - round toward nearest
  MCTLASS_HOST_DEVICE
  explicit bfloat16_t(float x) {

    //#if defined(__MACA_ARCH__)
    #if 0
    asm("cvt.rn.bf16.f32 %0, %1;\n" : "=h"(storage) : "f"(x));
    #else

    uint32_t bits;

    #if defined(__MACA_ARCH__)
    bits = reinterpret_cast<uint32_t &>(x);
    #else
    std::memcpy(&bits, &x, sizeof(bits));
    #endif

    if ((bits & 0x7f800000) != 0x7f800000) {

      bool mantissa_bit = ((bits & (1 << 16)) != 0);
      bool round_bit = ((bits & (1 << 15)) != 0);
      bool sticky_bit = ((bits & ((1 << 15) - 1)) != 0);

      if ((round_bit && sticky_bit) || (round_bit && mantissa_bit)) {
        bits += uint32_t(1 << 16);
      }
    }
    else if (bits & ~0xff800000) {
      bits = 0x7fffffff;
    }

    storage = uint16_t((bits >> 16) & 0xffff);
    #endif
  }

  /// Floating-point conversion - round toward nearest
  MCTLASS_HOST_DEVICE
  explicit bfloat16_t(double x): bfloat16_t(float(x)) {

  }

  /// Integer conversion - round toward nearest
  MCTLASS_HOST_DEVICE
  explicit bfloat16_t(int x) {
    float flt = static_cast<float>(x);
    uint32_t bits;

    #if defined(__MACA_ARCH__)
    bits = reinterpret_cast<uint32_t &>(flt);
    #else
    std::memcpy(&bits, &flt, sizeof(bits));
    #endif

    storage = uint16_t(bits >> 16);
  }

  /// Converts to float
  MCTLASS_HOST_DEVICE
  operator float() const {
    unsigned bits = (unsigned(storage) << 16);
    #if defined(__MACA_ARCH__)
    return reinterpret_cast<float const &>(bits);
    #else
    float flt;
    std::memcpy(&flt, &bits, sizeof(flt));
    return flt;
    #endif
  }

  /// Converts to float
  MCTLASS_HOST_DEVICE
  explicit operator double() const {
    return double(float(*this));
  }

  /// Converts to int
  MCTLASS_HOST_DEVICE
  explicit operator int() const {
    return int(float(*this));
  }

  /// Casts to bool
  MCTLASS_HOST_DEVICE
  explicit operator bool() const {
    return (float(*this) != 0.0f);
  }

  /// Bitcasts to __maca_bfloat16 type
  MCTLASS_HOST_DEVICE
  __maca_bfloat16 to_bfloat() const {
    #if defined(__MACA_ARCH__)
    return reinterpret_cast<__maca_bfloat16 const &>(storage);
    #else
    __maca_bfloat16_raw raw;
    std::memcpy(&raw.x, &storage, sizeof(raw.x));
    return __maca_bfloat16(raw);
    #endif
  }

  /// Bitcasts to MACA's half type
  #if defined(__MACA_ARCH__)
  MCTLASS_HOST_DEVICE
  const __fp16& to_macabfloat() const {
    return reinterpret_cast<__fp16 const &>(storage);
  }
  #endif

  /// Obtains raw bits
  MCTLASS_HOST_DEVICE
  uint16_t raw() const {
    return storage;
  }
    /// Returns the sign bit
  MCTLASS_HOST_DEVICE
  bool signbit() const {
    return ((raw() & 0x8000) != 0);
  }

  /// Returns the biased exponent
  MCTLASS_HOST_DEVICE
  int exponent_biased() const {
    return int((raw() >> 7) & 0x0ff);
  }

  /// Returns the unbiased exponent
  MCTLASS_HOST_DEVICE
  int exponent() const {
    return exponent_biased() - 127;
  }

  /// Returns the mantissa
  MCTLASS_HOST_DEVICE
  int mantissa() const {
    return int(raw() & 0x7f);
  }
};

///////////////////////////////////////////////////////////////////////////////////////////////////

MCTLASS_HOST_DEVICE
bool signbit(mctlass::bfloat16_t const& h) {
  return h.signbit();
}

MCTLASS_HOST_DEVICE
mctlass::bfloat16_t abs(mctlass::bfloat16_t const& h) {
  return mctlass::bfloat16_t::bitcast(h.raw() & 0x7fffffff);
}

MCTLASS_HOST_DEVICE
bool isnan(mctlass::bfloat16_t const& h) {
  return (h.exponent_biased() == 0x0ff) && h.mantissa();
}

MCTLASS_HOST_DEVICE
bool isfinite(mctlass::bfloat16_t const& h) {
  return (h.exponent_biased() != 0x0ff);
}

MCTLASS_HOST_DEVICE
mctlass::bfloat16_t nan_bf16(const char*) {
  return mctlass::bfloat16_t::bitcast(0x7fff);
}

MCTLASS_HOST_DEVICE
bool isinf(mctlass::bfloat16_t const& h) {
  return (h.exponent_biased() == 0x0ff) && !h.mantissa();
}

MCTLASS_HOST_DEVICE
bool isnormal(mctlass::bfloat16_t const& h) {
  return h.exponent_biased() && h.exponent_biased() != 0x0ff;
}

MCTLASS_HOST_DEVICE
int fpclassify(mctlass::bfloat16_t const& h) {
  int exp = h.exponent_biased();
  int mantissa = h.mantissa();
  if (exp == 0x0ff) {
    if (mantissa) {
      return FP_NAN;
    }
    else {
      return FP_INFINITE;
    }
  }
  else if (!exp) {
    if (mantissa) {
      return FP_SUBNORMAL;
    }
    else {
      return FP_ZERO;
    }
  }
  return FP_NORMAL;
}

MCTLASS_HOST_DEVICE
mctlass::bfloat16_t sqrt(mctlass::bfloat16_t const& h) {
#if defined(__MACACC_RTC__)
  return mctlass::bfloat16_t(sqrtf(float(h)));
#else
  return mctlass::bfloat16_t(std::sqrt(float(h)));
#endif
}

MCTLASS_HOST_DEVICE
bfloat16_t copysign(bfloat16_t const& a, bfloat16_t const& b) {

  uint16_t a_bits;
  uint16_t b_bits;

  #if defined(__MACA_ARCH__)
  a_bits = reinterpret_cast<uint16_t const &>(a);
  b_bits = reinterpret_cast<uint16_t const &>(b);
  #else
  std::memcpy(&a_bits, &a, sizeof(a_bits));
  std::memcpy(&b_bits, &b, sizeof(b_bits));
  #endif

  uint16_t a_mag = (a_bits & 0x7fff);
  uint16_t b_sign = (b_bits & 0x8000);
  uint16_t result = (a_mag | b_sign);

  return bfloat16_t::bitcast(result);
}

///////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace mctlass

///////////////////////////////////////////////////////////////////////////////////////////////////
//
// Standard Library operations and definitions
//
///////////////////////////////////////////////////////////////////////////////////////////////////

namespace std {

#if !defined(__MACACC_RTC__)
/// Numeric limits
template <>
struct numeric_limits<mctlass::bfloat16_t> {
  static bool const is_specialized = true;
  static bool const is_signed = true;
  static bool const is_integer = false;
  static bool const is_exact = false;
  static bool const has_infinity = true;
  static bool const has_quiet_NaN = true;
  static bool const has_signaling_NaN = false;
  static std::float_denorm_style const has_denorm = std::denorm_present;
  static bool const has_denorm_loss = true;
  static std::float_round_style const round_style = std::round_to_nearest;
  static bool const is_iec559 = false;
  static bool const is_bounded = true;
  static bool const is_modulo = false;
  static int const digits = 7;

  /// Least positive value
  MCTLASS_HOST_DEVICE
  static mctlass::bfloat16_t min() { return mctlass::bfloat16_t::bitcast(0x01); }

  /// Minimum finite value
  MCTLASS_HOST_DEVICE
  static mctlass::bfloat16_t lowest() { return mctlass::bfloat16_t::bitcast(0xff7f); }

  /// Maximum finite value
  MCTLASS_HOST_DEVICE
  static mctlass::bfloat16_t max() { return mctlass::bfloat16_t::bitcast(0x7f7f); }

  /// Returns smallest finite value
  MCTLASS_HOST_DEVICE
  static mctlass::bfloat16_t epsilon() { return mctlass::bfloat16_t::bitcast(0x1000); }

  /// Returns smallest finite value
  MCTLASS_HOST_DEVICE
  static mctlass::bfloat16_t round_error() { return mctlass::bfloat16_t(0.5f); }

  /// Returns smallest finite value
  MCTLASS_HOST_DEVICE
  static mctlass::bfloat16_t infinity() { return mctlass::bfloat16_t::bitcast(0x7f80); }

  /// Returns smallest finite value
  MCTLASS_HOST_DEVICE
  static mctlass::bfloat16_t quiet_NaN() { return mctlass::bfloat16_t::bitcast(0x7fff); }

  /// Returns smallest finite value
  MCTLASS_HOST_DEVICE
  static mctlass::bfloat16_t signaling_NaN() { return mctlass::bfloat16_t::bitcast(0x7fff); }

  /// Returns smallest finite value
  MCTLASS_HOST_DEVICE
  static mctlass::bfloat16_t denorm_min() { return mctlass::bfloat16_t::bitcast(0x1); }
};
#endif

} // namespace std

///////////////////////////////////////////////////////////////////////////////////////////////////
//
// Arithmetic operators
//
///////////////////////////////////////////////////////////////////////////////////////////////////

namespace mctlass {

///////////////////////////////////////////////////////////////////////////////////////////////////

MCTLASS_HOST_DEVICE
bool operator==(bfloat16_t const& lhs, bfloat16_t const& rhs) {
  return float(lhs) == float(rhs);
}

MCTLASS_HOST_DEVICE
bool operator!=(bfloat16_t const& lhs, bfloat16_t const& rhs) {
  return float(lhs) != float(rhs);
}

MCTLASS_HOST_DEVICE
bool operator<(bfloat16_t const& lhs, bfloat16_t const& rhs) {
  return float(lhs) < float(rhs);
}

MCTLASS_HOST_DEVICE
bool operator<=(bfloat16_t const& lhs, bfloat16_t const& rhs) {
  return float(lhs) <= float(rhs);
}

MCTLASS_HOST_DEVICE
bool operator>(bfloat16_t const& lhs, bfloat16_t const& rhs) {
  return float(lhs) > float(rhs);
}

MCTLASS_HOST_DEVICE
bool operator>=(bfloat16_t const& lhs, bfloat16_t const& rhs) {
  return float(lhs) >= float(rhs);
}

MCTLASS_HOST_DEVICE
bfloat16_t operator+(bfloat16_t const& lhs, bfloat16_t const& rhs) {
  return bfloat16_t(float(lhs) + float(rhs));
}

MCTLASS_HOST_DEVICE
bfloat16_t operator-(bfloat16_t const& lhs) {
  return bfloat16_t(-float(lhs));
}

MCTLASS_HOST_DEVICE
bfloat16_t operator-(bfloat16_t const& lhs, bfloat16_t const& rhs) {
  return bfloat16_t(float(lhs) - float(rhs));
}

MCTLASS_HOST_DEVICE
bfloat16_t operator*(bfloat16_t const& lhs, bfloat16_t const& rhs) {
  return bfloat16_t(float(lhs) * float(rhs));
}

MCTLASS_HOST_DEVICE
bfloat16_t operator/(bfloat16_t const& lhs, bfloat16_t const& rhs) {
  return bfloat16_t(float(lhs) / float(rhs));
}

MCTLASS_HOST_DEVICE
bfloat16_t& operator+=(bfloat16_t & lhs, bfloat16_t const& rhs) {
  lhs = bfloat16_t(float(lhs) + float(rhs));
  return lhs;
}

MCTLASS_HOST_DEVICE
bfloat16_t& operator-=(bfloat16_t & lhs, bfloat16_t const& rhs) {
  lhs = bfloat16_t(float(lhs) - float(rhs));
  return lhs;
}

MCTLASS_HOST_DEVICE
bfloat16_t& operator*=(bfloat16_t & lhs, bfloat16_t const& rhs) {
  lhs = bfloat16_t(float(lhs) * float(rhs));
  return lhs;
}

MCTLASS_HOST_DEVICE
bfloat16_t& operator/=(bfloat16_t & lhs, bfloat16_t const& rhs) {
  lhs = bfloat16_t(float(lhs) / float(rhs));
  return lhs;
}

MCTLASS_HOST_DEVICE
bfloat16_t& operator++(bfloat16_t & lhs) {
  float tmp(lhs);
  ++tmp;
  lhs = bfloat16_t(tmp);
  return lhs;
}

MCTLASS_HOST_DEVICE
bfloat16_t& operator--(bfloat16_t & lhs) {
  float tmp(lhs);
  --tmp;
  lhs = bfloat16_t(tmp);
  return lhs;
}

MCTLASS_HOST_DEVICE
bfloat16_t operator++(bfloat16_t & lhs, int) {
  bfloat16_t ret(lhs);
  float tmp(lhs);
  tmp++;
  lhs = bfloat16_t(tmp);
  return ret;
}

MCTLASS_HOST_DEVICE
bfloat16_t operator--(bfloat16_t & lhs, int) {
  bfloat16_t ret(lhs);
  float tmp(lhs);
  tmp--;
  lhs = bfloat16_t(tmp);
  return ret;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace mctlass

///////////////////////////////////////////////////////////////////////////////////////////////////

//
// User-defined literals
//

MCTLASS_HOST_DEVICE
mctlass::bfloat16_t operator "" _bf16(long double x) {
  return mctlass::bfloat16_t(float(x));
}

MCTLASS_HOST_DEVICE
mctlass::bfloat16_t operator "" _bf16(unsigned long long int x) {
  return mctlass::bfloat16_t(int(x));
}

/////////////////////////////////////////////////////////////////////////////////////////////////
