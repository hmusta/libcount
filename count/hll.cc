// Copyright 2015 The libcount Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License. See the AUTHORS file for names of
// contributors.

#include "count/hll.h"

#include <assert.h>
#include <errno.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#if __AVX2__
#include <immintrin.h>
#endif

#include <algorithm>

#include "count/empirical_data.h"
#include "count/utility.h"

namespace {

using libcount::CountLeadingZeroes;
using std::max;

// Helper that calculates cardinality according to LinearCounting
double LinearCounting(double register_count, double zeroed_registers) {
  return register_count * log(register_count / zeroed_registers);
}

// Helper to calculate the index into the table of registers from the hash
inline int RegisterIndexOf(uint64_t hash, int precision) {
  return (hash >> (64 - precision));
}

// Helper to count the leading zeros (less the bits used for the reg. index)
inline uint8_t ZeroCountOf(uint64_t hash, int precision) {
  // Make a mask for isolating the leading bits used for the register index.
  const uint64_t ONE = 1;
  const uint64_t mask = ~(((ONE << precision) - ONE) << (64 - precision));

  // Count zeroes, less the index bits we're masking off.
  return (CountLeadingZeroes(hash & mask) - static_cast<uint8_t>(precision));
}

#if __AVX2__

inline double haddall_pd(__m256d v) {
  // [ a, b, c, d ] -> [ a+b, 0, c+d, 0]
  __m256d pairs = _mm256_hadd_pd(v, _mm256_setzero_pd());

  // [ a+b, 0, c+d, 0] + [ c+d, 0, 0, 0] -> a+b+c+d
  return _mm256_cvtsd_f64(
      _mm256_add_pd(pairs, _mm256_permute4x64_pd(pairs, 0x56)));
}

inline uint64_t haddall_epi32(__m128i v) {
  // [ a, b, c, d ] -> [ a+b, c+d, c+d, a+b ]
  __m128i pairs = _mm_hadd_epi32(v, _mm_shuffle_epi32(v, 0xE));

  return _mm_extract_epi32(_mm_add_epi32(pairs, pairs), 0);
}

#endif

}  // namespace

namespace libcount {

HLL::HLL(int precision)
    : precision_(precision), register_count_(0), registers_(NULL) {
  // The precision is vetted by the Create() function.  Assertions nonetheless.
  assert(precision >= HLL_MIN_PRECISION);
  assert(precision <= HLL_MAX_PRECISION);

  // We employ (2 ^ precision) "registers" to store max leading zeroes.
  register_count_ = (1 << precision);

  // Allocate space for the registers. We can safely economize by using bytes
  // for the counters because we know the value can't ever exceed ~60.
  registers_ = new uint8_t[register_count_];
  memset(registers_, 0, register_count_ * sizeof(registers_[0]));
}

HLL::~HLL() { delete[] registers_; }

HLL* HLL::Create(int precision, int* error) {
  if ((precision < HLL_MIN_PRECISION) || (precision > HLL_MAX_PRECISION)) {
    MaybeAssign(error, EINVAL);
    return NULL;
  }
  return new HLL(precision);
}

void HLL::Update(uint64_t hash) {
  // TODO: is this worth rewriting with SIMD instructions?
  // Which register will potentially receive the zero count of this hash?
  const int index = RegisterIndexOf(hash, precision_);
  assert(index < register_count_);

  // Count the zeroes for the hash, and add one, per the algorithm spec.
  const uint8_t count = ZeroCountOf(hash, precision_) + 1;
  assert(count <= 64);

  // Update the appropriate register if the new count is greater than current.
  registers_[index] = max(registers_[index], count);
}

void HLL::Update(const uint64_t* hashes, uint64_t len) {
  // TODO: is this worth rewriting with SIMD instructions?
  for (uint64_t i = 0; i < len; ++i) {
    Update(hashes[i]);
  }
}

int HLL::Merge(const HLL* other) {
  assert(other != NULL);
  if (other == NULL) {
    return EINVAL;
  }

  // Ensure that the precision values of the two objects match.
  if (precision_ != other->precision_) {
    return EINVAL;
  }

  // Choose the maximum of corresponding registers from self, other and
  // store it back in self, effectively merging the state of the counters.
  // TODO: is this worth rewriting with SIMD instructions?
  for (int i = 0; i < register_count_; ++i) {
    registers_[i] = max(registers_[i], other->registers_[i]);
  }

  return 0;
}

std::pair<double, int> HLL::RawEstimate() const {
  // Let 'm' be the number of registers.
  const double m = static_cast<double>(register_count_);

  // For each register, let 'max' be the contents of the register.
  // Let 'term' be the reciprocal of 2 ^ max.
  // Finally, let 'sum' be the sum of all terms.
  double sum = 0.0;
  int zeroed_registers = 0;
  int i = 0;

#if __AVX2__

  static_assert(HLL_MAX_PRECISION < 32);

  __m128i zero_counts = _mm_setzero_si128();
  __m128i ones = _mm_set1_epi32(1);
  __m256d ones_pd = _mm256_set1_pd(1.0);
  __m256d sum_packed = _mm256_setzero_pd();
  for (; i + 4 <= register_count_; i += 4) {
    // Load 4 8-bit registers and sign-extend them to 32-bit
    __m128i pack = _mm256_castsi256_si128(_mm256_cvtepi8_epi32(
        _mm_set1_epi32(*reinterpret_cast<const uint32_t*>(&registers_[i]))));

    // Add zero counts
    zero_counts =
        _mm_add_epi32(zero_counts, _mm_cmpeq_epi32(pack, _mm_setzero_si128()));

    // Convert to packed doubles
    __m256d exps_pd = _mm256_cvtepi32_pd(_mm_sllv_epi32(ones, pack));

    sum_packed = _mm256_add_pd(sum_packed, _mm256_div_pd(ones_pd, exps_pd));
  }

  sum += haddall_pd(sum_packed);
  zeroed_registers += haddall_epi32(zero_counts);

#endif

  for (; i < register_count_; ++i) {
    if (registers_[i] == 0) {
      ++zeroed_registers;
    }
    sum += pow(2.0, -static_cast<double>(registers_[i]));
  }

  // Next, calculate the harmonic mean
  const double harmonic_mean = m * (1.0 / sum);
  assert(harmonic_mean >= 0.0);

  // The harmonic mean is scaled by a constant that depends on the precision.
  const double estimate = EmpiricalAlpha(precision_) * m * harmonic_mean;
  assert(estimate >= 0.0);

  return std::make_pair(estimate, zeroed_registers);
}

std::pair<double, int> HLL::RawEstimateUnion(const HLL* other) const {
  assert(register_count_ == other->register_count_);
  assert(precision_ == other->precision_);

  // Let 'm' be the number of registers.
  const double m = static_cast<double>(register_count_);

  // For each register, let 'max' be the contents of the register.
  // Let 'term' be the reciprocal of 2 ^ max.
  // Finally, let 'sum' be the sum of all terms.
  double sum = 0.0;
  int zeroed_registers = 0;
  int i = 0;

#if __AVX2__

  static_assert(HLL_MAX_PRECISION < 32);

  __m128i zero_counts = _mm_setzero_si128();
  __m128i ones = _mm_set1_epi32(1);
  __m256d ones_pd = _mm256_set1_pd(1.0);
  __m256d sum_packed = _mm256_setzero_pd();
  for (; i + 4 <= register_count_; i += 4) {
    // Load 4 8-bit registers and sign-extend them to 32-bit
    __m128i pack = _mm256_castsi256_si128(_mm256_cvtepi8_epi32(_mm_max_epu8(
        _mm_set1_epi32(*reinterpret_cast<const uint32_t*>(&registers_[i])),
        _mm_set1_epi32(
            *reinterpret_cast<const uint32_t*>(&other->registers_[i])))));

    // Add zero counts
    zero_counts =
        _mm_add_epi32(zero_counts, _mm_cmpeq_epi32(pack, _mm_setzero_si128()));

    // Convert to packed doubles
    __m256d exps_pd = _mm256_cvtepi32_pd(_mm_sllv_epi32(ones, pack));

    sum_packed = _mm256_add_pd(sum_packed, _mm256_div_pd(ones_pd, exps_pd));
  }

  sum += haddall_pd(sum_packed);
  zeroed_registers += haddall_epi32(zero_counts);

#endif

  for (; i < register_count_; ++i) {
    if (registers_[i] == 0 && other->registers_[i] == 0) {
      ++zeroed_registers;
    }
    sum += pow(2.0, -static_cast<double>(std::max(registers_[i], other->registers_[i])));
  }

  // Next, calculate the harmonic mean
  const double harmonic_mean = m * (1.0 / sum);
  assert(harmonic_mean >= 0.0);

  // The harmonic mean is scaled by a constant that depends on the precision.
  const double estimate = EmpiricalAlpha(precision_) * m * harmonic_mean;
  assert(estimate >= 0.0);

  return std::make_pair(estimate, zeroed_registers);
}

double HLL::CorrectEstimate(const std::pair<double, int>& EV) const {
  // TODO(tdial): The logic below was more or less copied from the research
  // paper, less the handling of the sparse register array, which is not
  // implemented at this time. It is correct, but seems a little awkward.
  // Have someone else review this.

  // Determine the threshold under which we apply a bias correction.
  const double BiasThreshold = 5 * register_count_;

  // Calculate E', the bias corrected estimate.
  const double EP = (EV.first < BiasThreshold)
                        ? (EV.first - EmpiricalBias(EV.first, precision_))
                        : EV.first;

  // H is either the LinearCounting estimate or the bias-corrected estimate.
  double H = 0.0;
  if (EV.second != 0) {
    H = LinearCounting(register_count_, EV.second);
  } else {
    H = EP;
  }

  // Under an empirically-determined threshold we return H, otherwise E'.
  if (H < EmpiricalThreshold(precision_)) {
    return H;
  } else {
    return EP;
  }
}

double HLL::Estimate() const {
  // Calculate the raw estimate and number of zerod registers per
  // original HyperLogLog. The number of zeroed registers decides whether we use
  // LinearCounting.
  return CorrectEstimate(RawEstimate());
}

double HLL::EstimateUnion(const HLL* other) const {
  return CorrectEstimate(RawEstimateUnion(other));
}

}  // namespace libcount
