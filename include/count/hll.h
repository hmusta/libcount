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

#ifndef INCLUDE_COUNT_HLL_H_
#define INCLUDE_COUNT_HLL_H_

#include <stdint.h>

#include <utility>

#include "count/hll_limits.h"

namespace libcount {

class HLL {
 public:
  ~HLL();

  // Create an instance of a HyperLogLog++ cardinality estimator. Valid values
  // for precision are [4..18] inclusive, and govern the precision of the
  // estimate. Returns NULL on failure. In the event of failure, the caller
  // may provide a pointer to an integer to learn the reason.
  static HLL* Create(int precision, int* error = 0);

  // Update the instance to record the observation of an element. It is
  // assumed that the caller uses a high-quality 64-bit hash function that
  // is free of bias. Empirically, using a subset of bits from a well-known
  // cryptographic hash function such as SHA1, is a good choice.
  void Update(uint64_t hash);

  // Add an array of hashes
  void Update(const uint64_t* hashes, uint64_t len);

  // Merge count tracking information from another instance into the object.
  // The object being merged in must have been instantiated with the same
  // precision. Returns 0 on success, EINVAL otherwise.
  int Merge(const HLL* other);

  // Compute the bias-corrected estimate using the HyperLogLog++ algorithm.
  double Estimate() const;

  // Compute the bias-corrected estimate of the union using the HyperLogLog++
  // algorithm.
  double EstimateUnion(const HLL* other) const;

  // Getters
  int get_precision() const { return precision_; }
  int get_register_count() const { return register_count_; }

  // Access underlying register array
  const uint8_t* data() const { return registers_; }
  uint8_t* data() { return registers_; }

 private:
  // No copying allowed
  HLL(const HLL& no_copy);
  HLL& operator=(const HLL& no_assign);

  // Constructor is private: we validate the precision in the Create function.
  explicit HLL(int precision);

  // Compute the raw estimate and number of zero registers based on the
  // HyperLogLog algorithm. The number of zeroed registers decides whether we
  // use LinearCounting.
  std::pair<double, int> RawEstimate() const;
  std::pair<double, int> RawEstimateUnion(const HLL* other) const;
  double CorrectEstimate(const std::pair<double, int>& EV) const;

  int precision_;
  int register_count_;
  uint8_t* registers_;
};

}  // namespace libcount

#endif  // INCLUDE_COUNT_HLL_H_
