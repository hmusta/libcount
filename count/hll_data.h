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

#ifndef INCLUDE_COUNT_HLL_DATA_H_
#define INCLUDE_COUNT_HLL_DATA_H_

// Two dimensional array containing estimate data at each precision level.
extern const double ESTIMATE_DATA[15][201];

// Two dimensional array containing bias for each estimate at each precision
// level. The indices into this array correspond 1:1 to the indices fof the
// data in ESTIMATE_DATA.
extern const double BIAS_DATA[15][201];

#endif  // INCLUDE_COUNT_HLL_DATA_H_

