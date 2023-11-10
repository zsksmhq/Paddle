// Copyright (c) 2023 CINN Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once
#include "paddle/cinn/ir/group_schedule/base_group_scheduler.h"

namespace cinn {
namespace ir {

/**
 * The class used for scheduling fusion groups with dynamic shape.
 * Note: Currently only CUDA backend is supported.
 */
class DynamicShapeGroupScheduler : public GroupScheduler {
 public:
  DynamicShapeGroupScheduler(
      ir::IRSchedule* ir_sch,
      const std::unordered_set<std::string>& output_tensor_names,
      const common::Target& target)
      : GroupScheduler(ir_sch, output_tensor_names, target) {}

  void Schedule() override;

  std::vector<std::pair<SymbolicCondition, ir::Expr>> GetIRs() override;

 private:
  std::vector<std::pair<SymbolicCondition, std::unique_ptr<ir::IRSchedule>>>
      ir_schs_;
};

}  // namespace ir
}  // namespace cinn