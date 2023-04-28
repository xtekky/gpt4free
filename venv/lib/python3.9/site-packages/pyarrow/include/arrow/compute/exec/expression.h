// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

// This API is EXPERIMENTAL.

#pragma once

#include <memory>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "arrow/compute/type_fwd.h"
#include "arrow/datum.h"
#include "arrow/type_fwd.h"
#include "arrow/util/small_vector.h"

namespace arrow {
namespace compute {

/// \defgroup expression-core Expressions to describe transformations in execution plans
///
/// @{

/// An unbound expression which maps a single Datum to another Datum.
/// An expression is one of
/// - A literal Datum.
/// - A reference to a single (potentially nested) field of the input Datum.
/// - A call to a compute function, with arguments specified by other Expressions.
class ARROW_EXPORT Expression {
 public:
  struct Call {
    std::string function_name;
    std::vector<Expression> arguments;
    std::shared_ptr<FunctionOptions> options;
    // Cached hash value
    size_t hash;

    // post-Bind properties:
    std::shared_ptr<Function> function;
    const Kernel* kernel = NULLPTR;
    std::shared_ptr<KernelState> kernel_state;
    TypeHolder type;

    void ComputeHash();
  };

  std::string ToString() const;
  bool Equals(const Expression& other) const;
  size_t hash() const;
  struct Hash {
    size_t operator()(const Expression& expr) const { return expr.hash(); }
  };

  /// Bind this expression to the given input type, looking up Kernels and field types.
  /// Some expression simplification may be performed and implicit casts will be inserted.
  /// Any state necessary for execution will be initialized and returned.
  Result<Expression> Bind(const TypeHolder& in, ExecContext* = NULLPTR) const;
  Result<Expression> Bind(const Schema& in_schema, ExecContext* = NULLPTR) const;

  // XXX someday
  // Clone all KernelState in this bound expression. If any function referenced by this
  // expression has mutable KernelState, it is not safe to execute or apply simplification
  // passes to it (or copies of it!) from multiple threads. Cloning state produces new
  // KernelStates where necessary to ensure that Expressions may be manipulated safely
  // on multiple threads.
  // Result<ExpressionState> CloneState() const;
  // Status SetState(ExpressionState);

  /// Return true if all an expression's field references have explicit types
  /// and all of its functions' kernels are looked up.
  bool IsBound() const;

  /// Return true if this expression is composed only of Scalar literals, field
  /// references, and calls to ScalarFunctions.
  bool IsScalarExpression() const;

  /// Return true if this expression is literal and entirely null.
  bool IsNullLiteral() const;

  /// Return true if this expression could evaluate to true. Will return true for any
  /// unbound, non-boolean, or unsimplified Expressions
  bool IsSatisfiable() const;

  // XXX someday
  // Result<PipelineGraph> GetPipelines();

  bool is_valid() const { return impl_ != NULLPTR; }

  /// Access a Call or return nullptr if this expression is not a call
  const Call* call() const;
  /// Access a Datum or return nullptr if this expression is not a literal
  const Datum* literal() const;
  /// Access a FieldRef or return nullptr if this expression is not a field_ref
  const FieldRef* field_ref() const;

  /// The type to which this expression will evaluate
  const DataType* type() const;
  // XXX someday
  // NullGeneralization::type nullable() const;

  struct Parameter {
    FieldRef ref;

    // post-bind properties
    TypeHolder type;
    ::arrow::internal::SmallVector<int, 2> indices;
  };
  const Parameter* parameter() const;

  Expression() = default;
  explicit Expression(Call call);
  explicit Expression(Datum literal);
  explicit Expression(Parameter parameter);

 private:
  using Impl = std::variant<Datum, Parameter, Call>;
  std::shared_ptr<Impl> impl_;

  ARROW_FRIEND_EXPORT friend bool Identical(const Expression& l, const Expression& r);
};

inline bool operator==(const Expression& l, const Expression& r) { return l.Equals(r); }
inline bool operator!=(const Expression& l, const Expression& r) { return !l.Equals(r); }

ARROW_EXPORT void PrintTo(const Expression&, std::ostream*);

// Factories

ARROW_EXPORT
Expression literal(Datum lit);

template <typename Arg>
Expression literal(Arg&& arg) {
  return literal(Datum(std::forward<Arg>(arg)));
}

ARROW_EXPORT
Expression field_ref(FieldRef ref);

ARROW_EXPORT
Expression call(std::string function, std::vector<Expression> arguments,
                std::shared_ptr<FunctionOptions> options = NULLPTR);

template <typename Options, typename = typename std::enable_if<
                                std::is_base_of<FunctionOptions, Options>::value>::type>
Expression call(std::string function, std::vector<Expression> arguments,
                Options options) {
  return call(std::move(function), std::move(arguments),
              std::make_shared<Options>(std::move(options)));
}

/// Assemble a list of all fields referenced by an Expression at any depth.
ARROW_EXPORT
std::vector<FieldRef> FieldsInExpression(const Expression&);

/// Check if the expression references any fields.
ARROW_EXPORT
bool ExpressionHasFieldRefs(const Expression&);

struct ARROW_EXPORT KnownFieldValues;

/// Assemble a mapping from field references to known values. This derives known values
/// from "equal" and "is_null" Expressions referencing a field and a literal.
ARROW_EXPORT
Result<KnownFieldValues> ExtractKnownFieldValues(
    const Expression& guaranteed_true_predicate);

/// @}

/// \defgroup expression-passes Functions for modification of Expressions
///
/// @{
///
/// These transform bound expressions. Some transforms utilize a guarantee, which is
/// provided as an Expression which is guaranteed to evaluate to true. The
/// guaranteed_true_predicate need not be bound, but canonicalization is currently
/// deferred to producers of guarantees. For example in order to be recognized as a
/// guarantee on a field value, an Expression must be a call to "equal" with field_ref LHS
/// and literal RHS. Flipping the arguments, "is_in" with a one-long value_set, ... or
/// other semantically identical Expressions will not be recognized.

/// Weak canonicalization which establishes guarantees for subsequent passes. Even
/// equivalent Expressions may result in different canonicalized expressions.
/// TODO this could be a strong canonicalization
ARROW_EXPORT
Result<Expression> Canonicalize(Expression, ExecContext* = NULLPTR);

/// Simplify Expressions based on literal arguments (for example, add(null, x) will always
/// be null so replace the call with a null literal). Includes early evaluation of all
/// calls whose arguments are entirely literal.
ARROW_EXPORT
Result<Expression> FoldConstants(Expression);

/// Simplify Expressions by replacing with known values of the fields which it references.
ARROW_EXPORT
Result<Expression> ReplaceFieldsWithKnownValues(const KnownFieldValues& known_values,
                                                Expression);

/// Simplify an expression by replacing subexpressions based on a guarantee:
/// a boolean expression which is guaranteed to evaluate to `true`. For example, this is
/// used to remove redundant function calls from a filter expression or to replace a
/// reference to a constant-value field with a literal.
ARROW_EXPORT
Result<Expression> SimplifyWithGuarantee(Expression,
                                         const Expression& guaranteed_true_predicate);

/// Replace all named field refs (e.g. "x" or "x.y") with field paths (e.g. [0] or [1,3])
///
/// This isn't usually needed and does not offer any simplification by itself.  However,
/// it can be useful to normalize an expression to paths to make it simpler to work with.
ARROW_EXPORT Result<Expression> RemoveNamedRefs(Expression expression);

/// @}

// Execution

/// Create an ExecBatch suitable for passing to ExecuteScalarExpression() from a
/// RecordBatch which may have missing or incorrectly ordered columns.
/// Missing fields will be replaced with null scalars.
ARROW_EXPORT Result<ExecBatch> MakeExecBatch(const Schema& full_schema,
                                             const Datum& partial,
                                             Expression guarantee = literal(true));

/// Execute a scalar expression against the provided state and input ExecBatch. This
/// expression must be bound.
ARROW_EXPORT
Result<Datum> ExecuteScalarExpression(const Expression&, const ExecBatch& input,
                                      ExecContext* = NULLPTR);

/// Convenience function for invoking against a RecordBatch
ARROW_EXPORT
Result<Datum> ExecuteScalarExpression(const Expression&, const Schema& full_schema,
                                      const Datum& partial_input, ExecContext* = NULLPTR);

// Serialization

ARROW_EXPORT
Result<std::shared_ptr<Buffer>> Serialize(const Expression&);

ARROW_EXPORT
Result<Expression> Deserialize(std::shared_ptr<Buffer>);

/// \defgroup expression-convenience Functions convenient expression creation
///
/// @{

ARROW_EXPORT Expression project(std::vector<Expression> values,
                                std::vector<std::string> names);

ARROW_EXPORT Expression equal(Expression lhs, Expression rhs);

ARROW_EXPORT Expression not_equal(Expression lhs, Expression rhs);

ARROW_EXPORT Expression less(Expression lhs, Expression rhs);

ARROW_EXPORT Expression less_equal(Expression lhs, Expression rhs);

ARROW_EXPORT Expression greater(Expression lhs, Expression rhs);

ARROW_EXPORT Expression greater_equal(Expression lhs, Expression rhs);

ARROW_EXPORT Expression is_null(Expression lhs, bool nan_is_null = false);

ARROW_EXPORT Expression is_valid(Expression lhs);

ARROW_EXPORT Expression and_(Expression lhs, Expression rhs);
ARROW_EXPORT Expression and_(const std::vector<Expression>&);
ARROW_EXPORT Expression or_(Expression lhs, Expression rhs);
ARROW_EXPORT Expression or_(const std::vector<Expression>&);
ARROW_EXPORT Expression not_(Expression operand);

/// @}

}  // namespace compute
}  // namespace arrow
