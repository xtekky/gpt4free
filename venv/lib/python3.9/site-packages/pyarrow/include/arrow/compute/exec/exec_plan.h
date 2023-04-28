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

#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "arrow/compute/exec.h"
#include "arrow/compute/type_fwd.h"
#include "arrow/type_fwd.h"
#include "arrow/util/future.h"
#include "arrow/util/macros.h"
#include "arrow/util/tracing.h"
#include "arrow/util/type_fwd.h"
#include "arrow/util/visibility.h"

namespace arrow {

namespace compute {

/// \addtogroup execnode-components
/// @{

class ARROW_EXPORT ExecPlan : public std::enable_shared_from_this<ExecPlan> {
 public:
  // This allows operators to rely on signed 16-bit indices
  static const uint32_t kMaxBatchSize = 1 << 15;
  using NodeVector = std::vector<ExecNode*>;

  virtual ~ExecPlan() = default;

  QueryContext* query_context();

  /// Make an empty exec plan
  static Result<std::shared_ptr<ExecPlan>> Make(
      QueryOptions options, ExecContext exec_context = *threaded_exec_context(),
      std::shared_ptr<const KeyValueMetadata> metadata = NULLPTR);

  static Result<std::shared_ptr<ExecPlan>> Make(
      ExecContext exec_context = *threaded_exec_context(),
      std::shared_ptr<const KeyValueMetadata> metadata = NULLPTR);

  static Result<std::shared_ptr<ExecPlan>> Make(
      QueryOptions options, ExecContext* exec_context,
      std::shared_ptr<const KeyValueMetadata> metadata = NULLPTR);

  static Result<std::shared_ptr<ExecPlan>> Make(
      ExecContext* exec_context,
      std::shared_ptr<const KeyValueMetadata> metadata = NULLPTR);

  ExecNode* AddNode(std::unique_ptr<ExecNode> node);

  template <typename Node, typename... Args>
  Node* EmplaceNode(Args&&... args) {
    std::unique_ptr<Node> node{new Node{std::forward<Args>(args)...}};
    auto out = node.get();
    AddNode(std::move(node));
    return out;
  }

  /// The initial inputs
  const NodeVector& sources() const;

  /// The final outputs
  const NodeVector& sinks() const;

  Status Validate();

  /// \brief Start producing on all nodes
  ///
  /// Nodes are started in reverse topological order, such that any node
  /// is started before all of its inputs.
  Status StartProducing();

  /// \brief Stop producing on all nodes
  ///
  /// Nodes are stopped in topological order, such that any node
  /// is stopped before all of its outputs.
  void StopProducing();

  /// \brief A future which will be marked finished when all nodes have stopped producing.
  Future<> finished();

  /// \brief Return whether the plan has non-empty metadata
  bool HasMetadata() const;

  /// \brief Return the plan's attached metadata
  std::shared_ptr<const KeyValueMetadata> metadata() const;

  std::string ToString() const;
};

class ARROW_EXPORT ExecNode {
 public:
  using NodeVector = std::vector<ExecNode*>;

  virtual ~ExecNode() = default;

  virtual const char* kind_name() const = 0;

  // The number of inputs/outputs expected by this node
  int num_inputs() const { return static_cast<int>(inputs_.size()); }
  int num_outputs() const { return num_outputs_; }

  /// This node's predecessors in the exec plan
  const NodeVector& inputs() const { return inputs_; }

  /// \brief Labels identifying the function of each input.
  const std::vector<std::string>& input_labels() const { return input_labels_; }

  /// This node's successors in the exec plan
  const NodeVector& outputs() const { return outputs_; }

  /// The datatypes for batches produced by this node
  const std::shared_ptr<Schema>& output_schema() const { return output_schema_; }

  /// This node's exec plan
  ExecPlan* plan() { return plan_; }

  /// \brief An optional label, for display and debugging
  ///
  /// There is no guarantee that this value is non-empty or unique.
  const std::string& label() const { return label_; }
  void SetLabel(std::string label) { label_ = std::move(label); }

  Status Validate() const;

  /// Upstream API:
  /// These functions are called by input nodes that want to inform this node
  /// about an updated condition (a new input batch, an error, an impeding
  /// end of stream).
  ///
  /// Implementation rules:
  /// - these may be called anytime after StartProducing() has succeeded
  ///   (and even during or after StopProducing())
  /// - these may be called concurrently
  /// - these are allowed to call back into PauseProducing(), ResumeProducing()
  ///   and StopProducing()

  /// Transfer input batch to ExecNode
  virtual void InputReceived(ExecNode* input, ExecBatch batch) = 0;

  /// Signal error to ExecNode
  virtual void ErrorReceived(ExecNode* input, Status error) = 0;

  /// Mark the inputs finished after the given number of batches.
  ///
  /// This may be called before all inputs are received.  This simply fixes
  /// the total number of incoming batches for an input, so that the ExecNode
  /// knows when it has received all input, regardless of order.
  virtual void InputFinished(ExecNode* input, int total_batches) = 0;

  /// \brief Perform any needed initialization
  ///
  /// This hook performs any actions in between creation of ExecPlan and the call to
  /// StartProducing. An example could be Bloom filter pushdown. The order of ExecNodes
  /// that executes this method is undefined, but the calls are made synchronously.
  ///
  /// At this point a node can rely on all inputs & outputs (and the input schemas)
  /// being well defined.
  virtual Status Init();

  /// Lifecycle API:
  /// - start / stop to initiate and terminate production
  /// - pause / resume to apply backpressure
  ///
  /// Implementation rules:
  /// - StartProducing() should not recurse into the inputs, as it is
  ///   handled by ExecPlan::StartProducing()
  /// - PauseProducing(), ResumeProducing(), StopProducing() may be called
  ///   concurrently (but only after StartProducing() has returned successfully)
  /// - PauseProducing(), ResumeProducing(), StopProducing() may be called
  ///   by the downstream nodes' InputReceived(), ErrorReceived(), InputFinished()
  ///   methods
  /// - StopProducing() should recurse into the inputs
  /// - StopProducing() must be idempotent

  // XXX What happens if StartProducing() calls an output's InputReceived()
  // synchronously, and InputReceived() decides to call back into StopProducing()
  // (or PauseProducing()) because it received enough data?
  //
  // Right now, since synchronous calls happen in both directions (input to
  // output and then output to input), a node must be careful to be reentrant
  // against synchronous calls from its output, *and* also concurrent calls from
  // other threads.  The most reliable solution is to update the internal state
  // first, and notify outputs only at the end.
  //
  // Alternate rules:
  // - StartProducing(), ResumeProducing() can call synchronously into
  //   its ouputs' consuming methods (InputReceived() etc.)
  // - InputReceived(), ErrorReceived(), InputFinished() can call asynchronously
  //   into its inputs' PauseProducing(), StopProducing()
  //
  // Alternate API:
  // - InputReceived(), ErrorReceived(), InputFinished() return a ProductionHint
  //   enum: either None (default), PauseProducing, ResumeProducing, StopProducing
  // - A method allows passing a ProductionHint asynchronously from an output node
  //   (replacing PauseProducing(), ResumeProducing(), StopProducing())

  // Concurrent calls to PauseProducing and ResumeProducing can be hard to sequence
  // as they may travel at different speeds through the plan.
  //
  // For example, consider a resume that comes quickly after a pause.  If the source
  // receives the resume before the pause the source may think the destination is full
  // and halt production which would lead to deadlock.
  //
  // To resolve this a counter is sent for all calls to pause/resume.  Only the call with
  // the highest counter value is valid.  So if a call to PauseProducing(5) comes after
  // a call to ResumeProducing(6) then the source should continue producing.
  //
  // If a node has multiple outputs it should emit a new counter value to its inputs
  // whenever any of its outputs changes which means the counters sent to inputs may be
  // larger than the counters received on its outputs.
  //
  // A node with multiple outputs will also need to ensure it is applying backpressure if
  // any of its outputs is asking to pause

  /// \brief Start producing
  ///
  /// This must only be called once.  If this fails, then other lifecycle
  /// methods must not be called.
  ///
  /// This is typically called automatically by ExecPlan::StartProducing().
  virtual Status StartProducing() = 0;

  /// \brief Pause producing temporarily
  ///
  /// \param output Pointer to the output that is full
  /// \param counter Counter used to sequence calls to pause/resume
  ///
  /// This call is a hint that an output node is currently not willing
  /// to receive data.
  ///
  /// This may be called any number of times after StartProducing() succeeds.
  /// However, the node is still free to produce data (which may be difficult
  /// to prevent anyway if data is produced using multiple threads).
  virtual void PauseProducing(ExecNode* output, int32_t counter) = 0;

  /// \brief Resume producing after a temporary pause
  ///
  /// \param output Pointer to the output that is now free
  /// \param counter Counter used to sequence calls to pause/resume
  ///
  /// This call is a hint that an output node is willing to receive data again.
  ///
  /// This may be called any number of times after StartProducing() succeeds.
  virtual void ResumeProducing(ExecNode* output, int32_t counter) = 0;

  /// \brief Stop producing definitively to a single output
  ///
  /// This call is a hint that an output node has completed and is not willing
  /// to receive any further data.
  virtual void StopProducing(ExecNode* output) = 0;

  /// \brief Stop producing definitively to all outputs
  virtual void StopProducing() = 0;

  /// \brief A future which will be marked finished when this node has stopped producing.
  virtual Future<> finished() { return finished_; }

  std::string ToString(int indent = 0) const;

 protected:
  ExecNode(ExecPlan* plan, NodeVector inputs, std::vector<std::string> input_labels,
           std::shared_ptr<Schema> output_schema, int num_outputs);

  // A helper method to send an error status to all outputs.
  // Returns true if the status was an error.
  bool ErrorIfNotOk(Status status);

  /// Provide extra info to include in the string representation.
  virtual std::string ToStringExtra(int indent) const;

  ExecPlan* plan_;
  std::string label_;

  NodeVector inputs_;
  std::vector<std::string> input_labels_;

  std::shared_ptr<Schema> output_schema_;
  int num_outputs_;
  NodeVector outputs_;

  // Future to sync finished
  Future<> finished_ = Future<>::Make();

  util::tracing::Span span_;
};

/// \brief An extensible registry for factories of ExecNodes
class ARROW_EXPORT ExecFactoryRegistry {
 public:
  using Factory = std::function<Result<ExecNode*>(ExecPlan*, std::vector<ExecNode*>,
                                                  const ExecNodeOptions&)>;

  virtual ~ExecFactoryRegistry() = default;

  /// \brief Get the named factory from this registry
  ///
  /// will raise if factory_name is not found
  virtual Result<Factory> GetFactory(const std::string& factory_name) = 0;

  /// \brief Add a factory to this registry with the provided name
  ///
  /// will raise if factory_name is already in the registry
  virtual Status AddFactory(std::string factory_name, Factory factory) = 0;
};

/// The default registry, which includes built-in factories.
ARROW_EXPORT
ExecFactoryRegistry* default_exec_factory_registry();

/// \brief Construct an ExecNode using the named factory
inline Result<ExecNode*> MakeExecNode(
    const std::string& factory_name, ExecPlan* plan, std::vector<ExecNode*> inputs,
    const ExecNodeOptions& options,
    ExecFactoryRegistry* registry = default_exec_factory_registry()) {
  ARROW_ASSIGN_OR_RAISE(auto factory, registry->GetFactory(factory_name));
  return factory(plan, std::move(inputs), options);
}

/// \brief Helper class for declaring sets of ExecNodes efficiently
///
/// A Declaration represents an unconstructed ExecNode (and potentially more since its
/// inputs may also be Declarations). The node can be constructed and added to a plan
/// with Declaration::AddToPlan, which will recursively construct any inputs as necessary.
struct ARROW_EXPORT Declaration {
  using Input = std::variant<ExecNode*, Declaration>;

  Declaration() {}

  Declaration(std::string factory_name, std::vector<Input> inputs,
              std::shared_ptr<ExecNodeOptions> options, std::string label)
      : factory_name{std::move(factory_name)},
        inputs{std::move(inputs)},
        options{std::move(options)},
        label{std::move(label)} {}

  template <typename Options>
  Declaration(std::string factory_name, std::vector<Input> inputs, Options options,
              std::string label)
      : Declaration{std::move(factory_name), std::move(inputs),
                    std::shared_ptr<ExecNodeOptions>(
                        std::make_shared<Options>(std::move(options))),
                    std::move(label)} {}

  template <typename Options>
  Declaration(std::string factory_name, std::vector<Input> inputs, Options options)
      : Declaration{std::move(factory_name), std::move(inputs), std::move(options),
                    /*label=*/""} {}

  template <typename Options>
  Declaration(std::string factory_name, Options options)
      : Declaration{std::move(factory_name), {}, std::move(options), /*label=*/""} {}

  template <typename Options>
  Declaration(std::string factory_name, Options options, std::string label)
      : Declaration{std::move(factory_name), {}, std::move(options), std::move(label)} {}

  /// \brief Convenience factory for the common case of a simple sequence of nodes.
  ///
  /// Each of decls will be appended to the inputs of the subsequent declaration,
  /// and the final modified declaration will be returned.
  ///
  /// Without this convenience factory, constructing a sequence would require explicit,
  /// difficult-to-read nesting:
  ///
  ///     Declaration{"n3",
  ///                   {
  ///                       Declaration{"n2",
  ///                                   {
  ///                                       Declaration{"n1",
  ///                                                   {
  ///                                                       Declaration{"n0", N0Opts{}},
  ///                                                   },
  ///                                                   N1Opts{}},
  ///                                   },
  ///                                   N2Opts{}},
  ///                   },
  ///                   N3Opts{}};
  ///
  /// An equivalent Declaration can be constructed more tersely using Sequence:
  ///
  ///     Declaration::Sequence({
  ///         {"n0", N0Opts{}},
  ///         {"n1", N1Opts{}},
  ///         {"n2", N2Opts{}},
  ///         {"n3", N3Opts{}},
  ///     });
  static Declaration Sequence(std::vector<Declaration> decls);

  Result<ExecNode*> AddToPlan(ExecPlan* plan, ExecFactoryRegistry* registry =
                                                  default_exec_factory_registry()) const;

  // Validate a declaration
  bool IsValid(ExecFactoryRegistry* registry = default_exec_factory_registry()) const;

  std::string factory_name;
  std::vector<Input> inputs;
  std::shared_ptr<ExecNodeOptions> options;
  std::string label;
};

/// \brief Utility method to run a declaration and collect the results into a table
///
/// \param declaration A declaration describing the plan to run
/// \param use_threads If `use_threads` is false then all CPU work will be done on the
///                    calling thread.  I/O tasks will still happen on the I/O executor
///                    and may be multi-threaded (but should not use significant CPU
///                    resources).
/// \param memory_pool The memory pool to use for allocations made while running the plan.
/// \param function_registry The function registry to use for function execution.  If null
///                          then the default function registry will be used.
///
/// This method will add a sink node to the declaration to collect results into a
/// table.  It will then create an ExecPlan from the declaration, start the exec plan,
/// block until the plan has finished, and return the created table.
ARROW_EXPORT Result<std::shared_ptr<Table>> DeclarationToTable(
    Declaration declaration, bool use_threads = true,
    MemoryPool* memory_pool = default_memory_pool(),
    FunctionRegistry* function_registry = NULLPTR);

/// \brief Asynchronous version of \see DeclarationToTable
///
/// \param declaration A declaration describing the plan to run
/// \param use_threads The behavior of use_threads is slightly different than the
///                    synchronous version since we cannot run synchronously on the
///                    calling thread. Instead, if use_threads=false then a new thread
///                    pool will be created with a single thread and this will be used for
///                    all compute work.
/// \param memory_pool The memory pool to use for allocations made while running the plan.
/// \param function_registry The function registry to use for function execution. If null
///                          then the default function registry will be used.
ARROW_EXPORT Future<std::shared_ptr<Table>> DeclarationToTableAsync(
    Declaration declaration, bool use_threads = true,
    MemoryPool* memory_pool = default_memory_pool(),
    FunctionRegistry* function_registry = NULLPTR);

/// \brief Overload of \see DeclarationToTableAsync accepting a custom exec context
///
/// The executor must be specified (cannot be null) and must be kept alive until the
/// returned future finishes.
ARROW_EXPORT Future<std::shared_ptr<Table>> DeclarationToTableAsync(
    Declaration declaration, ExecContext custom_exec_context);

/// \brief a collection of exec batches with a common schema
struct BatchesWithCommonSchema {
  std::vector<ExecBatch> batches;
  std::shared_ptr<Schema> schema;
};

/// \brief Utility method to run a declaration and collect the results into ExecBatch
/// vector
///
/// \see DeclarationToTable for details on threading & execution
ARROW_EXPORT Result<BatchesWithCommonSchema> DeclarationToExecBatches(
    Declaration declaration, bool use_threads = true,
    MemoryPool* memory_pool = default_memory_pool(),
    FunctionRegistry* function_registry = NULLPTR);

/// \brief Asynchronous version of \see DeclarationToExecBatches
///
/// \see DeclarationToTableAsync for details on threading & execution
ARROW_EXPORT Future<BatchesWithCommonSchema> DeclarationToExecBatchesAsync(
    Declaration declaration, bool use_threads = true,
    MemoryPool* memory_pool = default_memory_pool(),
    FunctionRegistry* function_registry = NULLPTR);

/// \brief Overload of \see DeclarationToExecBatchesAsync accepting a custom exec context
///
/// \see DeclarationToTableAsync for details on threading & execution
ARROW_EXPORT Future<BatchesWithCommonSchema> DeclarationToExecBatchesAsync(
    Declaration declaration, ExecContext custom_exec_context);

/// \brief Utility method to run a declaration and collect the results into a vector
///
/// \see DeclarationToTable for details on threading & execution
ARROW_EXPORT Result<std::vector<std::shared_ptr<RecordBatch>>> DeclarationToBatches(
    Declaration declaration, bool use_threads = true,
    MemoryPool* memory_pool = default_memory_pool(),
    FunctionRegistry* function_registry = NULLPTR);

/// \brief Asynchronous version of \see DeclarationToBatches
///
/// \see DeclarationToTableAsync for details on threading & execution
ARROW_EXPORT Future<std::vector<std::shared_ptr<RecordBatch>>> DeclarationToBatchesAsync(
    Declaration declaration, bool use_threads = true,
    MemoryPool* memory_pool = default_memory_pool(),
    FunctionRegistry* function_registry = NULLPTR);

/// \brief Overload of \see DeclarationToBatchesAsync accepting a custom exec context
///
/// \see DeclarationToTableAsync for details on threading & execution
ARROW_EXPORT Future<std::vector<std::shared_ptr<RecordBatch>>> DeclarationToBatchesAsync(
    Declaration declaration, ExecContext exec_context);

/// \brief Utility method to run a declaration and return results as a RecordBatchReader
///
/// If an exec context is not provided then a default exec context will be used based
/// on the value of `use_threads`.  If `use_threads` is false then the CPU exeuctor will
/// be a serial executor and all CPU work will be done on the calling thread.  I/O tasks
/// will still happen on the I/O executor and may be multi-threaded.
///
/// If `use_threads` is false then all CPU work will happen during the calls to
/// RecordBatchReader::Next and no CPU work will happen in the background.  If
/// `use_threads` is true then CPU work will happen on the CPU thread pool and tasks may
/// run in between calls to RecordBatchReader::Next.  If the returned reader is not
/// consumed quickly enough then the plan will eventually pause as the backpressure queue
/// fills up.
///
/// If a custom exec context is provided then the value of `use_threads` will be ignored.
ARROW_EXPORT Result<std::unique_ptr<RecordBatchReader>> DeclarationToReader(
    Declaration declaration, bool use_threads = true,
    MemoryPool* memory_pool = default_memory_pool(),
    FunctionRegistry* function_registry = NULLPTR);

/// \brief Overload of \see DeclarationToReader accepting a custom exec context
ARROW_EXPORT Result<std::unique_ptr<RecordBatchReader>> DeclarationToReader(
    Declaration declaration, ExecContext exec_context);

/// \brief Utility method to run a declaration and ignore results
///
/// This can be useful when the data are consumed as part of the plan itself, for
/// example, when the plan ends with a write node.
///
/// \see DeclarationToTable for details on threading & execution
ARROW_EXPORT Status DeclarationToStatus(Declaration declaration, bool use_threads = true,
                                        MemoryPool* memory_pool = default_memory_pool(),
                                        FunctionRegistry* function_registry = NULLPTR);

/// \brief Asynchronous version of \see DeclarationToStatus
///
/// This can be useful when the data are consumed as part of the plan itself, for
/// example, when the plan ends with a write node.
///
/// \see DeclarationToTableAsync for details on threading & execution
ARROW_EXPORT Future<> DeclarationToStatusAsync(
    Declaration declaration, bool use_threads = true,
    MemoryPool* memory_pool = default_memory_pool(),
    FunctionRegistry* function_registry = NULLPTR);

/// \brief Overload of \see DeclarationToStatusAsync accepting a custom exec context
///
/// \see DeclarationToTableAsync for details on threading & execution
ARROW_EXPORT Future<> DeclarationToStatusAsync(Declaration declaration,
                                               ExecContext exec_context);

/// \brief Wrap an ExecBatch generator in a RecordBatchReader.
///
/// The RecordBatchReader does not impose any ordering on emitted batches.
ARROW_EXPORT
std::shared_ptr<RecordBatchReader> MakeGeneratorReader(
    std::shared_ptr<Schema>, std::function<Future<std::optional<ExecBatch>>()>,
    MemoryPool*);

constexpr int kDefaultBackgroundMaxQ = 32;
constexpr int kDefaultBackgroundQRestart = 16;

/// \brief Make a generator of RecordBatchReaders
///
/// Useful as a source node for an Exec plan
ARROW_EXPORT
Result<std::function<Future<std::optional<ExecBatch>>()>> MakeReaderGenerator(
    std::shared_ptr<RecordBatchReader> reader, arrow::internal::Executor* io_executor,
    int max_q = kDefaultBackgroundMaxQ, int q_restart = kDefaultBackgroundQRestart);

/// @}

}  // namespace compute
}  // namespace arrow
