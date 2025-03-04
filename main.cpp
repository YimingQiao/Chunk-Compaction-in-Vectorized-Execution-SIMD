#include <iostream>
#include <random>

#include "base.h"
#include "chaining_ht.h"
#include "compactor.h"
#include "data_collection.h"
#include "negative_feedback.hpp"
#include "profiler.h"
#include "setting.h"

using namespace simd_compaction;

struct PipelineState {
  vector<unique_ptr<HashTable>> hts;
  vector<unique_ptr<DataChunk>> intermediates;
  vector<unique_ptr<Compactor>> compactors;

  PipelineState() : hts(kJoins), intermediates(kJoins), compactors(kJoins) {}
};

static void ExecutePipeline(DataChunk &input, PipelineState &state, DataCollection &result_table, size_t level);

void FlushPipelineCache(PipelineState &state, DataCollection &result_table, size_t level);

std::vector<size_t> ParseList(const std::string &s) {
  std::stringstream ss(s.substr(1, s.size() - 2));// Ignore brackets
  std::vector<size_t> result;
  std::string item;
  while (std::getline(ss, item, ',')) { result.push_back(std::stoi(item)); }
  return result;
}

int ParseParameters(int argc, char *argv[]);

// example: compaction --join-num 4 --chunk-factor 5 --lhs-size 20000000 --rhs-size 2000000 --payload-length=[0,0,0,0]
int main(int argc, char *argv[]) {
  if (ParseParameters(argc, argv)) return 0;

  // random generator
  std::random_device rd;
  std::mt19937 gen(2);
  std::uniform_int_distribution<> dist(0, kRHSTupleSize);

  // ---------------------------------------------- Query Setting ----------------------------------------------

  // create probe table: (id1, id2, ..., idn)
  vector<AttributeType> types;
  for (size_t i = 0; i < kJoins; ++i) types.push_back(AttributeType::INTEGER);
  simd_compaction::DataCollection table(types);
  vector<simd_compaction::Attribute> tuple(kJoins);
  for (size_t i = 0; i < kLHSTupleSize; ++i) {
    for (size_t j = 0; j < kJoins; ++j) { tuple[j] = size_t(dist(gen)); }
    table.AppendTuple(tuple);
  }

  // create rhs hash table, and the result_table chunk
  PipelineState state;
  auto &hts = state.hts;
  auto &intermediates = state.intermediates;
  auto &compactors = state.compactors;
  for (size_t i = 0; i < kJoins; ++i) {
    hts[i] = std::make_unique<HashTable>(kRHSTupleSize, kChunkFactor);
    types.push_back(AttributeType::INTEGER);
    types.push_back(AttributeType::INTEGER);
    intermediates[i] = std::make_unique<DataChunk>(types);
    compactors[i] = std::make_unique<Compactor>(types);
  }

  // create the result_table collection
  DataCollection result_table(types);

#ifdef flag_dynamic_compact
  // create MAB for each join operator
  for (size_t i = 0; i < kJoins; ++i) CompactTuner::Get().Initialize(size_t(&state.compactors[i]));
#endif
  // -----------------------------------------------------------------------------------------------------------

  double latency = 0;
  Profiler timer;
  {
    // Start process each chunk in the lhs table
    size_t num_chunk_size = kBlockSize;
    size_t start = 0;
    size_t end;
    do {
      end = std::min(start + num_chunk_size, kLHSTupleSize);
      // num_chunk_size = (num_chunk_size + 1) % kBlockSize;
      DataChunk chunk = table.FetchChunk(start, end);
      start = end;

      timer.Start();
      ExecutePipeline(chunk, state, result_table, 0);
      latency += timer.Elapsed();
    } while (end < kLHSTupleSize);

#if defined(flag_full_compact) || defined(flag_binary_compact) || defined(flag_dynamic_compact)
    timer.Start();
    { FlushPipelineCache(state, result_table, 0); }
    latency += timer.Elapsed();
#endif
  }

  std::cerr << "------------------ Statistic ------------------\n";
  std::cerr << "[Total Time]: " << latency << "s\n";
  BeeProfiler::Get().EndProfiling();
  ZebraProfiler::Get().ToCSV();
  CompactTuner::Get().Reset();

  if (flag_collect_tuples) {
    // show the joined result.
    std::cout << "Number of tuples in the result table: " << result_table.NumTuples() << "\n";
    result_table.Print(8);
  }

  return 0;
}

void ExecutePipeline(DataChunk &input, PipelineState &state, DataCollection &result_table, size_t level) {
  auto &hts = state.hts;
  auto &intermediates = state.intermediates;
  auto &compactors = state.compactors;

  // The last operator: ResultCollector
  if (level == hts.size()) {
    if (flag_collect_tuples) result_table.AppendChunk(input);
    return;
  }

  auto &join_key = input.data_[level];
  auto &result = intermediates[level];
  auto &compactor = compactors[level];

#ifdef flag_dynamic_compact
  // ---------------------------------- learn compaction thresholds -------------------------------------
  Profiler profiler;
  profiler.Start();
  // 1. Get a compaction threshold
  // 2. Start to record execution time
  size_t threshold = CompactTuner::Get().SelectArm(level);
  compactor->SetThreshold(threshold);

  BeeProfiler::Get().InsertStatRecord("[UCB Get Thresholds]", profiler.Elapsed());

  profiler.Start();
  // -----------------------------------------------------------------------------------------------------
#endif

  auto ss = hts[level]->Probe(join_key, input.count_, input.selection_vector_);
  while (ss.HasNext()) {
    ss.Next(join_key, input, *result);

#if defined(flag_full_compact) || defined(flag_binary_compact) || defined(flag_dynamic_compact)
    // A compactor sits here.
    compactor->Compact(result);
    if (result->count_ == 0) continue;
#endif

    ExecutePipeline(*result, state, result_table, level + 1);
  }

#ifdef flag_dynamic_compact
  // ---------------------------------- learn compaction thresholds -------------------------------------
  double time = profiler.Elapsed();
  profiler.Start();
  CompactTuner::Get().UpdateArm(level, compactor->GetThreshold(), 2 / time / 1e3);
  BeeProfiler::Get().InsertStatRecord("[UCB Update]", profiler.Elapsed());
  // -----------------------------------------------------------------------------------------------------
#endif
}

void FlushPipelineCache(PipelineState &state, DataCollection &result_table, size_t level) {
  auto &hts = state.hts;
  auto &intermediates = state.intermediates;
  auto &compactors = state.compactors;

  // The last operator: ResultCollector. It has no compactor.
  if (level == hts.size()) return;

  auto &result = intermediates[level];
  auto &compactor = compactors[level];

  // Fetch the remaining tuples in the cache.
  compactor->Flush(result);

  // Continue the pipeline execution.
  ExecutePipeline(*result, state, result_table, level + 1);

  // Flush the next level.
  FlushPipelineCache(state, result_table, level + 1);
}

void PrintHelp() {
  std::cerr << "Usage: [program_name] [options]\n";
  std::cerr << "Options:\n";
  std::cerr << "  --join-num [value]        Number of joins\n";
  std::cerr << "  --chunk-factor [value]    Chunk factor\n";
  std::cerr << "  --lhs-size [value]        Size of LHS tuples\n";
  std::cerr << "  --rhs-size [value]        Size of RHS tuples\n";
}

int ParseParameters(int argc, char **argv) {
  if (argc != 1) {
    for (int i = 1; i < argc; i++) {
      std::string arg(argv[i]);

      if (arg == "--join-num") {
        if (i + 1 < argc) {
          kJoins = std::stoi(argv[i + 1]);
          i++;
        }
      } else if (arg == "--chunk-factor") {
        if (i + 1 < argc) {
          kChunkFactor = std::stoi(argv[i + 1]);
          i++;
        }
      } else if (arg == "--lhs-size") {
        if (i + 1 < argc) {
          kLHSTupleSize = std::stoi(argv[i + 1]);
          i++;
        }
      } else if (arg == "--rhs-size") {
        if (i + 1 < argc) {
          kRHSTupleSize = std::stoi(argv[i + 1]);
          i++;
        }
      }
    }
  } else {
    PrintHelp();
    return 1;
  }

  // show the setting
  std::cerr << "------------------ Setting ------------------\n";
  std::cerr << "Strategy: " << strategy_name << "\n"
            << "Number of Joins: " << kJoins << "\n"
            << "Number of LHS Tuple: " << kLHSTupleSize << "\n"
            << "Number of RHS Tuple: " << kRHSTupleSize << "\n"
            << "Chunk Factor: " << kChunkFactor << "\n";

  return 0;
}
