#ifndef CAMPP_NATIVE_PIPELINE_PROCESS_RUNNER_H_
#define CAMPP_NATIVE_PIPELINE_PROCESS_RUNNER_H_

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace campp_pipeline {

struct ProcessResult {
    int exit_code = -1;
    int signal_number = 0;
};

class ProcessTreeMemoryTracker {
 public:
    ProcessTreeMemoryTracker();

    void Sample();
    uint64_t PipelinePeakRssBytes() const;
    uint64_t HostPeakRssBytes() const;

 private:
    int root_pid_;
    uint64_t pipeline_peak_rss_bytes_ = 0;
};

ProcessResult RunProcess(
    const std::vector<std::string> &arguments,
    const std::filesystem::path *stdout_path,
    ProcessTreeMemoryTracker *memory_tracker);

}  // namespace campp_pipeline

#endif  // CAMPP_NATIVE_PIPELINE_PROCESS_RUNNER_H_
