#include "process_runner.h"

#include <fcntl.h>
#include <signal.h>
#include <errno.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include <chrono>
#include <fstream>
#include <set>
#include <sstream>
#include <stdexcept>
#include <thread>

namespace campp_pipeline {
namespace {

uint64_t ReadStatusField(int pid, const std::string &name) {
    std::ifstream input(
        "/proc/" + std::to_string(pid) + "/status", std::ios::binary);
    if (!input) {
        return 0;
    }
    const std::string prefix = name + ":";
    std::string line;
    while (std::getline(input, line)) {
        if (line.rfind(prefix, 0) != 0) {
            continue;
        }
        std::istringstream fields(line.substr(prefix.size()));
        uint64_t kibibytes = 0;
        fields >> kibibytes;
        return fields ? kibibytes * 1024u : 0u;
    }
    return 0;
}

std::vector<int> ReadChildren(int pid) {
    std::ifstream input(
        "/proc/" + std::to_string(pid) + "/task/" +
        std::to_string(pid) + "/children",
        std::ios::binary);
    std::vector<int> children;
    int child = 0;
    while (input >> child) {
        if (child > 0) {
            children.push_back(child);
        }
    }
    return children;
}

std::vector<int> ProcessTree(int root_pid) {
    std::vector<int> pending{root_pid};
    std::set<int> seen;
    while (!pending.empty()) {
        const int pid = pending.back();
        pending.pop_back();
        if (!seen.insert(pid).second) {
            continue;
        }
        const std::vector<int> children = ReadChildren(pid);
        pending.insert(pending.end(), children.begin(), children.end());
    }
    return std::vector<int>(seen.begin(), seen.end());
}

std::vector<char *> BuildArgv(const std::vector<std::string> &arguments) {
    std::vector<char *> result;
    result.reserve(arguments.size() + 1);
    for (const std::string &argument : arguments) {
        result.push_back(const_cast<char *>(argument.c_str()));
    }
    result.push_back(nullptr);
    return result;
}

}  // namespace

ProcessTreeMemoryTracker::ProcessTreeMemoryTracker()
    : root_pid_(static_cast<int>(getpid())) {
    Sample();
}

void ProcessTreeMemoryTracker::Sample() {
    uint64_t total = 0;
    for (const int pid : ProcessTree(root_pid_)) {
        total += ReadStatusField(pid, "VmRSS");
    }
    if (total > pipeline_peak_rss_bytes_) {
        pipeline_peak_rss_bytes_ = total;
    }
}

uint64_t ProcessTreeMemoryTracker::PipelinePeakRssBytes() const {
    return pipeline_peak_rss_bytes_;
}

uint64_t ProcessTreeMemoryTracker::HostPeakRssBytes() const {
    return ReadStatusField(root_pid_, "VmHWM");
}

ProcessResult RunProcess(
    const std::vector<std::string> &arguments,
    const std::filesystem::path *stdout_path,
    ProcessTreeMemoryTracker *memory_tracker) {
    if (arguments.empty() || arguments[0].empty()) {
        throw std::runtime_error("cannot run an empty command");
    }
    const pid_t child = fork();
    if (child < 0) {
        throw std::runtime_error("fork failed for: " + arguments[0]);
    }
    if (child == 0) {
        if (stdout_path != nullptr) {
            const int output = open(
                stdout_path->c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0600);
            if (output < 0 || dup2(output, STDOUT_FILENO) < 0) {
                _exit(126);
            }
            close(output);
        }
        setenv("OMP_NUM_THREADS", "1", 1);
        setenv("ORT_NUM_THREADS", "1", 1);
        std::vector<char *> argv = BuildArgv(arguments);
        execvp(argv[0], argv.data());
        _exit(127);
    }

    int status = 0;
    while (true) {
        if (memory_tracker != nullptr) {
            memory_tracker->Sample();
        }
        const pid_t waited = waitpid(child, &status, WNOHANG);
        if (waited == child) {
            break;
        }
        if (waited < 0) {
            if (errno == EINTR) {
                continue;
            }
            throw std::runtime_error("waitpid failed for: " + arguments[0]);
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    if (memory_tracker != nullptr) {
        memory_tracker->Sample();
    }
    ProcessResult result;
    if (WIFEXITED(status)) {
        result.exit_code = WEXITSTATUS(status);
    } else if (WIFSIGNALED(status)) {
        result.signal_number = WTERMSIG(status);
    }
    return result;
}

}  // namespace campp_pipeline
