#include "json_lite.h"
#include "process_runner.h"
#include "integrity/sha256.h"

#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace fs = std::filesystem;

namespace campp_pipeline {
namespace {

constexpr int kEmbeddingDimension = 192;
constexpr int kSampleRate = 16000;
constexpr int kFbankBins = 80;

struct Options {
    std::string microphone_version;
    std::string speaker_embedding;
    int bucket_frames = 0;
    int countdown_seconds = 3;
    int warmup = 0;
    int repeat = 1;
    int threads = 1;
    bool dry_run = false;
    bool show_version = false;
    std::optional<fs::path> pipeline_root;
    std::optional<fs::path> microphone_config;
    std::optional<fs::path> runtime_binary;
    std::optional<fs::path> fbank_binary;
    std::optional<fs::path> asset_manifest;
    std::optional<fs::path> input_wav;
};

struct MicrophoneProfile {
    std::string version;
    std::string device;
};

struct AssetSet {
    std::string mode;
    int bucket_frames = 0;
    int audio_seconds = 0;
    fs::path runtime;
    fs::path frontend;
    fs::path application;
    fs::path model;
    fs::path plan;
    fs::path weights;
    fs::path schedule;
    std::string runtime_sha256;
    std::string frontend_sha256;
    std::string application_sha256;
    std::string model_sha256;
    std::string plan_sha256;
    std::string weights_sha256;
    std::string schedule_sha256;
};

struct RuntimeMetrics {
    double latency_mean_ms = 0.0;
    double rtf = 0.0;
    uint64_t runtime_peak_rss_bytes = 0;
    uint64_t weight_bytes = 0;
    uint64_t activation_bytes = 0;
};

int AudioSecondsForBucket(int bucket) {
    switch (bucket) {
        case 98: return 1;
        case 298: return 3;
        case 498: return 5;
        case 998: return 10;
        default: throw std::runtime_error("bucket must be 98, 298, 498, or 998");
    }
}

int ParseInteger(const std::string &value, const std::string &name) {
    size_t consumed = 0;
    long parsed = 0;
    try {
        parsed = std::stol(value, &consumed, 10);
    } catch (const std::exception &) {
        throw std::runtime_error("invalid " + name + ": " + value);
    }
    if (consumed != value.size() || parsed < 0 || parsed > INT32_MAX) {
        throw std::runtime_error("invalid " + name + ": " + value);
    }
    return static_cast<int>(parsed);
}

Options ParseOptions(int argc, char **argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string name = argv[index];
        if (name == "--dry-run") {
            options.dry_run = true;
            continue;
        }
        if (name == "--version") {
            options.show_version = true;
            continue;
        }
        if (name == "--help" || name == "-h") {
            std::cout
                << "usage: campp_speaker_verify --mic-version NAME "
                << "--speaker-embedding NAME --bucket 98|298|498|998 "
                << "[--countdown N] [--warmup N] [--repeat N] [--threads N] "
                << "[--input-wav WAV] [--dry-run]\n";
            std::exit(0);
        }
        if (index + 1 >= argc) {
            throw std::runtime_error("missing value for option: " + name);
        }
        const std::string value = argv[++index];
        if (name == "--mic-version") {
            options.microphone_version = value;
        } else if (name == "--speaker-embedding" || name == "--enrollment") {
            options.speaker_embedding = value;
        } else if (name == "--bucket") {
            options.bucket_frames = ParseInteger(value, "bucket");
        } else if (name == "--countdown") {
            options.countdown_seconds = ParseInteger(value, "countdown");
        } else if (name == "--warmup") {
            options.warmup = ParseInteger(value, "warmup");
        } else if (name == "--repeat") {
            options.repeat = ParseInteger(value, "repeat");
        } else if (name == "--threads") {
            options.threads = ParseInteger(value, "threads");
        } else if (name == "--pipeline-root") {
            options.pipeline_root = fs::path(value);
        } else if (name == "--microphone-config") {
            options.microphone_config = fs::path(value);
        } else if (name == "--runtime") {
            options.runtime_binary = fs::path(value);
        } else if (name == "--fbank") {
            options.fbank_binary = fs::path(value);
        } else if (name == "--asset-manifest") {
            options.asset_manifest = fs::path(value);
        } else if (name == "--input-wav") {
            options.input_wav = fs::path(value);
        } else {
            throw std::runtime_error("unknown option: " + name);
        }
    }
    if (options.show_version) {
        return options;
    }
    if (options.microphone_version.empty() ||
        options.speaker_embedding.empty() || options.bucket_frames == 0) {
        throw std::runtime_error(
            "--mic-version, --speaker-embedding, and --bucket are required");
    }
    AudioSecondsForBucket(options.bucket_frames);
    if (options.repeat <= 0 || options.threads <= 0) {
        throw std::runtime_error("repeat and threads must be positive");
    }
    return options;
}

fs::path ExecutablePath(const char *argv0) {
    std::error_code error;
    const fs::path proc = fs::read_symlink("/proc/self/exe", error);
    if (!error) {
        return fs::canonical(proc);
    }
    return fs::weakly_canonical(fs::absolute(argv0));
}

fs::path DetectPipelineRoot(
    const Options &options, const fs::path &executable) {
    if (options.pipeline_root.has_value()) {
        return fs::weakly_canonical(options.pipeline_root.value());
    }
    if (executable.parent_path().filename() == "runtime") {
        return executable.parent_path().parent_path();
    }
    const fs::path current = fs::current_path();
    if (fs::is_regular_file(current / "configs/microphones.json")) {
        return fs::weakly_canonical(current);
    }
    throw std::runtime_error(
        "cannot detect runtime_cam_pipeline; pass --pipeline-root");
}

bool IsWithin(const fs::path &boundary, const fs::path &candidate) {
    const fs::path relative = candidate.lexically_relative(boundary);
    if (relative.empty()) {
        return candidate == boundary;
    }
    return *relative.begin() != "..";
}

fs::path ResolvePipelinePath(
    const fs::path &pipeline_root, const fs::path &value,
    bool require_file = true) {
    const fs::path combined = value.is_absolute() ? value : pipeline_root / value;
    const fs::path resolved = fs::weakly_canonical(combined);
    if (!IsWithin(pipeline_root, resolved)) {
        throw std::runtime_error(
            "path escapes runtime_cam_pipeline: " + resolved.string());
    }
    if (require_file && !fs::is_regular_file(resolved)) {
        throw std::runtime_error("required file is missing: " + resolved.string());
    }
    return resolved;
}

fs::path ResolveManifestPath(
    const fs::path &pipeline_root, const fs::path &manifest,
    const std::string &value) {
    const fs::path path(value);
    return ResolvePipelinePath(
        pipeline_root,
        path.is_absolute() ? path : manifest.parent_path() / path);
}

MicrophoneProfile LoadMicrophone(
    const fs::path &config, const std::string &version) {
    const std::string document = ReadTextFile(config);
    const JsonRange root = JsonRoot(document);
    const JsonRange microphones = JsonObject(document, "microphones", root);
    const JsonRange profile = JsonObject(document, version, microphones);
    if (JsonString(document, "backend", profile) != "alsa_arecord" ||
        JsonUnsigned(document, "sample_rate_hz", profile) != kSampleRate ||
        JsonUnsigned(document, "channels", profile) != 1 ||
        JsonString(document, "sample_format", profile) != "S16_LE") {
        throw std::runtime_error(
            "microphone must use ALSA, 16 kHz mono, and S16_LE");
    }
    return MicrophoneProfile{
        version, JsonString(document, "device", profile)};
}

AssetSet LoadAssets(
    const fs::path &pipeline_root, const fs::path &manifest, int bucket) {
    const std::string document = ReadTextFile(manifest);
    const JsonRange root = JsonRoot(document);
    if (JsonString(document, "format", root) !=
        "campp-runtime-pipeline-assets-v1") {
        throw std::runtime_error("unsupported runtime asset manifest");
    }
    if (JsonUnsigned(document, "schema_version", root) != 1u ||
        JsonString(document, "selection_key", root) != "bucket_frames") {
        throw std::runtime_error("incompatible runtime asset manifest schema");
    }
    AssetSet assets;
    assets.mode = JsonString(document, "mode", root);
    assets.bucket_frames = bucket;
    assets.audio_seconds = AudioSecondsForBucket(bucket);

    const JsonRange runtime_contract = JsonObject(document, "runtime", root);
    assets.runtime = ResolveManifestPath(
        pipeline_root, manifest,
        JsonString(document, "binary", runtime_contract));
    assets.runtime_sha256 = JsonString(
        document, "sha256", runtime_contract);
    const JsonRange frontend_contract = JsonObject(document, "frontend", root);
    assets.frontend = ResolveManifestPath(
        pipeline_root, manifest,
        JsonString(document, "binary", frontend_contract));
    assets.frontend_sha256 = JsonString(
        document, "sha256", frontend_contract);
    if (JsonString(document, "backend", frontend_contract) !=
            "kaldi-native-fbank" ||
        JsonBoolean(document, "torch_required", frontend_contract)) {
        throw std::runtime_error("frontend contract is incompatible");
    }
    const JsonRange application_contract = JsonObject(
        document, "application", root);
    assets.application = ResolveManifestPath(
        pipeline_root, manifest,
        JsonString(document, "binary", application_contract));
    assets.application_sha256 = JsonString(
        document, "sha256", application_contract);
    if (JsonString(document, "backend", application_contract) !=
            "native-cpp-orchestrator" ||
        JsonBoolean(document, "python_required", application_contract) ||
        JsonBoolean(document, "numpy_required", application_contract)) {
        throw std::runtime_error("application contract is incompatible");
    }

    const JsonRange buckets = JsonObject(document, "buckets", root);
    const JsonRange row = JsonObject(document, std::to_string(bucket), buckets);
    if (JsonUnsigned(document, "audio_seconds", row) !=
        static_cast<uint64_t>(assets.audio_seconds)) {
        throw std::runtime_error("bucket audio duration is inconsistent");
    }
    if (assets.mode == "package") {
        assets.model = ResolveManifestPath(
            pipeline_root, manifest, JsonString(document, "model", row));
        assets.model_sha256 = JsonString(document, "sha256", row);
        const std::string suffix = "_" + std::to_string(bucket);
        const std::string stem = assets.model.stem().string();
        if (stem.size() < suffix.size() ||
            stem.compare(stem.size() - suffix.size(), suffix.size(), suffix) != 0) {
            throw std::runtime_error("model package name does not match bucket");
        }
    } else if (assets.mode == "windowed") {
        assets.plan = ResolveManifestPath(
            pipeline_root, manifest, JsonString(document, "plan", row));
        assets.weights = ResolveManifestPath(
            pipeline_root, manifest, JsonString(document, "weights", row));
        assets.schedule = ResolveManifestPath(
            pipeline_root, manifest, JsonString(document, "schedule", row));
        assets.plan_sha256 = JsonString(document, "plan_sha256", row);
        assets.weights_sha256 = JsonString(document, "weights_sha256", row);
        assets.schedule_sha256 = JsonString(document, "schedule_sha256", row);
        const std::string suffix = "_" + std::to_string(bucket);
        const std::vector<fs::path> paths{
            assets.plan, assets.weights, assets.schedule};
        for (const fs::path &path : paths) {
            const std::string stem = path.stem().string();
            if (stem.size() < suffix.size() || stem.compare(
                    stem.size() - suffix.size(), suffix.size(), suffix) != 0) {
                throw std::runtime_error(
                    "windowed asset name does not match bucket");
            }
            if (path.parent_path() != assets.plan.parent_path()) {
                throw std::runtime_error(
                    "windowed bucket assets must share one directory");
            }
        }
    } else {
        throw std::runtime_error("unsupported runtime mode: " + assets.mode);
    }
    return assets;
}

void RequireSameFile(
    const fs::path &actual, const fs::path &declared,
    const std::string &label) {
    if (fs::canonical(actual) != fs::canonical(declared)) {
        throw std::runtime_error(
            label + " does not match runtime/assets.json: " + actual.string());
    }
}

void RequireAssetChecksum(
    const fs::path &path, const std::string &expected,
    const std::string &label) {
    const CamppStatus status = campp_validate_file_sha256_hex(
        path.string().c_str(), expected.c_str());
    if (status != CAMPP_STATUS_OK) {
        throw std::runtime_error(
            label + " SHA-256 validation failed (" +
            campp_status_name(status) + "): " + path.string());
    }
}

void ValidateDeploymentAssets(
    const AssetSet &assets, const fs::path &runtime,
    const fs::path &frontend, const fs::path &application) {
    RequireSameFile(runtime, assets.runtime, "runtime binary");
    RequireSameFile(frontend, assets.frontend, "frontend binary");
    RequireSameFile(application, assets.application, "application binary");
    RequireAssetChecksum(
        assets.runtime, assets.runtime_sha256, "runtime binary");
    RequireAssetChecksum(
        assets.frontend, assets.frontend_sha256, "frontend binary");
    RequireAssetChecksum(
        assets.application, assets.application_sha256, "application binary");
    if (assets.mode == "package") {
        RequireAssetChecksum(
            assets.model, assets.model_sha256, "model package");
    } else {
        RequireAssetChecksum(assets.plan, assets.plan_sha256, "execution plan");
        RequireAssetChecksum(
            assets.weights, assets.weights_sha256, "packed weights");
        RequireAssetChecksum(
            assets.schedule, assets.schedule_sha256, "weight schedule");
    }
}

fs::path ResolveSpeakerEmbedding(
    const fs::path &pipeline_root, const std::string &value) {
    const fs::path direct = ResolvePipelinePath(
        pipeline_root, fs::path(value), false);
    std::vector<fs::path> candidates{
        direct,
        pipeline_root / "voice/embedded" / value,
        pipeline_root / "voice/embedded" / value / "mean_embedding.f32",
    };
    for (const fs::path &candidate : candidates) {
        const fs::path resolved = fs::weakly_canonical(candidate);
        if (IsWithin(pipeline_root, resolved) && fs::is_regular_file(resolved)) {
            if (resolved.extension() != ".f32") {
                throw std::runtime_error(
                    "native verification accepts raw .f32 embeddings only");
            }
            return resolved;
        }
    }
    throw std::runtime_error("cannot resolve speaker embedding: " + value);
}

bool ValidateEnrollmentContract(
    const fs::path &pipeline_root, const fs::path &template_path) {
    const fs::path metadata_path = template_path.parent_path() / "enrollment.json";
    if (template_path.filename() != "mean_embedding.f32" ||
        !fs::is_regular_file(metadata_path)) {
        return false;
    }
    const std::string document = ReadTextFile(metadata_path);
    const JsonRange root = JsonRoot(document);
    if (JsonUnsigned(document, "schema_version", root) != 1u ||
        JsonUnsigned(document, "recording_count", root) != 5u ||
        JsonUnsigned(document, "recording_seconds", root) != 10u ||
        JsonUnsigned(document, "bucket_frames", root) != 998u ||
        JsonUnsigned(document, "embedding_dimension", root) !=
            static_cast<uint64_t>(kEmbeddingDimension) ||
        JsonString(document, "aggregation", root) !=
            "mean_of_l2_normalized_then_l2_normalize") {
        throw std::runtime_error("speaker enrollment contract is incompatible");
    }
    const fs::path declared = ResolvePipelinePath(
        pipeline_root, JsonString(document, "mean_embedding", root));
    RequireSameFile(template_path, declared, "speaker enrollment template");
    RequireAssetChecksum(
        template_path,
        JsonString(document, "mean_embedding_sha256", root),
        "speaker enrollment template");
    return true;
}

std::vector<float> ReadEmbedding(const fs::path &path) {
    if (fs::file_size(path) != kEmbeddingDimension * sizeof(float)) {
        throw std::runtime_error(
            "embedding must contain exactly 192 float32 values: " +
            path.string());
    }
    std::ifstream input(path, std::ios::binary);
    std::vector<float> values(kEmbeddingDimension);
    if (!input.read(
            reinterpret_cast<char *>(values.data()),
            static_cast<std::streamsize>(values.size() * sizeof(float)))) {
        throw std::runtime_error("cannot read embedding: " + path.string());
    }
    for (const float value : values) {
        if (!std::isfinite(value)) {
            throw std::runtime_error("embedding contains non-finite values");
        }
    }
    return values;
}

double CosineSimilarity(
    const std::vector<float> &left, const std::vector<float> &right) {
    if (left.size() != right.size() || left.size() != kEmbeddingDimension) {
        throw std::runtime_error("embedding dimensions do not match");
    }
    double dot = 0.0;
    double left_square = 0.0;
    double right_square = 0.0;
    for (size_t index = 0; index < left.size(); ++index) {
        dot += static_cast<double>(left[index]) * right[index];
        left_square += static_cast<double>(left[index]) * left[index];
        right_square += static_cast<double>(right[index]) * right[index];
    }
    if (left_square <= 1.0e-24 || right_square <= 1.0e-24) {
        throw std::runtime_error("embedding norm is zero");
    }
    return std::clamp(dot / std::sqrt(left_square * right_square), -1.0, 1.0);
}

void RequireProcessSuccess(
    const ProcessResult &result, const std::string &name) {
    if (result.exit_code != 0 || result.signal_number != 0) {
        throw std::runtime_error(
            name + " failed: exit=" + std::to_string(result.exit_code) +
            " signal=" + std::to_string(result.signal_number));
    }
}

void ValidateCapabilities(
    const fs::path &runtime, const fs::path &fbank, int bucket,
    const std::string &mode, const fs::path &scratch,
    ProcessTreeMemoryTracker *memory) {
    const fs::path fbank_json = scratch / "fbank_capabilities.json";
    RequireProcessSuccess(
        RunProcess({fbank.string(), "--version"}, &fbank_json, memory),
        "native FBank capability check");
    const std::string fbank_document = ReadTextFile(fbank_json);
    const JsonRange fbank_root = JsonRoot(fbank_document);
    if (JsonString(fbank_document, "frontend", fbank_root) !=
            "campp-kaldi-native-fbank" ||
        JsonUnsigned(fbank_document, "api_version", fbank_root) != 1) {
        throw std::runtime_error("native FBank ABI is incompatible");
    }

    const fs::path runtime_json = scratch / "runtime_capabilities.json";
    RequireProcessSuccess(
        RunProcess({runtime.string(), "--capabilities"}, &runtime_json, memory),
        "C runtime capability check");
    const std::string runtime_document = ReadTextFile(runtime_json);
    const JsonRange root = JsonRoot(runtime_document);
    if (JsonString(runtime_document, "runtime", root) != "campp-c-runtime" ||
        JsonUnsigned(runtime_document, "effective_threads", root) != 1u ||
        JsonString(runtime_document, "optimization_suite", root) != "final") {
        throw std::runtime_error("C runtime does not contain the final suite");
    }
    if (mode == "package" && JsonString(
            runtime_document, "model_package_format", root) !=
            "camppmodel-v1") {
        throw std::runtime_error("C runtime cannot load camppmodel-v1");
    }
    const std::vector<double> plans = JsonNumberArray(
        runtime_document, "optimization_bucket_plans", root);
    if (std::find(plans.begin(), plans.end(), static_cast<double>(bucket)) ==
        plans.end()) {
        throw std::runtime_error("C runtime has no V3 plan for selected bucket");
    }
    if (mode == "windowed") {
        const std::vector<std::string> modes = JsonStringArray(
            runtime_document, "weight_residency_modes", root);
        if (std::find(modes.begin(), modes.end(), "windowed") == modes.end()) {
            throw std::runtime_error("C runtime has no windowed weight support");
        }
    }
}

RuntimeMetrics ParseRuntimeMetrics(
    const std::string &document, int bucket, int audio_seconds,
    const std::string &expected_mode) {
    const JsonRange root = JsonRoot(document);
    if (JsonString(document, "optimization_bucket_policy", root) !=
        "layer_hybrid_v3") {
        throw std::runtime_error("C runtime did not activate V3 hybrid policy");
    }
    const JsonRange configuration = JsonObject(document, "configuration", root);
    const std::string weight_mode = JsonString(
        document, "weight_mode", configuration);
    if (weight_mode != (expected_mode == "package" ? "malloc" : "windowed")) {
        throw std::runtime_error("C runtime activated the wrong weight mode");
    }
    const JsonRange model = JsonObject(document, "model", root);
    if (JsonUnsigned(document, "bucket_frames", model) !=
        static_cast<uint64_t>(bucket)) {
        throw std::runtime_error("C runtime loaded a different bucket");
    }
    const JsonRange warm = JsonObject(document, "warm", root);
    const std::vector<double> timings = JsonNumberArray(
        document, "timings_ms", warm);
    if (timings.empty()) {
        throw std::runtime_error("C runtime returned no timing samples");
    }
    const JsonRange memory = JsonObject(document, "memory", root);
    const JsonRange after = JsonObject(document, "after_measurement", memory);
    RuntimeMetrics metrics;
    metrics.latency_mean_ms = std::accumulate(
        timings.begin(), timings.end(), 0.0) / timings.size();
    metrics.rtf = metrics.latency_mean_ms / (audio_seconds * 1000.0);
    metrics.runtime_peak_rss_bytes = JsonUnsigned(
        document, "peak_rss_bytes", after);
    metrics.weight_bytes = JsonUnsigned(document, "weight_bytes", model);
    metrics.activation_bytes = JsonUnsigned(
        document, "activation_bytes", memory);
    return metrics;
}

std::string UniqueRunName() {
    const auto microseconds = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    return "native_" + std::to_string(microseconds) + "_" +
        std::to_string(getpid());
}

std::string Mib(uint64_t bytes) {
    std::ostringstream output;
    output << std::fixed << std::setprecision(2)
           << bytes / (1024.0 * 1024.0) << " MiB";
    return output.str();
}

std::string BuildReportJson(
    double score, int bucket, int audio_seconds,
    uint64_t pipeline_peak, uint64_t host_peak,
    const RuntimeMetrics &metrics, const fs::path &template_path,
    const fs::path &wav_path, const fs::path &feature_path,
    const fs::path &embedding_path, const AssetSet &assets,
    bool enrollment_sha256_verified) {
    std::ostringstream output;
    output << std::setprecision(9)
           << "{\n  \"schema_version\": 2,\n"
           << "  \"backend\": \"native-cpp-orchestrator\",\n"
           << "  \"final_score\": " << score << ",\n"
           << "  \"threshold\": null,\n"
           << "  \"threshold_status\": \"not_calibrated\",\n"
           << "  \"bucket_frames\": " << bucket << ",\n"
           << "  \"audio_seconds\": " << audio_seconds << ",\n"
           << "  \"speaker_embedding\": \""
           << JsonEscape(template_path.string()) << "\",\n"
           << "  \"runtime_mode\": \"" << JsonEscape(assets.mode) << "\",\n"
           << "  \"asset_integrity\": \"sha256_verified\",\n"
           << "  \"enrollment_sha256_verified\": "
           << (enrollment_sha256_verified ? "true" : "false") << ",\n"
           << "  \"frontend\": {\"backend\": \"kaldi-native-fbank\", "
              "\"torch_required\": false},\n"
           << "  \"memory\": {\n"
           << "    \"pipeline_total_peak_rss_bytes\": " << pipeline_peak << ",\n"
           << "    \"native_host_peak_rss_bytes\": " << host_peak << ",\n"
           << "    \"c_runtime_peak_rss_bytes\": "
           << metrics.runtime_peak_rss_bytes << ",\n"
           << "    \"logical_weight_bytes\": " << metrics.weight_bytes << ",\n"
           << "    \"logical_activation_bytes\": "
           << metrics.activation_bytes << "\n  },\n"
           << "  \"latency_mean_ms\": " << metrics.latency_mean_ms << ",\n"
           << "  \"rtf\": " << metrics.rtf << ",\n"
           << "  \"artifacts\": {\n"
           << "    \"wav\": \"" << JsonEscape(wav_path.string()) << "\",\n"
           << "    \"feature\": \"" << JsonEscape(feature_path.string())
           << "\",\n"
           << "    \"query_embedding\": \""
           << JsonEscape(embedding_path.string()) << "\"\n  }\n}\n";
    return output.str();
}

}  // namespace
}  // namespace campp_pipeline

int main(int argc, char **argv) {
    using namespace campp_pipeline;
    try {
        const Options options = ParseOptions(argc, argv);
        if (options.show_version) {
            std::cout
                << "{\"application\":\"campp_speaker_verify\","
                << "\"api_version\":1,\"python_required\":false}\n";
            return 0;
        }
        const fs::path executable = ExecutablePath(argv[0]);
        const fs::path pipeline_root = DetectPipelineRoot(options, executable);
        const fs::path microphone_config = ResolvePipelinePath(
            pipeline_root,
            options.microphone_config.value_or("configs/microphones.json"));
        const fs::path runtime = ResolvePipelinePath(
            pipeline_root,
            options.runtime_binary.value_or("runtime/campp_runtime"));
        const fs::path fbank = ResolvePipelinePath(
            pipeline_root,
            options.fbank_binary.value_or("runtime/campp_fbank"));
        const fs::path manifest = ResolvePipelinePath(
            pipeline_root,
            options.asset_manifest.value_or("runtime/assets.json"));
        const MicrophoneProfile microphone = LoadMicrophone(
            microphone_config, options.microphone_version);
        const AssetSet assets = LoadAssets(
            pipeline_root, manifest, options.bucket_frames);
        ValidateDeploymentAssets(assets, runtime, fbank, executable);
        const fs::path template_path = ResolveSpeakerEmbedding(
            pipeline_root, options.speaker_embedding);
        const bool enrollment_sha256_verified = ValidateEnrollmentContract(
            pipeline_root, template_path);

        ProcessTreeMemoryTracker memory;
        const fs::path scratch = pipeline_root / "runs" /
            (".native_preflight_" + UniqueRunName());
        fs::create_directories(scratch);
        try {
            ValidateCapabilities(
                runtime, fbank, options.bucket_frames, assets.mode,
                scratch, &memory);
            fs::remove_all(scratch);
        } catch (...) {
            fs::remove_all(scratch);
            throw;
        }

        if (options.dry_run) {
            std::cout
                << "{\"ready\":true,\"backend\":\"native-cpp-orchestrator\","
                << "\"python_required\":false,\"bucket_frames\":"
                << options.bucket_frames << ",\"audio_seconds\":"
                << assets.audio_seconds << ",\"runtime_mode\":\""
                << JsonEscape(assets.mode) << "\",\"speaker_embedding\":\""
                << JsonEscape(template_path.string())
                << "\",\"asset_sha256_verified\":true,"
                << "\"enrollment_sha256_verified\":"
                << (enrollment_sha256_verified ? "true" : "false") << "}\n";
            return 0;
        }

        const fs::path run_root = pipeline_root / "runs/inference" /
            UniqueRunName();
        fs::create_directories(run_root);
        const fs::path wav_path = run_root /
            ("query__" + std::to_string(options.bucket_frames) + ".wav");
        const fs::path feature_path = run_root /
            ("query__" + std::to_string(options.bucket_frames) + ".f32");
        const fs::path embedding_path = run_root /
            ("query_embedding__" + std::to_string(options.bucket_frames) +
             ".f32");
        const fs::path fbank_json = run_root / "fbank.json";
        const fs::path runtime_json = run_root / "runtime.json";
        const fs::path report_path = run_root / "report.json";

        if (options.input_wav.has_value()) {
            const fs::path source = ResolvePipelinePath(
                pipeline_root, options.input_wav.value());
            fs::copy_file(source, wav_path, fs::copy_options::overwrite_existing);
        } else {
            std::cout << "Recording " << assets.audio_seconds
                      << " seconds with microphone '" << microphone.version
                      << "'...\n" << std::flush;
            for (int remaining = options.countdown_seconds;
                 remaining > 0; --remaining) {
                std::cout << "  recording starts in " << remaining << "...\n"
                          << std::flush;
                std::this_thread::sleep_for(std::chrono::seconds(1));
            }
            RequireProcessSuccess(
                RunProcess({
                    "arecord", "-q", "-D", microphone.device,
                    "-t", "wav", "-f", "S16_LE", "-r", "16000",
                    "-c", "1", "-d", std::to_string(assets.audio_seconds),
                    wav_path.string(),
                }, nullptr, &memory),
                "microphone recording");
        }

        RequireProcessSuccess(
            RunProcess({
                fbank.string(),
                "--input", wav_path.string(),
                "--output", feature_path.string(),
                "--expected-frames", std::to_string(options.bucket_frames),
                "--audio-seconds", std::to_string(assets.audio_seconds),
            }, &fbank_json, &memory),
            "native FBank");
        if (fs::file_size(feature_path) !=
            static_cast<uintmax_t>(options.bucket_frames * kFbankBins * 4)) {
            throw std::runtime_error("native FBank produced an invalid feature file");
        }

        std::vector<std::string> runtime_command{runtime.string()};
        if (assets.mode == "package") {
            runtime_command.insert(runtime_command.end(), {
                "--model", assets.model.string()});
        } else {
            runtime_command.insert(runtime_command.end(), {
                "--plan", assets.plan.string(),
                "--weights", assets.weights.string(),
                "--weight-mode", "windowed",
                "--weight-schedule", assets.schedule.string(),
            });
        }
        runtime_command.insert(runtime_command.end(), {
            "--input", feature_path.string(),
            "--audio-seconds", std::to_string(assets.audio_seconds),
            "--warmup", std::to_string(options.warmup),
            "--repeat", std::to_string(options.repeat),
            "--threads", std::to_string(options.threads),
            "--embedding-output", embedding_path.string(),
        });
        RequireProcessSuccess(
            RunProcess(runtime_command, &runtime_json, &memory), "C runtime");

        const std::string runtime_document = ReadTextFile(runtime_json);
        const RuntimeMetrics metrics = ParseRuntimeMetrics(
            runtime_document, options.bucket_frames, assets.audio_seconds,
            assets.mode);
        const double score = CosineSimilarity(
            ReadEmbedding(template_path), ReadEmbedding(embedding_path));
        memory.Sample();
        const uint64_t pipeline_peak = memory.PipelinePeakRssBytes();
        const uint64_t host_peak = memory.HostPeakRssBytes();
        WriteTextFile(
            report_path,
            BuildReportJson(
                score, options.bucket_frames, assets.audio_seconds,
                pipeline_peak, host_peak, metrics, template_path,
                wav_path, feature_path, embedding_path, assets,
                enrollment_sha256_verified));

        std::cout << std::fixed << std::setprecision(6)
                  << "=================================\n"
                  << "=================================\n"
                  << "=====                         ======\n"
                  << "=====      final score: " << score << "      ======\n"
                  << "=====                         ======\n"
                  << "=================================\n"
                  << "=================================\n"
                  << "[report]\n"
                  << "Backend: native-cpp-orchestrator + campp-c-runtime\n"
                  << "Peak RAM\n"
                  << "- pipeline total peak: " << Mib(pipeline_peak) << "\n"
                  << "- native host peak: " << Mib(host_peak) << "\n"
                  << "- C runtime peak: "
                  << Mib(metrics.runtime_peak_rss_bytes) << "\n"
                  << "- logical weight: " << Mib(metrics.weight_bytes) << "\n"
                  << "- logical activation: "
                  << Mib(metrics.activation_bytes) << "\n"
                  << "Integrity: deployment assets SHA-256 verified\n"
                  << "RTF: " << metrics.rtf << "\n"
                  << "report JSON: " << report_path << "\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "speaker verification failed: " << error.what() << "\n";
        return 1;
    }
}
