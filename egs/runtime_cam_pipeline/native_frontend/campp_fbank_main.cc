// CAM++ fixed-bucket WAV to Kaldi-compatible FBank frontend.
//
// kaldi-native-fbank is Apache-2.0 software. See THIRD_PARTY.md and the
// dependency's LICENSE file. This wrapper is project code and deliberately
// keeps the deployment contract fixed to the CAM++ model frontend.

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "kaldi-native-fbank/csrc/feature-fbank.h"
#include "kaldi-native-fbank/csrc/online-feature.h"

namespace {

constexpr int32_t kSampleRate = 16000;
constexpr int32_t kMelBins = 80;
constexpr float kTargetPeak = 0.95f;

struct Options {
    std::string input_path;
    std::string output_path;
    int32_t expected_frames = 0;
    int32_t audio_seconds = 0;
};

uint16_t ReadU16(const uint8_t *value) {
    return static_cast<uint16_t>(value[0]) |
           (static_cast<uint16_t>(value[1]) << 8);
}

uint32_t ReadU32(const uint8_t *value) {
    return static_cast<uint32_t>(value[0]) |
           (static_cast<uint32_t>(value[1]) << 8) |
           (static_cast<uint32_t>(value[2]) << 16) |
           (static_cast<uint32_t>(value[3]) << 24);
}

bool Matches(const uint8_t *value, const char *text) {
    return std::memcmp(value, text, 4) == 0;
}

std::vector<uint8_t> ReadFile(const std::string &path) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) {
        throw std::runtime_error("cannot open input WAV: " + path);
    }
    const std::streamsize size = input.tellg();
    if (size < 0) {
        throw std::runtime_error("cannot determine WAV size: " + path);
    }
    input.seekg(0, std::ios::beg);
    std::vector<uint8_t> bytes(static_cast<size_t>(size));
    if (size != 0 && !input.read(
            reinterpret_cast<char *>(bytes.data()), size)) {
        throw std::runtime_error("cannot read input WAV: " + path);
    }
    return bytes;
}

std::vector<int16_t> ReadPcm16Mono(const std::string &path) {
    const std::vector<uint8_t> bytes = ReadFile(path);
    if (bytes.size() < 12 || !Matches(bytes.data(), "RIFF") ||
        !Matches(bytes.data() + 8, "WAVE")) {
        throw std::runtime_error("input is not a RIFF/WAVE file");
    }

    bool found_format = false;
    bool found_data = false;
    uint16_t audio_format = 0;
    uint16_t channels = 0;
    uint16_t bits_per_sample = 0;
    uint32_t sample_rate = 0;
    const uint8_t *data = nullptr;
    size_t data_size = 0;
    size_t offset = 12;
    while (offset + 8 <= bytes.size()) {
        const uint8_t *chunk = bytes.data() + offset;
        const size_t chunk_size = ReadU32(chunk + 4);
        const size_t payload = offset + 8;
        if (payload > bytes.size() || chunk_size > bytes.size() - payload) {
            throw std::runtime_error("WAV contains a truncated chunk");
        }
        if (Matches(chunk, "fmt ")) {
            if (chunk_size < 16) {
                throw std::runtime_error("WAV fmt chunk is too short");
            }
            audio_format = ReadU16(bytes.data() + payload);
            channels = ReadU16(bytes.data() + payload + 2);
            sample_rate = ReadU32(bytes.data() + payload + 4);
            bits_per_sample = ReadU16(bytes.data() + payload + 14);
            found_format = true;
        } else if (Matches(chunk, "data")) {
            data = bytes.data() + payload;
            data_size = chunk_size;
            found_data = true;
        }
        offset = payload + chunk_size + (chunk_size & 1u);
    }
    if (!found_format || !found_data) {
        throw std::runtime_error("WAV is missing fmt or data chunk");
    }
    if (audio_format != 1 || channels != 1 || sample_rate != kSampleRate ||
        bits_per_sample != 16 || (data_size & 1u) != 0) {
        throw std::runtime_error(
            "WAV must be PCM16 mono at exactly 16000 Hz");
    }

    std::vector<int16_t> samples(data_size / 2);
    for (size_t i = 0; i < samples.size(); ++i) {
        samples[i] = static_cast<int16_t>(ReadU16(data + i * 2));
    }
    return samples;
}

std::vector<float> NormalizeWaveform(
    std::vector<int16_t> samples, int32_t audio_seconds) {
    const size_t expected_samples =
        static_cast<size_t>(kSampleRate) * audio_seconds;
    samples.resize(expected_samples, 0);

    std::vector<float> waveform(samples.size());
    float sum = 0.0f;
    for (size_t i = 0; i < samples.size(); ++i) {
        waveform[i] = static_cast<float>(samples[i]) / 32768.0f;
        sum += waveform[i];
    }
    const float mean = waveform.empty()
        ? 0.0f : sum / static_cast<float>(waveform.size());
    float peak = 0.0f;
    for (float &value : waveform) {
        value -= mean;
        peak = std::max(peak, std::abs(value));
    }
    if (peak > 1.0e-8f) {
        const float scale = kTargetPeak / peak;
        for (float &value : waveform) {
            value = std::max(-1.0f, std::min(1.0f, value * scale));
        }
    }
    return waveform;
}

std::vector<float> ComputeFbank(
    const std::vector<float> &waveform, int32_t expected_frames) {
    knf::FbankOptions options;
    options.frame_opts.samp_freq = static_cast<float>(kSampleRate);
    options.frame_opts.frame_length_ms = 25.0f;
    options.frame_opts.frame_shift_ms = 10.0f;
    options.frame_opts.dither = 0.0f;
    options.frame_opts.preemph_coeff = 0.97f;
    options.frame_opts.remove_dc_offset = true;
    options.frame_opts.window_type = "hamming";
    options.frame_opts.round_to_power_of_two = true;
    options.frame_opts.snip_edges = true;
    options.mel_opts.num_bins = kMelBins;
    options.mel_opts.low_freq = 20.0f;
    options.mel_opts.high_freq = 0.0f;
    options.use_energy = false;
    options.use_log_fbank = true;
    options.use_power = true;

    knf::OnlineFbank fbank(options);
    fbank.AcceptWaveform(
        static_cast<float>(kSampleRate), waveform.data(),
        static_cast<int32_t>(waveform.size()));
    fbank.InputFinished();
    const int32_t frames = fbank.NumFramesReady();
    if (frames != expected_frames) {
        throw std::runtime_error(
            "native frontend produced " + std::to_string(frames) +
            " frames; expected " + std::to_string(expected_frames));
    }

    std::vector<float> output(
        static_cast<size_t>(frames) * static_cast<size_t>(kMelBins));
    std::vector<float> means(kMelBins, 0.0f);
    for (int32_t frame = 0; frame < frames; ++frame) {
        const float *source = fbank.GetFrame(frame);
        for (int32_t bin = 0; bin < kMelBins; ++bin) {
            const float value = source[bin];
            if (!std::isfinite(value)) {
                throw std::runtime_error("native frontend produced non-finite data");
            }
            output[static_cast<size_t>(frame) * kMelBins + bin] = value;
            means[bin] += value;
        }
    }
    for (float &mean : means) {
        mean /= static_cast<float>(frames);
    }
    for (int32_t frame = 0; frame < frames; ++frame) {
        for (int32_t bin = 0; bin < kMelBins; ++bin) {
            output[static_cast<size_t>(frame) * kMelBins + bin] -= means[bin];
        }
    }
    return output;
}

void WriteFeatures(const std::string &path, const std::vector<float> &features) {
    static_assert(sizeof(float) == 4, "CAM++ feature ABI requires float32");
    const uint16_t endian_probe = 1;
    if (*reinterpret_cast<const uint8_t *>(&endian_probe) != 1) {
        throw std::runtime_error("CAM++ feature ABI requires little-endian CPU");
    }
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output || !output.write(
            reinterpret_cast<const char *>(features.data()),
            static_cast<std::streamsize>(features.size() * sizeof(float)))) {
        throw std::runtime_error("cannot write FBank output: " + path);
    }
}

int32_t ParsePositiveInt(const std::string &text, const char *name) {
    size_t consumed = 0;
    long value = 0;
    try {
        value = std::stol(text, &consumed, 10);
    } catch (const std::exception &) {
        throw std::runtime_error(std::string("invalid ") + name);
    }
    if (consumed != text.size() || value <= 0 ||
        value > std::numeric_limits<int32_t>::max()) {
        throw std::runtime_error(std::string("invalid ") + name);
    }
    return static_cast<int32_t>(value);
}

Options ParseOptions(int argc, char **argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string argument = argv[i];
        if (argument == "--version") {
            std::cout
                << "{\"frontend\":\"campp-kaldi-native-fbank\","
                << "\"api_version\":1,\"kaldi_native_fbank\":\"1.22.3\","
                << "\"sample_rate\":16000,\"mel_bins\":80}\n";
            std::exit(0);
        }
        if (i + 1 >= argc) {
            throw std::runtime_error("missing value for " + argument);
        }
        const std::string value = argv[++i];
        if (argument == "--input") {
            options.input_path = value;
        } else if (argument == "--output") {
            options.output_path = value;
        } else if (argument == "--expected-frames") {
            options.expected_frames = ParsePositiveInt(value, "expected frames");
        } else if (argument == "--audio-seconds") {
            options.audio_seconds = ParsePositiveInt(value, "audio seconds");
        } else {
            throw std::runtime_error("unknown option: " + argument);
        }
    }
    if (options.input_path.empty() || options.output_path.empty() ||
        options.expected_frames <= 0 || options.audio_seconds <= 0) {
        throw std::runtime_error(
            "usage: campp_fbank --input WAV --output F32 "
            "--expected-frames N --audio-seconds N");
    }
    return options;
}

}  // namespace

int main(int argc, char **argv) {
    try {
        const Options options = ParseOptions(argc, argv);
        const std::vector<int16_t> samples = ReadPcm16Mono(options.input_path);
        const std::vector<float> waveform =
            NormalizeWaveform(samples, options.audio_seconds);
        const std::vector<float> features =
            ComputeFbank(waveform, options.expected_frames);
        WriteFeatures(options.output_path, features);
        std::cout
            << "{\"backend\":\"kaldi-native-fbank\",\"frames\":"
            << options.expected_frames << ",\"bins\":" << kMelBins
            << ",\"torch_required\":false}\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "campp_fbank failed: " << error.what() << "\n";
        return 1;
    }
}
