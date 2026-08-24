#ifndef CAMPP_NATIVE_PIPELINE_JSON_LITE_H_
#define CAMPP_NATIVE_PIPELINE_JSON_LITE_H_

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace campp_pipeline {

struct JsonRange {
    size_t begin = 0;
    size_t end = 0;
};

std::string ReadTextFile(const std::filesystem::path &path);
void WriteTextFile(const std::filesystem::path &path, const std::string &value);

JsonRange JsonRoot(const std::string &document);
JsonRange JsonObject(
    const std::string &document, const std::string &key,
    JsonRange range = JsonRange{});
std::string JsonString(
    const std::string &document, const std::string &key,
    JsonRange range = JsonRange{});
bool JsonBoolean(
    const std::string &document, const std::string &key,
    JsonRange range = JsonRange{});
uint64_t JsonUnsigned(
    const std::string &document, const std::string &key,
    JsonRange range = JsonRange{});
double JsonNumber(
    const std::string &document, const std::string &key,
    JsonRange range = JsonRange{});
std::vector<double> JsonNumberArray(
    const std::string &document, const std::string &key,
    JsonRange range = JsonRange{});
std::vector<std::string> JsonStringArray(
    const std::string &document, const std::string &key,
    JsonRange range = JsonRange{});

std::string JsonEscape(const std::string &value);

}  // namespace campp_pipeline

#endif  // CAMPP_NATIVE_PIPELINE_JSON_LITE_H_
