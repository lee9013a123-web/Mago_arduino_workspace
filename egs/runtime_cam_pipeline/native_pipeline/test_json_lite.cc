#include "json_lite.h"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

int main() {
    using namespace campp_pipeline;
    try {
        const std::string document =
            "{\"mode\":\"package\",\"buckets\":{\"298\":{"
            "\"audio_seconds\":3,\"model\":\"models/model.camppmodel\"}},"
            "\"timings_ms\":[1.25,2.75],\"enabled\":false,"
            "\"modes\":[\"malloc\",\"windowed\"]}";
        const JsonRange root = JsonRoot(document);
        if (JsonString(document, "mode", root) != "package") {
            throw std::runtime_error("string parse failed");
        }
        const JsonRange buckets = JsonObject(document, "buckets", root);
        const JsonRange row = JsonObject(document, "298", buckets);
        if (JsonUnsigned(document, "audio_seconds", row) != 3 ||
            JsonString(document, "model", row) != "models/model.camppmodel") {
            throw std::runtime_error("nested object parse failed");
        }
        const std::vector<double> timings = JsonNumberArray(
            document, "timings_ms", root);
        if (timings.size() != 2 || std::abs(timings[0] - 1.25) > 1.0e-12 ||
            std::abs(timings[1] - 2.75) > 1.0e-12) {
            throw std::runtime_error("number array parse failed");
        }
        const std::vector<std::string> modes = JsonStringArray(
            document, "modes", root);
        if (JsonBoolean(document, "enabled", root) || modes.size() != 2 ||
            modes[1] != "windowed") {
            throw std::runtime_error("boolean/string array parse failed");
        }
        if (JsonEscape("a\n\"b") != "a\\n\\\"b") {
            throw std::runtime_error("JSON escape failed");
        }
        const std::string shadowed =
            "{\"nested\":{\"mode\":\"wrong\"},\"mode\":\"package\"}";
        if (JsonString(shadowed, "mode", JsonRoot(shadowed)) != "package") {
            throw std::runtime_error("nested key shadowed a direct member");
        }
        bool duplicate_rejected = false;
        try {
            const std::string duplicate =
                "{\"mode\":\"package\",\"mode\":\"windowed\"}";
            (void)JsonString(duplicate, "mode", JsonRoot(duplicate));
        } catch (const std::exception &) {
            duplicate_rejected = true;
        }
        if (!duplicate_rejected) {
            throw std::runtime_error("duplicate direct key was accepted");
        }
        bool invalid_number_rejected = false;
        try {
            const std::string invalid = "{\"schema_version\":1garbage}";
            (void)JsonUnsigned(invalid, "schema_version", JsonRoot(invalid));
        } catch (const std::exception &) {
            invalid_number_rejected = true;
        }
        if (!invalid_number_rejected) {
            throw std::runtime_error("invalid numeric token was accepted");
        }
        std::cout << "PASS test_json_lite\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "FAIL test_json_lite: " << error.what() << "\n";
        return 1;
    }
}
