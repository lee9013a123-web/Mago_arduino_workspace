#include "json_lite.h"

#include <cctype>
#include <charconv>
#include <fstream>
#include <iomanip>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <system_error>

namespace campp_pipeline {
namespace {

JsonRange NormalizeRange(const std::string &document, JsonRange range) {
    if (range.end == 0 && range.begin == 0) {
        return JsonRange{0, document.size()};
    }
    if (range.begin > range.end || range.end > document.size()) {
        throw std::runtime_error("invalid JSON range");
    }
    return range;
}

size_t SkipWhitespace(
    const std::string &document, size_t offset, size_t end) {
    while (offset < end && std::isspace(
            static_cast<unsigned char>(document[offset])) != 0) {
        ++offset;
    }
    return offset;
}

std::string ParseString(
    const std::string &document, size_t begin, size_t end);

size_t SkipString(
    const std::string &document, size_t begin, size_t end) {
    if (begin >= end || document[begin] != '"') {
        throw std::runtime_error("JSON value is not a string");
    }
    bool escaped = false;
    for (size_t cursor = begin + 1; cursor < end; ++cursor) {
        const char value = document[cursor];
        if (escaped) {
            escaped = false;
        } else if (value == '\\') {
            escaped = true;
        } else if (value == '"') {
            return cursor + 1;
        }
    }
    throw std::runtime_error("unterminated JSON string");
}

size_t SkipComposite(
    const std::string &document, size_t begin, size_t end) {
    std::vector<char> expected;
    expected.push_back(document[begin] == '{' ? '}' : ']');
    bool in_string = false;
    bool escaped = false;
    for (size_t cursor = begin + 1; cursor < end; ++cursor) {
        const char value = document[cursor];
        if (in_string) {
            if (escaped) {
                escaped = false;
            } else if (value == '\\') {
                escaped = true;
            } else if (value == '"') {
                in_string = false;
            }
            continue;
        }
        if (value == '"') {
            in_string = true;
        } else if (value == '{') {
            expected.push_back('}');
        } else if (value == '[') {
            expected.push_back(']');
        } else if (value == '}' || value == ']') {
            if (expected.empty() || expected.back() != value) {
                throw std::runtime_error("mismatched JSON delimiter");
            }
            expected.pop_back();
            if (expected.empty()) {
                return cursor + 1;
            }
        }
    }
    throw std::runtime_error("unterminated JSON composite value");
}

size_t SkipValue(
    const std::string &document, size_t begin, size_t end) {
    begin = SkipWhitespace(document, begin, end);
    if (begin >= end) {
        throw std::runtime_error("missing JSON value");
    }
    if (document[begin] == '"') {
        return SkipString(document, begin, end);
    }
    if (document[begin] == '{' || document[begin] == '[') {
        return SkipComposite(document, begin, end);
    }
    size_t cursor = begin;
    while (cursor < end && document[cursor] != ',' &&
           document[cursor] != '}' && document[cursor] != ']') {
        ++cursor;
    }
    size_t value_end = cursor;
    while (value_end > begin && std::isspace(
            static_cast<unsigned char>(document[value_end - 1])) != 0) {
        --value_end;
    }
    if (value_end == begin) {
        throw std::runtime_error("missing JSON scalar value");
    }
    return value_end;
}

/* Only direct members of range are considered; nested duplicate names do not
 * satisfy a parent contract. Duplicate direct members are rejected. */
size_t FindValue(
    const std::string &document, const std::string &key, JsonRange range) {
    range = NormalizeRange(document, range);
    size_t cursor = SkipWhitespace(document, range.begin, range.end);
    if (cursor >= range.end || document[cursor] != '{') {
        throw std::runtime_error("JSON lookup range is not an object");
    }
    ++cursor;
    std::optional<size_t> match;
    for (;;) {
        cursor = SkipWhitespace(document, cursor, range.end);
        if (cursor >= range.end) {
            throw std::runtime_error("unterminated JSON object");
        }
        if (document[cursor] == '}') {
            break;
        }
        if (document[cursor] != '"') {
            throw std::runtime_error("JSON object member has no string key");
        }
        const std::string member = ParseString(document, cursor, range.end);
        cursor = SkipString(document, cursor, range.end);
        cursor = SkipWhitespace(document, cursor, range.end);
        if (cursor >= range.end || document[cursor] != ':') {
            throw std::runtime_error("JSON object member has no colon");
        }
        const size_t value = SkipWhitespace(document, cursor + 1, range.end);
        const size_t value_end = SkipValue(document, value, range.end);
        if (member == key) {
            if (match.has_value()) {
                throw std::runtime_error("duplicate JSON key: " + key);
            }
            match = value;
        }
        cursor = SkipWhitespace(document, value_end, range.end);
        if (cursor >= range.end) {
            throw std::runtime_error("unterminated JSON object");
        }
        if (document[cursor] == ',') {
            ++cursor;
            const size_t next = SkipWhitespace(document, cursor, range.end);
            if (next >= range.end || document[next] == '}') {
                throw std::runtime_error("trailing comma in JSON object");
            }
            continue;
        }
        if (document[cursor] == '}') {
            break;
        }
        throw std::runtime_error("invalid JSON object separator");
    }
    if (!match.has_value()) {
        throw std::runtime_error("JSON key is missing: " + key);
    }
    return match.value();
}

JsonRange BalancedObject(
    const std::string &document, size_t begin, size_t end) {
    if (begin >= end || document[begin] != '{') {
        throw std::runtime_error("JSON value is not an object");
    }
    int depth = 0;
    bool in_string = false;
    bool escaped = false;
    for (size_t cursor = begin; cursor < end; ++cursor) {
        const char value = document[cursor];
        if (in_string) {
            if (escaped) {
                escaped = false;
            } else if (value == '\\') {
                escaped = true;
            } else if (value == '"') {
                in_string = false;
            }
            continue;
        }
        if (value == '"') {
            in_string = true;
        } else if (value == '{') {
            ++depth;
        } else if (value == '}') {
            --depth;
            if (depth == 0) {
                return JsonRange{begin, cursor + 1};
            }
        }
    }
    throw std::runtime_error("unterminated JSON object");
}

std::string ParseString(
    const std::string &document, size_t begin, size_t end) {
    if (begin >= end || document[begin] != '"') {
        throw std::runtime_error("JSON value is not a string");
    }
    std::string output;
    for (size_t cursor = begin + 1; cursor < end; ++cursor) {
        const char value = document[cursor];
        if (value == '"') {
            return output;
        }
        if (value != '\\') {
            output.push_back(value);
            continue;
        }
        if (++cursor >= end) {
            break;
        }
        switch (document[cursor]) {
            case '"': output.push_back('"'); break;
            case '\\': output.push_back('\\'); break;
            case '/': output.push_back('/'); break;
            case 'b': output.push_back('\b'); break;
            case 'f': output.push_back('\f'); break;
            case 'n': output.push_back('\n'); break;
            case 'r': output.push_back('\r'); break;
            case 't': output.push_back('\t'); break;
            default:
                throw std::runtime_error("unsupported JSON string escape");
        }
    }
    throw std::runtime_error("unterminated JSON string");
}

double ParseNumberAt(
    const std::string &document, size_t begin, size_t end,
    size_t *consumed = nullptr) {
    size_t cursor = begin;
    if (cursor < end && document[cursor] == '-') {
        ++cursor;
    }
    if (cursor >= end) {
        throw std::runtime_error("JSON value is not numeric");
    }
    if (document[cursor] == '0') {
        ++cursor;
        if (cursor < end && std::isdigit(
                static_cast<unsigned char>(document[cursor])) != 0) {
            throw std::runtime_error("JSON number has a leading zero");
        }
    } else if (document[cursor] >= '1' && document[cursor] <= '9') {
        while (cursor < end && std::isdigit(
                static_cast<unsigned char>(document[cursor])) != 0) {
            ++cursor;
        }
    } else {
        throw std::runtime_error("JSON value is not numeric");
    }
    if (cursor < end && document[cursor] == '.') {
        ++cursor;
        const size_t fraction = cursor;
        while (cursor < end && std::isdigit(
                static_cast<unsigned char>(document[cursor])) != 0) {
            ++cursor;
        }
        if (cursor == fraction) {
            throw std::runtime_error("JSON number has an empty fraction");
        }
    }
    if (cursor < end && (document[cursor] == 'e' ||
                         document[cursor] == 'E')) {
        ++cursor;
        if (cursor < end && (document[cursor] == '+' ||
                             document[cursor] == '-')) {
            ++cursor;
        }
        const size_t exponent = cursor;
        while (cursor < end && std::isdigit(
                static_cast<unsigned char>(document[cursor])) != 0) {
            ++cursor;
        }
        if (cursor == exponent) {
            throw std::runtime_error("JSON number has an empty exponent");
        }
    }
    size_t parsed = 0;
    const double value = std::stod(document.substr(begin, cursor - begin), &parsed);
    if (parsed != cursor - begin) {
        throw std::runtime_error("invalid JSON number");
    }
    if (consumed != nullptr) {
        *consumed = cursor - begin;
    }
    return value;
}

}  // namespace

std::string ReadTextFile(const std::filesystem::path &path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("cannot read file: " + path.string());
    }
    std::ostringstream output;
    output << input.rdbuf();
    if (!input.good() && !input.eof()) {
        throw std::runtime_error("cannot finish reading file: " + path.string());
    }
    return output.str();
}

void WriteTextFile(const std::filesystem::path &path, const std::string &value) {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output || !output.write(
            value.data(), static_cast<std::streamsize>(value.size()))) {
        throw std::runtime_error("cannot write file: " + path.string());
    }
}

JsonRange JsonRoot(const std::string &document) {
    const size_t begin = SkipWhitespace(document, 0, document.size());
    const JsonRange root = BalancedObject(document, begin, document.size());
    if (SkipWhitespace(document, root.end, document.size()) != document.size()) {
        throw std::runtime_error("trailing data after JSON root");
    }
    return root;
}

JsonRange JsonObject(
    const std::string &document, const std::string &key, JsonRange range) {
    range = NormalizeRange(document, range);
    return BalancedObject(document, FindValue(document, key, range), range.end);
}

std::string JsonString(
    const std::string &document, const std::string &key, JsonRange range) {
    range = NormalizeRange(document, range);
    return ParseString(document, FindValue(document, key, range), range.end);
}

bool JsonBoolean(
    const std::string &document, const std::string &key, JsonRange range) {
    range = NormalizeRange(document, range);
    const size_t value = FindValue(document, key, range);
    const size_t end = SkipValue(document, value, range.end);
    const std::string token = document.substr(value, end - value);
    if (token == "true") {
        return true;
    }
    if (token == "false") {
        return false;
    }
    throw std::runtime_error("JSON value is not boolean: " + key);
}

double JsonNumber(
    const std::string &document, const std::string &key, JsonRange range) {
    range = NormalizeRange(document, range);
    const size_t begin = FindValue(document, key, range);
    size_t consumed = 0;
    const double value = ParseNumberAt(
        document, begin, range.end, &consumed);
    if (begin + consumed != SkipValue(document, begin, range.end)) {
        throw std::runtime_error("invalid JSON number: " + key);
    }
    return value;
}

uint64_t JsonUnsigned(
    const std::string &document, const std::string &key, JsonRange range) {
    range = NormalizeRange(document, range);
    const size_t begin = FindValue(document, key, range);
    const size_t end = SkipValue(document, begin, range.end);
    uint64_t value = 0;
    const char *first = document.data() + begin;
    const char *last = document.data() + end;
    const auto parsed = std::from_chars(first, last, value, 10);
    if (parsed.ec != std::errc() || parsed.ptr != last || first == last) {
        throw std::runtime_error("JSON value is not an unsigned integer: " + key);
    }
    return value;
}

std::vector<double> JsonNumberArray(
    const std::string &document, const std::string &key, JsonRange range) {
    range = NormalizeRange(document, range);
    size_t cursor = FindValue(document, key, range);
    if (cursor >= range.end || document[cursor] != '[') {
        throw std::runtime_error("JSON value is not an array: " + key);
    }
    ++cursor;
    std::vector<double> values;
    while (cursor < range.end) {
        cursor = SkipWhitespace(document, cursor, range.end);
        if (cursor < range.end && document[cursor] == ']') {
            return values;
        }
        size_t consumed = 0;
        values.push_back(ParseNumberAt(document, cursor, range.end, &consumed));
        cursor += consumed;
        cursor = SkipWhitespace(document, cursor, range.end);
        if (cursor < range.end && document[cursor] == ',') {
            ++cursor;
            const size_t next = SkipWhitespace(document, cursor, range.end);
            if (next >= range.end || document[next] == ']') {
                throw std::runtime_error(
                    "trailing comma in JSON numeric array: " + key);
            }
            continue;
        }
        if (cursor < range.end && document[cursor] == ']') {
            return values;
        }
        throw std::runtime_error("invalid JSON numeric array: " + key);
    }
    throw std::runtime_error("unterminated JSON array: " + key);
}

std::vector<std::string> JsonStringArray(
    const std::string &document, const std::string &key, JsonRange range) {
    range = NormalizeRange(document, range);
    size_t cursor = FindValue(document, key, range);
    if (cursor >= range.end || document[cursor] != '[') {
        throw std::runtime_error("JSON value is not an array: " + key);
    }
    ++cursor;
    std::vector<std::string> values;
    while (cursor < range.end) {
        cursor = SkipWhitespace(document, cursor, range.end);
        if (cursor < range.end && document[cursor] == ']') {
            return values;
        }
        values.push_back(ParseString(document, cursor, range.end));
        cursor = SkipString(document, cursor, range.end);
        cursor = SkipWhitespace(document, cursor, range.end);
        if (cursor < range.end && document[cursor] == ',') {
            ++cursor;
            const size_t next = SkipWhitespace(document, cursor, range.end);
            if (next >= range.end || document[next] == ']') {
                throw std::runtime_error(
                    "trailing comma in JSON string array: " + key);
            }
            continue;
        }
        if (cursor < range.end && document[cursor] == ']') {
            return values;
        }
        throw std::runtime_error("invalid JSON string array: " + key);
    }
    throw std::runtime_error("unterminated JSON array: " + key);
}

std::string JsonEscape(const std::string &value) {
    std::ostringstream output;
    for (const unsigned char character : value) {
        switch (character) {
            case '"': output << "\\\""; break;
            case '\\': output << "\\\\"; break;
            case '\b': output << "\\b"; break;
            case '\f': output << "\\f"; break;
            case '\n': output << "\\n"; break;
            case '\r': output << "\\r"; break;
            case '\t': output << "\\t"; break;
            default:
                if (character < 0x20) {
                    output << "\\u" << std::hex << std::setw(4)
                           << std::setfill('0') << static_cast<int>(character)
                           << std::dec;
                } else {
                    output << static_cast<char>(character);
                }
        }
    }
    return output.str();
}

}  // namespace campp_pipeline
