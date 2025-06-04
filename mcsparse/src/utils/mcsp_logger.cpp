#include "utils/mcsp_logger.h"

#include <cstdarg>
#include <cstring>
#include <mutex>
#include <ostream>
#include <sstream>
#include <string>
#include <vector>

namespace mcsparse {
namespace utils {

const char *Logger::LogLevelMap[] = {"FATAL", "ERR", "WARN", "INFO", "DBG"};

std::string basename(const std::string &in) {
    auto loc = in.find_last_of("/");
    if (loc == std::string::npos) {
        return in;
    }
    return in.substr(loc + 1);
}

Logger &Logger::GetLogger() {
    static Logger s_log;
    return s_log;
}

Logger::Logger() : ostream_(&file_ostream_) {
    char *env_log_enable = getenv(SPARSE_LOG_ENABLE);
    if (env_log_enable != nullptr) {
        log_enable_ = strtol(env_log_enable, nullptr, 10) != 0;
    } else {
        log_enable_ = true;
    }

    char *env_output = getenv(SPARSE_LOG_OUTPUT);
    if (env_output != nullptr) {
        if (strcmp(env_output, "stdout") == 0) {
            log_output_ = LOG_STDOUT;
            ostream_ = &std::cout;
        } else if (strcmp(env_output, "file") == 0) {
            log_output_ = LOG_FILE;
        } else if (strcmp(env_output, "syslog") == 0) {
            log_output_ = LOG_SYSLOG;
        }
    } else {
        log_output_ = LOG_STDOUT;
        ostream_ = &std::cout;
    }

    char *env_log_level = getenv(SPARSE_LOG_LEVEL);
    if (env_log_level != nullptr) {
        log_level_ = strtol(env_log_level, nullptr, 10);
        if (log_level_ > LOG_DEBUG) {
            log_level_ = LOG_DEBUG;
        }
    } else {
        log_level_ = LOG_INFO;
    }
}

uint32_t Logger::TimeStr(char *buf, uint32_t bufsize) {
    time_t t;
    struct tm *timeinfo;
    std::time(&t);
    timeinfo = std::localtime(&t);
    return std::strftime(buf, bufsize, "[%D %X %s]", timeinfo);
}

uint32_t Logger::FmtLogHeader(char *buf, uint32_t bufsize, int level) {
    uint32_t off = 0;
    off += snprintf(buf, bufsize, "[%s] ", LogLevelMap[level]);
    off += TimeStr(buf + off, bufsize - off);
    return off;
}

uint32_t Logger::FmtLogHeaderSimple(char *buf, uint32_t bufsize, int level) {
    uint32_t off = 0;
    off += snprintf(buf, bufsize, "[%s] ", LogLevelMap[level]);
    return off;
}

void Logger::IntervalCheck() {
    ++counter_;
    if (counter_ >= kCheckInterval) {
        counter_ = 0;
    }
}

void Logger::LogSimple(int level, const char *fmt, ...) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (log_output_ == LOG_FILE && !file_ostream_.is_open()) {
        file_ostream_.open(filename_.c_str());
    }

    if (log_output_ != LOG_SYSLOG) {
        char message[4096];
        uint32_t off = 0;

        off += FmtLogHeaderSimple(message, sizeof(message), level);

        va_list args;
        va_start(args, fmt);
        off += std::vsnprintf(message + off, sizeof(message) - off, fmt, args);
        va_end(args);

        *ostream_ << message;
    }

    IntervalCheck();
}

void Logger::Log(int level, const char *fmt, ...) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (log_output_ == LOG_FILE && !file_ostream_.is_open()) {
        file_ostream_.open(filename_.c_str());
    }

    if (log_output_ != LOG_SYSLOG) {
        char message[4096];
        uint32_t off = 0;

        off += FmtLogHeader(message, sizeof(message), level);

        va_list args;
        va_start(args, fmt);
        off += std::vsnprintf(message + off, sizeof(message) - off, fmt, args);
        va_end(args);

        *ostream_ << message;
    }

    IntervalCheck();
}

}  // namespace utils
}  // namespace mcsparse
