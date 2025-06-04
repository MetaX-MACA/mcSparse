#ifndef UTILS_LOGGER_H_
#define UTILS_LOGGER_H_

#include <chrono>
#include <ctime>
#include <fstream>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

#include "common/mcsp_types.h"
#ifdef __cplusplus
extern "C" {
#endif

typedef void (*mcsparseLoggerCallback_t)(int logLevel, const char *functionName, const char *message);

mcsparseStatus_t mcsparseLoggerSetCallback(mcsparseLoggerCallback_t callback);
mcsparseStatus_t mcsparseLoggerSetFile(FILE *file);
mcsparseStatus_t mcsparseLoggerOpenFile(const char *logFile);
mcsparseStatus_t mcsparseLoggerSetLevel(int level);
mcsparseStatus_t mcsparseLoggerSetMask(int mask);
mcsparseStatus_t mcsparseLoggerForceDisable(void);

#ifdef __cplusplus
}
#endif

// Log control env variables
#define SPARSE_LOG_ENABLE "SPARSE_LOG_ENABLE"
#define SPARSE_LOG_LEVEL "SPARSE_LOG_LEVEL"
#define SPARSE_LOG_OUTPUT "SPARSE_LOG_OUTPUT"

/**
 * @brief Set log control when you are debuging and want to run locally.
 */
#define LOG_RESET_TMP(enable, level, output, file) \
    mcsparse::utils::Logger::GetLogger().Reset(mcsparse::utils::level, mcsparse::utils::output, file)

/**
 * @brief Recover log control after LOG_RESET_TMP
 */
#define LOG_SET_RECOVER() mcsparse::utils::Logger::GetLogger().Recover()

/**
 * @brief  Log format
 * LOG_F(LOG_INFO, "%s\n", "enjoy");
 * Recommended api is LOG_INFO("%s\n", dflakjf);
 */
#define LOG_F(level, fmt, ...)                                                                                 \
    do {                                                                                                       \
        if (mcsparse::utils::Logger::Filter(mcsparse::utils::level)) {                                         \
            mcsparse::utils::Logger::GetLogger().Log(mcsparse::utils::level, "%s:%d | " fmt,                   \
                                                     mcsparse::utils::basename(std::string(__FILE__)).c_str(), \
                                                     __LINE__, ##__VA_ARGS__);                                 \
        }                                                                                                      \
    } while (false)

/**
 * @brief Logger api for different levels
 */
#define LOG_FATAL(fmt, ...) LOG_F(LOG_FATAL, fmt, ##__VA_ARGS__)
#define LOG_WARN(fmt, ...) LOG_F(LOG_WARNING, fmt, ##__VA_ARGS__)
#define LOG_ERR(fmt, ...) LOG_F(LOG_ERROR, fmt, ##__VA_ARGS__)
#define LOG_INFO(fmt, ...) LOG_F(LOG_INFO, fmt, ##__VA_ARGS__)
#define LOG_DEBUG(fmt, ...) LOG_F(LOG_DEBUG, fmt, ##__VA_ARGS__)

/**
 * @brief  Simplified log format
 * Use LOG_FS when you don't need header and timestamp, for example in a for loop
 */
#define LOG_FS(level, fmt, ...)                                                                         \
    do {                                                                                                \
        if (mcsparse::utils::Logger::Filter(mcsparse::utils::level)) {                                  \
            mcsparse::utils::Logger::GetLogger().LogSimple(mcsparse::utils::level, fmt, ##__VA_ARGS__); \
        }                                                                                               \
    } while (false)

/**
 * @brief Simplified logger api for different levels
 */
#define LOG_FS_FATAL(fmt, ...) LOG_FS(LOG_FATAL, fmt, ##__VA_ARGS__)
#define LOG_FS_WARN(fmt, ...) LOG_FS(LOG_WARNING, fmt, ##__VA_ARGS__)
#define LOG_FS_ERR(fmt, ...) LOG_FS(LOG_ERROR, fmt, ##__VA_ARGS__)
#define LOG_FS_INFO(fmt, ...) LOG_FS(LOG_INFO, fmt, ##__VA_ARGS__)
#define LOG_FS_DEBUG(fmt, ...) LOG_FS(LOG_DEBUG, fmt, ##__VA_ARGS__)

/**
 * @brief  Log stream
 *
 *  LOG_S(INFO) << a << b << "\n";
 *  LOG_S(WARNING) << a << b << "\n";
 */
#define LOG_S(level) mcsparse::utils::StreamLogger(mcsparse::utils::LOG_##level, __FILE__, __LINE__)

namespace mcsparse {
namespace utils {

enum LogLevel : uint32_t {
    LOG_FATAL = 0,  // fatal error occurs, which may lead to process or system crash
    LOG_ERROR,      // errors may lead to function working abnormally
    LOG_WARNING,    // functions is still working, but may have pool tolerance
    LOG_INFO,       // simple information to indicate process is working normally
    LOG_DEBUG,      // detailed information for debug only
};

enum LogOutput : uint32_t {
    LOG_STDOUT = 0,
    LOG_FILE,
    LOG_SYSLOG,
};

std::string basename(const std::string &in);

class Logger {
 public:
    Logger();
    void Reset(int level, int output, const std::string &filename) {
        old_enable_ = log_enable_;
        old_level_ = log_level_;
        old_output_ = log_output_;

        log_enable_ = true;
        log_level_ = level;
        log_output_ = output;
        if (file_ostream_.is_open()) {
            file_ostream_.close();
        }
        filename_ = filename;
        ostream_ = output == LOG_STDOUT ? &std::cout : &file_ostream_;
    }

    void Recover() {
        log_enable_ = old_enable_;
        log_level_ = old_level_;
        log_output_ = old_output_;
    }

    static Logger &GetLogger();
    static bool Filter(int level) { return GetLogger().log_enable_ && (level <= GetLogger().log_level_); }

    void Log(int level, const char *fmt, ...);
    void LogSimple(int level, const char *fmt, ...);

 private:
    using LoggerClock = std::chrono::high_resolution_clock;
    static uint64_t MicroSeconds() {
        std::chrono::duration<uint64_t, std::micro> ts =
            std::chrono::duration_cast<std::chrono::microseconds>(LoggerClock::now().time_since_epoch());
        return ts.count();
    }
    void IntervalCheck();

    static uint32_t TimeStr(char *buf, uint32_t bufsize);
    static uint32_t FmtLogHeader(char *buf, uint32_t bufsize, int level);
    static uint32_t FmtLogHeaderSimple(char *buf, uint32_t bufsize, int level);

    uint32_t log_level_ = LOG_INFO;
    uint32_t old_level_ = LOG_INFO;
    uint32_t log_output_ = LOG_STDOUT;
    uint32_t old_output_ = LOG_STDOUT;
    bool log_enable_ = false;
    bool old_enable_ = false;

    std::ofstream file_ostream_;
    std::ostream *ostream_;
    std::mutex mutex_;
    std::string filename_ = "mcsparse.txt";
    uint32_t counter_ = 0;
    static constexpr int kCheckInterval = 1000;
    static const char *LogLevelMap[];
};

class StreamLogger {
 public:
    StreamLogger(int level, const char *filename, int linenum)
        : level_(level), filename_(filename), linenum_(linenum) {}
    ~StreamLogger() { Flush(); }
    void Flush() {
        if (Filter()) {
            Logger::GetLogger().Log(level_, "%s:%d | %s", basename(filename_).c_str(), linenum_, ss_.str().c_str());
            ss_.clear();
        }
    }

    template <class T>
    StreamLogger &operator<<(const T &v) {
        if (Filter()) {
            ss_ << v;
        }
        return *this;
    }

    StreamLogger &operator<<(std::ostream &(*f)(std::ostream &)) {
        if (Filter()) {
            f(ss_);
        }
        return *this;
    }

 private:
    bool Filter() const { return Logger::Filter(level_); }
    int level_;
    std::string filename_;
    int linenum_;
    std::stringstream ss_;
};

}  // namespace utils
}  // namespace mcsparse

#endif  // UTILS_LOGGER_H_
