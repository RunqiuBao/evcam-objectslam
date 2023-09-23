#ifndef TDO_LOGGING_H
#define TDO_LOGGING_H

#include <log4cxx/logger.h>
#include <log4cxx/basicconfigurator.h>
#include <log4cxx/rolling/rollingfileappender.h>
#include <log4cxx/layout.h>
#include <log4cxx/file.h>
#include <log4cxx/patternlayout.h>
#include <log4cxx/consoleappender.h>
#include <log4cxx/net/syslogappender.h>
#include <log4cxx/spi/filter.h>
#include <log4cxx/logmanager.h>

#include <cstring>
#include <boost/format.hpp>

#define TDO_LOGGER(name) \
    static log4cxx::LoggerPtr logger = log4cxx::Logger::getLogger(name); \

#define TDO_GET_LOGGER() logger
#define __FILENAME__ (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)

#ifdef LOG4CXX_LOCATION
#undef LOG4CXX_LOCATION
#endif
#define LOG4CXX_LOCATION log4cxx::spi::LocationInfo(__FILENAME__, __FILENAME__, __PRETTY_FUNCTION__, __LINE__)


/**
Conditions to check if a log level has been enabled.
*/
#define TDO_LOG_CRITICAL_ENABLED (!!logger && logger->getEffectiveLevel()->toInt() <= log4cxx::Level::FATAL_INT)
#define TDO_LOG_ERROR_ENABLED    (!!logger && logger->getEffectiveLevel()->toInt() <= log4cxx::Level::ERROR_INT)
#define TDO_LOG_WARN_ENABLED     (!!logger && logger->getEffectiveLevel()->toInt() <= log4cxx::Level::WARN_INT)
#define TDO_LOG_INFO_ENABLED     (!!logger && logger->getEffectiveLevel()->toInt() <= log4cxx::Level::INFO_INT)
#define TDO_LOG_DEBUG_ENABLED    (!!logger && LOG4CXX_UNLIKELY(logger->getEffectiveLevel()->toInt() <= log4cxx::Level::DEBUG_INT))
#define TDO_LOG_VERBOSE_ENABLED  (!!logger && LOG4CXX_UNLIKELY(logger->getEffectiveLevel()->toInt() <= log4cxx::Level::TRACE_INT))


/**
Logs a message to a specified logger with the TRACE level.

@param logger the logger to be used.
@param message the message string to log.
*/
#define TDO_LOG_VERBOSE(message) LOG4CXX_TRACE(logger, message)
#define TDO_LOG_VERBOSE_FORMAT(fmt, params) { \
    if (TDO_LOG_VERBOSE_ENABLED) { \
        logger->forcedLog(log4cxx::Level::getTrace(), boost::str(boost::format(fmt)%params), LOG4CXX_LOCATION); }}


/**
Logs a message to a specified logger with the DEBUG level.

@param logger the logger to be used.
@param message the message string to log.
*/
#define TDO_LOG_DEBUG(message) LOG4CXX_DEBUG(logger, message)
#define TDO_LOG_DEBUG_FORMAT(fmt, params) { \
    if (TDO_LOG_DEBUG_ENABLED) { \
        logger->forcedLog(log4cxx::Level::getDebug(), boost::str(boost::format(fmt)%params), LOG4CXX_LOCATION); }}


/**
Logs a message to a specified logger with the INFO level.

@param logger the logger to be used.
@param message the message string to log.
*/
#define TDO_LOG_INFO(message) LOG4CXX_INFO(logger, message)
#define TDO_LOG_INFO_FORMAT(fmt, params) { \
    if (TDO_LOG_INFO_ENABLED) { \
        logger->forcedLog(log4cxx::Level::getInfo(), boost::str(boost::format(fmt)%params), LOG4CXX_LOCATION); }}


/**
Logs a message to a specified logger with the WARN level.

@param logger the logger to be used.
@param message the message string to log.
*/
#define TDO_LOG_WARN(message) LOG4CXX_WARN(logger, message)
#define TDO_LOG_WARN_FORMAT(fmt, params) { \
    if (TDO_LOG_WARN_ENABLED) { \
        logger->forcedLog(log4cxx::Level::getWarn(), boost::str(boost::format(fmt)%params), LOG4CXX_LOCATION); }}


/**
Logs a message to a specified logger with the ERROR level.

@param logger the logger to be used.
@param message the message string to log.
*/
#define TDO_LOG_ERROR(message) LOG4CXX_ERROR(logger, message)
#define TDO_LOG_ERROR_FORMAT(fmt, params) { \
    if (TDO_LOG_ERROR_ENABLED) { \
        logger->forcedLog(log4cxx::Level::getError(), boost::str(boost::format(fmt)%params), LOG4CXX_LOCATION); }}


/**
Logs a message to a specified logger with the FATAL level.

@param logger the logger to be used.
@param message the message string to log.
*/
#define TDO_LOG_CRITICAL(message) LOG4CXX_FATAL(logger, message)
#define TDO_LOG_CRITICAL_FORMAT(fmt, params) { \
    if (TDO_LOG_CRITICAL_ENABLED) { \
        logger->forcedLog(log4cxx::Level::getFatal(), boost::str(boost::format(fmt)%params), LOG4CXX_LOCATION); }}


void ConfigureRootLogger(const std::string& level, const std::string& outputformat, const std::string& logFilePath);

#endif
