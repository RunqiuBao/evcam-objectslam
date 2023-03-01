#include "logging.h"
#include <cstdlib>
#include <iostream>
#include <algorithm>
#include <stdlib.h>
#include <syslog.h>

#define COLOR_BLACK 0
#define COLOR_RED 1
#define COLOR_GREEN 2
#define COLOR_YELLOW 3
#define COLOR_BLUE 4
#define COLOR_MAGNETA 5
#define COLOR_CYAN 6
#define COLOR_WHITE 7


namespace log4cxx
{

class ColorLayout : public Layout {
public:
    DECLARE_LOG4CXX_OBJECT(ColorLayout)
    BEGIN_LOG4CXX_CAST_MAP()
        LOG4CXX_CAST_ENTRY(ColorLayout)
        LOG4CXX_CAST_ENTRY_CHAIN(Layout)
    END_LOG4CXX_CAST_MAP()

    ColorLayout();
    ColorLayout(const LayoutPtr& layout);
    virtual ~ColorLayout();

    virtual void activateOptions(helpers::Pool& p) { _layout->activateOptions(p); }
    virtual void setOption(const LogString& option, const LogString& value) { _layout->setOption(option, value); }
    virtual bool ignoresThrowable() const { return _layout->ignoresThrowable(); }

    virtual void format(LogString& output, const spi::LoggingEventPtr& event, helpers::Pool& pool) const;

protected:
    virtual LogString _Colorize(const spi::LoggingEventPtr& event) const;

    LayoutPtr _layout;
};

}

using namespace log4cxx;
using namespace log4cxx::net;

IMPLEMENT_LOG4CXX_OBJECT(ColorLayout);

ColorLayout::ColorLayout(): Layout()
{
}

ColorLayout::ColorLayout(const LayoutPtr& layout): Layout(), _layout(layout)
{
}

ColorLayout::~ColorLayout()
{
}

void ColorLayout::format(LogString& output, const spi::LoggingEventPtr& event, helpers::Pool& pool) const
{
    _layout->format(output, event, pool);

    // add color
    output.reserve(output.size() + 32);
    output.insert(0, _Colorize(event));
    output.append(LOG4CXX_STR("\x1b[0m"));
}

LogString ColorLayout::_Colorize(const spi::LoggingEventPtr& event) const
{
    int bg = -1;
    int fg = -1;
    bool bold = false;
    LogString csi;

    csi.reserve(32);

    if (event->getLevel()->isGreaterOrEqual(Level::getFatal())) {
        bg = COLOR_WHITE;
        fg = COLOR_MAGNETA;
        bold = true;
    } else if (event->getLevel()->isGreaterOrEqual(Level::getError())) {
        fg = COLOR_RED;
    } else if (event->getLevel()->isGreaterOrEqual(Level::getWarn())) {
        fg = COLOR_YELLOW;
    } else if (event->getLevel()->isGreaterOrEqual(Level::getInfo())) {
    } else if (event->getLevel()->isGreaterOrEqual(Level::getDebug())) {
        fg = COLOR_GREEN;
    } else {
        fg = COLOR_BLUE;
    }

    csi += LOG4CXX_STR("\x1b[0");

    if (bg >= 0) {
        csi += LOG4CXX_STR(';');
        csi += LOG4CXX_STR('4');
        csi += LOG4CXX_STR('0') + bg;
    }

    if (fg >= 0) {
        csi += LOG4CXX_STR(';');
        csi += LOG4CXX_STR('3');
        csi += LOG4CXX_STR('0') + fg;
    }

    if (bold) {
        csi += LOG4CXX_STR(';');
        csi += LOG4CXX_STR('1');
    }

    csi += LOG4CXX_STR('m');

    return csi;
}

static std::string _GetLogLevelName(const log4cxx::LevelPtr& level) {
    std::string levelName;
    if (!level) { 
    }
    else if (level == log4cxx::Level::getFatal()) {
        levelName = "CRITICAL";
    }
    else if (level == log4cxx::Level::getError()) {
        levelName = "ERROR";
    }
    else if (level == log4cxx::Level::getWarn()) {
        levelName = "WARNING";
    }
    else if (level == log4cxx::Level::getInfo()) {
        levelName = "INFO";
    }
    else if (level == log4cxx::Level::getDebug()) {
        levelName = "DEBUG";
    }
    else if (level == log4cxx::Level::getTrace()) {
        levelName = "VERBOSE";
    }
    return levelName;
}

static log4cxx::LevelPtr _GetLogLevel(const std::string& levelName) {
    LevelPtr level;
    std::string levelNameUpper(levelName);
    std::transform(levelNameUpper.begin(), levelNameUpper.end(), levelNameUpper.begin(), ::toupper);
    if (levelNameUpper.empty()) {
    }
    else if (levelNameUpper == "CRITICAL") {
        level = log4cxx::Level::getFatal();
    }
    else if (levelNameUpper == "ERROR") {
        level = log4cxx::Level::getError();
    }
    else if (levelNameUpper == "WARN" || levelNameUpper == "WARNING") {
        level = log4cxx::Level::getWarn();
    }
    else if (levelNameUpper == "INFO") {
        level = log4cxx::Level::getInfo();
    }
    else if (levelNameUpper == "DEBUG") {
        level = log4cxx::Level::getDebug();
    }
    else if (levelNameUpper == "VERBOSE") {
        level = log4cxx::Level::getTrace();
    }
    return level;
}

void tooldetectobject::ConfigureRootLogger(const std::string& level, const std::string& outputformat, const std::string& logFilePath)
{
    //
    // Parse log level.
    //

    std::string loglevel(level);
    if (loglevel.empty() && std::getenv("TDO_LOG_LEVEL") != NULL) {
        loglevel = std::getenv("TDO_LOG_LEVEL");
    }

    LevelPtr loglevelptr = _GetLogLevel(loglevel);
    if (!loglevelptr) {
        loglevelptr = Level::getDebug();
    }

    //
    // Root logger.
    //

    const char *outputformatstr = "%d %c [%p] [%F:%L %M] %m%n";
    if (outputformat.size() > 0) {
        outputformatstr = outputformat.c_str();
    }

    LayoutPtr patternLayout(new PatternLayout(LOG4CXX_STR(outputformatstr)));
    LayoutPtr colorLayout(new ColorLayout(patternLayout));
    ConsoleAppenderPtr consoleAppender(new ConsoleAppender(colorLayout, ConsoleAppender::getSystemErr()));
    File logFile(logFilePath);
    FileAppenderPtr rollingFileAppender(new FileAppender(colorLayout, logFile.getPath()));
    // rollingFileAppender->setMaxFileSize("5MB");
    // rollingFileAppender->setMaxBackupIndex(5);
    // rollingFileAppender->setLayout(colorLayout);
    // rollingFileAppender->setFile(logFilePath);
    // rollingFileAppender->setAppend(true);

    log4cxx::LoggerPtr root(log4cxx::Logger::getRootLogger());
    root->setLevel(loglevelptr);
    root->removeAllAppenders();
    root->addAppender(consoleAppender);
    root->addAppender(rollingFileAppender);
}