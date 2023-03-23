
#ifndef TDO_ARGPARSER_H
#define TDO_ARGPARSER_H

#include <vector>
#include <string>
#include <algorithm>

#include "logging.h"

class ArgumentParser {

public:
    ArgumentParser(int& argc, char** argv, const std::vector<std::string> optionLabels);

    const std::string& getCmdOption(const std::string& option) const;

    const int getCmdOptionIndex(const std::string& option) const;

    bool cmdOptionExists(const std::string &option) const {
        return std::find(this->_keys.begin(), this->_keys.end(), option) != this->_keys.end();
    }

private:
    std::vector<std::string> _values;
    std::vector<std::string> _keys;
};

#endif
