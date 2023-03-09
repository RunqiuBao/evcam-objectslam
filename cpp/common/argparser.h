
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
        return std::find(this->tokens.begin(), this->tokens.end(), option) != this->tokens.end();
    }

private:
    std::vector<std::string> tokens;
};

#endif