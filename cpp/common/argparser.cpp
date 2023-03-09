#include "argparser.h"

TDO_LOGGER("tdo.common.argparser")


ArgumentParser::ArgumentParser(int& argc, char** argv, const std::vector<std::string> optionLabels){
    TDO_LOG_INFO("Arguments: \n");
    for (int i=0; i < argc; i++){
        TDO_LOG_INFO_FORMAT("%s: %s", optionLabels[i]%std::string(argv[i]));
        this->tokens.push_back(std::string(argv[i]));
    }
}

const std::string& ArgumentParser::getCmdOption(const std::string& option) const{  // const at the end means this function does not change object states
    std::vector<std::string>::const_iterator itr;
    itr = std::find(this->tokens.begin(), this->tokens.end(), option);
    if (itr != this->tokens.end() && ++itr != this->tokens.end()){
        return *itr;
    }
    static const std::string emptyString("");
    return emptyString;
}

const int ArgumentParser::getCmdOptionIndex(const std::string& option) const{
    std::vector<std::string>::const_iterator itr;
    itr = std::find(this->tokens.begin(), this->tokens.end(), option);
    if (itr == this->tokens.end())
        return -1;
    return std::distance(this->tokens.begin(), itr);
}