#include "argparser.h"

TDO_LOGGER("tdo.common.argparser")


ArgumentParser::ArgumentParser(int& argc, char** argv, const std::vector<std::string> optionLabels){
    TDO_LOG_INFO("Arguments: \n");
    for (int i=0; i < argc; i++){
        TDO_LOG_INFO_FORMAT("%s: %s", optionLabels[i]%std::string(argv[i]));
        this->_values.push_back(std::string(argv[i]));
        this->_keys.push_back(optionLabels[i]);
    }
}

const std::string& ArgumentParser::getCmdOption(const std::string& option) const{  // const at the end means this function does not change object states
    int indexOption = getCmdOptionIndex(option);
    if (indexOption >= 0){
        return this->_values[indexOption];
    }
    else{
        static const std::string emptyString("");
        return emptyString;
    }
}

const int ArgumentParser::getCmdOptionIndex(const std::string& option) const{
    std::vector<std::string>::const_iterator itr;
    itr = std::find(this->_keys.begin(), this->_keys.end(), option);
    if (itr == this->_keys.end())
        return -1;
    return std::distance(this->_keys.begin(), itr);
}