
#ifndef TDO_ARGPARSER_H
#define TDO_ARGPARSER_H

#include <vector>
#include <string>
#include <algorithm>

class ArgumentParser {

public:
    ArgumentParser (int& argc, char** argv){
        for (int i=1; i < argc; i++){
            this->tokens.push_back(std::string(argv[i]));
        }
    }

    const std::string& getCmdOption(const std::string& option) const{  // const at the end means this function does not change object states
        std::vector<std::string>::const_iterator itr;
        itr = std::find(this->tokens.begin(), this->tokens.end(), option);
        if (itr != this->tokens.end() && ++itr != this->tokens.end()){
            return *itr;
        }
        static const std::string emptyString("");
        return emptyString;
    }

    const int getCmdOptionIndex(const std::string& option) const{
        std::vector<std::string>::const_iterator itr;
        itr = std::find(this->tokens.begin(), this->tokens.end(), option);
        if (itr == this->tokens.end())
            return -1;
        return std::distance(this->tokens.begin(), itr);
    }

    bool cmdOptionExists(const std::string &option) const {
        return std::find(this->tokens.begin(), this->tokens.end(), option) != this->tokens.end();
    }

private:
    std::vector<std::string> tokens;
};

#endif