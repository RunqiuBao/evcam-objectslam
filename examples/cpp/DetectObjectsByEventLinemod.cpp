#include <logging.h>
TDO_LOGGER("examples.DetectObjectsByEventLinemod")


int main(int argc, char* argv[]){
    log4cxx::BasicConfigurator::configure();
    TDO_LOG_DEBUG("Hellom, world!");
    return 0;
}
