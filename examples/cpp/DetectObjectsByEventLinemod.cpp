#include <logging.h>
TDO_LOGGER("examples.DetectObjectsByEventLinemod")


int main(int argc, char* argv[]){
    // log4cxx::BasicConfigurator::configure();
    tooldetectobject::ConfigureRootLogger("DEBUG", "", "./myapp.log");

    TDO_LOG_DEBUG("Hellom, world!");
    TDO_LOG_DEBUG_FORMAT("shallom! %s", "aloha");
    return 0;
}
