#include <log4cxx/logger.h>
#include <log4cxx/basicconfigurator.h>

int main(int argc, char* argv[]){
    log4cxx::BasicConfigurator::configure();
    log4cxx::LoggerPtr logger(log4cxx::Logger::getLogger("logloglog"));

    LOG4CXX_DEBUG(logger, "Hello, wolrd!");

    return 0;
}