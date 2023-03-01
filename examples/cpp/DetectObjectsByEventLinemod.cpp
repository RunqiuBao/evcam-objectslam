#include <Python.h>

#include <logging.h>
TDO_LOGGER("examples.DetectObjectsByEventLinemod")


int main(int argc, char* argv[]){
    // log4cxx::BasicConfigurator::configure();
    tooldetectobject::ConfigureRootLogger("DEBUG", "", "./myapp.log");

    TDO_LOG_DEBUG("Hellom, world!");
    TDO_LOG_DEBUG_FORMAT("shallom! %s", "aloha");

    Py_Initialize();
    PyRun_SimpleString("from time import time, ctime\n"
                        "print('Today is', ctime(time()))\n");
    Py_Finalize();
    return 0;
}
