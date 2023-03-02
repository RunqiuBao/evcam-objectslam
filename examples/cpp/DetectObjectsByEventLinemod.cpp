#include <Python.h>
#include <argparser.h>

#include <logging.h>
TDO_LOGGER("examples.DetectObjectsByEventLinemod")


int main(int argc, char** argv){
    /**
     *  argv:
     *    - inputdata: path to the event database
     *    - eventdatabaseformat: format of the event database
     *    - templatepath: path to the template images
     **/
    ArgumentParser argparser(argc, argv);
    // log4cxx::BasicConfigurator::configure();
    tooldetectobject::ConfigureRootLogger("DEBUG", "", "./detector.log");

    TDO_LOG_DEBUG("-------- start of the detector! --------");

    Py_Initialize();
    // PyRun_SimpleString("from time import time, ctime\n"
    //                     "print('Today is', ctime(time()))\n");
    Py_Finalize();
    return 0;
}
