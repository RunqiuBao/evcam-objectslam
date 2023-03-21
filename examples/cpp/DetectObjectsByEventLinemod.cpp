#include <Python.h>
#include <Eigen/Core>

#include <argparser.h>
#include <pybindcommon.h>

#include <logging.h>
TDO_LOGGER("examples.DetectObjectsByEventLinemod")


int main(int argc, char** argv){

    tooldetectobject::aaaaa();

    tooldetectobject::ConfigureRootLogger("DEBUG", "", "./detector.log");  // need to execute at the beginning before any logging call.
    /**
     *  argv:
     *    - inputdata: path to the event database
     *    - templatepath: path to the template images
     **/
    std::vector<std::string> options{
        "executable",
        "inputdata",
        "templatepath"
    };
    ArgumentParser argparser(argc, argv, options);
    // log4cxx::BasicConfigurator::configure();

    TDO_LOG_DEBUG("-------- start of the detector! --------");
    PyObject *myModuleName, *myModule, *myDict, *myPythonClass, *myPythonObject, *dataPathPy;

    Py_Initialize();
    myModuleName = PyUnicode_FromString(
        "tool_eventdetectobjects.io.file_format.eventsinethzformat"
    );
    dataPathPy = PyUnicode_FromString(
        argparser.getCmdOption("inputdata").c_str()
    );

    // load the module object
    myModule = PyImport_Import(myModuleName);
    if (myModule == nullptr){
        PyErr_Print();
        TDO_LOG_ERROR("Failed to import the python module.");
        return 0;
    }
    // Py_DECREF(myModuleName);

    // load the classes into a dict
    myDict = PyModule_GetDict(myModule);
    if (myDict == nullptr){
        PyErr_Print();
        TDO_LOG_ERROR("Failed to get the dictionary.");
        return 0;
    }
    // Py_DECREF(myModule);

    // build the name of a callable class
    myPythonClass = PyDict_GetItemString(myDict, "EventsInETHZFormat");
    if (myPythonClass == nullptr){
        PyErr_Print();
        TDO_LOG_ERROR("Failed to get the python class.");
        return 0;
    }
    // Py_DECREF(myDict);

    // create an instance of the class
    if (PyCallable_Check(myPythonClass)){
        myPythonObject = PyObject_CallObject(myPythonClass, dataPathPy);
        // Py_DECREF(myPythonClass);
    } else {
        TDO_LOG_ERROR("Can not instantiate the python class");
        // Py_DECREF(myPythonClass);
        return 0;
    }

    PyObject *functionAName;
    functionAName = PyUnicode_FromString(
        "PopOneTimeLimitedSbn"
    );
    for (int indexFrame=0; indexFrame < 100; indexFrame++){
        PyObject* pMysbn = PyObject_CallMethodObjArgs(myPythonObject, functionAName, 20000, 720, 1280);
        Eigen::MatrixXf mMysbn = tooldetectobject::pybindutils::GetEigenMatrixFromPyObject(pMysbn);
        TDO_LOG_INFO_FORMAT("mMysbn shape: h %d x w %d", mMysbn.rows()%mMysbn.cols());
    }

    Py_Finalize();
    return 0;
}
