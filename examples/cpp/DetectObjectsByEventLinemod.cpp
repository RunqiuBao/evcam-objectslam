#include <Python.h>

#include <argparser.h>
#include <pybindcommon.h>

#include <logging.h>
TDO_LOGGER("examples.DetectObjectsByEventLinemod")


int main(int argc, char** argv){

    py::scoped_interpreter guard{};  // Note: It creates a new interpreter state and sets it as the thread-local state for the calling thread, and then destroys the interpreter state when the scoped_interpreter object goes out of scope.
    py::module sys = py::module::import("sys");
    py::print(sys.attr("version"));

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

    std::string modulename = "tool_eventdetectobjects.io.file_format.eventsinethzformat";
    std::string classname = "EventsInETHZFormat";
    py::object dataloaderclass = py::module::import(modulename.c_str()).attr(classname.c_str());
    TDO_LOG_DEBUG_FORMAT("inputdata: %s", argparser.getCmdOption("inputdata"));
    py::object dataloader = dataloaderclass(py::cast(argparser.getCmdOption("inputdata")));

    for (int indexFrame=0; indexFrame < 100; indexFrame++){
        py::object mysbnPy = dataloader.attr("PopOneTimeLimitedSbn")(20000, 720, 1280);
        py::buffer_info mysbnPyBuffer = py::cast<py::array_t<float, py::array::c_style | py::array::forcecast>>(*const_cast<py::object*>(&mysbnPy)).request(false);
        float* mysbnData = static_cast<float*>(mysbnPyBuffer.ptr);
        std::vector<ssize_t> mysbnShape = mysbnPyBuffer.shape;
        Eigen::MatrixXf mMysbn;
        mMysbn = Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(mysbnData, mysbnShape[0], mysbnShape[1]);
        TDO_LOG_INFO_FORMAT("mMysbn shape: h %d x w %d", mMysbn.rows()%mMysbn.cols());
    }
    return 0;
}
