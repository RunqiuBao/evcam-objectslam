#include <opencv2/opencv.hpp>

#include <argparser.h>
#include <pybindcommon.h>
#include <eventlinemod.h>


#include <logging.h>
TDO_LOGGER("examples.DetectObjectsByEventLinemod")


int main(int argc, char** argv){

    py::scoped_interpreter guard{};  // Note: It creates a new interpreter state and sets it as the thread-local state for the calling thread, and then destroys the interpreter state when the scoped_interpreter object goes out of scope.
    py::module sys = py::module::import("sys");
    py::print(sys.attr("version"));

    ConfigureRootLogger("DEBUG", "", "./detector.log");  // need to execute at the beginning before any logging call.
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

    TDO_LOG_DEBUG("-------- start of the detector! --------");

    std::string modulename = "tool_eventdetectobjects.io.file_format.eventsinethzformat";
    std::string classname = "EventsInETHZFormat";
    py::object dataloaderclass = py::module::import(modulename.c_str()).attr(classname.c_str());
    TDO_LOG_DEBUG_FORMAT("inputdata: %s", argparser.getCmdOption("inputdata"));
    py::object dataloader = dataloaderclass(py::cast(argparser.getCmdOption("inputdata")));
    float scoreThreshold = 400.0;
    tooldetectobject::EventLineModDetector myDetector(argparser.getCmdOption("templatepath"), scoreThreshold);

    for (int indexFrame=0; indexFrame < 100; indexFrame++){
        py::object mysbnPy = dataloader.attr("PopOneTimeLimitedSbn")(20000, 720, 1280);
        py::buffer_info mysbnPyBuffer = py::cast<py::array_t<int, py::array::c_style | py::array::forcecast>>(*const_cast<py::object*>(&mysbnPy)).request(false);
        int* mysbnData = static_cast<int*>(mysbnPyBuffer.ptr);
        // Eigen::MatrixXi mMysbn;
        // mMysbn = Eigen::Map<Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(mysbnData, mysbnShape[0], mysbnShape[1]);
        // Eigen::MatrixXf mMysbnFloat = mMysbn.cast<float>();
        // mMysbnFloat = mMysbnFloat.array() - mMysbnFloat.minCoeff();
        // mMysbnFloat = mMysbnFloat / mMysbnFloat.maxCoeff() * 255.0;
        cv::Mat mysbnMat(mysbnPyBuffer.shape[0], mysbnPyBuffer.shape[1], CV_32SC1, mysbnData);
        cv::Mat mysbnFloat;
        mysbnMat.convertTo(mysbnFloat, CV_32FC1);
        double minVal, maxVal;
        cv::minMaxLoc(mysbnFloat, &minVal, &maxVal);
        TDO_LOG_DEBUG_FORMAT("mysbnFloat max min values: %f, %f", maxVal % minVal);
        mysbnFloat = mysbnFloat - minVal;
        cv::minMaxLoc(mysbnFloat, &minVal, &maxVal);
        mysbnFloat = mysbnFloat / maxVal * 255.0;
        cv::Mat mysbnUint;
        mysbnFloat.convertTo(mysbnUint, CV_8UC1);
        cv::imwrite("/home/runqiu/tmptmp/" + std::to_string(indexFrame) + ".png" , mysbnUint);
    }

    return 0;
}
