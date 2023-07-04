#include <argparser.h>
#include <pybindcommon.h>
#include <eventlinemod.h>
#include <system.h>

#include <logging.h>
TDO_LOGGER("examples.RunObjectSlam")


int main(int argc, char** argv){
    py::scoped_interpreter guard{};  // Note: It creates a new interpreter state and sets it as the thread-local state for the calling thread, and then destroys the interpreter state when the scoped_interpreter object goes out of scope.
    py::module sys = py::module::import("sys");
    py::print(sys.attr("version")); 

    /**
     *  argv:
     *    - stereoseqpath: path to the stereo sequence.
     *    - sysconfigpath: path to the system config.
     **/
    std::vector<std::string> options{
        "executable",
        "stereoseqpath",
        "sysconfigpath"
    };
    ArgumentParser argparser(argc, argv, options);

    TDO_LOG_DEBUG("-------- start run slam! --------");

    eventobjectslam::SystemConfig thisSysConfig(argparser.getCmdOption("sysconfigpath"));
    std::shared_ptr pThisSysConfig = std::make_shared<eventobjectslam::SystemConfig>(thisSysConfig);
    eventobjectslam::SLAMSystem thisSlamSys(pThisSysConfig);

    thisSlamSys.TestTrackStereoSequence(argparser.getCmdOption("stereoseqpath"));

    return;
}