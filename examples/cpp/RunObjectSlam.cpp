#include <argparser.h>
#include <system.h>

#include <logging.h>
TDO_LOGGER("examples.RunObjectSlam")


int main(int argc, char** argv){
    tooldetectobject::ConfigureRootLogger("DEBUG", "", "./detector.log");
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

    return 0;
}