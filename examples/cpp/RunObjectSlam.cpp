#include <argparser.h>
#include <system.h>
#include <pangolinviewer/viewer.h>
#include <thread>
#include <chrono>

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
    std::shared_ptr<eventobjectslam::SystemConfig> pThisSysConfig = std::make_shared<eventobjectslam::SystemConfig>(thisSysConfig);

    eventobjectslam::SLAMSystem thisSlamSys(pThisSysConfig);

    // run the SLAM in another thread
    std::thread thread([&]() {
        thisSlamSys.TestTrackStereoSequence(argparser.getCmdOption("stereoseqpath"));
    });
    TDO_LOG_DEBUG("until here");
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
    eventobjectslam::pangolinviewer::Viewer viewer(thisSlamSys._pMapDb);
    viewer.Run();

    thread.join();

    return 0;
}
