#include <argparser.h>
#include <system.h>
#include <pangolinviewer/viewer.h>
#include <thread>
#include <chrono>
#include <filesystem>

#include <logging.h>
TDO_LOGGER("examples.RunObjectSlam")


void LoadTimestamps(const std::string& tsFilePath, std::vector<std::string>& timestamps) {
    std::ifstream inFile(tsFilePath);
    if (!inFile) {
        TDO_LOG_DEBUG("Failed to open file: " << tsFilePath);
        return;
    }

    std::string line;
    while (std::getline(inFile, line)) {  // Note: \n is automatically removed by std::getline
        if (!line.empty()) {
            timestamps.push_back(line);
        }
    }
    inFile.close();
    TDO_LOG_DEBUG("Loaded " << timestamps.size() << " timestamps.");
    return;
}


void TestUnitreeSequence(const std::string sStereoSequencePath, eventobjectslam::SLAMSystem& thisSlamSys){
    // the dataset dir includes `detections` and `sysconfig.json`
    rapidjson::Document& sysConfigJson = thisSlamSys._cfg->_jsonConfigNode;

    eventobjectslam::Vec3_t objectSize;
    objectSize << sysConfigJson["objects"]["0"]["objectSize"][0].GetFloat(),
                  sysConfigJson["objects"]["0"]["objectSize"][1].GetFloat(),
                  sysConfigJson["objects"]["0"]["objectSize"][2].GetFloat();
    thisSlamSys._pMapDb->SetObjectSize(objectSize);

    Eigen::Matrix3f kk;
    rapidjson::Value& kkJson = sysConfigJson["camera"]["kk"];
    for (rapidjson::SizeType i = 0; i < kkJson.Size(); i++){
        rapidjson::Value& kkJsonRow = kkJson[i];
        for (rapidjson::SizeType j = 0; j < kkJsonRow.Size(); j++){
            kk(static_cast<int>(i), static_cast<int>(j)) = kkJsonRow[j].GetFloat();
        }
    }

    thisSlamSys.InitializeCameraAndTracker(
        0,
        sysConfigJson["camera"]["imageWidth"].GetInt(),
        sysConfigJson["camera"]["imageHeight"].GetInt(),
        kk,
        sysConfigJson["camera"]["baseline"].GetFloat(),
        sStereoSequencePath
    );

    std::filesystem::path timestampsFilePath = sStereoSequencePath;
    timestampsFilePath.append("timestamps.txt");
    TDO_LOG_DEBUG("entered test track");

    std::vector<std::string> timestamps;
    LoadTimestamps(timestampsFilePath.string(), timestamps);

    std::string objectName = sysConfigJson["objects"]["0"]["objectName"].GetString();
    TDO_LOG_DEBUG("objectName: " << objectName);
    eventobjectslam::object::ObjectBase theLandmarkObject(objectName);
    std::shared_ptr<eventobjectslam::object::ObjectBase> pTheLandmarkObject = std::make_shared<eventobjectslam::object::ObjectBase>(theLandmarkObject);

    // // seq0, seq2
    // Eigen::Quaternionf q;
    // q.x() = -0.49999999999999956;
    // q.y() = 0.5000000218556936;
    // q.z() = 0.49999999999999956;
    // q.w() = -0.49999997814430636; 
    // Eigen::Matrix3f rWorldToRealWorld = q.toRotationMatrix();
    // Eigen::Matrix4f worldInRealWorld = Eigen::Matrix4f::Identity();
    // worldInRealWorld.block(0, 0, 3, 3) = rWorldToRealWorld;
    // worldInRealWorld.block(0, 2, 3, 1) *= -1;
    // TDO_LOG_DEBUG("rWorldToRealWorld: " << rWorldToRealWorld);
    // Eigen::Vector3f tWorldToRealWorld(-9.255331993103027, 7.211221218109131, 2.187476634979248);
    // worldInRealWorld.block(0, 3, 3, 1) = tWorldToRealWorld;
    
    // //seq1
    // eventobjectslam:Mat44_t worldInRealWorld;
    // worldInRealWorld << -4.371139e-8, 1.736483e-1, -9.848077e-1, 4.,
    //                     1., 7.590408e-9,-4.304732e-8, -2.2000,
    //                     -0., -9.848077e-1, -1.736483e-1, 2.3650,
    //                     0., 0., 0., 1.;

    Eigen::Matrix4f worldInRealWorld = Eigen::Matrix4f::Identity();

    std::filesystem::path trackResultPath = sStereoSequencePath;
    trackResultPath.append("cameraTrackRealTime.txt");
    std::ofstream outputFile(trackResultPath.string());
    
    int frameCount = 0;

    Eigen::Matrix4f cameraInRealWorld = Eigen::Matrix4f::Identity();  // Note: Pose for comparing to ground truth

    // for debug purpose
    bool isDebug = true;
    std::vector<size_t> numFramesEachKeyframe;
    int keyframeCount = 0;
    for(const std::string& timestamp : timestamps){
        TDO_LOG_DEBUG(timestamp);

        auto starttime = std::chrono::steady_clock::now();

        // load detection results from txt file.
        std::vector<std::string> sDetections;
        std::filesystem::path detectionFilePath = sStereoSequencePath;
        detectionFilePath.append("detections/");
        std::ifstream detectionResult(detectionFilePath.append(timestamp + ".txt"));
        TDO_LOG_DEBUG_FORMAT("opening detection: %s", detectionFilePath.string());
        if (!detectionResult.is_open()) {
            TDO_LOG_DEBUG("Failed to open the detectionResult (" << timestamp << ").");
            Eigen::Quaternionf myQuaternion(cameraInRealWorld.block<3, 3>(0, 0));
            outputFile << std::to_string(frameCount) << " " << cameraInRealWorld(0, 3) << " " << cameraInRealWorld(1, 3) << " " << cameraInRealWorld(2, 3) << " " << myQuaternion.x() << " " << myQuaternion.y() << " " << myQuaternion.z() << " " << myQuaternion.w() << std::endl;
            frameCount++;
        }
        else {
            std::string sDetection;
            while (std::getline(detectionResult, sDetection)) {
                sDetections.push_back(sDetection); // Store each line in the vector
            }
        }
        detectionResult.close();

        const Eigen::Matrix4f frameInWorld = thisSlamSys.UpdateOneFrame(timestamp, sDetections, pTheLandmarkObject, isDebug);
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - starttime);
        TDO_LOG_INFO_FORMAT("frame(%s) tracking finished in %d milisec.", timestamp % duration.count());
        cameraInRealWorld = worldInRealWorld * frameInWorld;  //Note: think like there is a point in current frame, first it will be transformed into keyframe, then to world, then to realWorld.
        Eigen::Quaternionf myQuaternion(cameraInRealWorld.block<3, 3>(0, 0));
        outputFile << std::to_string(frameCount) << " " << cameraInRealWorld(0, 3) << " " << cameraInRealWorld(1, 3) << " " << cameraInRealWorld(2, 3) << " " << myQuaternion.x() << " " << myQuaternion.y() << " " << myQuaternion.z() << " " << myQuaternion.w() << std::endl;
        TDO_LOG_DEBUG("cameraInRealWorld: \n" << cameraInRealWorld);
        frameCount++;

        if (thisSlamSys._pKeyFrameStack.size() > keyframeCount) {
            numFramesEachKeyframe.push_back(thisSlamSys._pKeyFrameStack.back()->_vFrames_ids.size());  // debug code
            keyframeCount = thisSlamSys._pKeyFrameStack.size();
        }
    }
    outputFile.close();
    // FIXME: the first keyframe will be optimized and bias from (0, 0, 0). Compensate it before saving.
    eventobjectslam::SaveOptimizedTraj(sStereoSequencePath, thisSlamSys._allFramesStack, worldInRealWorld);
    eventobjectslam::SaveLandmarks(sStereoSequencePath, thisSlamSys._pMapDb, worldInRealWorld);

    // print debug infos
    if (numFramesEachKeyframe.size() > 0){
        size_t minNumFrames = *std::min_element(numFramesEachKeyframe.begin(), numFramesEachKeyframe.end());
        size_t maxNumFrames = *std::max_element(numFramesEachKeyframe.begin(), numFramesEachKeyframe.end());
        TDO_LOG_INFO_FORMAT("keyframes contain maximum %d frames and minimum %d frames.", maxNumFrames % minNumFrames);
    }
    else{
        TDO_LOG_INFO("keyframes contain zero frames. Something is wrong.");
    }

}

int main(int argc, char** argv){
    ConfigureRootLogger("DEBUG", "", "./detector.log");
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

    thisSlamSys.Startup();

    // run the viewer in another thread
    std::thread thread1([&]() {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
        eventobjectslam::pangolinviewer::Viewer viewer(thisSlamSys._pMapDb);
        viewer.Run();
    });

    TestUnitreeSequence(argparser.getCmdOption("stereoseqpath"), thisSlamSys);

    thread1.join();

    return 0;
}
