#include "system.h"
#include <filesystem>

#include <logging.h>
TDO_LOGGER("eventobjectslam.system")

namespace eventobjectslam {

void SLAMSystem::TestTrackStereoSequence(const std::string sStereoSequencePath){
    // the dataset dir includes `leftcam` and `rightcam` and `colorconeInfo.json`
    // in each cam folder, it includes `*.png`, `detectionId*/(yolos, including linemod based template selection)`, `semanticDescriptorRules.json`
    // `semanticDescriptorRules.json` can be: 
    // (1) cone: left, right, front side, each have range and accuracy definition; distance to start; distance to light source. 
    // (2) light source: left right front side.
    filesystem::path stereoSequencePath = sStereoSequencePath;
    for (const auto& oneFilePath : filesystem::directory_iterator(stereoSequencePath)) {
        if (filesystem::is_regular_file(oneFilePath) && oneFilePath.extension() == '.png') {
            TDO_LOG_VERBOSE_FORMAT(oneFilePath.string());
        }
    }
}

}