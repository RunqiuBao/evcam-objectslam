#include "mapdatabase.h"

#include <logging.h>
TDO_LOGGER("eventobjectslam.mapdatabase")


namespace eventobjectslam{

MapDataBase::MapDataBase(){
    TDO_LOG_INFO("Construct: MapDataBase");
}

void MapDataBase::AddLandMark(KeyFrame* keyFrame){
    std::lock_guard<std::mutex> lock(_mtxMapAccess);
    _keyFrames[keyFrame]
}

} // end of namespace eventobjectslam