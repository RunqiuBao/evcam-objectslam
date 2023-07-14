#include "object.h"

#include <logging.h>
TDO_LOGGER("examples.DetectObjectsByEventLinemod")


namespace eventobjectslam {

object::ObjectBase::ObjectBase(const std::string sTemplateInfoPath){
    rapidjson::Document jsonTemplateInfos;
    FILE* fp = fopen(sTemplateInfoPath.c_str(), "rb");
    char readBuffer[65536];
    rapidjson::FileReadStream frs(
        fp,
        readBuffer,
        sizeof(readBuffer)
    );
    jsonTemplateInfos.ParseStream(frs);
    fclose(fp);

    for (rapidjson::Value::ConstValueIterator itr = jsonTemplateInfos.Begin(); itr != jsonTemplateInfos.End(); ++itr){
        if (itr->HasMember("objectName")){
            _objectName = itr->["objectName"].GetString();
            _objectExtents = {
                itr->["objectExtents"][0].GetFloat(),
                itr->["objectExtents"][1].GetFloat(),
                itr->["objectExtents"][2].GetFloat()
            };
        }
        else {
            uint16_t templID = itr->["templID"].GetInt();
            Mat44_t simulationCameraInObjectTransform;
            rapidjson::Value& jsonSimulationCameraInObjectTransform = itr->["camInObjectTransformation"];
            for (rapidjson::SizeType i = 0; i < jsonSimulationCameraInObjectTransform.Size(); i++){
                simulationCameraInObjectTransform(static_cast<int>(i) / 4, static_cast<int>(i) % 4) = jsonSimulationCameraInObjectTransform[i].GetFloat();
            }
            _templates.push_back(ObjectTemplate(templID, simulationCameraInObjectTransform));
        }
    }

    TDO_LOG_DEBUG_FORMAT("Object %s (xSize %f m, ySize %f m, zSize %f m) initialized with %d templates!", _objectName % _objectExtents[0] % _objectExtents[1] % _objectExtents[2] % _templates.size());
}

}