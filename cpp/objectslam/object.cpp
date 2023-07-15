#include "object.h"
#include <rapidjson/document.h>
#include <rapidjson/filereadstream.h>
#include <filesystem>

#include <logging.h>
TDO_LOGGER("examples.DetectObjectsByEventLinemod")


namespace eventobjectslam {

object::ObjectBase::ObjectBase(const std::string sTemplatesPath){
    rapidjson::Document jsonTemplateInfos;
    std::filesystem::path templateInfoPath = sTemplatesPath;
    templateInfoPath.append("templateInfos.json");
    FILE* fp = fopen(templateInfoPath.string().c_str(), "rb");
    char readBuffer[65536];
    rapidjson::FileReadStream frs(
        fp,
        readBuffer,
        sizeof(readBuffer)
    );
    jsonTemplateInfos.ParseStream(frs);
    fclose(fp);

    TDO_LOG_DEBUG("start loading templates...");
    indicesInTemplatesArray.reserve(jsonTemplateInfos.Size() - 1);
    for (rapidjson::SizeType i = 0; i < jsonTemplateInfos.Size(); ++i) {
        const rapidjson::Value& jsonOneTemplate = jsonTemplateInfos[i];
        if (jsonOneTemplate.HasMember("objectName")){
            _objectName = jsonOneTemplate["objectName"].GetString();
            _objectExtents = {
                jsonOneTemplate["objectExtents"][0].GetFloat(),
                jsonOneTemplate["objectExtents"][1].GetFloat(),
                jsonOneTemplate["objectExtents"][2].GetFloat()
            };
        }
        else {
            uint16_t templID = jsonOneTemplate["templId"].GetInt();
            indicesInTemplatesArray[templID] = _templates.size(); 
            Mat44_t simulationCameraInObjectTransform;
            const rapidjson::Value& jsonSimulationCameraInObjectTransform = jsonOneTemplate["camInObjectTransformation"];
            for (rapidjson::SizeType i = 0; i < jsonSimulationCameraInObjectTransform.Size(); i++){
                const rapidjson::Value& oneRow = jsonSimulationCameraInObjectTransform[i];
                for (rapidjson::SizeType j = 0; j < oneRow.Size(); j++){
                    simulationCameraInObjectTransform(static_cast<int>(i), static_cast<int>(j)) = oneRow[j].GetFloat();
                }
            }
            _templates.push_back(ObjectTemplate(templID, simulationCameraInObjectTransform));
        }

    }

    TDO_LOG_DEBUG_FORMAT("Object %s (xSize %f m, ySize %f m, zSize %f m) initialized with %d templates!", _objectName % _objectExtents[0] % _objectExtents[1] % _objectExtents[2] % _templates.size());
}

}