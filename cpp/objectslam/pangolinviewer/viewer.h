#ifndef EVENTOBJECTSLAM_PANGOLINVIEWER_VIEWER_H
#define EVENTOBJECTSLAM_PANGOLINVIEWER_VIEWER_H

#include "mapdatabase.h"
#include "objectslam.h"

#include <pangolin/pangolin.h>

namespace eventobjectslam{

namespace pangolinviewer {

class Viewer {

public:
    Viewer(const std::shared_ptr<MapDataBase> pMapDatabase);

    // main loop for window refresh
    void Run();

private:
    const std::shared_ptr<MapDataBase> _pMapDatabase;

    // camera renderer
    std::unique_ptr<pangolin::OpenGlRenderState> _sCam;
};

} // end of namespace panglinviewer

}  // end of namespace eventobjectslam

#endif