#include "pangolinviewer/viewer.h"


static void DrawLine(const float x1, const float y1, const float z1,
                              const float x2, const float y2, const float z2) {
    glVertex3f(x1, y1, z1);
    glVertex3f(x2, y2, z2);
}

static void DrawFrustum(const float w) {
    const float h = w * 0.75f;
    const float z = w * 0.6f;
    // 四角錐の斜辺
    DrawLine(0.0f, 0.0f, 0.0f, w, h, z);
    DrawLine(0.0f, 0.0f, 0.0f, w, -h, z);
    DrawLine(0.0f, 0.0f, 0.0f, -w, -h, z);
    DrawLine(0.0f, 0.0f, 0.0f, -w, h, z);
    // 四角錐の底辺
    DrawLine(w, h, z, w, -h, z);
    DrawLine(-w, h, z, -w, -h, z);
    DrawLine(-w, h, z, w, h, z);
    DrawLine(-w, -h, z, w, -h, z);
}

static void DrawCamera(const pangolin::OpenGlMatrix& gl_camPoseCurrentInWorld, const float width) {
    glPushMatrix();
    glMultMatrixd(gl_camPoseCurrentInWorld.m);

    glBegin(GL_LINES);
    DrawFrustum(width);
    glEnd();

    glPopMatrix();
}

static void DrawCamera(const eventobjectslam::Mat44_t& camPoseCurrentInWorld, const float width) {
    glPushMatrix();
    glMultMatrixf(camPoseCurrentInWorld.transpose().cast<float>().eval().data());

    glBegin(GL_LINES);
    DrawFrustum(width);
    glEnd();

    glPopMatrix();
}


namespace eventobjectslam{

namespace pangolinviewer {

Viewer::Viewer(const std::shared_ptr<MapDataBase> pMapDatabase)
:_pMapDatabase(pMapDatabase){}

void Viewer::Run(
    std::vector<std::shared_ptr<KeyFrame>> vKeyFrames,
    std::vector<std::shared_ptr<LandMark>> vLandmarks
){
    const std::string mapViewerName{"PangolinViewer: Map Viewer"};
    pangolin::CreateWindowAndBind(mapViewerName, 1024, 768);
    static constexpr float mapViewerWidth = 1024;
    static constexpr float mapViewerHeight = 768;
    const float viewpointF = 2000.;
    const float viewpointX = 0.;
    const float viewpointY = -10.0;
    const float viewpointZ = -0.1;


    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glEnable(GL_DEPTH_TEST);  // depth testing to be enabled for 3D mouse handler

    // setup camera renderer
    _sCam = std::unique_ptr<pangolin::OpenGlRenderState>(new pangolin::OpenGlRenderState(
        pangolin::ProjectionMatrix(mapViewerWidth, mapViewerHeight, viewpointF, viewpointF,
                                   mapViewerWidth / 2, mapViewerHeight / 2, 0.1, 1e6),
        pangolin::ModelViewLookAt(viewpointX, viewpointY, viewpointZ, 0, 0, 0, 0.0, -1.0, 0.0)));

    // create map window
    pangolin::View& dCam = pangolin::CreateDisplay()
                                .SetBounds(0.0, 1.0, pangolin::Attach::Pix(175), 1.0, -mapViewerWidth / mapViewerHeight)
                                .SetHandler(new pangolin::Handler3D(*_sCam));
    pangolin::OpenGlMatrix gl_camPoseCurrentInWorld;
    gl_camPoseCurrentInWorld.SetIdentity();

    for (std::shared_ptr<KeyFrame> oneKeyFrame : vKeyFrames){
        const pangolin::OpenGlMatrix gl_camPoseCurrentInWorld(oneKeyFrame->_poseCurrentFrameInWorld.inverse().eval());
    }

}

} // end of namespace panglinviewer

}  // end of namespace eventobjectslam