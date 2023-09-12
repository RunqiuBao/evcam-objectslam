#include "pangolinviewer/viewer.h"


#include <logging.h>
TDO_LOGGER("eventobjectslam.pangolinviewer.viewer")


#define PI 3.14159265358979
#define Cos(th) std::cos(PI/180*(th))
#define Sin(th) std::sin(PI/180*(th))
#define DEF_D 5
void DrawCylinder(Eigen::Matrix4f cylinder_pose_wo){
    float scale = 0.1;
    glBegin(GL_QUAD_STRIP);
    for (int j=0; j<=360;j+=DEF_D){
        glColor3f(1.0, 0.5, 0.0);
        Eigen::Vector4f vertex1;
        vertex1[0] = scale * Cos(j);
        vertex1[1] = scale * Sin(j);
        vertex1[2] = +2 * scale;
        vertex1[3] = 1;
        Eigen::Vector4f vertex1_w = cylinder_pose_wo * vertex1;
        glVertex3f(vertex1_w[0], vertex1_w[1], vertex1_w[2]);
        vertex1[2] = -2 * scale;
        Eigen::Vector4f vertex2_w = cylinder_pose_wo * vertex1;
        glVertex3f(vertex2_w[0], vertex2_w[1], vertex2_w[2]);
    }
    glEnd();
    glBegin(GL_TRIANGLE_FAN);
    for (int j=0; j<=360;j+=DEF_D){
        glColor3f(0., 1., 1.);
        Eigen::Vector4f vertex1;
        vertex1[0] = scale * Cos(j);
        vertex1[1] = scale * Sin(j);
        vertex1[2] = +2 * scale;
        vertex1[3] = 1;
        Eigen::Vector4f vertex1_w = cylinder_pose_wo * vertex1;
        glVertex3f(vertex1_w[0], vertex1_w[1], vertex1_w[2]);
    }
    glEnd();
    glBegin(GL_TRIANGLE_FAN);
    for (int j=0; j<=360;j+=DEF_D){
        glColor3f(0., 1., 1.);
        Eigen::Vector4f vertex1;
        vertex1[0] = scale * Cos(j);
        vertex1[1] = scale * Sin(j);
        vertex1[2] = -2 * scale;
        vertex1[3] = 1;
        Eigen::Vector4f vertex1_w = cylinder_pose_wo * vertex1;
        glVertex3f(vertex1_w[0], vertex1_w[1], vertex1_w[2]);
    }
    glEnd();
}


static void DrawLine(const float x1, const float y1, const float z1,
                              const float x2, const float y2, const float z2) {
    glVertex3f(x1, y1, z1);
    glVertex3f(x2, y2, z2);
}

static void DrawHorizontalGrid() {
    Eigen::Matrix4f origin;
    // origin << 0, 0, 1, 0, -1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 1;
    glPushMatrix();
    // glMultTransposeMatrixf(origin.data());

    glLineWidth(1);
    std::array<float, 3> grid_ = {{0.3f, 0.3f, 0.3f}};
    glColor3fv(grid_.data());

    glBegin(GL_LINES);

    float expandingFactor = 2;
    float interval_ratio = 0.1;
    float grid_min = -100.0f * interval_ratio * expandingFactor;
    float grid_max = 100.0f * interval_ratio * expandingFactor;

    for (int x = -10 * expandingFactor; x <= 10 * expandingFactor; x += 1) {
        DrawLine(x * 10.0f * interval_ratio, grid_min, 0, x * 10.0f * interval_ratio, grid_max, 0);
    }
    for (int y = -10 * expandingFactor; y <= 10 * expandingFactor; y += 1) {
        DrawLine(grid_min, y * 10.0f * interval_ratio, 0, grid_max, y * 10.0f * interval_ratio, 0);
    }

    glEnd();

    glPopMatrix();
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

void DrawCurrentCamPose(const pangolin::OpenGlMatrix& gl_cam_pose_wc) {
    // frustum size of the frame
    const float camera_size_ = 0.15;
    const float w = camera_size_;
    const float camera_line_width_ = 2;

    glLineWidth(camera_line_width_);
    std::array<float, 3> curr_cam_ = {{0.7f, 0.7f, 1.0f}};
    glColor3fv(curr_cam_.data());
    DrawCamera(gl_cam_pose_wc, w);
}


namespace eventobjectslam{

namespace pangolinviewer {

Viewer::Viewer(const std::shared_ptr<MapDataBase> pMapDatabase)
:_pMapDatabase(pMapDatabase){}

void Viewer::Run(){
    const std::string mapViewerName{"PangolinViewer: Map Viewer"};
    pangolin::CreateWindowAndBind(mapViewerName, 1024, 768);
    static constexpr float mapViewerWidth = 1024;
    static constexpr float mapViewerHeight = 768;
    const float viewpointF = 2000.;
    const float viewpointX = 0.;
    const float viewpointY = 0.0;
    const float viewpointZ = 50.;

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

    Mat44_t worldInRenderTransform;
    worldInRenderTransform << 1, 0, 0, 0,
                            0, 0, 1, 0,
                            0, -1, 0, 1.5,  // Note: cam height 1.5 is manually observed.
                            0, 0, 0, 1;
    while (!pangolin::ShouldQuit())
    {
        // Clear screen and activate view to render into
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        dCam.Activate(*_sCam);

        // Render ground
        DrawHorizontalGrid();

        std::vector<std::shared_ptr<KeyFrame>> vKeyFrames = _pMapDatabase->GetAllKeyframes();
        std::vector<std::shared_ptr<LandMark>> vLandmarks = _pMapDatabase->GetAllLandmarks();
        for (std::shared_ptr<KeyFrame> oneKeyFrame : vKeyFrames){
            Mat44_t camPoseInRender = worldInRenderTransform * oneKeyFrame->_poseCurrentFrameInWorld;
            const pangolin::OpenGlMatrix gl_camPoseCurrentInWorld(camPoseInRender.eval());
            DrawCurrentCamPose(gl_camPoseCurrentInWorld);
        }
        for (std::shared_ptr<LandMark> oneLandmark : vLandmarks){
            Mat44_t landmarkPoseInRender = worldInRenderTransform * oneLandmark->_poseLandmarkInWorld;
            DrawCylinder(landmarkPoseInRender);
        }

        // Swap frames and Process Events
        pangolin::FinishFrame();
    }

}

} // end of namespace panglinviewer

}  // end of namespace eventobjectslam