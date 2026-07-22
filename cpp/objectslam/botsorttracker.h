#ifndef EVENTOBJECTSLAM_BOTSORTTRACKER_H
#define EVENTOBJECTSLAM_BOTSORTTRACKER_H

#include <Eigen/Core>

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace eventobjectslam {

// BoT-SORT-style multi-object tracker (motion-only variant), replacing frame-to-frame IoU chaining:
//  - per-track kalman filter on (cx, cy, w, h) with constant velocity: predicted boxes absorb camera motion,
//    so fast pans no longer break the association.
//  - hungarian assignment on IoU of predicted boxes vs detections (globally optimal, exclusive).
//  - BYTE two-stage matching: high-score detections first, then low-score ones against remaining tracks.
//  - lost tracks are kept for a buffer period and re-activated on re-appearance (stable IDs across occlusions).
// No appearance/ReID features and no image-based camera motion compensation: this pipeline is detection-only.

struct BoTSortDetection {
    float centerX;
    float centerY;
    float width;
    float height;
    float score;
};

namespace botsort_detail {

// kalman filter over state (cx, cy, w, h, vcx, vcy, vw, vh), dt = 1 frame. noise scales with box size
// (standard ByteTrack/BoT-SORT weights).
class BoxKalman {
public:
    using Vec8 = Eigen::Matrix<double, 8, 1, Eigen::DontAlign>;
    using Mat88 = Eigen::Matrix<double, 8, 8, Eigen::DontAlign>;
    using Vec4 = Eigen::Matrix<double, 4, 1, Eigen::DontAlign>;

    void Initiate(const Vec4& measurement){
        _x.setZero();
        _x.head<4>() = measurement;
        const double w = measurement(2), h = measurement(3);
        Vec8 std;
        std << 2 * kStdWeightPosition * w, 2 * kStdWeightPosition * h, 2 * kStdWeightPosition * w, 2 * kStdWeightPosition * h,
               10 * kStdWeightVelocity * w, 10 * kStdWeightVelocity * h, 10 * kStdWeightVelocity * w, 10 * kStdWeightVelocity * h;
        _P = std.cwiseProduct(std).asDiagonal();
    }

    void Predict(){
        const double w = _x(2), h = _x(3);
        Mat88 F = Mat88::Identity();
        F.block<4, 4>(0, 4) = Eigen::Matrix4d::Identity();
        _x = F * _x;
        _x(2) = std::max(_x(2), 1.0);  // keep box size positive.
        _x(3) = std::max(_x(3), 1.0);
        Vec8 std;
        std << kStdWeightPosition * w, kStdWeightPosition * h, kStdWeightPosition * w, kStdWeightPosition * h,
               kStdWeightVelocity * w, kStdWeightVelocity * h, kStdWeightVelocity * w, kStdWeightVelocity * h;
        _P = (F * _P * F.transpose()).eval();
        _P.diagonal() += std.cwiseProduct(std);
    }

    void Update(const Vec4& measurement){
        const double w = _x(2), h = _x(3);
        Vec4 std;
        std << kStdWeightPosition * w, kStdWeightPosition * h, kStdWeightPosition * w, kStdWeightPosition * h;
        Eigen::Matrix4d S = _P.topLeftCorner<4, 4>();
        S.diagonal() += std.cwiseProduct(std);
        const Eigen::Matrix<double, 8, 4, Eigen::DontAlign> K = _P.leftCols<4>() * S.inverse();
        _x += K * (measurement - _x.head<4>());
        Mat88 I_KH = Mat88::Identity();
        I_KH.leftCols<4>() -= K;
        _P = (I_KH * _P).eval();
    }

    Vec4 PredictedBox() const { return _x.head<4>(); }

private:
    static constexpr double kStdWeightPosition = 1.0 / 20.0;
    static constexpr double kStdWeightVelocity = 1.0 / 160.0;
    Vec8 _x = Vec8::Zero();
    Mat88 _P = Mat88::Identity();
};

// IoU with an optional buffer (C-BIoU style): both boxes are expanded by `bufferScale`, which keeps
// associations alive under image motion that the kalman velocity has not learned yet (young tracks
// during fast camera pans).
inline float BoxIoU(const Eigen::Matrix<double, 4, 1, Eigen::DontAlign>& a, const BoTSortDetection& b, const double bufferScale = 1.0){
    const double aw = a(2) * bufferScale, ah = a(3) * bufferScale;
    const double bw = b.width * bufferScale, bh = b.height * bufferScale;
    const double ax0 = a(0) - aw / 2, ay0 = a(1) - ah / 2, ax1 = a(0) + aw / 2, ay1 = a(1) + ah / 2;
    const double bx0 = b.centerX - bw / 2., by0 = b.centerY - bh / 2.;
    const double bx1 = b.centerX + bw / 2., by1 = b.centerY + bh / 2.;
    const double iw = std::min(ax1, bx1) - std::max(ax0, bx0);
    const double ih = std::min(ay1, by1) - std::max(ay0, by0);
    if (iw <= 0 || ih <= 0){
        return 0.f;
    }
    const double inter = iw * ih;
    const double uni = aw * ah + bw * bh - inter;
    return (uni > 0)? static_cast<float>(inter / uni) : 0.f;
}

// hungarian algorithm (shortest augmenting path, O(n^3)); cost matrix numRows x numCols, INF forbids a pair.
// returns per-row assigned column (-1 if unassigned or only INF available).
inline std::vector<int> SolveAssignment(const std::vector<std::vector<double>>& cost){
    const double INF = std::numeric_limits<double>::infinity();
    const int numRows = static_cast<int>(cost.size());
    const int numCols = numRows? static_cast<int>(cost[0].size()) : 0;
    const int n = std::max(numRows, numCols);
    if (n == 0){
        return {};
    }
    // pad to square with large-but-finite cost so the algorithm terminates; INF entries stay INF-like large.
    const double BIG = 1e7;
    std::vector<std::vector<double>> a(n + 1, std::vector<double>(n + 1, BIG));
    for (int i = 0; i < numRows; i++){
        for (int j = 0; j < numCols; j++){
            a[i + 1][j + 1] = std::isinf(cost[i][j])? BIG : cost[i][j];
        }
    }
    std::vector<double> u(n + 1), v(n + 1);
    std::vector<int> p(n + 1), way(n + 1);
    for (int i = 1; i <= n; ++i){
        p[0] = i;
        int j0 = 0;
        std::vector<double> minv(n + 1, INF);
        std::vector<char> used(n + 1, false);
        do {
            used[j0] = true;
            int i0 = p[j0], j1 = 0;
            double delta = INF;
            for (int j = 1; j <= n; ++j){
                if (!used[j]){
                    const double cur = a[i0][j] - u[i0] - v[j];
                    if (cur < minv[j]){
                        minv[j] = cur;
                        way[j] = j0;
                    }
                    if (minv[j] < delta){
                        delta = minv[j];
                        j1 = j;
                    }
                }
            }
            for (int j = 0; j <= n; ++j){
                if (used[j]){
                    u[p[j]] += delta;
                    v[j] -= delta;
                }
                else {
                    minv[j] -= delta;
                }
            }
            j0 = j1;
        } while (p[j0] != 0);
        do {
            const int j1 = way[j0];
            p[j0] = p[j1];
            j0 = j1;
        } while (j0);
    }
    std::vector<int> rowToCol(numRows, -1);
    for (int j = 1; j <= n; ++j){
        const int i = p[j];
        if (i >= 1 && i <= numRows && j <= numCols && a[i][j] < BIG){
            rowToCol[i - 1] = j - 1;
        }
    }
    return rowToCol;
}

}  // namespace botsort_detail

class BoTSortTracker {
public:
    struct Result {
        std::vector<int> trackIDs;    // per input detection; -1 when no track was assigned.
        std::vector<int> trackHits;   // per input detection: number of matched frames of its track (0 if none).
    };

    // process one frame of detections; call exactly once per frame (also with an empty list on
    // no-detection frames, so that lost-track aging advances).
    Result Update(const std::vector<BoTSortDetection>& detections){
        using botsort_detail::BoxIoU;
        using botsort_detail::SolveAssignment;
        const double INF = std::numeric_limits<double>::infinity();
        _frameCount++;

        // 1. predict all live tracks.
        for (Track& track : _tracks){
            track.kalman.Predict();
        }

        // 2. split detections by score.
        std::vector<int> highDets, lowDets;
        for (int i = 0; i < static_cast<int>(detections.size()); i++){
            if (detections[i].score >= kTrackHighThresh){
                highDets.push_back(i);
            }
            else if (detections[i].score >= kTrackLowThresh){
                lowDets.push_back(i);
            }
        }

        Result result;
        result.trackIDs.assign(detections.size(), -1);
        result.trackHits.assign(detections.size(), 0);
        std::vector<bool> trackMatched(_tracks.size(), false);

        // 3. first association: all live tracks (tracked + lost) vs high-score detections.
        {
            std::vector<int> trackPool;
            for (int t = 0; t < static_cast<int>(_tracks.size()); t++){
                trackPool.push_back(t);
            }
            Associate(trackPool, highDets, detections, kFirstMatchMinIoU, kFirstMatchBufferScale, trackMatched, result);
        }

        // 4. second association (BYTE): remaining *tracked* (not lost) tracks vs low-score detections.
        {
            std::vector<int> trackPool;
            for (int t = 0; t < static_cast<int>(_tracks.size()); t++){
                if (!trackMatched[t] && !_tracks[t].isLost){
                    trackPool.push_back(t);
                }
            }
            Associate(trackPool, lowDets, detections, kSecondMatchMinIoU, 1.0, trackMatched, result);
        }

        // 5. unmatched tracks become lost; overdue lost tracks are removed.
        std::vector<Track> survivors;
        for (int t = 0; t < static_cast<int>(_tracks.size()); t++){
            Track& track = _tracks[t];
            if (trackMatched[t]){
                track.isLost = false;
                track.framesSinceUpdate = 0;
                survivors.push_back(track);
            }
            else {
                track.isLost = true;
                track.framesSinceUpdate++;
                if (track.framesSinceUpdate <= kTrackBufferFrames){
                    survivors.push_back(track);
                }
            }
        }
        _tracks = std::move(survivors);

        // 6. new tracks from unmatched high-score detections.
        for (const int d : highDets){
            if (result.trackIDs[d] >= 0 || detections[d].score < kNewTrackThresh){
                continue;
            }
            Track track;
            track.trackID = _nextTrackID++;
            track.hits = 1;
            track.isLost = false;
            track.framesSinceUpdate = 0;
            track.kalman.Initiate(ToMeasurement(detections[d]));
            _tracks.push_back(track);
            result.trackIDs[d] = track.trackID;
            result.trackHits[d] = 1;
        }
        return result;
    }

private:
    struct Track {
        int trackID = -1;
        int hits = 0;                // number of frames this track was matched (confirmation counter).
        int framesSinceUpdate = 0;
        bool isLost = false;
        botsort_detail::BoxKalman kalman;
    };

    static botsort_detail::BoxKalman::Vec4 ToMeasurement(const BoTSortDetection& det){
        botsort_detail::BoxKalman::Vec4 z;
        z << det.centerX, det.centerY, det.width, det.height;
        return z;
    }

    void Associate(const std::vector<int>& trackPool, std::vector<int>& detPool,
                   const std::vector<BoTSortDetection>& detections, const float minIoU, const double bufferScale,
                   std::vector<bool>& trackMatched, Result& result){
        using botsort_detail::BoxIoU;
        if (trackPool.empty() || detPool.empty()){
            return;
        }
        const double INF = std::numeric_limits<double>::infinity();
        std::vector<std::vector<double>> cost(trackPool.size(), std::vector<double>(detPool.size(), INF));
        for (size_t r = 0; r < trackPool.size(); r++){
            const auto predictedBox = _tracks[trackPool[r]].kalman.PredictedBox();
            for (size_t c = 0; c < detPool.size(); c++){
                const float iou = BoxIoU(predictedBox, detections[detPool[c]], bufferScale);
                if (iou >= minIoU){
                    cost[r][c] = 1.0 - iou;
                }
            }
        }
        const std::vector<int> rowToCol = botsort_detail::SolveAssignment(cost);
        std::vector<int> remainingDets;
        std::vector<bool> detTaken(detPool.size(), false);
        for (size_t r = 0; r < trackPool.size(); r++){
            const int c = rowToCol[r];
            if (c < 0){
                continue;
            }
            Track& track = _tracks[trackPool[r]];
            const int d = detPool[c];
            track.kalman.Update(ToMeasurement(detections[d]));
            track.hits++;
            trackMatched[trackPool[r]] = true;
            detTaken[c] = true;
            result.trackIDs[d] = track.trackID;
            result.trackHits[d] = track.hits;
        }
        for (size_t c = 0; c < detPool.size(); c++){
            if (!detTaken[c]){
                remainingDets.push_back(detPool[c]);
            }
        }
        detPool = std::move(remainingDets);
    }

    // BoT-SORT/BYTE default thresholds.
    static constexpr float kTrackHighThresh = 0.6f;
    static constexpr float kTrackLowThresh = 0.1f;
    static constexpr float kNewTrackThresh = 0.7f;
    static constexpr float kFirstMatchMinIoU = 0.2f;
    static constexpr float kSecondMatchMinIoU = 0.5f;
    static constexpr int kTrackBufferFrames = 30;
    static constexpr double kFirstMatchBufferScale = 1.5;  // C-BIoU box expansion for the first association.

    std::vector<Track> _tracks;
    int _nextTrackID = 0;
    unsigned long long _frameCount = 0;
};

}  // end of namespace eventobjectslam

#endif  // EVENTOBJECTSLAM_BOTSORTTRACKER_H
