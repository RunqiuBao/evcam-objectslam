#pragma once

#include <vector>

namespace tooldetectobject{

struct EventLineModDetection{

/** \brief Constructor. */
EventLineModDetection () : x (0), y (0), templateIndex (0), score (0.0f), scale (1.0f) {}

/** \brief x-position of the detection. */
int x;
/** \brief y-position of the detection. */
int y;
/** \brief index (ID) of the detected template. */
int templateIndex;
/** \brief score of the detection. */
float score;
/** \brief scale at which the template was detected. */
float scale;

};

class EventLineMod{

public:
    /** \brief Constructor */
    EventLineMod ();

    /** \brief Destructor */
    virtual ~EventLineMod ();

    int AddTemplates();

    void DetectTemplatesSemiScaleInvariant(
        // const EventLineModality & evmodality,
        std::vector<EventLineModDetection> & detections,
        float min_scale = 0.6944444f,
        float max_scale = 1.44f,
        float scale_multiplier = 1.2f
    ) const;

};

};  // end of namespace tooldetect
