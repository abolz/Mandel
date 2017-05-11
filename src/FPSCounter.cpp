#include "FPSCounter.h"

#ifdef _WIN32
#include <windows.h>
#include <mmsystem.h>
#else
#include <time.h>
#endif

//--------------------------------------------------------------------------------------------------
//
//--------------------------------------------------------------------------------------------------
FPSCounter::FPSCounter()
    : m_numframes(0)
    , m_start(0.0)
    , m_last(0.0)
    , m_fps(0.0)
{
#ifdef _WIN32
    timeBeginPeriod(1);
#endif
}

//--------------------------------------------------------------------------------------------------
//
//--------------------------------------------------------------------------------------------------
FPSCounter::~FPSCounter()
{
#ifdef _WIN32
    timeEndPeriod(1);
#endif
}

//--------------------------------------------------------------------------------------------------
//
//--------------------------------------------------------------------------------------------------
double FPSCounter::getTime()
{
#ifdef _WIN32
    return (1.0/1000.0) * (double)timeGetTime();
#else
    struct timespec ts;

    clock_gettime(CLOCK_REALTIME, &ts);

    return (double)ts.tv_sec + (1.0/1000000000.0) * (double)ts.tv_nsec;
#endif
}

//--------------------------------------------------------------------------------------------------
//
//--------------------------------------------------------------------------------------------------
double FPSCounter::registerFrame()
{
    // increase frame counter
    m_numframes++;

    double now = getTime();

    // update FPS value every few seconds
    double elapsed = now - m_start;
    if (elapsed > 0.25)
    {
        m_fps = (double)m_numframes / elapsed;
        m_numframes = 0;
        m_start = now;
    }

    // returns the time in seconds since the last call to registerFrame()
    double dt = now - m_last;
    m_last = now;

    return dt;
}
