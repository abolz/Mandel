#pragma once

class FPSCounter
{
public:
    FPSCounter();
    ~FPSCounter();

    // register a frame
    // returns the frame-delta, ie. the time in seconds since the last call to registerFrame()
    double registerFrame();

    // returns current fps
    double getFPS() const {
        return m_fps;
    }

private:
    // returns the current system time in seconds
    static double getTime();

private:
    // number of frames rendered since the last call to update()
    int m_numframes;
    // last update() time
    double m_start;
    // last registerFrame() time
    double m_last;
    // current frames per second
    double m_fps;
};
