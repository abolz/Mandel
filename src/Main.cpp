#include "OpenGL/OpenGL.h"

#include <SFML/Graphics.hpp>
#include <SFML/OpenGL.hpp>

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <iostream>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include "FPSCounter.h"
#include "Mandel.h"

#ifdef _MSC_VER
#pragma warning(disable : 4996)
#endif

static const GLenum kTexInternalFormat = GL_RGBA8;
static const GLenum kTexFormat = GL_BGRA;
static const GLenum kTexType = GL_UNSIGNED_INT_8_8_8_8_REV;

static sf::RenderWindow window;
static FPSCounter       fpscounter;
static bool             vsync = true;
static bool             autoRedraw = true;
static bool             needsRedraw = true;
static bool             useDoublePrecision = false;

static unsigned int     cudaArch = 0x0100;
static unsigned int     cudaSMCount = 0;

static int              imageW = 800;
static int              imageH = 600;

static int              clientW = 0;
static int              clientH = 0;
static int              lastX = 0;
static int              lastY = 0;
static int              downX = 0;
static int              downY = 0;
static int              buttonsDown = 0;

static int              swapInterval = 0;
static int              fitClient = 1;
static int              precisionGuard = 1;
static int              blockSceduling = 1;

static double           cenX = -0.75;
static double           cenY = 0.0;
static double           radius = 4.0;
static unsigned int     maxit = 500;
static int              algorithm = MANDEL_LEVELSET;

static GLuint           tex = 0;
static GLuint           pub = 0;
static cudaGraphicsResource* pubCudaResource = nullptr;

#if 0
#define CUDA_SAFE_CALL(x) x
#else
#define CUDA_SAFE_CALL(x)                                                                               \
    if (1) {                                                                                            \
        cudaError_t err = x;                                                                            \
        if (err != cudaSuccess) {                                                                       \
            fprintf(stdout, "%s(%d) : CUDA error: %s\n", __FILE__, __LINE__, cudaGetErrorString(err));  \
            fflush(stdout);                                                                             \
            exit(-1);                                                                                   \
        }                                                                                               \
    } else ((void)(0))
#endif

//--------------------------------------------------------------------------------------------------
//
//--------------------------------------------------------------------------------------------------
static int CoresPerMultiprocessor(unsigned int arch)
{
    if (arch >= 0x0500)
        return 128;
    if (arch >= 0x0300)
        return 192;
    if (arch >= 0x0201)
        return 48;
    if (arch >= 0x0200)
        return 32;
    return 8;
}

//--------------------------------------------------------------------------------------------------
//
//--------------------------------------------------------------------------------------------------
static int ChooseDevice(int& device, unsigned int& arch, unsigned int reqArch)
{
    const double KHzToGHz = 1.0 / (1000.0 * 1000.0);

    int             dev             = -1;
    unsigned int    devArch         =  0;
    int             devCores        =  0;
    double          devClockRate    =  0;
    double          devGFlops       =  0;
    int             devCount        =  0;

    // Query number of devices with compute capability 1.0 or higher.
    // Always returns cudaSuccess.
    cudaGetDeviceCount(&devCount);

    if (devCount == 0)
    {
        return -1;
    }

    for (int i = 0; i < devCount; i++)
    {
        cudaDeviceProp prop;

        // Query device properties
        // Always succeeds
        cudaGetDeviceProperties(&prop, i);

        // Device 0 with compute capability 9999.9999 indicates emulation mode
        if (i == 0 && (prop.major == 9999 && prop.minor == 9999))
        {
            arch = 0xffff;
            device = 0;

            return 0;
        }

        // get compute capability
        unsigned int arch = (prop.major << 8) | prop.minor;

        int cores = prop.multiProcessorCount * CoresPerMultiprocessor(arch);
        double clock_rate = prop.clockRate * KHzToGHz;
        double gflops = cores * clock_rate;

        fprintf(stdout, "CUDA device %d:\n", i);
        fprintf(stdout, "  compute capability : %d.%d\n", (arch >> 8) & 0xff, arch & 0xff);
        fprintf(stdout, "  cores              : %d\n", cores);
        fprintf(stdout, "  clock rate         : %.4f GHz\n", clock_rate);
        fprintf(stdout, "  flops              : %.2f GFLOPS\n", gflops);
        fprintf(stdout, "\n");

        if (arch >= reqArch && gflops > devGFlops)
        {
            dev             = i;
            devArch         = arch;
            devGFlops       = gflops;
            devCores        = cores;
            devClockRate    = clock_rate;
        }
    }

    if (dev < 0)
    {
        fprintf(stdout, "No compatible CUDA device found.\n");
        return -1;
    }

    arch = devArch;
    device = dev;

    return 0;
}

//--------------------------------------------------------------------------------------------------
//
//--------------------------------------------------------------------------------------------------
static void Home()
{
#if 0
    cenX = -7.46456342e-001;
    cenY =  9.88973268e-002;
    radius =  9.24323546e-005;
#else
    cenX = -0.75;
    cenY = 0.0;
    radius = 4.0;
#endif
}

//--------------------------------------------------------------------------------------------------
//
//--------------------------------------------------------------------------------------------------
static void PrintPosition()
{
    fprintf(stdout, "position:\n");
    fprintf(stdout, "  x = % g\n", cenX);
    fprintf(stdout, "  y = % g\n", cenY);
    fprintf(stdout, "  r = % g\n", radius);
}

//--------------------------------------------------------------------------------------------------
//
//--------------------------------------------------------------------------------------------------
static void CreateBuffers(int w, int h)
{
    assert( tex == 0 );
    assert( pub == 0 );

    if (w <= 0 || h <= 0)
        return;

    imageW = w;
    imageH = h;

    //
    // Create texture
    //
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexImage2D(GL_TEXTURE_2D, 0, kTexInternalFormat, w, h, 0, kTexFormat, kTexType, 0);
//  glBindTexture(GL_TEXTURE_2D, 0);

    //
    // Create pixel unpack buffer
    //
    glGenBuffers(1, &pub);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pub);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, w * h * 4 * sizeof(unsigned char), 0, GL_DYNAMIC_DRAW);
//  glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    //
    // Register pixel unpack buffer with CUDA
    //
    CUDA_SAFE_CALL( cudaGraphicsGLRegisterBuffer(&pubCudaResource, pub, cudaGraphicsMapFlagsWriteDiscard) );
}

//--------------------------------------------------------------------------------------------------
//
//--------------------------------------------------------------------------------------------------
static void DeleteBuffers(void)
{
    assert( tex != 0 );
    assert( pub != 0 );

    //
    // Unregister buffer object with CUDA
    //
    CUDA_SAFE_CALL( cudaGraphicsUnregisterResource(pubCudaResource) );

    glDeleteTextures(1, &tex);
    glDeleteBuffers(1, &pub);

    tex = 0;
    pub = 0;
    pubCudaResource = 0;
}

//--------------------------------------------------------------------------------------------------
//
//--------------------------------------------------------------------------------------------------
static void SetSwapInterval(int interval)
{
    swapInterval = interval;

    window.setVerticalSyncEnabled(swapInterval ? true : false);

    fprintf(stdout, "vsync: %s\n", (swapInterval ? "on" : "off"));
}

//--------------------------------------------------------------------------------------------------
//
//--------------------------------------------------------------------------------------------------
static void SetPrecisionGuard(int enable)
{
    precisionGuard = (enable & 1);

    fprintf(stdout, "precision-guard: %s\n", precisionGuard ? "on" : "off");
}

//--------------------------------------------------------------------------------------------------
//
//--------------------------------------------------------------------------------------------------
static void SetFitClient(int enable)
{
    fitClient = (enable & 1);
    if (fitClient)
    {
        DeleteBuffers();
        CreateBuffers(clientW, clientH);
    }

    fprintf(stdout, "fit-client: %s\n", fitClient ? "on" : "off");
}

//--------------------------------------------------------------------------------------------------
//
//--------------------------------------------------------------------------------------------------
static void SetBlockSceduling(int enable)
{
    blockSceduling = (enable & 1);

    fprintf(stdout, "block-sceduling: %s\n", blockSceduling ? "on" : "off");
}

//--------------------------------------------------------------------------------------------------
//
//--------------------------------------------------------------------------------------------------
static void SetUseDoublePrecision(bool enable)
{
    useDoublePrecision = enable;

    fprintf(stdout, "using double-precision: %s\n", useDoublePrecision ? "yes" : "no");
}

//--------------------------------------------------------------------------------------------------
//
//--------------------------------------------------------------------------------------------------
static int InitGL()
{
    //gl::init(std::cerr);
    gl::init();

    glClearColor(0.0f, 0.5f, 1.0f, 1.0f);

    glEnable(GL_TEXTURE_2D);

    return 0;
}

//--------------------------------------------------------------------------------------------------
//
//--------------------------------------------------------------------------------------------------
static int InitCuda()
{
    int device = 0;
    if (ChooseDevice(device, cudaArch, 0x0305) < 0 || cudaArch == 0xffff/*emu*/)
    {
        return -1;
    }

    //
    // Records device as the device on which the active host thread executes the device code.
    // Records the thread as using OpenGL interopability.
    // If the host thread has already initialized the CUDA runtime by calling non-device management
    // runtime functions, this call returns cudaErrorSetOnActiveProcess.
    //
    if (cudaSuccess != cudaGLSetGLDevice(device))
    {
        return -1;
    }

    //
    // Get number of multi-processors; used for block-scheduling
    //

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device); // never fails

    cudaSMCount = prop.multiProcessorCount;

    //
    // Allocate buffers for rendering
    //
    CreateBuffers(imageW, imageH);

    return 0;
}

//--------------------------------------------------------------------------------------------------
//
//--------------------------------------------------------------------------------------------------
static int InitApp()
{
    SetSwapInterval(swapInterval);
    SetPrecisionGuard(precisionGuard);
    SetFitClient(fitClient);
    SetBlockSceduling(blockSceduling);
    SetUseDoublePrecision(useDoublePrecision);

    Home();

    return 0;
}

//--------------------------------------------------------------------------------------------------
//
//--------------------------------------------------------------------------------------------------
static void Cleanup()
{
    DeleteBuffers();
}

//--------------------------------------------------------------------------------------------------
//
//--------------------------------------------------------------------------------------------------
#define BTN_LEFT        0x01
#define BTN_RIGHT       0x02
#define BTN_MIDDLE      0x04

static unsigned GetMouseButtons()
{
    unsigned btns = 0;

    if (sf::Mouse::isButtonPressed(sf::Mouse::Left))
        btns |= BTN_LEFT;
    if (sf::Mouse::isButtonPressed(sf::Mouse::Right))
        btns |= BTN_RIGHT;
    if (sf::Mouse::isButtonPressed(sf::Mouse::Middle))
        btns |= BTN_MIDDLE;

    return btns;
}

//--------------------------------------------------------------------------------------------------
//
//--------------------------------------------------------------------------------------------------
static void UpdateTitle()
{
    static const size_t kBufSize = 200;

    char buf[kBufSize];
    snprintf(buf, kBufSize, "Mandel %.2f FPS (%s, maxit %d)", 
        fpscounter.getFPS(), useDoublePrecision ? "F64" : "F32", maxit);

    window.setTitle(buf);
}

//--------------------------------------------------------------------------------------------------
//
//--------------------------------------------------------------------------------------------------
static void UpdateViewport(int w, int h)
{
    clientW = w;
    clientH = h;

    glViewport(0, 0, clientW, clientH);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0.0, 1.0, 1.0, 0.0, 0.0, 1.0);

    if (fitClient)
    {
        DeleteBuffers();
        CreateBuffers(clientW, clientH);
    }

    needsRedraw = true;
}

//--------------------------------------------------------------------------------------------------
//
//--------------------------------------------------------------------------------------------------
static bool UpdateScene(double dt)
{
    const auto pos = sf::Mouse::getPosition(window);
    const auto btn = GetMouseButtons();

    // Drag the image
    if (btn & BTN_LEFT)
    {
        cenX += radius * (double)(lastX - pos.x) / (double)clientW;
        cenY -= radius * (double)(lastY - pos.y) / (double)clientW;
    }

    // Zoom in/out?
    if (btn & BTN_RIGHT)
    {
        const double eps = useDoublePrecision 
            ? std::numeric_limits<double>::epsilon()
            : std::numeric_limits<float>::epsilon();

        if (!precisionGuard || (downY > lastY/*zooming out*/ || radius > imageW * eps))
        {
            radius += 4.0 * dt * (downY - lastY) * radius / clientW;
        }
    }

    lastX = pos.x;
    lastY = pos.y;

    return true;
}

//--------------------------------------------------------------------------------------------------
//
//--------------------------------------------------------------------------------------------------
static void RenderMandelbrotSet()
{
    assert( tex != 0 );
    assert( pub != 0 );

    color4* image_d = 0;

    // Map the PBO into CUDA's address space
    size_t num_bytes;

    CUDA_SAFE_CALL( cudaGraphicsMapResources(1, &pubCudaResource, 0) );
    CUDA_SAFE_CALL( cudaGraphicsResourceGetMappedPointer((void**)&image_d, &num_bytes, pubCudaResource) );

    // Compute the image
    if (useDoublePrecision)
        cuda_mandel64(image_d, imageW, imageH, cenX, cenY, radius, maxit, algorithm, blockSceduling, cudaSMCount);
    else
        cuda_mandel32(image_d, imageW, imageH, (float)cenX, (float)cenY, (float)radius, maxit, algorithm, blockSceduling, cudaSMCount);

    // Unmap the PBO - Blocks!
    CUDA_SAFE_CALL( cudaGraphicsUnmapResources(1, &pubCudaResource, 0) );
}

//--------------------------------------------------------------------------------------------------
//
//--------------------------------------------------------------------------------------------------
static void Render()
{
    double dt = fpscounter.registerFrame();

    UpdateScene(dt);

    // Render the mandelbrot image
    RenderMandelbrotSet();

    // Unpack the PBO into the texture
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, imageW, imageH, kTexFormat, kTexType, 0);

    // Draw full screen quad
    glBegin(GL_QUADS);
    glTexCoord2f(0.0f, 0.0f); glVertex2f(0.0f, 0.0f);
    glTexCoord2f(1.0f, 0.0f); glVertex2f(1.0f, 0.0f);
    glTexCoord2f(1.0f, 1.0f); glVertex2f(1.0f, 1.0f);
    glTexCoord2f(0.0f, 1.0f); glVertex2f(0.0f, 1.0f);
    glEnd();
}

//--------------------------------------------------------------------------------------------------
//
//--------------------------------------------------------------------------------------------------
static void OnKeyPressed(sf::Event::KeyEvent const& e)
{
    switch (e.code)
    {
    case sf::Keyboard::P:
        PrintPosition();
        break;
    case sf::Keyboard::H:
        Home();
        needsRedraw = true;
        break;
    case sf::Keyboard::L:
        algorithm = MANDEL_LEVELSET;
        needsRedraw = true;
        break;
    case sf::Keyboard::C:
        algorithm = MANDEL_LEVELSETCONT;
        needsRedraw = true;
        break;
    case sf::Keyboard::D:
        algorithm = MANDEL_DISTEST;
        needsRedraw = true;
        break;
    case sf::Keyboard::N:
        algorithm = MANDEL_COMPARE;
        needsRedraw = true;
        break;
    case sf::Keyboard::V:
        vsync = !vsync;
        SetSwapInterval(vsync ? 1 : 0);
        break;
    case sf::Keyboard::A:
        SetPrecisionGuard(precisionGuard ^ 1);
        needsRedraw = true;
        break;
    case sf::Keyboard::B:
        SetBlockSceduling(blockSceduling ^ 1);
        needsRedraw = true;
        break;
    case sf::Keyboard::X:
        SetUseDoublePrecision(!useDoublePrecision);
        needsRedraw = true;
        break;
    case sf::Keyboard::Up:
        maxit += 100;
        if (maxit > 20000)
            maxit = 20000;
        needsRedraw = true;
        break;
    case sf::Keyboard::Down:
        maxit -= 100;
        if (maxit < 100)
            maxit = 100;
        needsRedraw = true;
        break;
    case sf::Keyboard::F:
        SetFitClient(fitClient ^ 1);
        needsRedraw = true;
        break;
    default:
        break;
    }
}

//--------------------------------------------------------------------------------------------------
//
//--------------------------------------------------------------------------------------------------
static void OnKeyReleased(sf::Event::KeyEvent const& /*e*/)
{
}

//--------------------------------------------------------------------------------------------------
//
//--------------------------------------------------------------------------------------------------
static void OnMouseButtonDown(sf::Event::MouseButtonEvent const& e)
{
    downX = lastX = e.x;
    downY = lastY = e.y;
}

//--------------------------------------------------------------------------------------------------
//
//--------------------------------------------------------------------------------------------------
static void OnMouseButtonUp(sf::Event::MouseButtonEvent const& /*e*/)
{
}

//--------------------------------------------------------------------------------------------------
//
//--------------------------------------------------------------------------------------------------
static void OnMouseMove(sf::Event::MouseMoveEvent const& /*e*/)
{
}

//--------------------------------------------------------------------------------------------------
//
//--------------------------------------------------------------------------------------------------
static void OnMouseWheelMoved(sf::Event::MouseWheelEvent const& e)
{
}

//--------------------------------------------------------------------------------------------------
//
//--------------------------------------------------------------------------------------------------
static void OnResize(sf::Event::SizeEvent const& e)
{
    int w = static_cast<int>(e.width);
    int h = static_cast<int>(e.height);

    UpdateViewport(w, h);
}

//--------------------------------------------------------------------------------------------------
//
//--------------------------------------------------------------------------------------------------
static bool ProcessEvent(sf::Event const& e)
{
    switch (e.type)
    {
    case sf::Event::Closed:
        return false;
    case sf::Event::Resized:
        OnResize(e.size);
        break;
    case sf::Event::KeyPressed:
        OnKeyPressed(e.key);
        break;
    case sf::Event::KeyReleased:
        OnKeyReleased(e.key);
        break;
    case sf::Event::MouseButtonPressed:
        OnMouseButtonDown(e.mouseButton);
        break;
    case sf::Event::MouseButtonReleased:
        OnMouseButtonUp(e.mouseButton);
        break;
    case sf::Event::MouseMoved:
        OnMouseMove(e.mouseMove);
        needsRedraw = GetMouseButtons() != 0;
        break;
    case sf::Event::MouseWheelMoved:
        break;
    default:
        break;
    }

    return true;
}

//--------------------------------------------------------------------------------------------------
//
//--------------------------------------------------------------------------------------------------
int main(int argc, char* argv[])
{
    fprintf(stdout,
        "Key map:\n"
        "  P        print position\n"
        "  L        level-set algorithm\n"
        "  C        continuous level-set algorithm\n"
        "  N        comparing neighbors algorithm\n"
        "  D        distance-estimate algorithm\n"
        "  H        reset position\n"
        "  V        enable/disable vsync\n"
        "  A        enable/disable precision guard\n"
        "  B        enable/disable block sceduling\n"
        "  F        toggle fit client\n"
        "  X        switch between single- and double-precision kernels\n"
        "  up/down  increase/decrease maximum number of iterations\n"
        "\n"
        "Drag the image using the left mouse button. Press the right mouse\n"
        "button and move the mouse down/up to zoom in/out.\n"
        "\n"
        );

    // Save client size
    // setFitClient is called in initApp() and needs these properly set
    clientW = imageW;
    clientH = imageH;

    sf::ContextSettings settings;

    settings.majorVersion = 4;
    settings.minorVersion = 3;
    settings.depthBits = 24;
    settings.stencilBits = 8;
    settings.antialiasingLevel = 0;

    window.create(sf::VideoMode(clientW, clientH), "Mandel", sf::Style::Default, settings);
    window.setVerticalSyncEnabled(vsync);

    printf("OpenGL:\n");
    printf("  Version  : %s\n", (char const*)glGetString(GL_VERSION));
    printf("  Renderer : %s\n", (char const*)glGetString(GL_RENDERER));
    printf("  Vendor   : %s\n", (char const*)glGetString(GL_VENDOR));
    printf("\n");

    InitGL();
    if (InitCuda() < 0)
    {
        return -1;
    }
    InitApp();

    UpdateViewport(window.getSize().x, window.getSize().y);

    sf::Clock clock;
    for (;;)
    {
        if (clock.getElapsedTime().asSeconds() >= 1.0)
        {
            clock.restart();
            UpdateTitle();
        }

        Render();

        window.display();

        bool doRedraw = autoRedraw || needsRedraw;

        needsRedraw = false;

        sf::Event e;
        if (doRedraw)
        {
            while (window.pollEvent(e))
            {
                if (!ProcessEvent(e))
                    goto L_break_outer;
            }
        }
        else
        {
            window.waitEvent(e);
            do
            {
                if (!ProcessEvent(e))
                    goto L_break_outer;
            }
            while (window.pollEvent(e));
        }
    }
L_break_outer:

    Cleanup();

    window.close();

    return 0;
}
