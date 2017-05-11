#include "OpenGL.h"

#ifndef NOMINMAX
#define NOMINMAX
#endif

#if defined(_WIN32)
#include <windows.h>
#elif defined(__APPLE__)
#include <OpenGL/glx.h>
#else
#include <GL/glx.h>
#endif
#include <cassert>
#include <algorithm>
#include <ostream>

#ifdef _MSC_VER
#pragma warning(disable : 4996)
#endif

using namespace gl;

namespace {

class FunctionPointer
{
public:
#ifdef _WIN32
    using PointerType = PROC;
#else
    using PointerType = void (*)();
#endif

    FunctionPointer(PointerType pointer)
        : pointer(pointer)
    {
    }

    template <class T>
    operator T*() const
    {
        return reinterpret_cast<T*>(pointer);
    }

    explicit operator bool() const
    {
        return pointer != nullptr;
    }

private:
    PointerType pointer;
};

} // namespace

// Returns a pointer to the given GL function
static FunctionPointer GetGLProcAddress(char const* name)
{
#ifdef _WIN32
    return wglGetProcAddress(name);
#else
#ifdef GLX_VERSION_1_4
    return glXGetProcAddress(reinterpret_cast<GLubyte const*>(name));
#else
    return glXGetProcAddressARB(reinterpret_cast<GLubyte const*>(name));
#endif
#endif
}

// Define function pointers
#define GLFUNC(NAME, RET, ARGS) decltype(NAME) NAME = 0;
#include "gl_extensions.def.h"

void InitGL(std::ostream* pout)
{
#ifdef _WIN32
    assert( wglGetCurrentContext() );
#else
    assert( glXGetCurrentContext() );
#endif

    #define GLFUNC(NAME, RET, ARGS)     NAME = GetGLProcAddress(#NAME);
    #define GLALIAS(NAME, ALIAS)        if (NAME == 0) NAME = GetGLProcAddress(#ALIAS);
    #define GLFALLBACK(NAME, FALLBACK)  if (NAME == 0) NAME = FALLBACK;

    #include "gl_extensions.def.h"

    if (pout)
    {
        auto& out = *pout;

        #define GLEXT(NAME)                 out << "Extension: " #NAME "\n";
        #define GLFUNC(NAME, RET, ARGS)     if (NAME == 0) out << "    not available: " #NAME "\n";

        #include "gl_extensions.def.h"
    }
}

void gl::init()
{
    InitGL(nullptr);
}

void gl::init(std::ostream& out)
{
    InitGL(&out);
}

unsigned gl::getVersion()
{
    // First try to use the new method to query the context version.

    int major = glGetInteger(GL_MAJOR_VERSION);
    int minor = glGetInteger(GL_MINOR_VERSION);

    if (major == 0)
    {
        // Failed.
        // Use the old method and parse the extension string.

        auto verstr = (char const*)glGetString(GL_VERSION);

        // Skip non-digits.
        // Version string has format "<major>.<minor>" or "OpenGL ES <major>.<minor>".
        while (*verstr && !('0' <= *verstr && *verstr <= '9'))
        {
            ++verstr;
        }

        if (2 != sscanf(verstr, "%d.%d", &major, &minor))
        {
            major = 0;
            minor = 0;
        }
    }

    // Validate version
    if (major <= 0 || major >= 100 || minor < 0 || minor >= 100)
    {
        major = 0;
        minor = 0;
    }

    return 100 * major + minor;
}

template <class Callback>
static bool ParseExtensionString(std::string const& str, Callback func)
{
    for (size_t S = 0, E = str.size(); S < E; /**/)
    {
        size_t I = str.find(' ', S);

        if (I == std::string::npos)
            I = E;

        if (I > S && !func(std::string(str.data() + S, I - S)))
            return false;

        S = I + 1;
    }

    return true;
}

Extensions gl::getExtensions(bool addVersions)
{
    Extensions E;

    if (OPENGL_AVAILABLE(glGetStringi))
    {
        for (int i = 0, e = glGetInteger(GL_NUM_EXTENSIONS); i != e; ++i)
        {
            E.push_back((char const*)glGetStringi(GL_EXTENSIONS, i));
        }
    }
    else
    {
        auto insert = [&](std::string str) -> bool {
            E.emplace_back(std::move(str));
            return true;
        };

        // Parse the string, add extensions to the list
        ParseExtensionString((char const*)glGetString(GL_EXTENSIONS), insert);
    }

    if (addVersions)
    {
        auto ver = getVersion();

        if (ver >= 100) E.push_back("GL_VERSION_1_0");
        if (ver >= 101) E.push_back("GL_VERSION_1_1");
        if (ver >= 102) E.push_back("GL_VERSION_1_2");
        if (ver >= 103) E.push_back("GL_VERSION_1_3");
        if (ver >= 104) E.push_back("GL_VERSION_1_4");
        if (ver >= 105) E.push_back("GL_VERSION_1_5");
        if (ver >= 200) E.push_back("GL_VERSION_2_0");
        if (ver >= 201) E.push_back("GL_VERSION_2_1");
        if (ver >= 300) E.push_back("GL_VERSION_3_0");
        if (ver >= 301) E.push_back("GL_VERSION_3_1");
        if (ver >= 302) E.push_back("GL_VERSION_3_2");
        if (ver >= 303) E.push_back("GL_VERSION_3_3");
        if (ver >= 400) E.push_back("GL_VERSION_4_0");
        if (ver >= 401) E.push_back("GL_VERSION_4_1");
        if (ver >= 402) E.push_back("GL_VERSION_4_2");
        if (ver >= 403) E.push_back("GL_VERSION_4_3");
        if (ver >= 404) E.push_back("GL_VERSION_4_4");
        if (ver >= 405) E.push_back("GL_VERSION_4_5");
//      if (ver >= 406) E.push_back("GL_VERSION_4_6");
    }

    std::sort(E.begin(), E.end());

    // Remove possible duplicates
    E.erase(std::unique(E.begin(), E.end()), E.end());

    return E;
}

bool gl::supports(Extensions const& E, std::string const& str)
{
    return std::binary_search(E.begin(), E.end(), str);
}

bool gl::supportsAll(Extensions const& E, std::string const& str)
{
    auto test = [&](std::string const& s) {
        return supports(E, s);
    };

    return ParseExtensionString(str, test);
}

bool gl::supportsAny(Extensions const& E, std::string const& str)
{
    auto test = [&](std::string const& s) {
        return !supports(E, s);
    };

    return !ParseExtensionString(str, test);
}
