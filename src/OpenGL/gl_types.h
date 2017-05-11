#ifndef OPENGL_GL_TYPES_H
#define OPENGL_GL_TYPES_H 1

#include <stddef.h>
#include <stdint.h>

#ifndef APIENTRY
#ifdef _WIN32
#ifndef WINAPI
#define WINAPI __stdcall
#endif
#define APIENTRY WINAPI
#else
#define APIENTRY
#endif
#endif

#ifndef GLAPIENTRY
#define GLAPIENTRY APIENTRY
#endif

#ifdef _WIN32
#ifndef DECLSPEC_IMPORT
#define DECLSPEC_IMPORT __declspec(dllimport)
#endif
#ifndef WINGDIAPI
#define WINGDIAPI DECLSPEC_IMPORT
#endif
#endif

#ifndef GL_GLEXT_LEGACY
#define GL_GLEXT_LEGACY
#endif

#ifdef __APPLE__
#include <OpenGL/gl.h>
#else
#include <GL/gl.h>
#endif

// OpenGL 1.0
/* typedef void GLvoid; */
typedef unsigned int        GLenum;
typedef float               GLfloat;
typedef int                 GLint;
typedef int                 GLsizei;
typedef unsigned int        GLbitfield;
typedef double              GLdouble;
typedef unsigned int        GLuint;
typedef unsigned char       GLboolean;
typedef unsigned char       GLubyte;

// OpenGL 1.5
typedef ptrdiff_t           GLsizeiptr;
typedef ptrdiff_t           GLintptr;

// OpenGL 2.0
typedef char                GLchar;
typedef short               GLshort;
typedef signed char         GLbyte;
typedef unsigned short      GLushort;

// OpenGL 3.0
typedef unsigned short      GLhalf;

// OpenGL 3.2
typedef struct __GLsync *   GLsync;
typedef uint64_t            GLuint64;
typedef int64_t             GLint64;
typedef uint64_t            GLuint64EXT;
typedef int64_t             GLint64EXT;

// OpenGL 4.3
typedef void (GLAPIENTRY *GLDEBUGPROC) (GLenum source,
                                        GLenum type,
                                        GLuint id,
                                        GLenum severity,
                                        GLsizei length,
                                        GLchar const* message,
                                        GLvoid const* userParam);

#endif // OPENGL_GL_TYPES_H
