#pragma once

//------------------------------------------------------------------------------
// Types/Enums
//

#include "gl_types.h"
#include "gl_enums.h"

//------------------------------------------------------------------------------
// Extensions
//

#include "gl_fallbacks.h"

// Declare function pointers
#define GLFUNC(NAME, RET, ARGS) extern RET (GLAPIENTRY* NAME) ARGS;
#include "gl_extensions.def.h"

// Test whether the given function is available
#define OPENGL_AVAILABLE(NAME) (NAME != nullptr)

//------------------------------------------------------------------------------
// Some more useful "extensions"
// In the global namespace (like the standard gl* functions)
//

inline GLint glGetInteger(GLenum pname)
{
    GLint value = 0;
    glGetIntegerv(pname, &value);
    return value;
}

inline GLint glGetInteger(GLenum pname, GLuint index)
{
    GLint value = 0;
    glGetIntegeri_v(pname, index, &value);
    return value;
}

inline GLfloat glGetFloat(GLenum pname)
{
    GLfloat value = 0;
    glGetFloatv(pname, &value);
    return value;
}

inline GLfloat glGetFloat(GLenum pname, GLuint index)
{
    GLfloat value = 0;
    glGetFloati_v(pname, index, &value);
    return value;
}

inline void glSetEnabled(GLenum cap, GLboolean enable)
{
    if (enable)
        glEnable(cap);
    else
        glDisable(cap);
}
