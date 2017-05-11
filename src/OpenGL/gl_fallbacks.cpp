#include "OpenGL.h"

#include <cassert>

static GLenum GetTextureTargetBinding(GLenum target)
{
    switch (target)
    {
    case GL_TEXTURE_1D:                     return GL_TEXTURE_BINDING_1D;
    case GL_TEXTURE_1D_ARRAY:               return GL_TEXTURE_BINDING_1D_ARRAY;
    case GL_TEXTURE_2D:                     return GL_TEXTURE_BINDING_2D;
    case GL_TEXTURE_2D_ARRAY:               return GL_TEXTURE_BINDING_2D_ARRAY;
    case GL_TEXTURE_2D_MULTISAMPLE:         return GL_TEXTURE_BINDING_2D_MULTISAMPLE;
    case GL_TEXTURE_2D_MULTISAMPLE_ARRAY:   return GL_TEXTURE_BINDING_2D_MULTISAMPLE_ARRAY;
    case GL_TEXTURE_3D:                     return GL_TEXTURE_BINDING_3D;
    case GL_TEXTURE_BUFFER:                 return GL_TEXTURE_BINDING_BUFFER;
    case GL_TEXTURE_CUBE_MAP:               return GL_TEXTURE_BINDING_CUBE_MAP;
    case GL_TEXTURE_CUBE_MAP_ARRAY:         return GL_TEXTURE_BINDING_CUBE_MAP_ARRAY;
    case GL_TEXTURE_RECTANGLE:              return GL_TEXTURE_BINDING_RECTANGLE;
    default:
        assert(!"invalid texture target");
        return 0;
    }
}

extern "C"
{

// OpenGL 4.5

void GLAPIENTRY glEnableVertexArrayAttrib_IMPL(GLuint va, GLuint index)
{
    auto curr = glGetInteger(GL_VERTEX_ARRAY_BINDING);

    glBindVertexArray(va);
    glEnableVertexAttribArray(index);
    glBindVertexArray(curr);
}

void GLAPIENTRY glDisableVertexArrayAttrib_IMPL(GLuint va, GLuint index)
{
    auto curr = glGetInteger(GL_VERTEX_ARRAY_BINDING);

    glBindVertexArray(va);
    glDisableVertexAttribArray(index);
    glBindVertexArray(curr);
}

void GLAPIENTRY glVertexArrayElementBuffer_IMPL(GLuint va, GLuint buffer)
{
    auto curr = glGetInteger(GL_VERTEX_ARRAY_BINDING);

    glBindVertexArray(va);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, buffer);
    glBindVertexArray(curr);
}

void GLAPIENTRY glVertexArrayVertexAttribOffset_IMPL(GLuint va, GLuint buffer, GLuint index,
        GLint size, GLenum type, GLboolean normalized, GLsizei stride, GLintptr offset)
{
    auto currVA = glGetInteger(GL_VERTEX_ARRAY_BINDING);
    auto currAB = glGetInteger(GL_ARRAY_BUFFER_BINDING);

    glBindVertexArray(va);
    glBindBuffer(GL_ARRAY_BUFFER, buffer);
    glVertexAttribPointer(index, size, type, normalized, stride, (const void*)offset);
    glBindVertexArray(currVA);
    glBindBuffer(GL_ARRAY_BUFFER, currAB);
}

void GLAPIENTRY glCreateBuffers_IMPL(GLsizei n, GLuint* buffers)
{
    glGenBuffers(n, buffers);

    auto curr = glGetInteger(GL_ARRAY_BUFFER_BINDING);

    for (GLsizei i = 0; i < n; ++i)
        glBindBuffer(GL_ARRAY_BUFFER, buffers[n]);

    glBindBuffer(GL_ARRAY_BUFFER, curr);
}

void GLAPIENTRY glCreateRenderbuffers_IMPL(GLsizei n, GLuint* renderbuffers)
{
    glGenRenderbuffers(n, renderbuffers);

    auto curr = glGetInteger(GL_RENDERBUFFER_BINDING);

    for (GLsizei i = 0; i < n; ++i)
        glBindRenderbuffer(GL_RENDERBUFFER, renderbuffers[n]);

    glBindRenderbuffer(GL_RENDERBUFFER, curr);
}

void GLAPIENTRY glCreateTextures_IMPL(GLenum target, GLsizei n, GLuint* textures)
{
    // Texture objects may also be created with the command
    //
    //      void CreateTextures(enum target, sizei n, uint *textures);
    //
    // CreateTextures returns <n> previously unused texture names in
    // <textures>, each representing a new texture object that is a state
    // vector comprising all the state and with the same initial values listed
    // in section 8.22. The new texture objects are and remain textures of the
    // dimensionality and type specified by <target> until they are deleted.
    //
    // Errors
    //
    // An INVALID_VALUE error is generated if <n> is negative.

    glGenTextures(n, textures);

    auto binding = GetTextureTargetBinding(target);
    auto curr = glGetInteger(binding);

    for (GLsizei i = 0; i < n; ++i)
        glBindTexture(target, textures[n]);

    glBindTexture(target, curr);
}

void GLAPIENTRY glCreateVertexArrays_IMPL(GLsizei n, GLuint* arrays)
{
    glGenVertexArrays(n, arrays);

    auto curr = glGetInteger(GL_VERTEX_ARRAY_BINDING);

    for (GLsizei i = 0; i < n; ++i)
        glBindVertexArray(arrays[n]);

    glBindVertexArray(curr);
}

void GLAPIENTRY glCreateProgramPipelines_IMPL(GLsizei n, GLuint* pipelines)
{
    glGenProgramPipelines(n, pipelines);

    auto curr = glGetInteger(GL_PROGRAM_PIPELINE_BINDING);

    for (GLsizei i = 0; i < n; ++i)
        glBindProgramPipeline(pipelines[n]);

    glBindProgramPipeline(curr);
}

void GLAPIENTRY glCreateSamplers_IMPL(GLsizei n, GLuint* samplers)
{
    // Sampler objects may also be created with the command
    //
    //      void CreateSamplers(sizei n, uint *samplers);
    //
    // CreateSamplers returns <n> previously unused sampler names in
    // <samplers>, each representing a new sampler object which is a state
    // vector comprising all the state and with the same initial values listed
    // in table 23.18.
    //
    // Errors
    //
    // An INVALID_VALUE error is generated if <n> is negative.

    glGenSamplers(n, samplers);

    auto currActiveTexture = glGetInteger(GL_ACTIVE_TEXTURE);

    glActiveTexture(GL_TEXTURE0);

    auto currSampler = glGetInteger(GL_SAMPLER_BINDING);

    for (GLsizei i = 0; i < n; ++i)
        glBindSampler(GL_TEXTURE0, samplers[n]);

    glBindSampler(GL_TEXTURE0, currSampler);

    glActiveTexture(currActiveTexture);
}

void GLAPIENTRY glCreateQueries_IMPL(GLenum target, GLsizei n, GLuint* queries)
{
    // Query objects may also be created with the command
    //
    //      void CreateQueries(enum target, sizei n, uint *ids);
    //
    // CreateQueries returns <n> previously unused query object names in <ids>,
    // each representing a new query object with the specified <target>.
    // <target> may be one of SAMPLES_PASSED, ANY_SAMPLES_PASSED,
    // ANY_SAMPLES_PASSED_CONSERVATIVE, TIME_ELAPSED, TIMESTAMP,
    // PRIMITIVES_GENERATED, and TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN.
    //
    // For all values of <target>, the resulting query object will report
    // QUERY_RESULT_AVALIABLE as TRUE and QUERY_RESULT as zero.
    //
    // Errors
    //
    // An INVALID_ENUM error is generated if <target> is not one of the targets
    // listed above.
    //
    // An INVALID_VALUE error is generated if <n> is negative.

    glGenQueries(n, queries);

    for (GLsizei i = 0; i < n; ++i)
    {
        glBeginQuery(target, queries[n]);
        glEndQuery(target);
    }
}

void GLAPIENTRY glCreateFramebuffers_IMPL(GLsizei n, GLuint* framebuffers)
{
    glGenFramebuffers(n, framebuffers);

    auto curr = glGetInteger(GL_READ_FRAMEBUFFER_BINDING);

    for (GLsizei i = 0; i < n; ++i)
        glBindFramebuffer(GL_READ_FRAMEBUFFER, framebuffers[n]);

    glBindFramebuffer(GL_READ_FRAMEBUFFER, curr);
}

// GL_EXT_direct_state_access

void GLAPIENTRY glBindMultiTextureEXT_IMPL(GLenum texunit, GLenum target, GLuint texture)
{
    auto curr = glGetInteger(GL_ACTIVE_TEXTURE);

    glActiveTexture(texunit);
    glBindTexture(target, texture);
    glActiveTexture(curr);
}

void GLAPIENTRY glTextureParameteriEXT_IMPL(GLuint texture, GLenum target, GLenum pname, GLint param)
{
    auto curr = glGetInteger(GetTextureTargetBinding(target));

    glBindTexture(target, texture);
    glTexParameteri(target, pname, param);
    glBindTexture(target, curr);
}

void GLAPIENTRY glTextureParameterivEXT_IMPL(GLuint texture, GLenum target, GLenum pname, const GLint* param)
{
    auto curr = glGetInteger(GetTextureTargetBinding(target));

    glBindTexture(target, texture);
    glTexParameteriv(target, pname, param);
    glBindTexture(target, curr);
}

void GLAPIENTRY glTextureParameterfEXT_IMPL(GLuint texture, GLenum target, GLenum pname, GLfloat param)
{
    auto curr = glGetInteger(GetTextureTargetBinding(target));

    glBindTexture(target, texture);
    glTexParameterf(target, pname, param);
    glBindTexture(target, curr);
}

void GLAPIENTRY glTextureParameterfvEXT_IMPL(GLuint texture, GLenum target, GLenum pname, const GLfloat* param)
{
    auto curr = glGetInteger(GetTextureTargetBinding(target));

    glBindTexture(target, texture);
    glTexParameterfv(target, pname, param);
    glBindTexture(target, curr);
}

void GLAPIENTRY glGetTextureParameterivEXT_IMPL(GLuint texture, GLenum target, GLenum pname, GLint* params)
{
    auto curr = glGetInteger(GetTextureTargetBinding(target));

    glBindTexture(target, texture);
    glGetTexParameteriv(target, pname, params);
    glBindTexture(target, curr);
}

void GLAPIENTRY glGetTextureParameterfvEXT_IMPL(GLuint texture, GLenum target, GLenum pname, GLfloat* params)
{
    auto curr = glGetInteger(GetTextureTargetBinding(target));

    glBindTexture(target, texture);
    glGetTexParameterfv(target, pname, params);
    glBindTexture(target, curr);
}

void GLAPIENTRY glGetTextureLevelParameterivEXT_IMPL(
        GLuint texture, GLenum target, GLint level, GLenum pname, GLint* params)
{
    auto curr = glGetInteger(GetTextureTargetBinding(target));

    glBindTexture(target, texture);
    glGetTexLevelParameteriv(target, level, pname, params);
    glBindTexture(target, curr);
}

void GLAPIENTRY glGetTextureLevelParameterfvEXT_IMPL(
        GLuint texture, GLenum target, GLint level, GLenum pname, GLfloat* params)
{
    auto curr = glGetInteger(GetTextureTargetBinding(target));

    glBindTexture(target, texture);
    glGetTexLevelParameterfv(target, level, pname, params);
    glBindTexture(target, curr);
}

void GLAPIENTRY glGetTextureImageEXT_IMPL(
        GLuint texture, GLenum target, GLint level, GLenum format, GLenum type, void* pixels)
{
    auto curr = glGetInteger(GetTextureTargetBinding(target));

    glBindTexture(target, texture);
    glGetTexImage(target, level, format, type, pixels);
    glBindTexture(target, curr);
}

void GLAPIENTRY glTextureStorage1DEXT_IMPL(
        GLuint texture, GLenum target, GLsizei levels, GLenum internalformat, GLsizei width)
{
    auto curr = glGetInteger(GetTextureTargetBinding(target));

    glBindTexture(target, texture);
    glTexStorage1D(target, levels, internalformat, width);
    glBindTexture(target, curr);
}

void GLAPIENTRY glTextureStorage2DEXT_IMPL(
        GLuint texture, GLenum target, GLsizei levels, GLenum internalformat, GLsizei width, GLsizei height)
{
    auto curr = glGetInteger(GetTextureTargetBinding(target));

    glBindTexture(target, texture);
    glTexStorage2D(target, levels, internalformat, width, height);
    glBindTexture(target, curr);
}

void GLAPIENTRY glTextureStorage3DEXT_IMPL(
        GLuint texture, GLenum target, GLsizei levels, GLenum internalformat,
        GLsizei width, GLsizei height, GLsizei depth)
{
    auto curr = glGetInteger(GetTextureTargetBinding(target));

    glBindTexture(target, texture);
    glTexStorage3D(target, levels, internalformat, width, height, depth);
    glBindTexture(target, curr);
}

void GLAPIENTRY glTextureSubImage1DEXT_IMPL(
        GLuint texture, GLenum target, GLint level, GLint xoffset, GLsizei width,
        GLenum format, GLenum type, const void* pixels)
{
    auto curr = glGetInteger(GetTextureTargetBinding(target));

    glBindTexture(target, texture);
    glTexSubImage1D(target, level, xoffset, width, format, type, pixels);
    glBindTexture(target, curr);
}

void GLAPIENTRY glTextureSubImage2DEXT_IMPL(
        GLuint texture, GLenum target, GLint level, GLint xoffset, GLint yoffset,
        GLsizei width, GLsizei height, GLenum format, GLenum type, const void* pixels)
{
    auto curr = glGetInteger(GetTextureTargetBinding(target));

    glBindTexture(target, texture);
    glTexSubImage2D(target, level, xoffset, yoffset, width, height, format, type, pixels);
    glBindTexture(target, curr);
}

void GLAPIENTRY glTextureSubImage3DEXT_IMPL(
        GLuint texture, GLenum target, GLint level, GLint xoffset, GLint yoffset,
        GLint zoffset, GLsizei width, GLsizei height, GLsizei depth, GLenum format, GLenum type, const void* pixels)
{
    auto curr = glGetInteger(GetTextureTargetBinding(target));

    glBindTexture(target, texture);
    glTexSubImage3D(target, level, xoffset, yoffset, zoffset, width, height, depth, format, type, pixels);
    glBindTexture(target, curr);
}

void GLAPIENTRY glGenerateTextureMipmapEXT_IMPL(GLenum texture, GLenum target)
{
    auto curr = glGetInteger(GetTextureTargetBinding(target));

    glBindTexture(target, texture);
    glGenerateMipmap(target);
    glBindTexture(target, curr);
}

void GLAPIENTRY glNamedBufferDataEXT_IMPL(GLuint buffer, GLsizeiptr size, const void* data, GLenum usage)
{
    auto curr = glGetInteger(GL_ARRAY_BUFFER_BINDING);

    glBindBuffer(GL_ARRAY_BUFFER, buffer);
    glBufferData(GL_ARRAY_BUFFER, size, data, usage);
    glBindBuffer(GL_ARRAY_BUFFER, curr);
}

void GLAPIENTRY glNamedBufferSubDataEXT_IMPL(GLuint buffer, GLintptr offset, GLsizeiptr size, const void* data)
{
    auto curr = glGetInteger(GL_ARRAY_BUFFER_BINDING);

    glBindBuffer(GL_ARRAY_BUFFER, buffer);
    glBufferSubData(GL_ARRAY_BUFFER, offset, size, data);
    glBindBuffer(GL_ARRAY_BUFFER, curr);
}

void* GLAPIENTRY glMapNamedBufferEXT_IMPL(GLuint buffer, GLenum access)
{
    auto curr = glGetInteger(GL_ARRAY_BUFFER_BINDING);

    glBindBuffer(GL_ARRAY_BUFFER, buffer);
    auto r = glMapBuffer(GL_ARRAY_BUFFER, access);
    glBindBuffer(GL_ARRAY_BUFFER, curr);

    return r;
}

GLboolean GLAPIENTRY glUnmapNamedBufferEXT_IMPL(GLuint buffer)
{
    auto curr = glGetInteger(GL_ARRAY_BUFFER_BINDING);

    glBindBuffer(GL_ARRAY_BUFFER, buffer);
    auto r = glUnmapBuffer(GL_ARRAY_BUFFER);
    glBindBuffer(GL_ARRAY_BUFFER, curr);

    return r;
}

void GLAPIENTRY glGetNamedBufferParameterivEXT_IMPL(GLuint buffer, GLenum pname, GLint* params)
{
    auto curr = glGetInteger(GL_ARRAY_BUFFER_BINDING);

    glBindBuffer(GL_ARRAY_BUFFER, buffer);
    glGetBufferParameteriv(GL_ARRAY_BUFFER, pname, params);
    glBindBuffer(GL_ARRAY_BUFFER, curr);
}

void GLAPIENTRY glGetNamedBufferSubDataEXT_IMPL(GLuint buffer, GLintptr offset, GLsizeiptr size, void* data)
{
    auto curr = glGetInteger(GL_ARRAY_BUFFER_BINDING);

    glBindBuffer(GL_ARRAY_BUFFER, buffer);
    glGetBufferSubData(GL_ARRAY_BUFFER, offset, size, data);
    glBindBuffer(GL_ARRAY_BUFFER, curr);
}

} // extern "C"
