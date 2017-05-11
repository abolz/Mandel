#pragma once

extern "C" {

// OpenGL 4.5

void            GLAPIENTRY glEnableVertexArrayAttrib_IMPL               (GLuint va, GLuint index);
void            GLAPIENTRY glDisableVertexArrayAttrib_IMPL              (GLuint va, GLuint index);
void            GLAPIENTRY glVertexArrayElementBuffer_IMPL              (GLuint va, GLuint buffer);
void            GLAPIENTRY glVertexArrayVertexAttribOffset_IMPL         (GLuint va, GLuint buffer, GLuint index, GLint size, GLenum type, GLboolean normalized, GLsizei stride, GLintptr offset);
void            GLAPIENTRY glCreateBuffers_IMPL                         (GLsizei n, GLuint* buffers);
void            GLAPIENTRY glCreateRenderbuffers_IMPL                   (GLsizei n, GLuint* renderbuffers);
void            GLAPIENTRY glCreateTextures_IMPL                        (GLenum target, GLsizei n, GLuint* textures);
void            GLAPIENTRY glCreateVertexArrays_IMPL                    (GLsizei n, GLuint* arrays);
void            GLAPIENTRY glCreateProgramPipelines_IMPL                (GLsizei n, GLuint* pipelines);
void            GLAPIENTRY glCreateSamplers_IMPL                        (GLsizei n, GLuint* samplers);
void            GLAPIENTRY glCreateQueries_IMPL                         (GLenum target, GLsizei n, GLuint* queries);
void            GLAPIENTRY glCreateFramebuffers_IMPL                    (GLsizei n, GLuint* framebuffers);

// GL_EXT_direct_state_access

void            GLAPIENTRY glBindMultiTextureEXT_IMPL                   (GLenum texunit, GLenum target, GLuint texture);
void            GLAPIENTRY glTextureParameteriEXT_IMPL                  (GLuint texture, GLenum target, GLenum pname, GLint param);
void            GLAPIENTRY glTextureParameterivEXT_IMPL                 (GLuint texture, GLenum target, GLenum pname, GLint const* param);
void            GLAPIENTRY glTextureParameterfEXT_IMPL                  (GLuint texture, GLenum target, GLenum pname, GLfloat param);
void            GLAPIENTRY glTextureParameterfvEXT_IMPL                 (GLuint texture, GLenum target, GLenum pname, GLfloat const* param);
void            GLAPIENTRY glGetTextureParameterivEXT_IMPL              (GLuint texture, GLenum target, GLenum pname, GLint* params);
void            GLAPIENTRY glGetTextureParameterfvEXT_IMPL              (GLuint texture, GLenum target, GLenum pname, GLfloat* params);
void            GLAPIENTRY glGetTextureLevelParameterivEXT_IMPL         (GLuint texture, GLenum target, GLint level, GLenum pname, GLint* params);
void            GLAPIENTRY glGetTextureLevelParameterfvEXT_IMPL         (GLuint texture, GLenum target, GLint level, GLenum pname, GLfloat* params);
void            GLAPIENTRY glGetTextureImageEXT_IMPL                    (GLuint texture, GLenum target, GLint level, GLenum format, GLenum type, void* pixels);
void            GLAPIENTRY glTextureStorage1DEXT_IMPL                   (GLuint texture, GLenum target, GLsizei levels, GLenum internalformat, GLsizei width);
void            GLAPIENTRY glTextureStorage2DEXT_IMPL                   (GLuint texture, GLenum target, GLsizei levels, GLenum internalformat, GLsizei width, GLsizei height);
void            GLAPIENTRY glTextureStorage3DEXT_IMPL                   (GLuint texture, GLenum target, GLsizei levels, GLenum internalformat, GLsizei width, GLsizei height, GLsizei depth);
void            GLAPIENTRY glTextureSubImage1DEXT_IMPL                  (GLuint texture, GLenum target, GLint level, GLint xoffset, GLsizei width, GLenum format, GLenum type, void const* pixels);
void            GLAPIENTRY glTextureSubImage2DEXT_IMPL                  (GLuint texture, GLenum target, GLint level, GLint xoffset, GLint yoffset, GLsizei width, GLsizei height, GLenum format, GLenum type, void const* pixels);
void            GLAPIENTRY glTextureSubImage3DEXT_IMPL                  (GLuint texture, GLenum target, GLint level, GLint xoffset, GLint yoffset, GLint zoffset, GLsizei width, GLsizei height, GLsizei depth, GLenum format, GLenum type, void const* pixels);
void            GLAPIENTRY glGenerateTextureMipmapEXT_IMPL              (GLuint texture, GLenum target);
void            GLAPIENTRY glNamedBufferDataEXT_IMPL                    (GLuint buffer, GLsizeiptr size, void const* data, GLenum usage);
void            GLAPIENTRY glNamedBufferSubDataEXT_IMPL                 (GLuint buffer, GLintptr offset, GLsizeiptr size, void const* data);
void*           GLAPIENTRY glMapNamedBufferEXT_IMPL                     (GLuint buffer, GLenum access);
GLboolean       GLAPIENTRY glUnmapNamedBufferEXT_IMPL                   (GLuint buffer);
void            GLAPIENTRY glGetNamedBufferParameterivEXT_IMPL          (GLuint buffer, GLenum pname, GLint* params);
void            GLAPIENTRY glGetNamedBufferSubDataEXT_IMPL              (GLuint buffer, GLintptr offset, GLsizeiptr size, void* data);

} // extern "C"
