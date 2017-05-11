// Derived from glext.h. Original license follows.

/*
** Copyright (c) 2013-2014 The Khronos Group Inc.
**
** Permission is hereby granted, free of charge, to any person obtaining a
** copy of this software and/or associated documentation files (the
** "Materials"), to deal in the Materials without restriction, including
** without limitation the rights to use, copy, modify, merge, publish,
** distribute, sublicense, and/or sell copies of the Materials, and to
** permit persons to whom the Materials are furnished to do so, subject to
** the following conditions:
**
** The above copyright notice and this permission notice shall be included
** in all copies or substantial portions of the Materials.
**
** THE MATERIALS ARE PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
** EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
** MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
** IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
** CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
** TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
** MATERIALS OR THE USE OR OTHER DEALINGS IN THE MATERIALS.
*/

#ifndef GLEXT
#define GLEXT(NAME)
#endif

#ifndef GLFUNC
#define GLFUNC(NAME, RET, ARGS)
#endif

#ifndef GLFALLBACK
#define GLFALLBACK(NAME, FALLBACK)
#endif

#ifndef GLALIAS
#define GLALIAS(NAME, ALIAS)
#endif

//------------------------------------------------------------------------------
// OpenGL 1.2
//

// EXT_bgra
// EXT_draw_range_elements
// EXT_packed_pixels
// EXT_rescale_normal
// EXT_separate_specular_color
// EXT_texture3D
// SGIS_texture_edge_clamp
// SGIS_texture_lod

#if !defined(GL_VERSION_1_2) || (GL_VERSION_1_2 == 0xDLL)
#define GL_VERSION_1_2 0xDLL

GLEXT(GL_VERSION_1_2)

GLFUNC(glDrawRangeElements,                                     void,               (GLenum mode, GLuint start, GLuint end, GLsizei count, GLenum type, const void *indices))
GLFUNC(glTexImage3D,                                            void,               (GLenum target, GLint level, GLint internalformat, GLsizei width, GLsizei height, GLsizei depth, GLint border, GLenum format, GLenum type, const void *pixels))
GLFUNC(glTexSubImage3D,                                         void,               (GLenum target, GLint level, GLint xoffset, GLint yoffset, GLint zoffset, GLsizei width, GLsizei height, GLsizei depth, GLenum format, GLenum type, const void *pixels))
GLFUNC(glCopyTexSubImage3D,                                     void,               (GLenum target, GLint level, GLint xoffset, GLint yoffset, GLint zoffset, GLint x, GLint y, GLsizei width, GLsizei height))

#endif

//------------------------------------------------------------------------------
// OpenGL 1.3
//

// ARB_multisample
// ARB_multitexture
// ARB_texture_border_clamp
// ARB_texture_compression
// ARB_texture_cube_map
// ARB_texture_env_add
// ARB_texture_env_combine
// ARB_texture_env_dot3
// ARB_transpose_matrix

#if !defined(GL_VERSION_1_3) || (GL_VERSION_1_3 == 0xDLL)
#define GL_VERSION_1_3 0xDLL

GLEXT(GL_VERSION_1_3)

GLFUNC(glActiveTexture,                                         void,               (GLenum texture))
GLFUNC(glSampleCoverage,                                        void,               (GLfloat value, GLboolean invert))
GLFUNC(glCompressedTexImage3D,                                  void,               (GLenum target, GLint level, GLenum internalformat, GLsizei width, GLsizei height, GLsizei depth, GLint border, GLsizei imageSize, const void *data))
GLFUNC(glCompressedTexImage2D,                                  void,               (GLenum target, GLint level, GLenum internalformat, GLsizei width, GLsizei height, GLint border, GLsizei imageSize, const void *data))
GLFUNC(glCompressedTexImage1D,                                  void,               (GLenum target, GLint level, GLenum internalformat, GLsizei width, GLint border, GLsizei imageSize, const void *data))
GLFUNC(glCompressedTexSubImage3D,                               void,               (GLenum target, GLint level, GLint xoffset, GLint yoffset, GLint zoffset, GLsizei width, GLsizei height, GLsizei depth, GLenum format, GLsizei imageSize, const void *data))
GLFUNC(glCompressedTexSubImage2D,                               void,               (GLenum target, GLint level, GLint xoffset, GLint yoffset, GLsizei width, GLsizei height, GLenum format, GLsizei imageSize, const void *data))
GLFUNC(glCompressedTexSubImage1D,                               void,               (GLenum target, GLint level, GLint xoffset, GLsizei width, GLenum format, GLsizei imageSize, const void *data))
GLFUNC(glGetCompressedTexImage,                                 void,               (GLenum target, GLint level, void *img))
//GLFUNC(glLoadTransposeMatrixf,                                  void,               (const GLfloat *m))
//GLFUNC(glLoadTransposeMatrixd,                                  void,               (const GLdouble *m))
//GLFUNC(glMultTransposeMatrixf,                                  void,               (const GLfloat *m))
//GLFUNC(glMultTransposeMatrixd,                                  void,               (const GLdouble *m))

#endif

//------------------------------------------------------------------------------
// OpenGL 1.4
//

// ARB_depth_texture
// ARB_point_parameters
// ARB_shadow
// ARB_texture_env_crossbar
// ARB_texture_mirrored_repeat
// ARB_window_pos
// EXT_blend_color
// EXT_blend_func_separate
// EXT_blend_minmax
// EXT_blend_subtract
// EXT_fog_coord
// EXT_multi_draw_arrays
// EXT_secondary_color
// EXT_stencil_wrap
// NV_blend_square
// SGIS_generate_mipmap

#if !defined(GL_VERSION_1_4) || (GL_VERSION_1_4 == 0xDLL)
#define GL_VERSION_1_4 0xDLL

GLEXT(GL_VERSION_1_4)

GLFUNC(glBlendFuncSeparate,                                     void,               (GLenum sfactorRGB, GLenum dfactorRGB, GLenum sfactorAlpha, GLenum dfactorAlpha))
GLFUNC(glMultiDrawArrays,                                       void,               (GLenum mode, const GLint *first, const GLsizei *count, GLsizei drawcount))
GLFUNC(glMultiDrawElements,                                     void,               (GLenum mode, const GLsizei *count, GLenum type, const void *const*indices, GLsizei drawcount))
//GLFUNC(glPointParameterf,                                       void,               (GLenum pname, GLfloat param))
//GLFUNC(glPointParameterfv,                                      void,               (GLenum pname, const GLfloat *params))
//GLFUNC(glPointParameteri,                                       void,               (GLenum pname, GLint param))
//GLFUNC(glPointParameteriv,                                      void,               (GLenum pname, const GLint *params))
GLFUNC(glBlendColor,                                            void,               (GLfloat red, GLfloat green, GLfloat blue, GLfloat alpha))
GLFUNC(glBlendEquation,                                         void,               (GLenum mode))

#endif

//------------------------------------------------------------------------------
// OpenGL 1.5
//

// ARB_occlusion_query
// ARB_vertex_buffer_object
// EXT_shadow_funcs

#if !defined(GL_VERSION_1_5) || (GL_VERSION_1_5 == 0xDLL)
#define GL_VERSION_1_5 0xDLL

GLEXT(GL_VERSION_1_5)

GLFUNC(glGenQueries,                                            void,               (GLsizei n, GLuint *ids))
GLFUNC(glDeleteQueries,                                         void,               (GLsizei n, const GLuint *ids))
GLFUNC(glIsQuery,                                               GLboolean,          (GLuint id))
GLFUNC(glBeginQuery,                                            void,               (GLenum target, GLuint id))
GLFUNC(glEndQuery,                                              void,               (GLenum target))
GLFUNC(glGetQueryiv,                                            void,               (GLenum target, GLenum pname, GLint *params))
GLFUNC(glGetQueryObjectiv,                                      void,               (GLuint id, GLenum pname, GLint *params))
GLFUNC(glGetQueryObjectuiv,                                     void,               (GLuint id, GLenum pname, GLuint *params))
GLFUNC(glBindBuffer,                                            void,               (GLenum target, GLuint buffer))
GLFUNC(glDeleteBuffers,                                         void,               (GLsizei n, const GLuint *buffers))
GLFUNC(glGenBuffers,                                            void,               (GLsizei n, GLuint *buffers))
GLFUNC(glIsBuffer,                                              GLboolean,          (GLuint buffer))
GLFUNC(glBufferData,                                            void,               (GLenum target, GLsizeiptr size, const void *data, GLenum usage))
GLFUNC(glBufferSubData,                                         void,               (GLenum target, GLintptr offset, GLsizeiptr size, const void *data))
GLFUNC(glGetBufferSubData,                                      void,               (GLenum target, GLintptr offset, GLsizeiptr size, void *data))
GLFUNC(glMapBuffer,                                             void *,             (GLenum target, GLenum access))
GLFUNC(glUnmapBuffer,                                           GLboolean,          (GLenum target))
GLFUNC(glGetBufferParameteriv,                                  void,               (GLenum target, GLenum pname, GLint *params))
GLFUNC(glGetBufferPointerv,                                     void,               (GLenum target, GLenum pname, void **params))

#endif

//------------------------------------------------------------------------------
// OpenGL 2.0
//

// ARB_draw_buffers
// ARB_fragment_shader
// ARB_point_sprite
// ARB_shader_objects
// ARB_shading_language_100
// ARB_texture_non_power_of_two
// ARB_vertex_shader
// ATI_separate_stencil
// EXT_blend_equation_separate
// EXT_stencil_two_side

#if !defined(GL_VERSION_2_0) || (GL_VERSION_2_0 == 0xDLL)
#define GL_VERSION_2_0 0xDLL

GLEXT(GL_VERSION_2_0)

GLFUNC(glBlendEquationSeparate,                                 void,               (GLenum modeRGB, GLenum modeAlpha))
GLFUNC(glDrawBuffers,                                           void,               (GLsizei n, const GLenum *bufs))
GLFUNC(glStencilOpSeparate,                                     void,               (GLenum face, GLenum sfail, GLenum dpfail, GLenum dppass))
GLFUNC(glStencilFuncSeparate,                                   void,               (GLenum face, GLenum func, GLint ref, GLuint mask))
GLFUNC(glStencilMaskSeparate,                                   void,               (GLenum face, GLuint mask))
GLFUNC(glAttachShader,                                          void,               (GLuint program, GLuint shader))
GLFUNC(glBindAttribLocation,                                    void,               (GLuint program, GLuint index, const GLchar *name))
GLFUNC(glCompileShader,                                         void,               (GLuint shader))
GLFUNC(glCreateProgram,                                         GLuint,             (void))
GLFUNC(glCreateShader,                                          GLuint,             (GLenum type))
GLFUNC(glDeleteProgram,                                         void,               (GLuint program))
GLFUNC(glDeleteShader,                                          void,               (GLuint shader))
GLFUNC(glDetachShader,                                          void,               (GLuint program, GLuint shader))
GLFUNC(glDisableVertexAttribArray,                              void,               (GLuint index))
GLFUNC(glEnableVertexAttribArray,                               void,               (GLuint index))
GLFUNC(glGetActiveAttrib,                                       void,               (GLuint program, GLuint index, GLsizei bufSize, GLsizei *length, GLint *size, GLenum *type, GLchar *name))
GLFUNC(glGetActiveUniform,                                      void,               (GLuint program, GLuint index, GLsizei bufSize, GLsizei *length, GLint *size, GLenum *type, GLchar *name))
GLFUNC(glGetAttachedShaders,                                    void,               (GLuint program, GLsizei maxCount, GLsizei *count, GLuint *shaders))
GLFUNC(glGetAttribLocation,                                     GLint,              (GLuint program, const GLchar *name))
GLFUNC(glGetProgramiv,                                          void,               (GLuint program, GLenum pname, GLint *params))
GLFUNC(glGetProgramInfoLog,                                     void,               (GLuint program, GLsizei bufSize, GLsizei *length, GLchar *infoLog))
GLFUNC(glGetShaderiv,                                           void,               (GLuint shader, GLenum pname, GLint *params))
GLFUNC(glGetShaderInfoLog,                                      void,               (GLuint shader, GLsizei bufSize, GLsizei *length, GLchar *infoLog))
GLFUNC(glGetShaderSource,                                       void,               (GLuint shader, GLsizei bufSize, GLsizei *length, GLchar *source))
GLFUNC(glGetUniformLocation,                                    GLint,              (GLuint program, const GLchar *name))
GLFUNC(glGetUniformfv,                                          void,               (GLuint program, GLint location, GLfloat *params))
GLFUNC(glGetUniformiv,                                          void,               (GLuint program, GLint location, GLint *params))
GLFUNC(glGetVertexAttribdv,                                     void,               (GLuint index, GLenum pname, GLdouble *params))
GLFUNC(glGetVertexAttribfv,                                     void,               (GLuint index, GLenum pname, GLfloat *params))
GLFUNC(glGetVertexAttribiv,                                     void,               (GLuint index, GLenum pname, GLint *params))
GLFUNC(glGetVertexAttribPointerv,                               void,               (GLuint index, GLenum pname, void **pointer))
GLFUNC(glIsProgram,                                             GLboolean,          (GLuint program))
GLFUNC(glIsShader,                                              GLboolean,          (GLuint shader))
GLFUNC(glLinkProgram,                                           void,               (GLuint program))
GLFUNC(glShaderSource,                                          void,               (GLuint shader, GLsizei count, const GLchar *const*string, const GLint *length))
GLFUNC(glUseProgram,                                            void,               (GLuint program))
GLFUNC(glUniform1f,                                             void,               (GLint location, GLfloat v0))
GLFUNC(glUniform2f,                                             void,               (GLint location, GLfloat v0, GLfloat v1))
GLFUNC(glUniform3f,                                             void,               (GLint location, GLfloat v0, GLfloat v1, GLfloat v2))
GLFUNC(glUniform4f,                                             void,               (GLint location, GLfloat v0, GLfloat v1, GLfloat v2, GLfloat v3))
GLFUNC(glUniform1i,                                             void,               (GLint location, GLint v0))
GLFUNC(glUniform2i,                                             void,               (GLint location, GLint v0, GLint v1))
GLFUNC(glUniform3i,                                             void,               (GLint location, GLint v0, GLint v1, GLint v2))
GLFUNC(glUniform4i,                                             void,               (GLint location, GLint v0, GLint v1, GLint v2, GLint v3))
GLFUNC(glUniform1fv,                                            void,               (GLint location, GLsizei count, const GLfloat *value))
GLFUNC(glUniform2fv,                                            void,               (GLint location, GLsizei count, const GLfloat *value))
GLFUNC(glUniform3fv,                                            void,               (GLint location, GLsizei count, const GLfloat *value))
GLFUNC(glUniform4fv,                                            void,               (GLint location, GLsizei count, const GLfloat *value))
GLFUNC(glUniform1iv,                                            void,               (GLint location, GLsizei count, const GLint *value))
GLFUNC(glUniform2iv,                                            void,               (GLint location, GLsizei count, const GLint *value))
GLFUNC(glUniform3iv,                                            void,               (GLint location, GLsizei count, const GLint *value))
GLFUNC(glUniform4iv,                                            void,               (GLint location, GLsizei count, const GLint *value))
GLFUNC(glUniformMatrix2fv,                                      void,               (GLint location, GLsizei count, GLboolean transpose, const GLfloat *value))
GLFUNC(glUniformMatrix3fv,                                      void,               (GLint location, GLsizei count, GLboolean transpose, const GLfloat *value))
GLFUNC(glUniformMatrix4fv,                                      void,               (GLint location, GLsizei count, GLboolean transpose, const GLfloat *value))
GLFUNC(glValidateProgram,                                       void,               (GLuint program))
GLFUNC(glVertexAttrib1d,                                        void,               (GLuint index, GLdouble x))
GLFUNC(glVertexAttrib1dv,                                       void,               (GLuint index, const GLdouble *v))
GLFUNC(glVertexAttrib1f,                                        void,               (GLuint index, GLfloat x))
GLFUNC(glVertexAttrib1fv,                                       void,               (GLuint index, const GLfloat *v))
GLFUNC(glVertexAttrib1s,                                        void,               (GLuint index, GLshort x))
GLFUNC(glVertexAttrib1sv,                                       void,               (GLuint index, const GLshort *v))
GLFUNC(glVertexAttrib2d,                                        void,               (GLuint index, GLdouble x, GLdouble y))
GLFUNC(glVertexAttrib2dv,                                       void,               (GLuint index, const GLdouble *v))
GLFUNC(glVertexAttrib2f,                                        void,               (GLuint index, GLfloat x, GLfloat y))
GLFUNC(glVertexAttrib2fv,                                       void,               (GLuint index, const GLfloat *v))
GLFUNC(glVertexAttrib2s,                                        void,               (GLuint index, GLshort x, GLshort y))
GLFUNC(glVertexAttrib2sv,                                       void,               (GLuint index, const GLshort *v))
GLFUNC(glVertexAttrib3d,                                        void,               (GLuint index, GLdouble x, GLdouble y, GLdouble z))
GLFUNC(glVertexAttrib3dv,                                       void,               (GLuint index, const GLdouble *v))
GLFUNC(glVertexAttrib3f,                                        void,               (GLuint index, GLfloat x, GLfloat y, GLfloat z))
GLFUNC(glVertexAttrib3fv,                                       void,               (GLuint index, const GLfloat *v))
GLFUNC(glVertexAttrib3s,                                        void,               (GLuint index, GLshort x, GLshort y, GLshort z))
GLFUNC(glVertexAttrib3sv,                                       void,               (GLuint index, const GLshort *v))
GLFUNC(glVertexAttrib4Nbv,                                      void,               (GLuint index, const GLbyte *v))
GLFUNC(glVertexAttrib4Niv,                                      void,               (GLuint index, const GLint *v))
GLFUNC(glVertexAttrib4Nsv,                                      void,               (GLuint index, const GLshort *v))
GLFUNC(glVertexAttrib4Nub,                                      void,               (GLuint index, GLubyte x, GLubyte y, GLubyte z, GLubyte w))
GLFUNC(glVertexAttrib4Nubv,                                     void,               (GLuint index, const GLubyte *v))
GLFUNC(glVertexAttrib4Nuiv,                                     void,               (GLuint index, const GLuint *v))
GLFUNC(glVertexAttrib4Nusv,                                     void,               (GLuint index, const GLushort *v))
GLFUNC(glVertexAttrib4bv,                                       void,               (GLuint index, const GLbyte *v))
GLFUNC(glVertexAttrib4d,                                        void,               (GLuint index, GLdouble x, GLdouble y, GLdouble z, GLdouble w))
GLFUNC(glVertexAttrib4dv,                                       void,               (GLuint index, const GLdouble *v))
GLFUNC(glVertexAttrib4f,                                        void,               (GLuint index, GLfloat x, GLfloat y, GLfloat z, GLfloat w))
GLFUNC(glVertexAttrib4fv,                                       void,               (GLuint index, const GLfloat *v))
GLFUNC(glVertexAttrib4iv,                                       void,               (GLuint index, const GLint *v))
GLFUNC(glVertexAttrib4s,                                        void,               (GLuint index, GLshort x, GLshort y, GLshort z, GLshort w))
GLFUNC(glVertexAttrib4sv,                                       void,               (GLuint index, const GLshort *v))
GLFUNC(glVertexAttrib4ubv,                                      void,               (GLuint index, const GLubyte *v))
GLFUNC(glVertexAttrib4uiv,                                      void,               (GLuint index, const GLuint *v))
GLFUNC(glVertexAttrib4usv,                                      void,               (GLuint index, const GLushort *v))
GLFUNC(glVertexAttribPointer,                                   void,               (GLuint index, GLint size, GLenum type, GLboolean normalized, GLsizei stride, const void *pointer))

#endif

//------------------------------------------------------------------------------
// OpenGL 2.1
//

// ARB_pixel_buffer_object
// EXT_texture_sRGB

#if !defined(GL_VERSION_2_1) || (GL_VERSION_2_1 == 0xDLL)
#define GL_VERSION_2_1 0xDLL

GLEXT(GL_VERSION_2_1)

GLFUNC(glUniformMatrix2x3fv,                                    void,               (GLint location, GLsizei count, GLboolean transpose, const GLfloat *value))
GLFUNC(glUniformMatrix3x2fv,                                    void,               (GLint location, GLsizei count, GLboolean transpose, const GLfloat *value))
GLFUNC(glUniformMatrix2x4fv,                                    void,               (GLint location, GLsizei count, GLboolean transpose, const GLfloat *value))
GLFUNC(glUniformMatrix4x2fv,                                    void,               (GLint location, GLsizei count, GLboolean transpose, const GLfloat *value))
GLFUNC(glUniformMatrix3x4fv,                                    void,               (GLint location, GLsizei count, GLboolean transpose, const GLfloat *value))
GLFUNC(glUniformMatrix4x3fv,                                    void,               (GLint location, GLsizei count, GLboolean transpose, const GLfloat *value))

#endif

//------------------------------------------------------------------------------
// OpenGL 3.0
//

// ARB_depth_buffer_float
// ARB_framebuffer_object
// ARB_framebuffer_sRGB
// ARB_half_float_vertex
// ARB_map_buffer_range
// ARB_texture_compression_rgtc
// ARB_texture_rg
// ARB_vertex_array_object

#if !defined(GL_VERSION_3_0) || (GL_VERSION_3_0 == 0xDLL)
#define GL_VERSION_3_0 0xDLL

GLEXT(GL_VERSION_3_0)

GLFUNC(glColorMaski,                                            void,               (GLuint index, GLboolean r, GLboolean g, GLboolean b, GLboolean a))
GLFUNC(glGetBooleani_v,                                         void,               (GLenum target, GLuint index, GLboolean *data))
GLFUNC(glGetIntegeri_v,                                         void,               (GLenum target, GLuint index, GLint *data))
GLFUNC(glEnablei,                                               void,               (GLenum target, GLuint index))
GLFUNC(glDisablei,                                              void,               (GLenum target, GLuint index))
GLFUNC(glIsEnabledi,                                            GLboolean,          (GLenum target, GLuint index))
GLFUNC(glBeginTransformFeedback,                                void,               (GLenum primitiveMode))
GLFUNC(glEndTransformFeedback,                                  void,               (void))
GLFUNC(glBindBufferRange,                                       void,               (GLenum target, GLuint index, GLuint buffer, GLintptr offset, GLsizeiptr size))
GLFUNC(glBindBufferBase,                                        void,               (GLenum target, GLuint index, GLuint buffer))
GLFUNC(glTransformFeedbackVaryings,                             void,               (GLuint program, GLsizei count, const GLchar *const*varyings, GLenum bufferMode))
GLFUNC(glGetTransformFeedbackVarying,                           void,               (GLuint program, GLuint index, GLsizei bufSize, GLsizei *length, GLsizei *size, GLenum *type, GLchar *name))
GLFUNC(glClampColor,                                            void,               (GLenum target, GLenum clamp))
GLFUNC(glBeginConditionalRender,                                void,               (GLuint id, GLenum mode))
GLFUNC(glEndConditionalRender,                                  void,               (void))
GLFUNC(glVertexAttribIPointer,                                  void,               (GLuint index, GLint size, GLenum type, GLsizei stride, const void *pointer))
GLFUNC(glGetVertexAttribIiv,                                    void,               (GLuint index, GLenum pname, GLint *params))
GLFUNC(glGetVertexAttribIuiv,                                   void,               (GLuint index, GLenum pname, GLuint *params))
GLFUNC(glVertexAttribI1i,                                       void,               (GLuint index, GLint x))
GLFUNC(glVertexAttribI2i,                                       void,               (GLuint index, GLint x, GLint y))
GLFUNC(glVertexAttribI3i,                                       void,               (GLuint index, GLint x, GLint y, GLint z))
GLFUNC(glVertexAttribI4i,                                       void,               (GLuint index, GLint x, GLint y, GLint z, GLint w))
GLFUNC(glVertexAttribI1ui,                                      void,               (GLuint index, GLuint x))
GLFUNC(glVertexAttribI2ui,                                      void,               (GLuint index, GLuint x, GLuint y))
GLFUNC(glVertexAttribI3ui,                                      void,               (GLuint index, GLuint x, GLuint y, GLuint z))
GLFUNC(glVertexAttribI4ui,                                      void,               (GLuint index, GLuint x, GLuint y, GLuint z, GLuint w))
GLFUNC(glVertexAttribI1iv,                                      void,               (GLuint index, const GLint *v))
GLFUNC(glVertexAttribI2iv,                                      void,               (GLuint index, const GLint *v))
GLFUNC(glVertexAttribI3iv,                                      void,               (GLuint index, const GLint *v))
GLFUNC(glVertexAttribI4iv,                                      void,               (GLuint index, const GLint *v))
GLFUNC(glVertexAttribI1uiv,                                     void,               (GLuint index, const GLuint *v))
GLFUNC(glVertexAttribI2uiv,                                     void,               (GLuint index, const GLuint *v))
GLFUNC(glVertexAttribI3uiv,                                     void,               (GLuint index, const GLuint *v))
GLFUNC(glVertexAttribI4uiv,                                     void,               (GLuint index, const GLuint *v))
GLFUNC(glVertexAttribI4bv,                                      void,               (GLuint index, const GLbyte *v))
GLFUNC(glVertexAttribI4sv,                                      void,               (GLuint index, const GLshort *v))
GLFUNC(glVertexAttribI4ubv,                                     void,               (GLuint index, const GLubyte *v))
GLFUNC(glVertexAttribI4usv,                                     void,               (GLuint index, const GLushort *v))
GLFUNC(glGetUniformuiv,                                         void,               (GLuint program, GLint location, GLuint *params))
GLFUNC(glBindFragDataLocation,                                  void,               (GLuint program, GLuint color, const GLchar *name))
GLFUNC(glGetFragDataLocation,                                   GLint,              (GLuint program, const GLchar *name))
GLFUNC(glUniform1ui,                                            void,               (GLint location, GLuint v0))
GLFUNC(glUniform2ui,                                            void,               (GLint location, GLuint v0, GLuint v1))
GLFUNC(glUniform3ui,                                            void,               (GLint location, GLuint v0, GLuint v1, GLuint v2))
GLFUNC(glUniform4ui,                                            void,               (GLint location, GLuint v0, GLuint v1, GLuint v2, GLuint v3))
GLFUNC(glUniform1uiv,                                           void,               (GLint location, GLsizei count, const GLuint *value))
GLFUNC(glUniform2uiv,                                           void,               (GLint location, GLsizei count, const GLuint *value))
GLFUNC(glUniform3uiv,                                           void,               (GLint location, GLsizei count, const GLuint *value))
GLFUNC(glUniform4uiv,                                           void,               (GLint location, GLsizei count, const GLuint *value))
GLFUNC(glTexParameterIiv,                                       void,               (GLenum target, GLenum pname, const GLint *params))
GLFUNC(glTexParameterIuiv,                                      void,               (GLenum target, GLenum pname, const GLuint *params))
GLFUNC(glGetTexParameterIiv,                                    void,               (GLenum target, GLenum pname, GLint *params))
GLFUNC(glGetTexParameterIuiv,                                   void,               (GLenum target, GLenum pname, GLuint *params))
GLFUNC(glClearBufferiv,                                         void,               (GLenum buffer, GLint drawbuffer, const GLint *value))
GLFUNC(glClearBufferuiv,                                        void,               (GLenum buffer, GLint drawbuffer, const GLuint *value))
GLFUNC(glClearBufferfv,                                         void,               (GLenum buffer, GLint drawbuffer, const GLfloat *value))
GLFUNC(glClearBufferfi,                                         void,               (GLenum buffer, GLint drawbuffer, GLfloat depth, GLint stencil))
GLFUNC(glGetStringi,                                            const GLubyte *,    (GLenum name, GLuint index))
GLFUNC(glIsRenderbuffer,                                        GLboolean,          (GLuint renderbuffer))
GLFUNC(glBindRenderbuffer,                                      void,               (GLenum target, GLuint renderbuffer))
GLFUNC(glDeleteRenderbuffers,                                   void,               (GLsizei n, const GLuint *renderbuffers))
GLFUNC(glGenRenderbuffers,                                      void,               (GLsizei n, GLuint *renderbuffers))
GLFUNC(glRenderbufferStorage,                                   void,               (GLenum target, GLenum internalformat, GLsizei width, GLsizei height))
GLFUNC(glGetRenderbufferParameteriv,                            void,               (GLenum target, GLenum pname, GLint *params))
GLFUNC(glIsFramebuffer,                                         GLboolean,          (GLuint framebuffer))
GLFUNC(glBindFramebuffer,                                       void,               (GLenum target, GLuint framebuffer))
GLFUNC(glDeleteFramebuffers,                                    void,               (GLsizei n, const GLuint *framebuffers))
GLFUNC(glGenFramebuffers,                                       void,               (GLsizei n, GLuint *framebuffers))
GLFUNC(glCheckFramebufferStatus,                                GLenum,             (GLenum target))
GLFUNC(glFramebufferTexture1D,                                  void,               (GLenum target, GLenum attachment, GLenum textarget, GLuint texture, GLint level))
GLFUNC(glFramebufferTexture2D,                                  void,               (GLenum target, GLenum attachment, GLenum textarget, GLuint texture, GLint level))
GLFUNC(glFramebufferTexture3D,                                  void,               (GLenum target, GLenum attachment, GLenum textarget, GLuint texture, GLint level, GLint zoffset))
GLFUNC(glFramebufferRenderbuffer,                               void,               (GLenum target, GLenum attachment, GLenum renderbuffertarget, GLuint renderbuffer))
GLFUNC(glGetFramebufferAttachmentParameteriv,                   void,               (GLenum target, GLenum attachment, GLenum pname, GLint *params))
GLFUNC(glGenerateMipmap,                                        void,               (GLenum target))
GLFUNC(glBlitFramebuffer,                                       void,               (GLint srcX0, GLint srcY0, GLint srcX1, GLint srcY1, GLint dstX0, GLint dstY0, GLint dstX1, GLint dstY1, GLbitfield mask, GLenum filter))
GLFUNC(glRenderbufferStorageMultisample,                        void,               (GLenum target, GLsizei samples, GLenum internalformat, GLsizei width, GLsizei height))
GLFUNC(glFramebufferTextureLayer,                               void,               (GLenum target, GLenum attachment, GLuint texture, GLint level, GLint layer))
GLFUNC(glMapBufferRange,                                        void *,             (GLenum target, GLintptr offset, GLsizeiptr length, GLbitfield access))
GLFUNC(glFlushMappedBufferRange,                                void,               (GLenum target, GLintptr offset, GLsizeiptr length))
GLFUNC(glBindVertexArray,                                       void,               (GLuint array))
GLFUNC(glDeleteVertexArrays,                                    void,               (GLsizei n, const GLuint *arrays))
GLFUNC(glGenVertexArrays,                                       void,               (GLsizei n, GLuint *arrays))
GLFUNC(glIsVertexArray,                                         GLboolean,          (GLuint array))

// Reuse EXT_framebuffer_object
//GLALIAS(glIsRenderbuffer,                                       glIsRenderbufferEXT)
//GLALIAS(glBindRenderbuffer,                                     glBindRenderbufferEXT)
//GLALIAS(glDeleteRenderbuffers,                                  glDeleteRenderbuffersEXT)
//GLALIAS(glGenRenderbuffers,                                     glGenRenderbuffersEXT)
//GLALIAS(glRenderbufferStorage,                                  glRenderbufferStorageEXT)
//GLALIAS(glGetRenderbufferParameteriv,                           glGetRenderbufferParameterivEXT)
//GLALIAS(glIsFramebuffer,                                        glIsFramebufferEXT)
//GLALIAS(glBindFramebuffer,                                      glBindFramebufferEXT)
//GLALIAS(glDeleteFramebuffers,                                   glDeleteFramebuffersEXT)
//GLALIAS(glGenFramebuffers,                                      glGenFramebuffersEXT)
//GLALIAS(glCheckFramebufferStatus,                               glCheckFramebufferStatusEXT)
//GLALIAS(glFramebufferTexture1D,                                 glFramebufferTexture1DEXT)
//GLALIAS(glFramebufferTexture2D,                                 glFramebufferTexture2DEXT)
//GLALIAS(glFramebufferTexture3D,                                 glFramebufferTexture3DEXT)
//GLALIAS(glFramebufferRenderbuffer,                              glFramebufferRenderbufferEXT)
//GLALIAS(glGetFramebufferAttachmentParameteriv,                  glGetFramebufferAttachmentParameterivEXT)
//GLALIAS(glGenerateMipmap,                                       glGenerateMipmapEXT)

#endif

//------------------------------------------------------------------------------
// OpenGL 3.1
//

// ARB_copy_buffer
// ARB_draw_instanced
// ARB_uniform_buffer_object

#if !defined(GL_VERSION_3_1) || (GL_VERSION_3_1 == 0xDLL)
#define GL_VERSION_3_1 0xDLL

GLEXT(GL_VERSION_3_1)

GLFUNC(glDrawArraysInstanced,                                   void,               (GLenum mode, GLint first, GLsizei count, GLsizei instancecount))
GLFUNC(glDrawElementsInstanced,                                 void,               (GLenum mode, GLsizei count, GLenum type, const void *indices, GLsizei instancecount))
GLFUNC(glTexBuffer,                                             void,               (GLenum target, GLenum internalformat, GLuint buffer))
GLFUNC(glPrimitiveRestartIndex,                                 void,               (GLuint index))
GLFUNC(glCopyBufferSubData,                                     void,               (GLenum readTarget, GLenum writeTarget, GLintptr readOffset, GLintptr writeOffset, GLsizeiptr size))
GLFUNC(glGetUniformIndices,                                     void,               (GLuint program, GLsizei uniformCount, const GLchar *const*uniformNames, GLuint *uniformIndices))
GLFUNC(glGetActiveUniformsiv,                                   void,               (GLuint program, GLsizei uniformCount, const GLuint *uniformIndices, GLenum pname, GLint *params))
GLFUNC(glGetActiveUniformName,                                  void,               (GLuint program, GLuint uniformIndex, GLsizei bufSize, GLsizei *length, GLchar *uniformName))
GLFUNC(glGetUniformBlockIndex,                                  GLuint,             (GLuint program, const GLchar *uniformBlockName))
GLFUNC(glGetActiveUniformBlockiv,                               void,               (GLuint program, GLuint uniformBlockIndex, GLenum pname, GLint *params))
GLFUNC(glGetActiveUniformBlockName,                             void,               (GLuint program, GLuint uniformBlockIndex, GLsizei bufSize, GLsizei *length, GLchar *uniformBlockName))
GLFUNC(glUniformBlockBinding,                                   void,               (GLuint program, GLuint uniformBlockIndex, GLuint uniformBlockBinding))

// Reuse ARB_draw_instanced
GLALIAS(glDrawArraysInstanced,                                  glDrawArraysInstancedARB)
GLALIAS(glDrawElementsInstanced,                                glDrawElementsInstancedARB)

#endif

//------------------------------------------------------------------------------
// OpenGL 3.2
//

// ARB_depth_clamp
// ARB_draw_elements_base_vertex
// ARB_fragment_coord_conventions
// ARB_provoking_vertex
// ARB_seamless_cube_map
// ARB_sync
// ARB_texture_multisample

#if !defined(GL_VERSION_3_2) || (GL_VERSION_3_2 == 0xDLL)
#define GL_VERSION_3_2 0xDLL

GLEXT(GL_VERSION_3_2)

GLFUNC(glDrawElementsBaseVertex,                                void,               (GLenum mode, GLsizei count, GLenum type, const void *indices, GLint basevertex))
GLFUNC(glDrawRangeElementsBaseVertex,                           void,               (GLenum mode, GLuint start, GLuint end, GLsizei count, GLenum type, const void *indices, GLint basevertex))
GLFUNC(glDrawElementsInstancedBaseVertex,                       void,               (GLenum mode, GLsizei count, GLenum type, const void *indices, GLsizei instancecount, GLint basevertex))
GLFUNC(glMultiDrawElementsBaseVertex,                           void,               (GLenum mode, const GLsizei *count, GLenum type, const void *const*indices, GLsizei drawcount, const GLint *basevertex))
GLFUNC(glProvokingVertex,                                       void,               (GLenum mode))
GLFUNC(glFenceSync,                                             GLsync,             (GLenum condition, GLbitfield flags))
GLFUNC(glIsSync,                                                GLboolean,          (GLsync sync))
GLFUNC(glDeleteSync,                                            void,               (GLsync sync))
GLFUNC(glClientWaitSync,                                        GLenum,             (GLsync sync, GLbitfield flags, GLuint64 timeout))
GLFUNC(glWaitSync,                                              void,               (GLsync sync, GLbitfield flags, GLuint64 timeout))
GLFUNC(glGetInteger64v,                                         void,               (GLenum pname, GLint64 *data))
GLFUNC(glGetSynciv,                                             void,               (GLsync sync, GLenum pname, GLsizei bufSize, GLsizei *length, GLint *values))
GLFUNC(glGetInteger64i_v,                                       void,               (GLenum target, GLuint index, GLint64 *data))
GLFUNC(glGetBufferParameteri64v,                                void,               (GLenum target, GLenum pname, GLint64 *params))
GLFUNC(glFramebufferTexture,                                    void,               (GLenum target, GLenum attachment, GLuint texture, GLint level))
GLFUNC(glTexImage2DMultisample,                                 void,               (GLenum target, GLsizei samples, GLenum internalformat, GLsizei width, GLsizei height, GLboolean fixedsamplelocations))
GLFUNC(glTexImage3DMultisample,                                 void,               (GLenum target, GLsizei samples, GLenum internalformat, GLsizei width, GLsizei height, GLsizei depth, GLboolean fixedsamplelocations))
GLFUNC(glGetMultisamplefv,                                      void,               (GLenum pname, GLuint index, GLfloat *val))
GLFUNC(glSampleMaski,                                           void,               (GLuint maskNumber, GLbitfield mask))

#endif

//------------------------------------------------------------------------------
// OpenGL 3.3
//

// ARB_blend_func_extended
// ARB_explicit_attrib_location
// ARB_occlusion_query2
// ARB_sampler_objects
// ARB_shader_bit_encoding
// ARB_texture_rgb10_a2ui
// ARB_texture_swizzle
// ARB_timer_query
// ARB_vertex_type_2_10_10_10_rev

#if !defined(GL_VERSION_3_3) || (GL_VERSION_3_3 == 0xDLL)
#define GL_VERSION_3_3 0xDLL

GLEXT(GL_VERSION_3_3)

GLFUNC(glBindFragDataLocationIndexed,                           void,               (GLuint program, GLuint colorNumber, GLuint index, const GLchar *name))
GLFUNC(glGetFragDataIndex,                                      GLint,              (GLuint program, const GLchar *name))
GLFUNC(glGenSamplers,                                           void,               (GLsizei count, GLuint *samplers))
GLFUNC(glDeleteSamplers,                                        void,               (GLsizei count, const GLuint *samplers))
GLFUNC(glIsSampler,                                             GLboolean,          (GLuint sampler))
GLFUNC(glBindSampler,                                           void,               (GLuint unit, GLuint sampler))
GLFUNC(glSamplerParameteri,                                     void,               (GLuint sampler, GLenum pname, GLint param))
GLFUNC(glSamplerParameteriv,                                    void,               (GLuint sampler, GLenum pname, const GLint *param))
GLFUNC(glSamplerParameterf,                                     void,               (GLuint sampler, GLenum pname, GLfloat param))
GLFUNC(glSamplerParameterfv,                                    void,               (GLuint sampler, GLenum pname, const GLfloat *param))
GLFUNC(glSamplerParameterIiv,                                   void,               (GLuint sampler, GLenum pname, const GLint *param))
GLFUNC(glSamplerParameterIuiv,                                  void,               (GLuint sampler, GLenum pname, const GLuint *param))
GLFUNC(glGetSamplerParameteriv,                                 void,               (GLuint sampler, GLenum pname, GLint *params))
GLFUNC(glGetSamplerParameterIiv,                                void,               (GLuint sampler, GLenum pname, GLint *params))
GLFUNC(glGetSamplerParameterfv,                                 void,               (GLuint sampler, GLenum pname, GLfloat *params))
GLFUNC(glGetSamplerParameterIuiv,                               void,               (GLuint sampler, GLenum pname, GLuint *params))
GLFUNC(glQueryCounter,                                          void,               (GLuint id, GLenum target))
GLFUNC(glGetQueryObjecti64v,                                    void,               (GLuint id, GLenum pname, GLint64 *params))
GLFUNC(glGetQueryObjectui64v,                                   void,               (GLuint id, GLenum pname, GLuint64 *params))
GLFUNC(glVertexAttribDivisor,                                   void,               (GLuint index, GLuint divisor))
GLFUNC(glVertexAttribP1ui,                                      void,               (GLuint index, GLenum type, GLboolean normalized, GLuint value))
GLFUNC(glVertexAttribP1uiv,                                     void,               (GLuint index, GLenum type, GLboolean normalized, const GLuint *value))
GLFUNC(glVertexAttribP2ui,                                      void,               (GLuint index, GLenum type, GLboolean normalized, GLuint value))
GLFUNC(glVertexAttribP2uiv,                                     void,               (GLuint index, GLenum type, GLboolean normalized, const GLuint *value))
GLFUNC(glVertexAttribP3ui,                                      void,               (GLuint index, GLenum type, GLboolean normalized, GLuint value))
GLFUNC(glVertexAttribP3uiv,                                     void,               (GLuint index, GLenum type, GLboolean normalized, const GLuint *value))
GLFUNC(glVertexAttribP4ui,                                      void,               (GLuint index, GLenum type, GLboolean normalized, GLuint value))
GLFUNC(glVertexAttribP4uiv,                                     void,               (GLuint index, GLenum type, GLboolean normalized, const GLuint *value))

#endif

//------------------------------------------------------------------------------
// OpenGL 4.0
//

// ARB_draw_buffers_blend
// ARB_draw_indirect
// ARB_gpu_shader5
// ARB_gpu_shader_fp64
// ARB_shader_subroutine
// ARB_tessellation_shader
// ARB_texture_buffer_object_rgb32
// ARB_texture_cube_map_array
// ARB_texture_gather
// ARB_texture_query_lod
// ARB_transform_feedback2
// ARB_transform_feedback3

#if !defined(GL_VERSION_4_0) || (GL_VERSION_4_0 == 0xDLL)
#define GL_VERSION_4_0 0xDLL

GLEXT(GL_VERSION_4_0)

GLFUNC(glMinSampleShading,                                      void,               (GLfloat value))
GLFUNC(glBlendEquationi,                                        void,               (GLuint buf, GLenum mode))
GLFUNC(glBlendEquationSeparatei,                                void,               (GLuint buf, GLenum modeRGB, GLenum modeAlpha))
GLFUNC(glBlendFunci,                                            void,               (GLuint buf, GLenum src, GLenum dst))
GLFUNC(glBlendFuncSeparatei,                                    void,               (GLuint buf, GLenum srcRGB, GLenum dstRGB, GLenum srcAlpha, GLenum dstAlpha))
GLFUNC(glDrawArraysIndirect,                                    void,               (GLenum mode, const void *indirect))
GLFUNC(glDrawElementsIndirect,                                  void,               (GLenum mode, GLenum type, const void *indirect))
GLFUNC(glUniform1d,                                             void,               (GLint location, GLdouble x))
GLFUNC(glUniform2d,                                             void,               (GLint location, GLdouble x, GLdouble y))
GLFUNC(glUniform3d,                                             void,               (GLint location, GLdouble x, GLdouble y, GLdouble z))
GLFUNC(glUniform4d,                                             void,               (GLint location, GLdouble x, GLdouble y, GLdouble z, GLdouble w))
GLFUNC(glUniform1dv,                                            void,               (GLint location, GLsizei count, const GLdouble *value))
GLFUNC(glUniform2dv,                                            void,               (GLint location, GLsizei count, const GLdouble *value))
GLFUNC(glUniform3dv,                                            void,               (GLint location, GLsizei count, const GLdouble *value))
GLFUNC(glUniform4dv,                                            void,               (GLint location, GLsizei count, const GLdouble *value))
GLFUNC(glUniformMatrix2dv,                                      void,               (GLint location, GLsizei count, GLboolean transpose, const GLdouble *value))
GLFUNC(glUniformMatrix3dv,                                      void,               (GLint location, GLsizei count, GLboolean transpose, const GLdouble *value))
GLFUNC(glUniformMatrix4dv,                                      void,               (GLint location, GLsizei count, GLboolean transpose, const GLdouble *value))
GLFUNC(glUniformMatrix2x3dv,                                    void,               (GLint location, GLsizei count, GLboolean transpose, const GLdouble *value))
GLFUNC(glUniformMatrix2x4dv,                                    void,               (GLint location, GLsizei count, GLboolean transpose, const GLdouble *value))
GLFUNC(glUniformMatrix3x2dv,                                    void,               (GLint location, GLsizei count, GLboolean transpose, const GLdouble *value))
GLFUNC(glUniformMatrix3x4dv,                                    void,               (GLint location, GLsizei count, GLboolean transpose, const GLdouble *value))
GLFUNC(glUniformMatrix4x2dv,                                    void,               (GLint location, GLsizei count, GLboolean transpose, const GLdouble *value))
GLFUNC(glUniformMatrix4x3dv,                                    void,               (GLint location, GLsizei count, GLboolean transpose, const GLdouble *value))
GLFUNC(glGetUniformdv,                                          void,               (GLuint program, GLint location, GLdouble *params))
GLFUNC(glGetSubroutineUniformLocation,                          GLint,              (GLuint program, GLenum shadertype, const GLchar *name))
GLFUNC(glGetSubroutineIndex,                                    GLuint,             (GLuint program, GLenum shadertype, const GLchar *name))
GLFUNC(glGetActiveSubroutineUniformiv,                          void,               (GLuint program, GLenum shadertype, GLuint index, GLenum pname, GLint *values))
GLFUNC(glGetActiveSubroutineUniformName,                        void,               (GLuint program, GLenum shadertype, GLuint index, GLsizei bufsize, GLsizei *length, GLchar *name))
GLFUNC(glGetActiveSubroutineName,                               void,               (GLuint program, GLenum shadertype, GLuint index, GLsizei bufsize, GLsizei *length, GLchar *name))
GLFUNC(glUniformSubroutinesuiv,                                 void,               (GLenum shadertype, GLsizei count, const GLuint *indices))
GLFUNC(glGetUniformSubroutineuiv,                               void,               (GLenum shadertype, GLint location, GLuint *params))
GLFUNC(glGetProgramStageiv,                                     void,               (GLuint program, GLenum shadertype, GLenum pname, GLint *values))
GLFUNC(glPatchParameteri,                                       void,               (GLenum pname, GLint value))
GLFUNC(glPatchParameterfv,                                      void,               (GLenum pname, const GLfloat *values))
GLFUNC(glBindTransformFeedback,                                 void,               (GLenum target, GLuint id))
GLFUNC(glDeleteTransformFeedbacks,                              void,               (GLsizei n, const GLuint *ids))
GLFUNC(glGenTransformFeedbacks,                                 void,               (GLsizei n, GLuint *ids))
GLFUNC(glIsTransformFeedback,                                   GLboolean,          (GLuint id))
GLFUNC(glPauseTransformFeedback,                                void,               (void))
GLFUNC(glResumeTransformFeedback,                               void,               (void))
GLFUNC(glDrawTransformFeedback,                                 void,               (GLenum mode, GLuint id))
GLFUNC(glDrawTransformFeedbackStream,                           void,               (GLenum mode, GLuint id, GLuint stream))
GLFUNC(glBeginQueryIndexed,                                     void,               (GLenum target, GLuint index, GLuint id))
GLFUNC(glEndQueryIndexed,                                       void,               (GLenum target, GLuint index))
GLFUNC(glGetQueryIndexediv,                                     void,               (GLenum target, GLuint index, GLenum pname, GLint *params))

// Reuse GL_ARB_draw_buffers_blend
GLALIAS(glBlendEquationi,                                       glBlendEquationiARB)
GLALIAS(glBlendEquationSeparatei,                               glBlendEquationSeparateiARB)
GLALIAS(glBlendFunci,                                           glBlendFunciARB)
GLALIAS(glBlendFuncSeparatei,                                   glBlendFuncSeparateiARB)

#endif

//------------------------------------------------------------------------------
// OpenGL 4.1
//

// ARB_ES2_compatibility
// ARB_get_program_binary
// ARB_separate_shader_objects
// ARB_shader_precision
// ARB_vertex_attrib_64bit
// ARB_viewport_array

#if !defined(GL_VERSION_4_1) || (GL_VERSION_4_1 == 0xDLL)
#define GL_VERSION_4_1 0xDLL

GLEXT(GL_VERSION_4_1)

GLFUNC(glReleaseShaderCompiler,                                 void,               (void))
GLFUNC(glShaderBinary,                                          void,               (GLsizei count, const GLuint *shaders, GLenum binaryformat, const void *binary, GLsizei length))
GLFUNC(glGetShaderPrecisionFormat,                              void,               (GLenum shadertype, GLenum precisiontype, GLint *range, GLint *precision))
GLFUNC(glDepthRangef,                                           void,               (GLfloat n, GLfloat f))
GLFUNC(glClearDepthf,                                           void,               (GLfloat d))
GLFUNC(glGetProgramBinary,                                      void,               (GLuint program, GLsizei bufSize, GLsizei *length, GLenum *binaryFormat, void *binary))
GLFUNC(glProgramBinary,                                         void,               (GLuint program, GLenum binaryFormat, const void *binary, GLsizei length))
GLFUNC(glProgramParameteri,                                     void,               (GLuint program, GLenum pname, GLint value))
GLFUNC(glUseProgramStages,                                      void,               (GLuint pipeline, GLbitfield stages, GLuint program))
GLFUNC(glActiveShaderProgram,                                   void,               (GLuint pipeline, GLuint program))
GLFUNC(glCreateShaderProgramv,                                  GLuint,             (GLenum type, GLsizei count, const GLchar *const*strings))
GLFUNC(glBindProgramPipeline,                                   void,               (GLuint pipeline))
GLFUNC(glDeleteProgramPipelines,                                void,               (GLsizei n, const GLuint *pipelines))
GLFUNC(glGenProgramPipelines,                                   void,               (GLsizei n, GLuint *pipelines))
GLFUNC(glIsProgramPipeline,                                     GLboolean,          (GLuint pipeline))
GLFUNC(glGetProgramPipelineiv,                                  void,               (GLuint pipeline, GLenum pname, GLint *params))
GLFUNC(glProgramUniform1i,                                      void,               (GLuint program, GLint location, GLint v0))
GLFUNC(glProgramUniform1iv,                                     void,               (GLuint program, GLint location, GLsizei count, const GLint *value))
GLFUNC(glProgramUniform1f,                                      void,               (GLuint program, GLint location, GLfloat v0))
GLFUNC(glProgramUniform1fv,                                     void,               (GLuint program, GLint location, GLsizei count, const GLfloat *value))
GLFUNC(glProgramUniform1d,                                      void,               (GLuint program, GLint location, GLdouble v0))
GLFUNC(glProgramUniform1dv,                                     void,               (GLuint program, GLint location, GLsizei count, const GLdouble *value))
GLFUNC(glProgramUniform1ui,                                     void,               (GLuint program, GLint location, GLuint v0))
GLFUNC(glProgramUniform1uiv,                                    void,               (GLuint program, GLint location, GLsizei count, const GLuint *value))
GLFUNC(glProgramUniform2i,                                      void,               (GLuint program, GLint location, GLint v0, GLint v1))
GLFUNC(glProgramUniform2iv,                                     void,               (GLuint program, GLint location, GLsizei count, const GLint *value))
GLFUNC(glProgramUniform2f,                                      void,               (GLuint program, GLint location, GLfloat v0, GLfloat v1))
GLFUNC(glProgramUniform2fv,                                     void,               (GLuint program, GLint location, GLsizei count, const GLfloat *value))
GLFUNC(glProgramUniform2d,                                      void,               (GLuint program, GLint location, GLdouble v0, GLdouble v1))
GLFUNC(glProgramUniform2dv,                                     void,               (GLuint program, GLint location, GLsizei count, const GLdouble *value))
GLFUNC(glProgramUniform2ui,                                     void,               (GLuint program, GLint location, GLuint v0, GLuint v1))
GLFUNC(glProgramUniform2uiv,                                    void,               (GLuint program, GLint location, GLsizei count, const GLuint *value))
GLFUNC(glProgramUniform3i,                                      void,               (GLuint program, GLint location, GLint v0, GLint v1, GLint v2))
GLFUNC(glProgramUniform3iv,                                     void,               (GLuint program, GLint location, GLsizei count, const GLint *value))
GLFUNC(glProgramUniform3f,                                      void,               (GLuint program, GLint location, GLfloat v0, GLfloat v1, GLfloat v2))
GLFUNC(glProgramUniform3fv,                                     void,               (GLuint program, GLint location, GLsizei count, const GLfloat *value))
GLFUNC(glProgramUniform3d,                                      void,               (GLuint program, GLint location, GLdouble v0, GLdouble v1, GLdouble v2))
GLFUNC(glProgramUniform3dv,                                     void,               (GLuint program, GLint location, GLsizei count, const GLdouble *value))
GLFUNC(glProgramUniform3ui,                                     void,               (GLuint program, GLint location, GLuint v0, GLuint v1, GLuint v2))
GLFUNC(glProgramUniform3uiv,                                    void,               (GLuint program, GLint location, GLsizei count, const GLuint *value))
GLFUNC(glProgramUniform4i,                                      void,               (GLuint program, GLint location, GLint v0, GLint v1, GLint v2, GLint v3))
GLFUNC(glProgramUniform4iv,                                     void,               (GLuint program, GLint location, GLsizei count, const GLint *value))
GLFUNC(glProgramUniform4f,                                      void,               (GLuint program, GLint location, GLfloat v0, GLfloat v1, GLfloat v2, GLfloat v3))
GLFUNC(glProgramUniform4fv,                                     void,               (GLuint program, GLint location, GLsizei count, const GLfloat *value))
GLFUNC(glProgramUniform4d,                                      void,               (GLuint program, GLint location, GLdouble v0, GLdouble v1, GLdouble v2, GLdouble v3))
GLFUNC(glProgramUniform4dv,                                     void,               (GLuint program, GLint location, GLsizei count, const GLdouble *value))
GLFUNC(glProgramUniform4ui,                                     void,               (GLuint program, GLint location, GLuint v0, GLuint v1, GLuint v2, GLuint v3))
GLFUNC(glProgramUniform4uiv,                                    void,               (GLuint program, GLint location, GLsizei count, const GLuint *value))
GLFUNC(glProgramUniformMatrix2fv,                               void,               (GLuint program, GLint location, GLsizei count, GLboolean transpose, const GLfloat *value))
GLFUNC(glProgramUniformMatrix3fv,                               void,               (GLuint program, GLint location, GLsizei count, GLboolean transpose, const GLfloat *value))
GLFUNC(glProgramUniformMatrix4fv,                               void,               (GLuint program, GLint location, GLsizei count, GLboolean transpose, const GLfloat *value))
GLFUNC(glProgramUniformMatrix2dv,                               void,               (GLuint program, GLint location, GLsizei count, GLboolean transpose, const GLdouble *value))
GLFUNC(glProgramUniformMatrix3dv,                               void,               (GLuint program, GLint location, GLsizei count, GLboolean transpose, const GLdouble *value))
GLFUNC(glProgramUniformMatrix4dv,                               void,               (GLuint program, GLint location, GLsizei count, GLboolean transpose, const GLdouble *value))
GLFUNC(glProgramUniformMatrix2x3fv,                             void,               (GLuint program, GLint location, GLsizei count, GLboolean transpose, const GLfloat *value))
GLFUNC(glProgramUniformMatrix3x2fv,                             void,               (GLuint program, GLint location, GLsizei count, GLboolean transpose, const GLfloat *value))
GLFUNC(glProgramUniformMatrix2x4fv,                             void,               (GLuint program, GLint location, GLsizei count, GLboolean transpose, const GLfloat *value))
GLFUNC(glProgramUniformMatrix4x2fv,                             void,               (GLuint program, GLint location, GLsizei count, GLboolean transpose, const GLfloat *value))
GLFUNC(glProgramUniformMatrix3x4fv,                             void,               (GLuint program, GLint location, GLsizei count, GLboolean transpose, const GLfloat *value))
GLFUNC(glProgramUniformMatrix4x3fv,                             void,               (GLuint program, GLint location, GLsizei count, GLboolean transpose, const GLfloat *value))
GLFUNC(glProgramUniformMatrix2x3dv,                             void,               (GLuint program, GLint location, GLsizei count, GLboolean transpose, const GLdouble *value))
GLFUNC(glProgramUniformMatrix3x2dv,                             void,               (GLuint program, GLint location, GLsizei count, GLboolean transpose, const GLdouble *value))
GLFUNC(glProgramUniformMatrix2x4dv,                             void,               (GLuint program, GLint location, GLsizei count, GLboolean transpose, const GLdouble *value))
GLFUNC(glProgramUniformMatrix4x2dv,                             void,               (GLuint program, GLint location, GLsizei count, GLboolean transpose, const GLdouble *value))
GLFUNC(glProgramUniformMatrix3x4dv,                             void,               (GLuint program, GLint location, GLsizei count, GLboolean transpose, const GLdouble *value))
GLFUNC(glProgramUniformMatrix4x3dv,                             void,               (GLuint program, GLint location, GLsizei count, GLboolean transpose, const GLdouble *value))
GLFUNC(glValidateProgramPipeline,                               void,               (GLuint pipeline))
GLFUNC(glGetProgramPipelineInfoLog,                             void,               (GLuint pipeline, GLsizei bufSize, GLsizei *length, GLchar *infoLog))
GLFUNC(glVertexAttribL1d,                                       void,               (GLuint index, GLdouble x))
GLFUNC(glVertexAttribL2d,                                       void,               (GLuint index, GLdouble x, GLdouble y))
GLFUNC(glVertexAttribL3d,                                       void,               (GLuint index, GLdouble x, GLdouble y, GLdouble z))
GLFUNC(glVertexAttribL4d,                                       void,               (GLuint index, GLdouble x, GLdouble y, GLdouble z, GLdouble w))
GLFUNC(glVertexAttribL1dv,                                      void,               (GLuint index, const GLdouble *v))
GLFUNC(glVertexAttribL2dv,                                      void,               (GLuint index, const GLdouble *v))
GLFUNC(glVertexAttribL3dv,                                      void,               (GLuint index, const GLdouble *v))
GLFUNC(glVertexAttribL4dv,                                      void,               (GLuint index, const GLdouble *v))
GLFUNC(glVertexAttribLPointer,                                  void,               (GLuint index, GLint size, GLenum type, GLsizei stride, const void *pointer))
GLFUNC(glGetVertexAttribLdv,                                    void,               (GLuint index, GLenum pname, GLdouble *params))
GLFUNC(glViewportArrayv,                                        void,               (GLuint first, GLsizei count, const GLfloat *v))
GLFUNC(glViewportIndexedf,                                      void,               (GLuint index, GLfloat x, GLfloat y, GLfloat w, GLfloat h))
GLFUNC(glViewportIndexedfv,                                     void,               (GLuint index, const GLfloat *v))
GLFUNC(glScissorArrayv,                                         void,               (GLuint first, GLsizei count, const GLint *v))
GLFUNC(glScissorIndexed,                                        void,               (GLuint index, GLint left, GLint bottom, GLsizei width, GLsizei height))
GLFUNC(glScissorIndexedv,                                       void,               (GLuint index, const GLint *v))
GLFUNC(glDepthRangeArrayv,                                      void,               (GLuint first, GLsizei count, const GLdouble *v))
GLFUNC(glDepthRangeIndexed,                                     void,               (GLuint index, GLdouble n, GLdouble f))
GLFUNC(glGetFloati_v,                                           void,               (GLenum target, GLuint index, GLfloat *data))
GLFUNC(glGetDoublei_v,                                          void,               (GLenum target, GLuint index, GLdouble *data))

// Reuse EXT_direct_state_access
GLALIAS(glProgramUniform1f,                                     glProgramUniform1fEXT)
GLALIAS(glProgramUniform2f,                                     glProgramUniform2fEXT)
GLALIAS(glProgramUniform3f,                                     glProgramUniform3fEXT)
GLALIAS(glProgramUniform4f,                                     glProgramUniform4fEXT)
GLALIAS(glProgramUniform1i,                                     glProgramUniform1iEXT)
GLALIAS(glProgramUniform2i,                                     glProgramUniform2iEXT)
GLALIAS(glProgramUniform3i,                                     glProgramUniform3iEXT)
GLALIAS(glProgramUniform4i,                                     glProgramUniform4iEXT)
GLALIAS(glProgramUniform1ui,                                    glProgramUniform1uiEXT)
GLALIAS(glProgramUniform2ui,                                    glProgramUniform2uiEXT)
GLALIAS(glProgramUniform3ui,                                    glProgramUniform3uiEXT)
GLALIAS(glProgramUniform4ui,                                    glProgramUniform4uiEXT)
GLALIAS(glProgramUniform1fv,                                    glProgramUniform1fvEXT)
GLALIAS(glProgramUniform2fv,                                    glProgramUniform2fvEXT)
GLALIAS(glProgramUniform3fv,                                    glProgramUniform3fvEXT)
GLALIAS(glProgramUniform4fv,                                    glProgramUniform4fvEXT)
GLALIAS(glProgramUniform1iv,                                    glProgramUniform1ivEXT)
GLALIAS(glProgramUniform2iv,                                    glProgramUniform2ivEXT)
GLALIAS(glProgramUniform3iv,                                    glProgramUniform3ivEXT)
GLALIAS(glProgramUniform4iv,                                    glProgramUniform4ivEXT)
GLALIAS(glProgramUniform1uiv,                                   glProgramUniform1uivEXT)
GLALIAS(glProgramUniform2uiv,                                   glProgramUniform2uivEXT)
GLALIAS(glProgramUniform3uiv,                                   glProgramUniform3uivEXT)
GLALIAS(glProgramUniform4uiv,                                   glProgramUniform4uivEXT)
GLALIAS(glProgramUniformMatrix2fv,                              glProgramUniformMatrix2fvEXT)
GLALIAS(glProgramUniformMatrix3fv,                              glProgramUniformMatrix3fvEXT)
GLALIAS(glProgramUniformMatrix4fv,                              glProgramUniformMatrix4fvEXT)
GLALIAS(glProgramUniformMatrix2x3fv,                            glProgramUniformMatrix2x3fvEXT)
GLALIAS(glProgramUniformMatrix3x2fv,                            glProgramUniformMatrix3x2fvEXT)
GLALIAS(glProgramUniformMatrix2x4fv,                            glProgramUniformMatrix2x4fvEXT)
GLALIAS(glProgramUniformMatrix4x2fv,                            glProgramUniformMatrix4x2fvEXT)
GLALIAS(glProgramUniformMatrix3x4fv,                            glProgramUniformMatrix3x4fvEXT)
GLALIAS(glProgramUniformMatrix4x3fv,                            glProgramUniformMatrix4x3fvEXT)

#endif

//------------------------------------------------------------------------------
// OpenGL 4.2
//

// ARB_base_instance
// ARB_compressed_texture_pixel_storage
// ARB_conservative_depth
// ARB_internalformat_query
// ARB_map_buffer_alignment
// ARB_shader_atomic_counters
// ARB_shader_image_load_store
// ARB_shading_language_420pack
// ARB_shading_language_packing
// ARB_texture_storage
// ARB_transform_feedback_instanced

#if !defined(GL_VERSION_4_2) || (GL_VERSION_4_2 == 0xDLL)
#define GL_VERSION_4_2 0xDLL

GLEXT(GL_VERSION_4_2)

GLFUNC(glDrawArraysInstancedBaseInstance,                       void,               (GLenum mode, GLint first, GLsizei count, GLsizei instancecount, GLuint baseinstance))
GLFUNC(glDrawElementsInstancedBaseInstance,                     void,               (GLenum mode, GLsizei count, GLenum type, const void *indices, GLsizei instancecount, GLuint baseinstance))
GLFUNC(glDrawElementsInstancedBaseVertexBaseInstance,           void,               (GLenum mode, GLsizei count, GLenum type, const void *indices, GLsizei instancecount, GLint basevertex, GLuint baseinstance))
GLFUNC(glGetInternalformativ,                                   void,               (GLenum target, GLenum internalformat, GLenum pname, GLsizei bufSize, GLint *params))
GLFUNC(glGetActiveAtomicCounterBufferiv,                        void,               (GLuint program, GLuint bufferIndex, GLenum pname, GLint *params))
GLFUNC(glBindImageTexture,                                      void,               (GLuint unit, GLuint texture, GLint level, GLboolean layered, GLint layer, GLenum access, GLenum format))
GLFUNC(glMemoryBarrier,                                         void,               (GLbitfield barriers))
GLFUNC(glTexStorage1D,                                          void,               (GLenum target, GLsizei levels, GLenum internalformat, GLsizei width))
GLFUNC(glTexStorage2D,                                          void,               (GLenum target, GLsizei levels, GLenum internalformat, GLsizei width, GLsizei height))
GLFUNC(glTexStorage3D,                                          void,               (GLenum target, GLsizei levels, GLenum internalformat, GLsizei width, GLsizei height, GLsizei depth))
GLFUNC(glDrawTransformFeedbackInstanced,                        void,               (GLenum mode, GLuint id, GLsizei instancecount))
GLFUNC(glDrawTransformFeedbackStreamInstanced,                  void,               (GLenum mode, GLuint id, GLuint stream, GLsizei instancecount))

#endif

//------------------------------------------------------------------------------
// OpenGL 4.3
//

// ARB_arrays_of_arrays
// ARB_clear_buffer_object
// ARB_compute_shader
// ARB_copy_image
// ARB_ES3_compatibility
// ARB_explicit_uniform_location
// ARB_fragment_layer_viewport
// ARB_framebuffer_no_attachments
// ARB_internalformat_query2
// ARB_invalidate_subdata
// ARB_multi_draw_indirect
// ARB_program_interface_query
// ARB_robust_buffer_access_behavior
// ARB_shader_image_size
// ARB_shader_storage_buffer_object
// ARB_stencil_texturing
// ARB_texture_buffer_range
// ARB_texture_query_levels
// ARB_texture_storage_multisample
// ARB_texture_view
// ARB_vertex_attrib_binding
// KHR_debug

#if !defined(GL_VERSION_4_3) || (GL_VERSION_4_3 == 0xDLL)
#define GL_VERSION_4_3 0xDLL

GLEXT(GL_VERSION_4_3)

GLFUNC(glClearBufferData,                                       void,               (GLenum target, GLenum internalformat, GLenum format, GLenum type, const void *data))
GLFUNC(glClearBufferSubData,                                    void,               (GLenum target, GLenum internalformat, GLintptr offset, GLsizeiptr size, GLenum format, GLenum type, const void *data))
GLFUNC(glDispatchCompute,                                       void,               (GLuint num_groups_x, GLuint num_groups_y, GLuint num_groups_z))
GLFUNC(glDispatchComputeIndirect,                               void,               (GLintptr indirect))
GLFUNC(glCopyImageSubData,                                      void,               (GLuint srcName, GLenum srcTarget, GLint srcLevel, GLint srcX, GLint srcY, GLint srcZ, GLuint dstName, GLenum dstTarget, GLint dstLevel, GLint dstX, GLint dstY, GLint dstZ, GLsizei srcWidth, GLsizei srcHeight, GLsizei srcDepth))
GLFUNC(glFramebufferParameteri,                                 void,               (GLenum target, GLenum pname, GLint param))
GLFUNC(glGetFramebufferParameteriv,                             void,               (GLenum target, GLenum pname, GLint *params))
GLFUNC(glGetInternalformati64v,                                 void,               (GLenum target, GLenum internalformat, GLenum pname, GLsizei bufSize, GLint64 *params))
GLFUNC(glInvalidateTexSubImage,                                 void,               (GLuint texture, GLint level, GLint xoffset, GLint yoffset, GLint zoffset, GLsizei width, GLsizei height, GLsizei depth))
GLFUNC(glInvalidateTexImage,                                    void,               (GLuint texture, GLint level))
GLFUNC(glInvalidateBufferSubData,                               void,               (GLuint buffer, GLintptr offset, GLsizeiptr length))
GLFUNC(glInvalidateBufferData,                                  void,               (GLuint buffer))
GLFUNC(glInvalidateFramebuffer,                                 void,               (GLenum target, GLsizei numAttachments, const GLenum *attachments))
GLFUNC(glInvalidateSubFramebuffer,                              void,               (GLenum target, GLsizei numAttachments, const GLenum *attachments, GLint x, GLint y, GLsizei width, GLsizei height))
GLFUNC(glMultiDrawArraysIndirect,                               void,               (GLenum mode, const void *indirect, GLsizei drawcount, GLsizei stride))
GLFUNC(glMultiDrawElementsIndirect,                             void,               (GLenum mode, GLenum type, const void *indirect, GLsizei drawcount, GLsizei stride))
GLFUNC(glGetProgramInterfaceiv,                                 void,               (GLuint program, GLenum programInterface, GLenum pname, GLint *params))
GLFUNC(glGetProgramResourceIndex,                               GLuint,             (GLuint program, GLenum programInterface, const GLchar *name))
GLFUNC(glGetProgramResourceName,                                void,               (GLuint program, GLenum programInterface, GLuint index, GLsizei bufSize, GLsizei *length, GLchar *name))
GLFUNC(glGetProgramResourceiv,                                  void,               (GLuint program, GLenum programInterface, GLuint index, GLsizei propCount, const GLenum *props, GLsizei bufSize, GLsizei *length, GLint *params))
GLFUNC(glGetProgramResourceLocation,                            GLint,              (GLuint program, GLenum programInterface, const GLchar *name))
GLFUNC(glGetProgramResourceLocationIndex,                       GLint,              (GLuint program, GLenum programInterface, const GLchar *name))
GLFUNC(glShaderStorageBlockBinding,                             void,               (GLuint program, GLuint storageBlockIndex, GLuint storageBlockBinding))
GLFUNC(glTexBufferRange,                                        void,               (GLenum target, GLenum internalformat, GLuint buffer, GLintptr offset, GLsizeiptr size))
GLFUNC(glTexStorage2DMultisample,                               void,               (GLenum target, GLsizei samples, GLenum internalformat, GLsizei width, GLsizei height, GLboolean fixedsamplelocations))
GLFUNC(glTexStorage3DMultisample,                               void,               (GLenum target, GLsizei samples, GLenum internalformat, GLsizei width, GLsizei height, GLsizei depth, GLboolean fixedsamplelocations))
GLFUNC(glTextureView,                                           void,               (GLuint texture, GLenum target, GLuint origtexture, GLenum internalformat, GLuint minlevel, GLuint numlevels, GLuint minlayer, GLuint numlayers))
GLFUNC(glBindVertexBuffer,                                      void,               (GLuint bindingindex, GLuint buffer, GLintptr offset, GLsizei stride))
GLFUNC(glVertexAttribFormat,                                    void,               (GLuint attribindex, GLint size, GLenum type, GLboolean normalized, GLuint relativeoffset))
GLFUNC(glVertexAttribIFormat,                                   void,               (GLuint attribindex, GLint size, GLenum type, GLuint relativeoffset))
GLFUNC(glVertexAttribLFormat,                                   void,               (GLuint attribindex, GLint size, GLenum type, GLuint relativeoffset))
GLFUNC(glVertexAttribBinding,                                   void,               (GLuint attribindex, GLuint bindingindex))
GLFUNC(glVertexBindingDivisor,                                  void,               (GLuint bindingindex, GLuint divisor))
GLFUNC(glDebugMessageControl,                                   void,               (GLenum source, GLenum type, GLenum severity, GLsizei count, const GLuint *ids, GLboolean enabled))
GLFUNC(glDebugMessageInsert,                                    void,               (GLenum source, GLenum type, GLuint id, GLenum severity, GLsizei length, const GLchar *buf))
GLFUNC(glDebugMessageCallback,                                  void,               (GLDEBUGPROC callback, const void *userParam))
GLFUNC(glGetDebugMessageLog,                                    GLuint,             (GLuint count, GLsizei bufSize, GLenum *sources, GLenum *types, GLuint *ids, GLenum *severities, GLsizei *lengths, GLchar *messageLog))
GLFUNC(glPushDebugGroup,                                        void,               (GLenum source, GLuint id, GLsizei length, const GLchar *message))
GLFUNC(glPopDebugGroup,                                         void,               (void))
GLFUNC(glObjectLabel,                                           void,               (GLenum identifier, GLuint name, GLsizei length, const GLchar *label))
GLFUNC(glGetObjectLabel,                                        void,               (GLenum identifier, GLuint name, GLsizei bufSize, GLsizei *length, GLchar *label))
GLFUNC(glObjectPtrLabel,                                        void,               (const void *ptr, GLsizei length, const GLchar *label))
GLFUNC(glGetObjectPtrLabel,                                     void,               (const void *ptr, GLsizei bufSize, GLsizei *length, GLchar *label))

#endif

//------------------------------------------------------------------------------
// OpenGL 4.4
//

// ARB_buffer_storage
// ARB_clear_texture
// ARB_enhanced_layouts
// ARB_multi_bind
// ARB_query_buffer_object
// ARB_texture_mirror_clamp_to_edge
// ARB_texture_stencil8
// ARB_vertex_type_10f_11f_11f_rev

#if !defined(GL_VERSION_4_4) || (GL_VERSION_4_4 == 0xDLL)
#define GL_VERSION_4_4 0xDLL

GLEXT(GL_VERSION_4_4)

GLFUNC(glBufferStorage,                                         void,               (GLenum target, GLsizeiptr size, const void *data, GLbitfield flags))
GLFUNC(glClearTexImage,                                         void,               (GLuint texture, GLint level, GLenum format, GLenum type, const void *data))
GLFUNC(glClearTexSubImage,                                      void,               (GLuint texture, GLint level, GLint xoffset, GLint yoffset, GLint zoffset, GLsizei width, GLsizei height, GLsizei depth, GLenum format, GLenum type, const void *data))
GLFUNC(glBindBuffersBase,                                       void,               (GLenum target, GLuint first, GLsizei count, const GLuint *buffers))
GLFUNC(glBindBuffersRange,                                      void,               (GLenum target, GLuint first, GLsizei count, const GLuint *buffers, const GLintptr *offsets, const GLsizeiptr *sizes))
GLFUNC(glBindTextures,                                          void,               (GLuint first, GLsizei count, const GLuint *textures))
GLFUNC(glBindSamplers,                                          void,               (GLuint first, GLsizei count, const GLuint *samplers))
GLFUNC(glBindImageTextures,                                     void,               (GLuint first, GLsizei count, const GLuint *textures))
GLFUNC(glBindVertexBuffers,                                     void,               (GLuint first, GLsizei count, const GLuint *buffers, const GLintptr *offsets, const GLsizei *strides))

#endif

//------------------------------------------------------------------------------
// OpenGL 4.5
//

// ARB_clip_control
// ARB_conditional_render_inverted
// ARB_cull_distance
// ARB_derivative_control
// ARB_direct_state_access
// ARB_ES3_1_compatibility
// ARB_get_texture_sub_image
// ARB_robustness
// ARB_shader_texture_image_samples
// ARB_texture_barrier

#if !defined(GL_VERSION_4_5) || (GL_VERSION_4_5 == 0xDLL)
#define GL_VERSION_4_5 0xDLL

GLEXT(GL_VERSION_4_5)

GLFUNC(glClipControl,                                           void,               (GLenum origin, GLenum depth))
GLFUNC(glCreateTransformFeedbacks,                              void,               (GLsizei n, GLuint *ids))
GLFUNC(glTransformFeedbackBufferBase,                           void,               (GLuint xfb, GLuint index, GLuint buffer))
GLFUNC(glTransformFeedbackBufferRange,                          void,               (GLuint xfb, GLuint index, GLuint buffer, GLintptr offset, GLsizei size))
GLFUNC(glGetTransformFeedbackiv,                                void,               (GLuint xfb, GLenum pname, GLint *param))
GLFUNC(glGetTransformFeedbacki_v,                               void,               (GLuint xfb, GLenum pname, GLuint index, GLint *param))
GLFUNC(glGetTransformFeedbacki64_v,                             void,               (GLuint xfb, GLenum pname, GLuint index, GLint64 *param))
GLFUNC(glCreateBuffers,                                         void,               (GLsizei n, GLuint *buffers))
GLFUNC(glNamedBufferStorage,                                    void,               (GLuint buffer, GLsizei size, const void *data, GLbitfield flags))
GLFUNC(glNamedBufferData,                                       void,               (GLuint buffer, GLsizei size, const void *data, GLenum usage))
GLFUNC(glNamedBufferSubData,                                    void,               (GLuint buffer, GLintptr offset, GLsizei size, const void *data))
GLFUNC(glCopyNamedBufferSubData,                                void,               (GLuint readBuffer, GLuint writeBuffer, GLintptr readOffset, GLintptr writeOffset, GLsizei size))
GLFUNC(glClearNamedBufferData,                                  void,               (GLuint buffer, GLenum internalformat, GLenum format, GLenum type, const void *data))
GLFUNC(glClearNamedBufferSubData,                               void,               (GLuint buffer, GLenum internalformat, GLintptr offset, GLsizei size, GLenum format, GLenum type, const void *data))
GLFUNC(glMapNamedBuffer,                                        void *,             (GLuint buffer, GLenum access))
GLFUNC(glMapNamedBufferRange,                                   void *,             (GLuint buffer, GLintptr offset, GLsizei length, GLbitfield access))
GLFUNC(glUnmapNamedBuffer,                                      GLboolean,          (GLuint buffer))
GLFUNC(glFlushMappedNamedBufferRange,                           void,               (GLuint buffer, GLintptr offset, GLsizei length))
GLFUNC(glGetNamedBufferParameteriv,                             void,               (GLuint buffer, GLenum pname, GLint *params))
GLFUNC(glGetNamedBufferParameteri64v,                           void,               (GLuint buffer, GLenum pname, GLint64 *params))
GLFUNC(glGetNamedBufferPointerv,                                void,               (GLuint buffer, GLenum pname, void **params))
GLFUNC(glGetNamedBufferSubData,                                 void,               (GLuint buffer, GLintptr offset, GLsizei size, void *data))
GLFUNC(glCreateFramebuffers,                                    void,               (GLsizei n, GLuint *framebuffers))
GLFUNC(glNamedFramebufferRenderbuffer,                          void,               (GLuint framebuffer, GLenum attachment, GLenum renderbuffertarget, GLuint renderbuffer))
GLFUNC(glNamedFramebufferParameteri,                            void,               (GLuint framebuffer, GLenum pname, GLint param))
GLFUNC(glNamedFramebufferTexture,                               void,               (GLuint framebuffer, GLenum attachment, GLuint texture, GLint level))
GLFUNC(glNamedFramebufferTextureLayer,                          void,               (GLuint framebuffer, GLenum attachment, GLuint texture, GLint level, GLint layer))
GLFUNC(glNamedFramebufferDrawBuffer,                            void,               (GLuint framebuffer, GLenum buf))
GLFUNC(glNamedFramebufferDrawBuffers,                           void,               (GLuint framebuffer, GLsizei n, const GLenum *bufs))
GLFUNC(glNamedFramebufferReadBuffer,                            void,               (GLuint framebuffer, GLenum src))
GLFUNC(glInvalidateNamedFramebufferData,                        void,               (GLuint framebuffer, GLsizei numAttachments, const GLenum *attachments))
GLFUNC(glInvalidateNamedFramebufferSubData,                     void,               (GLuint framebuffer, GLsizei numAttachments, const GLenum *attachments, GLint x, GLint y, GLsizei width, GLsizei height))
GLFUNC(glClearNamedFramebufferiv,                               void,               (GLuint framebuffer, GLenum buffer, GLint drawbuffer, const GLint *value))
GLFUNC(glClearNamedFramebufferuiv,                              void,               (GLuint framebuffer, GLenum buffer, GLint drawbuffer, const GLuint *value))
GLFUNC(glClearNamedFramebufferfv,                               void,               (GLuint framebuffer, GLenum buffer, GLint drawbuffer, const GLfloat *value))
GLFUNC(glClearNamedFramebufferfi,                               void,               (GLuint framebuffer, GLenum buffer, const GLfloat depth, GLint stencil))
GLFUNC(glBlitNamedFramebuffer,                                  void,               (GLuint readFramebuffer, GLuint drawFramebuffer, GLint srcX0, GLint srcY0, GLint srcX1, GLint srcY1, GLint dstX0, GLint dstY0, GLint dstX1, GLint dstY1, GLbitfield mask, GLenum filter))
GLFUNC(glCheckNamedFramebufferStatus,                           GLenum,             (GLuint framebuffer, GLenum target))
GLFUNC(glGetNamedFramebufferParameteriv,                        void,               (GLuint framebuffer, GLenum pname, GLint *param))
GLFUNC(glGetNamedFramebufferAttachmentParameteriv,              void,               (GLuint framebuffer, GLenum attachment, GLenum pname, GLint *params))
GLFUNC(glCreateRenderbuffers,                                   void,               (GLsizei n, GLuint *renderbuffers))
GLFUNC(glNamedRenderbufferStorage,                              void,               (GLuint renderbuffer, GLenum internalformat, GLsizei width, GLsizei height))
GLFUNC(glNamedRenderbufferStorageMultisample,                   void,               (GLuint renderbuffer, GLsizei samples, GLenum internalformat, GLsizei width, GLsizei height))
GLFUNC(glGetNamedRenderbufferParameteriv,                       void,               (GLuint renderbuffer, GLenum pname, GLint *params))
GLFUNC(glCreateTextures,                                        void,               (GLenum target, GLsizei n, GLuint *textures))
GLFUNC(glTextureBuffer,                                         void,               (GLuint texture, GLenum internalformat, GLuint buffer))
GLFUNC(glTextureBufferRange,                                    void,               (GLuint texture, GLenum internalformat, GLuint buffer, GLintptr offset, GLsizei size))
GLFUNC(glTextureStorage1D,                                      void,               (GLuint texture, GLsizei levels, GLenum internalformat, GLsizei width))
GLFUNC(glTextureStorage2D,                                      void,               (GLuint texture, GLsizei levels, GLenum internalformat, GLsizei width, GLsizei height))
GLFUNC(glTextureStorage3D,                                      void,               (GLuint texture, GLsizei levels, GLenum internalformat, GLsizei width, GLsizei height, GLsizei depth))
GLFUNC(glTextureStorage2DMultisample,                           void,               (GLuint texture, GLsizei samples, GLenum internalformat, GLsizei width, GLsizei height, GLboolean fixedsamplelocations))
GLFUNC(glTextureStorage3DMultisample,                           void,               (GLuint texture, GLsizei samples, GLenum internalformat, GLsizei width, GLsizei height, GLsizei depth, GLboolean fixedsamplelocations))
GLFUNC(glTextureSubImage1D,                                     void,               (GLuint texture, GLint level, GLint xoffset, GLsizei width, GLenum format, GLenum type, const void *pixels))
GLFUNC(glTextureSubImage2D,                                     void,               (GLuint texture, GLint level, GLint xoffset, GLint yoffset, GLsizei width, GLsizei height, GLenum format, GLenum type, const void *pixels))
GLFUNC(glTextureSubImage3D,                                     void,               (GLuint texture, GLint level, GLint xoffset, GLint yoffset, GLint zoffset, GLsizei width, GLsizei height, GLsizei depth, GLenum format, GLenum type, const void *pixels))
GLFUNC(glCompressedTextureSubImage1D,                           void,               (GLuint texture, GLint level, GLint xoffset, GLsizei width, GLenum format, GLsizei imageSize, const void *data))
GLFUNC(glCompressedTextureSubImage2D,                           void,               (GLuint texture, GLint level, GLint xoffset, GLint yoffset, GLsizei width, GLsizei height, GLenum format, GLsizei imageSize, const void *data))
GLFUNC(glCompressedTextureSubImage3D,                           void,               (GLuint texture, GLint level, GLint xoffset, GLint yoffset, GLint zoffset, GLsizei width, GLsizei height, GLsizei depth, GLenum format, GLsizei imageSize, const void *data))
GLFUNC(glCopyTextureSubImage1D,                                 void,               (GLuint texture, GLint level, GLint xoffset, GLint x, GLint y, GLsizei width))
GLFUNC(glCopyTextureSubImage2D,                                 void,               (GLuint texture, GLint level, GLint xoffset, GLint yoffset, GLint x, GLint y, GLsizei width, GLsizei height))
GLFUNC(glCopyTextureSubImage3D,                                 void,               (GLuint texture, GLint level, GLint xoffset, GLint yoffset, GLint zoffset, GLint x, GLint y, GLsizei width, GLsizei height))
GLFUNC(glTextureParameterf,                                     void,               (GLuint texture, GLenum pname, GLfloat param))
GLFUNC(glTextureParameterfv,                                    void,               (GLuint texture, GLenum pname, const GLfloat *param))
GLFUNC(glTextureParameteri,                                     void,               (GLuint texture, GLenum pname, GLint param))
GLFUNC(glTextureParameterIiv,                                   void,               (GLuint texture, GLenum pname, const GLint *params))
GLFUNC(glTextureParameterIuiv,                                  void,               (GLuint texture, GLenum pname, const GLuint *params))
GLFUNC(glTextureParameteriv,                                    void,               (GLuint texture, GLenum pname, const GLint *param))
GLFUNC(glGenerateTextureMipmap,                                 void,               (GLuint texture))
GLFUNC(glBindTextureUnit,                                       void,               (GLuint unit, GLuint texture))
GLFUNC(glGetTextureImage,                                       void,               (GLuint texture, GLint level, GLenum format, GLenum type, GLsizei bufSize, void *pixels))
GLFUNC(glGetCompressedTextureImage,                             void,               (GLuint texture, GLint level, GLsizei bufSize, void *pixels))
GLFUNC(glGetTextureLevelParameterfv,                            void,               (GLuint texture, GLint level, GLenum pname, GLfloat *params))
GLFUNC(glGetTextureLevelParameteriv,                            void,               (GLuint texture, GLint level, GLenum pname, GLint *params))
GLFUNC(glGetTextureParameterfv,                                 void,               (GLuint texture, GLenum pname, GLfloat *params))
GLFUNC(glGetTextureParameterIiv,                                void,               (GLuint texture, GLenum pname, GLint *params))
GLFUNC(glGetTextureParameterIuiv,                               void,               (GLuint texture, GLenum pname, GLuint *params))
GLFUNC(glGetTextureParameteriv,                                 void,               (GLuint texture, GLenum pname, GLint *params))
GLFUNC(glCreateVertexArrays,                                    void,               (GLsizei n, GLuint *arrays))
GLFUNC(glDisableVertexArrayAttrib,                              void,               (GLuint vaobj, GLuint index))
GLFUNC(glEnableVertexArrayAttrib,                               void,               (GLuint vaobj, GLuint index))
GLFUNC(glVertexArrayElementBuffer,                              void,               (GLuint vaobj, GLuint buffer))
GLFUNC(glVertexArrayVertexBuffer,                               void,               (GLuint vaobj, GLuint bindingindex, GLuint buffer, GLintptr offset, GLsizei stride))
GLFUNC(glVertexArrayVertexBuffers,                              void,               (GLuint vaobj, GLuint first, GLsizei count, const GLuint *buffers, const GLintptr *offsets, const GLsizei *strides))
GLFUNC(glVertexArrayAttribBinding,                              void,               (GLuint vaobj, GLuint attribindex, GLuint bindingindex))
GLFUNC(glVertexArrayAttribFormat,                               void,               (GLuint vaobj, GLuint attribindex, GLint size, GLenum type, GLboolean normalized, GLuint relativeoffset))
GLFUNC(glVertexArrayAttribIFormat,                              void,               (GLuint vaobj, GLuint attribindex, GLint size, GLenum type, GLuint relativeoffset))
GLFUNC(glVertexArrayAttribLFormat,                              void,               (GLuint vaobj, GLuint attribindex, GLint size, GLenum type, GLuint relativeoffset))
GLFUNC(glVertexArrayBindingDivisor,                             void,               (GLuint vaobj, GLuint bindingindex, GLuint divisor))
GLFUNC(glGetVertexArrayiv,                                      void,               (GLuint vaobj, GLenum pname, GLint *param))
GLFUNC(glGetVertexArrayIndexediv,                               void,               (GLuint vaobj, GLuint index, GLenum pname, GLint *param))
GLFUNC(glGetVertexArrayIndexed64iv,                             void,               (GLuint vaobj, GLuint index, GLenum pname, GLint64 *param))
GLFUNC(glCreateSamplers,                                        void,               (GLsizei n, GLuint *samplers))
GLFUNC(glCreateProgramPipelines,                                void,               (GLsizei n, GLuint *pipelines))
GLFUNC(glCreateQueries,                                         void,               (GLenum target, GLsizei n, GLuint *ids))
GLFUNC(glGetQueryBufferObjecti64v,                              void,               (GLuint id, GLuint buffer, GLenum pname, GLintptr offset))
GLFUNC(glGetQueryBufferObjectiv,                                void,               (GLuint id, GLuint buffer, GLenum pname, GLintptr offset))
GLFUNC(glGetQueryBufferObjectui64v,                             void,               (GLuint id, GLuint buffer, GLenum pname, GLintptr offset))
GLFUNC(glGetQueryBufferObjectuiv,                               void,               (GLuint id, GLuint buffer, GLenum pname, GLintptr offset))
GLFUNC(glMemoryBarrierByRegion,                                 void,               (GLbitfield barriers))
GLFUNC(glGetTextureSubImage,                                    void,               (GLuint texture, GLint level, GLint xoffset, GLint yoffset, GLint zoffset, GLsizei width, GLsizei height, GLsizei depth, GLenum format, GLenum type, GLsizei bufSize, void *pixels))
GLFUNC(glGetCompressedTextureSubImage,                          void,               (GLuint texture, GLint level, GLint xoffset, GLint yoffset, GLint zoffset, GLsizei width, GLsizei height, GLsizei depth, GLsizei bufSize, void *pixels))
GLFUNC(glGetGraphicsResetStatus,                                GLenum,             (void))
GLFUNC(glGetnCompressedTexImage,                                void,               (GLenum target, GLint lod, GLsizei bufSize, void *pixels))
GLFUNC(glGetnTexImage,                                          void,               (GLenum target, GLint level, GLenum format, GLenum type, GLsizei bufSize, void *pixels))
GLFUNC(glGetnUniformdv,                                         void,               (GLuint program, GLint location, GLsizei bufSize, GLdouble *params))
GLFUNC(glGetnUniformfv,                                         void,               (GLuint program, GLint location, GLsizei bufSize, GLfloat *params))
GLFUNC(glGetnUniformiv,                                         void,               (GLuint program, GLint location, GLsizei bufSize, GLint *params))
GLFUNC(glGetnUniformuiv,                                        void,               (GLuint program, GLint location, GLsizei bufSize, GLuint *params))
GLFUNC(glReadnPixels,                                           void,               (GLint x, GLint y, GLsizei width, GLsizei height, GLenum format, GLenum type, GLsizei bufSize, void *data))
GLFUNC(glTextureBarrier,                                        void,               (void))

GLALIAS(glEnableVertexArrayAttrib,                              glEnableVertexArrayAttribEXT)
GLALIAS(glDisableVertexArrayAttrib,                             glDisableVertexArrayAttribEXT)

GLFALLBACK(glEnableVertexArrayAttrib,                           glEnableVertexArrayAttrib_IMPL)
GLFALLBACK(glDisableVertexArrayAttrib,                          glDisableVertexArrayAttrib_IMPL)
GLFALLBACK(glVertexArrayElementBuffer,                          glVertexArrayElementBuffer_IMPL)
GLFALLBACK(glCreateBuffers,                                     glCreateBuffers_IMPL)
GLFALLBACK(glCreateRenderbuffers,                               glCreateRenderbuffers_IMPL)
GLFALLBACK(glCreateTextures,                                    glCreateTextures_IMPL)
GLFALLBACK(glCreateVertexArrays,                                glCreateVertexArrays_IMPL)
GLFALLBACK(glCreateProgramPipelines,                            glCreateProgramPipelines_IMPL)
GLFALLBACK(glCreateSamplers,                                    glCreateSamplers_IMPL)
GLFALLBACK(glCreateQueries,                                     glCreateQueries_IMPL)
GLFALLBACK(glCreateFramebuffers,                                glCreateFramebuffers_IMPL)

#endif

//------------------------------------------------------------------------------
// GL_ARB_bindless_texture
//

#if !defined(GL_ARB_bindless_texture) || (GL_ARB_bindless_texture == 0xDLL)
#define GL_ARB_bindless_texture 0xDLL

GLEXT(GL_ARB_bindless_texture)

GLFUNC(glGetTextureHandleARB,                                   GLuint64,           (GLuint texture))
GLFUNC(glGetTextureSamplerHandleARB,                            GLuint64,           (GLuint texture, GLuint sampler))
GLFUNC(glMakeTextureHandleResidentARB,                          void,               (GLuint64 handle))
GLFUNC(glMakeTextureHandleNonResidentARB,                       void,               (GLuint64 handle))
GLFUNC(glGetImageHandleARB,                                     GLuint64,           (GLuint texture, GLint level, GLboolean layered, GLint layer, GLenum format))
GLFUNC(glMakeImageHandleResidentARB,                            void,               (GLuint64 handle, GLenum access))
GLFUNC(glMakeImageHandleNonResidentARB,                         void,               (GLuint64 handle))
GLFUNC(glUniformHandleui64ARB,                                  void,               (GLint location, GLuint64 value))
GLFUNC(glUniformHandleui64vARB,                                 void,               (GLint location, GLsizei count, const GLuint64 *value))
GLFUNC(glProgramUniformHandleui64ARB,                           void,               (GLuint program, GLint location, GLuint64 value))
GLFUNC(glProgramUniformHandleui64vARB,                          void,               (GLuint program, GLint location, GLsizei count, const GLuint64 *values))
GLFUNC(glIsTextureHandleResidentARB,                            GLboolean,          (GLuint64 handle))
GLFUNC(glIsImageHandleResidentARB,                              GLboolean,          (GLuint64 handle))
GLFUNC(glVertexAttribL1ui64ARB,                                 void,               (GLuint index, GLuint64EXT x))
GLFUNC(glVertexAttribL1ui64vARB,                                void,               (GLuint index, const GLuint64EXT *v))
GLFUNC(glGetVertexAttribLui64vARB,                              void,               (GLuint index, GLenum pname, GLuint64EXT *params))

#endif

//------------------------------------------------------------------------------
// GL_ARB_shading_language_include
//

#if !defined(GL_ARB_shading_language_include) || (GL_ARB_shading_language_include == 0xDLL)
#define GL_ARB_shading_language_include 0xDLL

GLEXT(GL_ARB_shading_language_include)

GLFUNC(glNamedStringARB,                                        void,               (GLenum type, GLint namelen, const GLchar *name, GLint stringlen, const GLchar *string))
GLFUNC(glDeleteNamedStringARB,                                  void,               (GLint namelen, const GLchar *name))
GLFUNC(glCompileShaderIncludeARB,                               void,               (GLuint shader, GLsizei count, const GLchar *const*path, const GLint *length))
GLFUNC(glIsNamedStringARB,                                      GLboolean,          (GLint namelen, const GLchar *name))
GLFUNC(glGetNamedStringARB,                                     void,               (GLint namelen, const GLchar *name, GLsizei bufSize, GLint *stringlen, GLchar *string))
GLFUNC(glGetNamedStringivARB,                                   void,               (GLint namelen, const GLchar *name, GLenum pname, GLint *params))

#endif

//------------------------------------------------------------------------------
// GL_EXT_direct_state_access
//

#if !defined(GL_EXT_direct_state_access) || (GL_EXT_direct_state_access == 0xDLL)
#define GL_EXT_direct_state_access 0xDLL

GLEXT(GL_EXT_direct_state_access)

//GLFUNC(glMatrixLoadfEXT,                                        void,               (GLenum mode, const GLfloat *m))
//GLFUNC(glMatrixLoaddEXT,                                        void,               (GLenum mode, const GLdouble *m))
//GLFUNC(glMatrixMultfEXT,                                        void,               (GLenum mode, const GLfloat *m))
//GLFUNC(glMatrixMultdEXT,                                        void,               (GLenum mode, const GLdouble *m))
//GLFUNC(glMatrixLoadIdentityEXT,                                 void,               (GLenum mode))
//GLFUNC(glMatrixRotatefEXT,                                      void,               (GLenum mode, GLfloat angle, GLfloat x, GLfloat y, GLfloat z))
//GLFUNC(glMatrixRotatedEXT,                                      void,               (GLenum mode, GLdouble angle, GLdouble x, GLdouble y, GLdouble z))
//GLFUNC(glMatrixScalefEXT,                                       void,               (GLenum mode, GLfloat x, GLfloat y, GLfloat z))
//GLFUNC(glMatrixScaledEXT,                                       void,               (GLenum mode, GLdouble x, GLdouble y, GLdouble z))
//GLFUNC(glMatrixTranslatefEXT,                                   void,               (GLenum mode, GLfloat x, GLfloat y, GLfloat z))
//GLFUNC(glMatrixTranslatedEXT,                                   void,               (GLenum mode, GLdouble x, GLdouble y, GLdouble z))
//GLFUNC(glMatrixFrustumEXT,                                      void,               (GLenum mode, GLdouble left, GLdouble right, GLdouble bottom, GLdouble top, GLdouble zNear, GLdouble zFar))
//GLFUNC(glMatrixOrthoEXT,                                        void,               (GLenum mode, GLdouble left, GLdouble right, GLdouble bottom, GLdouble top, GLdouble zNear, GLdouble zFar))
//GLFUNC(glMatrixPopEXT,                                          void,               (GLenum mode))
//GLFUNC(glMatrixPushEXT,                                         void,               (GLenum mode))
GLFUNC(glClientAttribDefaultEXT,                                void,               (GLbitfield mask))
GLFUNC(glPushClientAttribDefaultEXT,                            void,               (GLbitfield mask))
GLFUNC(glTextureParameterfEXT,                                  void,               (GLuint texture, GLenum target, GLenum pname, GLfloat param))
GLFUNC(glTextureParameterfvEXT,                                 void,               (GLuint texture, GLenum target, GLenum pname, const GLfloat *params))
GLFUNC(glTextureParameteriEXT,                                  void,               (GLuint texture, GLenum target, GLenum pname, GLint param))
GLFUNC(glTextureParameterivEXT,                                 void,               (GLuint texture, GLenum target, GLenum pname, const GLint *params))
//GLFUNC(glTextureImage1DEXT,                                     void,               (GLuint texture, GLenum target, GLint level, GLint internalformat, GLsizei width, GLint border, GLenum format, GLenum type, const void *pixels))
//GLFUNC(glTextureImage2DEXT,                                     void,               (GLuint texture, GLenum target, GLint level, GLint internalformat, GLsizei width, GLsizei height, GLint border, GLenum format, GLenum type, const void *pixels))
GLFUNC(glTextureSubImage1DEXT,                                  void,               (GLuint texture, GLenum target, GLint level, GLint xoffset, GLsizei width, GLenum format, GLenum type, const void *pixels))
GLFUNC(glTextureSubImage2DEXT,                                  void,               (GLuint texture, GLenum target, GLint level, GLint xoffset, GLint yoffset, GLsizei width, GLsizei height, GLenum format, GLenum type, const void *pixels))
GLFUNC(glCopyTextureImage1DEXT,                                 void,               (GLuint texture, GLenum target, GLint level, GLenum internalformat, GLint x, GLint y, GLsizei width, GLint border))
GLFUNC(glCopyTextureImage2DEXT,                                 void,               (GLuint texture, GLenum target, GLint level, GLenum internalformat, GLint x, GLint y, GLsizei width, GLsizei height, GLint border))
GLFUNC(glCopyTextureSubImage1DEXT,                              void,               (GLuint texture, GLenum target, GLint level, GLint xoffset, GLint x, GLint y, GLsizei width))
GLFUNC(glCopyTextureSubImage2DEXT,                              void,               (GLuint texture, GLenum target, GLint level, GLint xoffset, GLint yoffset, GLint x, GLint y, GLsizei width, GLsizei height))
GLFUNC(glGetTextureImageEXT,                                    void,               (GLuint texture, GLenum target, GLint level, GLenum format, GLenum type, void *pixels))
GLFUNC(glGetTextureParameterfvEXT,                              void,               (GLuint texture, GLenum target, GLenum pname, GLfloat *params))
GLFUNC(glGetTextureParameterivEXT,                              void,               (GLuint texture, GLenum target, GLenum pname, GLint *params))
GLFUNC(glGetTextureLevelParameterfvEXT,                         void,               (GLuint texture, GLenum target, GLint level, GLenum pname, GLfloat *params))
GLFUNC(glGetTextureLevelParameterivEXT,                         void,               (GLuint texture, GLenum target, GLint level, GLenum pname, GLint *params))
//GLFUNC(glTextureImage3DEXT,                                     void,               (GLuint texture, GLenum target, GLint level, GLint internalformat, GLsizei width, GLsizei height, GLsizei depth, GLint border, GLenum format, GLenum type, const void *pixels))
GLFUNC(glTextureSubImage3DEXT,                                  void,               (GLuint texture, GLenum target, GLint level, GLint xoffset, GLint yoffset, GLint zoffset, GLsizei width, GLsizei height, GLsizei depth, GLenum format, GLenum type, const void *pixels))
GLFUNC(glCopyTextureSubImage3DEXT,                              void,               (GLuint texture, GLenum target, GLint level, GLint xoffset, GLint yoffset, GLint zoffset, GLint x, GLint y, GLsizei width, GLsizei height))
GLFUNC(glBindMultiTextureEXT,                                   void,               (GLenum texunit, GLenum target, GLuint texture))
//GLFUNC(glMultiTexCoordPointerEXT,                               void,               (GLenum texunit, GLint size, GLenum type, GLsizei stride, const void *pointer))
//GLFUNC(glMultiTexEnvfEXT,                                       void,               (GLenum texunit, GLenum target, GLenum pname, GLfloat param))
//GLFUNC(glMultiTexEnvfvEXT,                                      void,               (GLenum texunit, GLenum target, GLenum pname, const GLfloat *params))
//GLFUNC(glMultiTexEnviEXT,                                       void,               (GLenum texunit, GLenum target, GLenum pname, GLint param))
//GLFUNC(glMultiTexEnvivEXT,                                      void,               (GLenum texunit, GLenum target, GLenum pname, const GLint *params))
//GLFUNC(glMultiTexGendEXT,                                       void,               (GLenum texunit, GLenum coord, GLenum pname, GLdouble param))
//GLFUNC(glMultiTexGendvEXT,                                      void,               (GLenum texunit, GLenum coord, GLenum pname, const GLdouble *params))
//GLFUNC(glMultiTexGenfEXT,                                       void,               (GLenum texunit, GLenum coord, GLenum pname, GLfloat param))
//GLFUNC(glMultiTexGenfvEXT,                                      void,               (GLenum texunit, GLenum coord, GLenum pname, const GLfloat *params))
//GLFUNC(glMultiTexGeniEXT,                                       void,               (GLenum texunit, GLenum coord, GLenum pname, GLint param))
//GLFUNC(glMultiTexGenivEXT,                                      void,               (GLenum texunit, GLenum coord, GLenum pname, const GLint *params))
//GLFUNC(glGetMultiTexEnvfvEXT,                                   void,               (GLenum texunit, GLenum target, GLenum pname, GLfloat *params))
//GLFUNC(glGetMultiTexEnvivEXT,                                   void,               (GLenum texunit, GLenum target, GLenum pname, GLint *params))
//GLFUNC(glGetMultiTexGendvEXT,                                   void,               (GLenum texunit, GLenum coord, GLenum pname, GLdouble *params))
//GLFUNC(glGetMultiTexGenfvEXT,                                   void,               (GLenum texunit, GLenum coord, GLenum pname, GLfloat *params))
//GLFUNC(glGetMultiTexGenivEXT,                                   void,               (GLenum texunit, GLenum coord, GLenum pname, GLint *params))
//GLFUNC(glMultiTexParameteriEXT,                                 void,               (GLenum texunit, GLenum target, GLenum pname, GLint param))
//GLFUNC(glMultiTexParameterivEXT,                                void,               (GLenum texunit, GLenum target, GLenum pname, const GLint *params))
//GLFUNC(glMultiTexParameterfEXT,                                 void,               (GLenum texunit, GLenum target, GLenum pname, GLfloat param))
//GLFUNC(glMultiTexParameterfvEXT,                                void,               (GLenum texunit, GLenum target, GLenum pname, const GLfloat *params))
//GLFUNC(glMultiTexImage1DEXT,                                    void,               (GLenum texunit, GLenum target, GLint level, GLint internalformat, GLsizei width, GLint border, GLenum format, GLenum type, const void *pixels))
//GLFUNC(glMultiTexImage2DEXT,                                    void,               (GLenum texunit, GLenum target, GLint level, GLint internalformat, GLsizei width, GLsizei height, GLint border, GLenum format, GLenum type, const void *pixels))
//GLFUNC(glMultiTexSubImage1DEXT,                                 void,               (GLenum texunit, GLenum target, GLint level, GLint xoffset, GLsizei width, GLenum format, GLenum type, const void *pixels))
//GLFUNC(glMultiTexSubImage2DEXT,                                 void,               (GLenum texunit, GLenum target, GLint level, GLint xoffset, GLint yoffset, GLsizei width, GLsizei height, GLenum format, GLenum type, const void *pixels))
//GLFUNC(glCopyMultiTexImage1DEXT,                                void,               (GLenum texunit, GLenum target, GLint level, GLenum internalformat, GLint x, GLint y, GLsizei width, GLint border))
//GLFUNC(glCopyMultiTexImage2DEXT,                                void,               (GLenum texunit, GLenum target, GLint level, GLenum internalformat, GLint x, GLint y, GLsizei width, GLsizei height, GLint border))
//GLFUNC(glCopyMultiTexSubImage1DEXT,                             void,               (GLenum texunit, GLenum target, GLint level, GLint xoffset, GLint x, GLint y, GLsizei width))
//GLFUNC(glCopyMultiTexSubImage2DEXT,                             void,               (GLenum texunit, GLenum target, GLint level, GLint xoffset, GLint yoffset, GLint x, GLint y, GLsizei width, GLsizei height))
//GLFUNC(glGetMultiTexImageEXT,                                   void,               (GLenum texunit, GLenum target, GLint level, GLenum format, GLenum type, void *pixels))
//GLFUNC(glGetMultiTexParameterfvEXT,                             void,               (GLenum texunit, GLenum target, GLenum pname, GLfloat *params))
//GLFUNC(glGetMultiTexParameterivEXT,                             void,               (GLenum texunit, GLenum target, GLenum pname, GLint *params))
//GLFUNC(glGetMultiTexLevelParameterfvEXT,                        void,               (GLenum texunit, GLenum target, GLint level, GLenum pname, GLfloat *params))
//GLFUNC(glGetMultiTexLevelParameterivEXT,                        void,               (GLenum texunit, GLenum target, GLint level, GLenum pname, GLint *params))
//GLFUNC(glMultiTexImage3DEXT,                                    void,               (GLenum texunit, GLenum target, GLint level, GLint internalformat, GLsizei width, GLsizei height, GLsizei depth, GLint border, GLenum format, GLenum type, const void *pixels))
//GLFUNC(glMultiTexSubImage3DEXT,                                 void,               (GLenum texunit, GLenum target, GLint level, GLint xoffset, GLint yoffset, GLint zoffset, GLsizei width, GLsizei height, GLsizei depth, GLenum format, GLenum type, const void *pixels))
//GLFUNC(glCopyMultiTexSubImage3DEXT,                             void,               (GLenum texunit, GLenum target, GLint level, GLint xoffset, GLint yoffset, GLint zoffset, GLint x, GLint y, GLsizei width, GLsizei height))
GLFUNC(glEnableClientStateIndexedEXT,                           void,               (GLenum array, GLuint index))
GLFUNC(glDisableClientStateIndexedEXT,                          void,               (GLenum array, GLuint index))
GLFUNC(glGetFloatIndexedvEXT,                                   void,               (GLenum target, GLuint index, GLfloat *data))
GLFUNC(glGetDoubleIndexedvEXT,                                  void,               (GLenum target, GLuint index, GLdouble *data))
GLFUNC(glGetPointerIndexedvEXT,                                 void,               (GLenum target, GLuint index, void **data))
GLFUNC(glEnableIndexedEXT,                                      void,               (GLenum target, GLuint index))
GLFUNC(glDisableIndexedEXT,                                     void,               (GLenum target, GLuint index))
GLFUNC(glIsEnabledIndexedEXT,                                   GLboolean,          (GLenum target, GLuint index))
GLFUNC(glGetIntegerIndexedvEXT,                                 void,               (GLenum target, GLuint index, GLint *data))
GLFUNC(glGetBooleanIndexedvEXT,                                 void,               (GLenum target, GLuint index, GLboolean *data))
//GLFUNC(glCompressedTextureImage3DEXT,                           void,               (GLuint texture, GLenum target, GLint level, GLenum internalformat, GLsizei width, GLsizei height, GLsizei depth, GLint border, GLsizei imageSize, const void *bits))
//GLFUNC(glCompressedTextureImage2DEXT,                           void,               (GLuint texture, GLenum target, GLint level, GLenum internalformat, GLsizei width, GLsizei height, GLint border, GLsizei imageSize, const void *bits))
//GLFUNC(glCompressedTextureImage1DEXT,                           void,               (GLuint texture, GLenum target, GLint level, GLenum internalformat, GLsizei width, GLint border, GLsizei imageSize, const void *bits))
GLFUNC(glCompressedTextureSubImage3DEXT,                        void,               (GLuint texture, GLenum target, GLint level, GLint xoffset, GLint yoffset, GLint zoffset, GLsizei width, GLsizei height, GLsizei depth, GLenum format, GLsizei imageSize, const void *bits))
GLFUNC(glCompressedTextureSubImage2DEXT,                        void,               (GLuint texture, GLenum target, GLint level, GLint xoffset, GLint yoffset, GLsizei width, GLsizei height, GLenum format, GLsizei imageSize, const void *bits))
GLFUNC(glCompressedTextureSubImage1DEXT,                        void,               (GLuint texture, GLenum target, GLint level, GLint xoffset, GLsizei width, GLenum format, GLsizei imageSize, const void *bits))
GLFUNC(glGetCompressedTextureImageEXT,                          void,               (GLuint texture, GLenum target, GLint lod, void *img))
//GLFUNC(glCompressedMultiTexImage3DEXT,                          void,               (GLenum texunit, GLenum target, GLint level, GLenum internalformat, GLsizei width, GLsizei height, GLsizei depth, GLint border, GLsizei imageSize, const void *bits))
//GLFUNC(glCompressedMultiTexImage2DEXT,                          void,               (GLenum texunit, GLenum target, GLint level, GLenum internalformat, GLsizei width, GLsizei height, GLint border, GLsizei imageSize, const void *bits))
//GLFUNC(glCompressedMultiTexImage1DEXT,                          void,               (GLenum texunit, GLenum target, GLint level, GLenum internalformat, GLsizei width, GLint border, GLsizei imageSize, const void *bits))
//GLFUNC(glCompressedMultiTexSubImage3DEXT,                       void,               (GLenum texunit, GLenum target, GLint level, GLint xoffset, GLint yoffset, GLint zoffset, GLsizei width, GLsizei height, GLsizei depth, GLenum format, GLsizei imageSize, const void *bits))
//GLFUNC(glCompressedMultiTexSubImage2DEXT,                       void,               (GLenum texunit, GLenum target, GLint level, GLint xoffset, GLint yoffset, GLsizei width, GLsizei height, GLenum format, GLsizei imageSize, const void *bits))
//GLFUNC(glCompressedMultiTexSubImage1DEXT,                       void,               (GLenum texunit, GLenum target, GLint level, GLint xoffset, GLsizei width, GLenum format, GLsizei imageSize, const void *bits))
//GLFUNC(glGetCompressedMultiTexImageEXT,                         void,               (GLenum texunit, GLenum target, GLint lod, void *img))
//GLFUNC(glMatrixLoadTransposefEXT,                               void,               (GLenum mode, const GLfloat *m))
//GLFUNC(glMatrixLoadTransposedEXT,                               void,               (GLenum mode, const GLdouble *m))
//GLFUNC(glMatrixMultTransposefEXT,                               void,               (GLenum mode, const GLfloat *m))
//GLFUNC(glMatrixMultTransposedEXT,                               void,               (GLenum mode, const GLdouble *m))
GLFUNC(glNamedBufferDataEXT,                                    void,               (GLuint buffer, GLsizeiptr size, const void *data, GLenum usage))
GLFUNC(glNamedBufferSubDataEXT,                                 void,               (GLuint buffer, GLintptr offset, GLsizeiptr size, const void *data))
GLFUNC(glMapNamedBufferEXT,                                     void *,             (GLuint buffer, GLenum access))
GLFUNC(glUnmapNamedBufferEXT,                                   GLboolean,          (GLuint buffer))
GLFUNC(glGetNamedBufferParameterivEXT,                          void,               (GLuint buffer, GLenum pname, GLint *params))
GLFUNC(glGetNamedBufferPointervEXT,                             void,               (GLuint buffer, GLenum pname, void **params))
GLFUNC(glGetNamedBufferSubDataEXT,                              void,               (GLuint buffer, GLintptr offset, GLsizeiptr size, void *data))
GLFUNC(glProgramUniform1fEXT,                                   void,               (GLuint program, GLint location, GLfloat v0))
GLFUNC(glProgramUniform2fEXT,                                   void,               (GLuint program, GLint location, GLfloat v0, GLfloat v1))
GLFUNC(glProgramUniform3fEXT,                                   void,               (GLuint program, GLint location, GLfloat v0, GLfloat v1, GLfloat v2))
GLFUNC(glProgramUniform4fEXT,                                   void,               (GLuint program, GLint location, GLfloat v0, GLfloat v1, GLfloat v2, GLfloat v3))
GLFUNC(glProgramUniform1iEXT,                                   void,               (GLuint program, GLint location, GLint v0))
GLFUNC(glProgramUniform2iEXT,                                   void,               (GLuint program, GLint location, GLint v0, GLint v1))
GLFUNC(glProgramUniform3iEXT,                                   void,               (GLuint program, GLint location, GLint v0, GLint v1, GLint v2))
GLFUNC(glProgramUniform4iEXT,                                   void,               (GLuint program, GLint location, GLint v0, GLint v1, GLint v2, GLint v3))
GLFUNC(glProgramUniform1fvEXT,                                  void,               (GLuint program, GLint location, GLsizei count, const GLfloat *value))
GLFUNC(glProgramUniform2fvEXT,                                  void,               (GLuint program, GLint location, GLsizei count, const GLfloat *value))
GLFUNC(glProgramUniform3fvEXT,                                  void,               (GLuint program, GLint location, GLsizei count, const GLfloat *value))
GLFUNC(glProgramUniform4fvEXT,                                  void,               (GLuint program, GLint location, GLsizei count, const GLfloat *value))
GLFUNC(glProgramUniform1ivEXT,                                  void,               (GLuint program, GLint location, GLsizei count, const GLint *value))
GLFUNC(glProgramUniform2ivEXT,                                  void,               (GLuint program, GLint location, GLsizei count, const GLint *value))
GLFUNC(glProgramUniform3ivEXT,                                  void,               (GLuint program, GLint location, GLsizei count, const GLint *value))
GLFUNC(glProgramUniform4ivEXT,                                  void,               (GLuint program, GLint location, GLsizei count, const GLint *value))
GLFUNC(glProgramUniformMatrix2fvEXT,                            void,               (GLuint program, GLint location, GLsizei count, GLboolean transpose, const GLfloat *value))
GLFUNC(glProgramUniformMatrix3fvEXT,                            void,               (GLuint program, GLint location, GLsizei count, GLboolean transpose, const GLfloat *value))
GLFUNC(glProgramUniformMatrix4fvEXT,                            void,               (GLuint program, GLint location, GLsizei count, GLboolean transpose, const GLfloat *value))
GLFUNC(glProgramUniformMatrix2x3fvEXT,                          void,               (GLuint program, GLint location, GLsizei count, GLboolean transpose, const GLfloat *value))
GLFUNC(glProgramUniformMatrix3x2fvEXT,                          void,               (GLuint program, GLint location, GLsizei count, GLboolean transpose, const GLfloat *value))
GLFUNC(glProgramUniformMatrix2x4fvEXT,                          void,               (GLuint program, GLint location, GLsizei count, GLboolean transpose, const GLfloat *value))
GLFUNC(glProgramUniformMatrix4x2fvEXT,                          void,               (GLuint program, GLint location, GLsizei count, GLboolean transpose, const GLfloat *value))
GLFUNC(glProgramUniformMatrix3x4fvEXT,                          void,               (GLuint program, GLint location, GLsizei count, GLboolean transpose, const GLfloat *value))
GLFUNC(glProgramUniformMatrix4x3fvEXT,                          void,               (GLuint program, GLint location, GLsizei count, GLboolean transpose, const GLfloat *value))
GLFUNC(glTextureBufferEXT,                                      void,               (GLuint texture, GLenum target, GLenum internalformat, GLuint buffer))
//GLFUNC(glMultiTexBufferEXT,                                     void,               (GLenum texunit, GLenum target, GLenum internalformat, GLuint buffer))
GLFUNC(glTextureParameterIivEXT,                                void,               (GLuint texture, GLenum target, GLenum pname, const GLint *params))
GLFUNC(glTextureParameterIuivEXT,                               void,               (GLuint texture, GLenum target, GLenum pname, const GLuint *params))
GLFUNC(glGetTextureParameterIivEXT,                             void,               (GLuint texture, GLenum target, GLenum pname, GLint *params))
GLFUNC(glGetTextureParameterIuivEXT,                            void,               (GLuint texture, GLenum target, GLenum pname, GLuint *params))
//GLFUNC(glMultiTexParameterIivEXT,                               void,               (GLenum texunit, GLenum target, GLenum pname, const GLint *params))
//GLFUNC(glMultiTexParameterIuivEXT,                              void,               (GLenum texunit, GLenum target, GLenum pname, const GLuint *params))
//GLFUNC(glGetMultiTexParameterIivEXT,                            void,               (GLenum texunit, GLenum target, GLenum pname, GLint *params))
//GLFUNC(glGetMultiTexParameterIuivEXT,                           void,               (GLenum texunit, GLenum target, GLenum pname, GLuint *params))
GLFUNC(glProgramUniform1uiEXT,                                  void,               (GLuint program, GLint location, GLuint v0))
GLFUNC(glProgramUniform2uiEXT,                                  void,               (GLuint program, GLint location, GLuint v0, GLuint v1))
GLFUNC(glProgramUniform3uiEXT,                                  void,               (GLuint program, GLint location, GLuint v0, GLuint v1, GLuint v2))
GLFUNC(glProgramUniform4uiEXT,                                  void,               (GLuint program, GLint location, GLuint v0, GLuint v1, GLuint v2, GLuint v3))
GLFUNC(glProgramUniform1uivEXT,                                 void,               (GLuint program, GLint location, GLsizei count, const GLuint *value))
GLFUNC(glProgramUniform2uivEXT,                                 void,               (GLuint program, GLint location, GLsizei count, const GLuint *value))
GLFUNC(glProgramUniform3uivEXT,                                 void,               (GLuint program, GLint location, GLsizei count, const GLuint *value))
GLFUNC(glProgramUniform4uivEXT,                                 void,               (GLuint program, GLint location, GLsizei count, const GLuint *value))
//GLFUNC(glNamedProgramLocalParameters4fvEXT,                     void,               (GLuint program, GLenum target, GLuint index, GLsizei count, const GLfloat *params))
//GLFUNC(glNamedProgramLocalParameterI4iEXT,                      void,               (GLuint program, GLenum target, GLuint index, GLint x, GLint y, GLint z, GLint w))
//GLFUNC(glNamedProgramLocalParameterI4ivEXT,                     void,               (GLuint program, GLenum target, GLuint index, const GLint *params))
//GLFUNC(glNamedProgramLocalParametersI4ivEXT,                    void,               (GLuint program, GLenum target, GLuint index, GLsizei count, const GLint *params))
//GLFUNC(glNamedProgramLocalParameterI4uiEXT,                     void,               (GLuint program, GLenum target, GLuint index, GLuint x, GLuint y, GLuint z, GLuint w))
//GLFUNC(glNamedProgramLocalParameterI4uivEXT,                    void,               (GLuint program, GLenum target, GLuint index, const GLuint *params))
//GLFUNC(glNamedProgramLocalParametersI4uivEXT,                   void,               (GLuint program, GLenum target, GLuint index, GLsizei count, const GLuint *params))
//GLFUNC(glGetNamedProgramLocalParameterIivEXT,                   void,               (GLuint program, GLenum target, GLuint index, GLint *params))
//GLFUNC(glGetNamedProgramLocalParameterIuivEXT,                  void,               (GLuint program, GLenum target, GLuint index, GLuint *params))
GLFUNC(glEnableClientStateiEXT,                                 void,               (GLenum array, GLuint index))
GLFUNC(glDisableClientStateiEXT,                                void,               (GLenum array, GLuint index))
GLFUNC(glGetFloati_vEXT,                                        void,               (GLenum pname, GLuint index, GLfloat *params))
GLFUNC(glGetDoublei_vEXT,                                       void,               (GLenum pname, GLuint index, GLdouble *params))
GLFUNC(glGetPointeri_vEXT,                                      void,               (GLenum pname, GLuint index, void **params))
GLFUNC(glNamedProgramStringEXT,                                 void,               (GLuint program, GLenum target, GLenum format, GLsizei len, const void *string))
//GLFUNC(glNamedProgramLocalParameter4dEXT,                       void,               (GLuint program, GLenum target, GLuint index, GLdouble x, GLdouble y, GLdouble z, GLdouble w))
//GLFUNC(glNamedProgramLocalParameter4dvEXT,                      void,               (GLuint program, GLenum target, GLuint index, const GLdouble *params))
//GLFUNC(glNamedProgramLocalParameter4fEXT,                       void,               (GLuint program, GLenum target, GLuint index, GLfloat x, GLfloat y, GLfloat z, GLfloat w))
//GLFUNC(glNamedProgramLocalParameter4fvEXT,                      void,               (GLuint program, GLenum target, GLuint index, const GLfloat *params))
//GLFUNC(glGetNamedProgramLocalParameterdvEXT,                    void,               (GLuint program, GLenum target, GLuint index, GLdouble *params))
//GLFUNC(glGetNamedProgramLocalParameterfvEXT,                    void,               (GLuint program, GLenum target, GLuint index, GLfloat *params))
GLFUNC(glGetNamedProgramivEXT,                                  void,               (GLuint program, GLenum target, GLenum pname, GLint *params))
GLFUNC(glGetNamedProgramStringEXT,                              void,               (GLuint program, GLenum target, GLenum pname, void *string))
GLFUNC(glNamedRenderbufferStorageEXT,                           void,               (GLuint renderbuffer, GLenum internalformat, GLsizei width, GLsizei height))
GLFUNC(glGetNamedRenderbufferParameterivEXT,                    void,               (GLuint renderbuffer, GLenum pname, GLint *params))
GLFUNC(glNamedRenderbufferStorageMultisampleEXT,                void,               (GLuint renderbuffer, GLsizei samples, GLenum internalformat, GLsizei width, GLsizei height))
GLFUNC(glNamedRenderbufferStorageMultisampleCoverageEXT,        void,               (GLuint renderbuffer, GLsizei coverageSamples, GLsizei colorSamples, GLenum internalformat, GLsizei width, GLsizei height))
GLFUNC(glCheckNamedFramebufferStatusEXT,                        GLenum,             (GLuint framebuffer, GLenum target))
GLFUNC(glNamedFramebufferTexture1DEXT,                          void,               (GLuint framebuffer, GLenum attachment, GLenum textarget, GLuint texture, GLint level))
GLFUNC(glNamedFramebufferTexture2DEXT,                          void,               (GLuint framebuffer, GLenum attachment, GLenum textarget, GLuint texture, GLint level))
GLFUNC(glNamedFramebufferTexture3DEXT,                          void,               (GLuint framebuffer, GLenum attachment, GLenum textarget, GLuint texture, GLint level, GLint zoffset))
GLFUNC(glNamedFramebufferRenderbufferEXT,                       void,               (GLuint framebuffer, GLenum attachment, GLenum renderbuffertarget, GLuint renderbuffer))
GLFUNC(glGetNamedFramebufferAttachmentParameterivEXT,           void,               (GLuint framebuffer, GLenum attachment, GLenum pname, GLint *params))
GLFUNC(glGenerateTextureMipmapEXT,                              void,               (GLuint texture, GLenum target))
//GLFUNC(glGenerateMultiTexMipmapEXT,                             void,               (GLenum texunit, GLenum target))
GLFUNC(glFramebufferDrawBufferEXT,                              void,               (GLuint framebuffer, GLenum mode))
GLFUNC(glFramebufferDrawBuffersEXT,                             void,               (GLuint framebuffer, GLsizei n, const GLenum *bufs))
GLFUNC(glFramebufferReadBufferEXT,                              void,               (GLuint framebuffer, GLenum mode))
GLFUNC(glGetFramebufferParameterivEXT,                          void,               (GLuint framebuffer, GLenum pname, GLint *params))
GLFUNC(glNamedCopyBufferSubDataEXT,                             void,               (GLuint readBuffer, GLuint writeBuffer, GLintptr readOffset, GLintptr writeOffset, GLsizeiptr size))
GLFUNC(glNamedFramebufferTextureEXT,                            void,               (GLuint framebuffer, GLenum attachment, GLuint texture, GLint level))
GLFUNC(glNamedFramebufferTextureLayerEXT,                       void,               (GLuint framebuffer, GLenum attachment, GLuint texture, GLint level, GLint layer))
GLFUNC(glNamedFramebufferTextureFaceEXT,                        void,               (GLuint framebuffer, GLenum attachment, GLuint texture, GLint level, GLenum face))
GLFUNC(glTextureRenderbufferEXT,                                void,               (GLuint texture, GLenum target, GLuint renderbuffer))
//GLFUNC(glMultiTexRenderbufferEXT,                               void,               (GLenum texunit, GLenum target, GLuint renderbuffer))
GLFUNC(glVertexArrayVertexOffsetEXT,                            void,               (GLuint vaobj, GLuint buffer, GLint size, GLenum type, GLsizei stride, GLintptr offset))
//GLFUNC(glVertexArrayColorOffsetEXT,                             void,               (GLuint vaobj, GLuint buffer, GLint size, GLenum type, GLsizei stride, GLintptr offset))
//GLFUNC(glVertexArrayEdgeFlagOffsetEXT,                          void,               (GLuint vaobj, GLuint buffer, GLsizei stride, GLintptr offset))
//GLFUNC(glVertexArrayIndexOffsetEXT,                             void,               (GLuint vaobj, GLuint buffer, GLenum type, GLsizei stride, GLintptr offset))
//GLFUNC(glVertexArrayNormalOffsetEXT,                            void,               (GLuint vaobj, GLuint buffer, GLenum type, GLsizei stride, GLintptr offset))
//GLFUNC(glVertexArrayTexCoordOffsetEXT,                          void,               (GLuint vaobj, GLuint buffer, GLint size, GLenum type, GLsizei stride, GLintptr offset))
//GLFUNC(glVertexArrayMultiTexCoordOffsetEXT,                     void,               (GLuint vaobj, GLuint buffer, GLenum texunit, GLint size, GLenum type, GLsizei stride, GLintptr offset))
//GLFUNC(glVertexArrayFogCoordOffsetEXT,                          void,               (GLuint vaobj, GLuint buffer, GLenum type, GLsizei stride, GLintptr offset))
//GLFUNC(glVertexArraySecondaryColorOffsetEXT,                    void,               (GLuint vaobj, GLuint buffer, GLint size, GLenum type, GLsizei stride, GLintptr offset))
GLFUNC(glVertexArrayVertexAttribOffsetEXT,                      void,               (GLuint vaobj, GLuint buffer, GLuint index, GLint size, GLenum type, GLboolean normalized, GLsizei stride, GLintptr offset))
GLFUNC(glVertexArrayVertexAttribIOffsetEXT,                     void,               (GLuint vaobj, GLuint buffer, GLuint index, GLint size, GLenum type, GLsizei stride, GLintptr offset))
GLFUNC(glEnableVertexArrayEXT,                                  void,               (GLuint vaobj, GLenum array))
GLFUNC(glDisableVertexArrayEXT,                                 void,               (GLuint vaobj, GLenum array))
GLFUNC(glEnableVertexArrayAttribEXT,                            void,               (GLuint vaobj, GLuint index))
GLFUNC(glDisableVertexArrayAttribEXT,                           void,               (GLuint vaobj, GLuint index))
GLFUNC(glGetVertexArrayIntegervEXT,                             void,               (GLuint vaobj, GLenum pname, GLint *param))
GLFUNC(glGetVertexArrayPointervEXT,                             void,               (GLuint vaobj, GLenum pname, void **param))
GLFUNC(glGetVertexArrayIntegeri_vEXT,                           void,               (GLuint vaobj, GLuint index, GLenum pname, GLint *param))
GLFUNC(glGetVertexArrayPointeri_vEXT,                           void,               (GLuint vaobj, GLuint index, GLenum pname, void **param))
GLFUNC(glMapNamedBufferRangeEXT,                                void *,             (GLuint buffer, GLintptr offset, GLsizeiptr length, GLbitfield access))
GLFUNC(glFlushMappedNamedBufferRangeEXT,                        void,               (GLuint buffer, GLintptr offset, GLsizeiptr length))
GLFUNC(glNamedBufferStorageEXT,                                 void,               (GLuint buffer, GLsizeiptr size, const void *data, GLbitfield flags))
GLFUNC(glClearNamedBufferDataEXT,                               void,               (GLuint buffer, GLenum internalformat, GLenum format, GLenum type, const void *data))
GLFUNC(glClearNamedBufferSubDataEXT,                            void,               (GLuint buffer, GLenum internalformat, GLsizeiptr offset, GLsizeiptr size, GLenum format, GLenum type, const void *data))
GLFUNC(glNamedFramebufferParameteriEXT,                         void,               (GLuint framebuffer, GLenum pname, GLint param))
GLFUNC(glGetNamedFramebufferParameterivEXT,                     void,               (GLuint framebuffer, GLenum pname, GLint *params))
GLFUNC(glProgramUniform1dEXT,                                   void,               (GLuint program, GLint location, GLdouble x))
GLFUNC(glProgramUniform2dEXT,                                   void,               (GLuint program, GLint location, GLdouble x, GLdouble y))
GLFUNC(glProgramUniform3dEXT,                                   void,               (GLuint program, GLint location, GLdouble x, GLdouble y, GLdouble z))
GLFUNC(glProgramUniform4dEXT,                                   void,               (GLuint program, GLint location, GLdouble x, GLdouble y, GLdouble z, GLdouble w))
GLFUNC(glProgramUniform1dvEXT,                                  void,               (GLuint program, GLint location, GLsizei count, const GLdouble *value))
GLFUNC(glProgramUniform2dvEXT,                                  void,               (GLuint program, GLint location, GLsizei count, const GLdouble *value))
GLFUNC(glProgramUniform3dvEXT,                                  void,               (GLuint program, GLint location, GLsizei count, const GLdouble *value))
GLFUNC(glProgramUniform4dvEXT,                                  void,               (GLuint program, GLint location, GLsizei count, const GLdouble *value))
GLFUNC(glProgramUniformMatrix2dvEXT,                            void,               (GLuint program, GLint location, GLsizei count, GLboolean transpose, const GLdouble *value))
GLFUNC(glProgramUniformMatrix3dvEXT,                            void,               (GLuint program, GLint location, GLsizei count, GLboolean transpose, const GLdouble *value))
GLFUNC(glProgramUniformMatrix4dvEXT,                            void,               (GLuint program, GLint location, GLsizei count, GLboolean transpose, const GLdouble *value))
GLFUNC(glProgramUniformMatrix2x3dvEXT,                          void,               (GLuint program, GLint location, GLsizei count, GLboolean transpose, const GLdouble *value))
GLFUNC(glProgramUniformMatrix2x4dvEXT,                          void,               (GLuint program, GLint location, GLsizei count, GLboolean transpose, const GLdouble *value))
GLFUNC(glProgramUniformMatrix3x2dvEXT,                          void,               (GLuint program, GLint location, GLsizei count, GLboolean transpose, const GLdouble *value))
GLFUNC(glProgramUniformMatrix3x4dvEXT,                          void,               (GLuint program, GLint location, GLsizei count, GLboolean transpose, const GLdouble *value))
GLFUNC(glProgramUniformMatrix4x2dvEXT,                          void,               (GLuint program, GLint location, GLsizei count, GLboolean transpose, const GLdouble *value))
GLFUNC(glProgramUniformMatrix4x3dvEXT,                          void,               (GLuint program, GLint location, GLsizei count, GLboolean transpose, const GLdouble *value))
GLFUNC(glTextureBufferRangeEXT,                                 void,               (GLuint texture, GLenum target, GLenum internalformat, GLuint buffer, GLintptr offset, GLsizeiptr size))
GLFUNC(glTextureStorage1DEXT,                                   void,               (GLuint texture, GLenum target, GLsizei levels, GLenum internalformat, GLsizei width))
GLFUNC(glTextureStorage2DEXT,                                   void,               (GLuint texture, GLenum target, GLsizei levels, GLenum internalformat, GLsizei width, GLsizei height))
GLFUNC(glTextureStorage3DEXT,                                   void,               (GLuint texture, GLenum target, GLsizei levels, GLenum internalformat, GLsizei width, GLsizei height, GLsizei depth))
GLFUNC(glTextureStorage2DMultisampleEXT,                        void,               (GLuint texture, GLenum target, GLsizei samples, GLenum internalformat, GLsizei width, GLsizei height, GLboolean fixedsamplelocations))
GLFUNC(glTextureStorage3DMultisampleEXT,                        void,               (GLuint texture, GLenum target, GLsizei samples, GLenum internalformat, GLsizei width, GLsizei height, GLsizei depth, GLboolean fixedsamplelocations))
GLFUNC(glVertexArrayBindVertexBufferEXT,                        void,               (GLuint vaobj, GLuint bindingindex, GLuint buffer, GLintptr offset, GLsizei stride))
GLFUNC(glVertexArrayVertexAttribFormatEXT,                      void,               (GLuint vaobj, GLuint attribindex, GLint size, GLenum type, GLboolean normalized, GLuint relativeoffset))
GLFUNC(glVertexArrayVertexAttribIFormatEXT,                     void,               (GLuint vaobj, GLuint attribindex, GLint size, GLenum type, GLuint relativeoffset))
GLFUNC(glVertexArrayVertexAttribLFormatEXT,                     void,               (GLuint vaobj, GLuint attribindex, GLint size, GLenum type, GLuint relativeoffset))
GLFUNC(glVertexArrayVertexAttribBindingEXT,                     void,               (GLuint vaobj, GLuint attribindex, GLuint bindingindex))
GLFUNC(glVertexArrayVertexBindingDivisorEXT,                    void,               (GLuint vaobj, GLuint bindingindex, GLuint divisor))
GLFUNC(glVertexArrayVertexAttribLOffsetEXT,                     void,               (GLuint vaobj, GLuint buffer, GLuint index, GLint size, GLenum type, GLsizei stride, GLintptr offset))
GLFUNC(glTexturePageCommitmentEXT,                              void,               (GLuint texture, GLint level, GLint xoffset, GLint yoffset, GLint zoffset, GLsizei width, GLsizei height, GLsizei depth, GLboolean resident))
GLFUNC(glVertexArrayVertexAttribDivisorEXT,                     void,               (GLuint vaobj, GLuint index, GLuint divisor))

GLFALLBACK(glBindMultiTextureEXT,                               glBindMultiTextureEXT_IMPL)
GLFALLBACK(glTextureParameteriEXT,                              glTextureParameteriEXT_IMPL)
GLFALLBACK(glTextureParameterivEXT,                             glTextureParameterivEXT_IMPL)
GLFALLBACK(glTextureParameterfEXT,                              glTextureParameterfEXT_IMPL)
GLFALLBACK(glTextureParameterfvEXT,                             glTextureParameterfvEXT_IMPL)
GLFALLBACK(glGetTextureParameterivEXT,                          glGetTextureParameterivEXT_IMPL)
GLFALLBACK(glGetTextureParameterfvEXT,                          glGetTextureParameterfvEXT_IMPL)
GLFALLBACK(glGetTextureLevelParameterivEXT,                     glGetTextureLevelParameterivEXT_IMPL)
GLFALLBACK(glGetTextureLevelParameterfvEXT,                     glGetTextureLevelParameterfvEXT_IMPL)
GLFALLBACK(glGetTextureImageEXT,                                glGetTextureImageEXT_IMPL)
GLFALLBACK(glTextureStorage1DEXT,                               glTextureStorage1DEXT_IMPL)
GLFALLBACK(glTextureStorage2DEXT,                               glTextureStorage2DEXT_IMPL)
GLFALLBACK(glTextureStorage3DEXT,                               glTextureStorage3DEXT_IMPL)
GLFALLBACK(glTextureSubImage1DEXT,                              glTextureSubImage1DEXT_IMPL)
GLFALLBACK(glTextureSubImage2DEXT,                              glTextureSubImage2DEXT_IMPL)
GLFALLBACK(glTextureSubImage3DEXT,                              glTextureSubImage3DEXT_IMPL)
GLFALLBACK(glGenerateTextureMipmapEXT,                          glGenerateTextureMipmapEXT_IMPL)

GLFALLBACK(glNamedBufferDataEXT,                                glNamedBufferDataEXT_IMPL)
GLFALLBACK(glNamedBufferSubDataEXT,                             glNamedBufferSubDataEXT_IMPL)
GLFALLBACK(glMapNamedBufferEXT,                                 glMapNamedBufferEXT_IMPL)
GLFALLBACK(glUnmapNamedBufferEXT,                               glUnmapNamedBufferEXT_IMPL)
GLFALLBACK(glGetNamedBufferParameterivEXT,                      glGetNamedBufferParameterivEXT_IMPL)
GLFALLBACK(glGetNamedBufferSubDataEXT,                          glGetNamedBufferSubDataEXT_IMPL)

#endif

//------------------------------------------------------------------------------
// GL_NV_bindless_texture
//

//#if !defined(GL_NV_bindless_texture) || (GL_NV_bindless_texture == 0xDLL)
//#define GL_NV_bindless_texture 0xDLL
//
//GLEXT(GL_NV_bindless_texture)
//
//GLFUNC(glGetTextureHandleNV,                                    GLuint64,           (GLuint texture))
//GLFUNC(glGetTextureSamplerHandleNV,                             GLuint64,           (GLuint texture, GLuint sampler))
//GLFUNC(glMakeTextureHandleResidentNV,                           void,               (GLuint64 handle))
//GLFUNC(glMakeTextureHandleNonResidentNV,                        void,               (GLuint64 handle))
//GLFUNC(glGetImageHandleNV,                                      GLuint64,           (GLuint texture, GLint level, GLboolean layered, GLint layer, GLenum format))
//GLFUNC(glMakeImageHandleResidentNV,                             void,               (GLuint64 handle, GLenum access))
//GLFUNC(glMakeImageHandleNonResidentNV,                          void,               (GLuint64 handle))
//GLFUNC(glUniformHandleui64NV,                                   void,               (GLint location, GLuint64 value))
//GLFUNC(glUniformHandleui64vNV,                                  void,               (GLint location, GLsizei count, const GLuint64 *value))
//GLFUNC(glProgramUniformHandleui64NV,                            void,               (GLuint program, GLint location, GLuint64 value))
//GLFUNC(glProgramUniformHandleui64vNV,                           void,               (GLuint program, GLint location, GLsizei count, const GLuint64 *values))
//GLFUNC(glIsTextureHandleResidentNV,                             GLboolean,          (GLuint64 handle))
//GLFUNC(glIsImageHandleResidentNV,                               GLboolean,          (GLuint64 handle))
//
//#endif

#undef GLALIAS
#undef GLFALLBACK
#undef GLFUNC
#undef GLEXT
