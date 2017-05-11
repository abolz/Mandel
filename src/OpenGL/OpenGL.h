#pragma once

#include "gl_impl.h"

#include <iosfwd>
#include <string>
#include <vector>

namespace gl {

//------------------------------------------------------------------------------

// Initialize function pointers for the current context.
// Must be called before any other function in namespace gl.
void init();

// Initialize function pointers for the current context.
// Must be called before any other function in namespace gl.
void init(std::ostream& out);

//------------------------------------------------------------------------------

// Returns the OpenGL version of the current context.
// Format is MMmm.
unsigned getVersion();

//------------------------------------------------------------------------------

using Extensions = std::vector<std::string>;

// Returns the list of extensions for the current context.
// This adds "GL_VERSION_2_0" etc. if addVersions is true, so one can check
// for supported versions easily.
Extensions getExtensions(bool addVersions = true);

// Returns whether a single extension is supported
bool supports(Extensions const& E, std::string const& str);

// Returns whether the given extensions are supported.
// The list must be a space-separated list of extension names.
bool supportsAll(Extensions const& E, std::string const& str);

// Returns whether any of the given extensions is supported.
// The list must be a space-separated list of extension names.
bool supportsAny(Extensions const& E, std::string const& str);

} // namespace gl
