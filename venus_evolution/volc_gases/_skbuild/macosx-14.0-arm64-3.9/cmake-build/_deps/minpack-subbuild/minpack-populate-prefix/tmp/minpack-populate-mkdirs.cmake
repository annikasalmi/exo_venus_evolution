# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION ${CMAKE_VERSION}) # this file comes with cmake

# If CMAKE_DISABLE_SOURCE_CHANGES is set to true and the source directory is an
# existing directory in our source tree, calling file(MAKE_DIRECTORY) on it
# would cause a fatal error, even though it would be a no-op.
if(NOT EXISTS "/Users/annikasalmi/funding_project_venus/venus_evolution/volc_gases/_skbuild/macosx-14.0-arm64-3.9/cmake-build/_deps/minpack-src")
  file(MAKE_DIRECTORY "/Users/annikasalmi/funding_project_venus/venus_evolution/volc_gases/_skbuild/macosx-14.0-arm64-3.9/cmake-build/_deps/minpack-src")
endif()
file(MAKE_DIRECTORY
  "/Users/annikasalmi/funding_project_venus/venus_evolution/volc_gases/_skbuild/macosx-14.0-arm64-3.9/cmake-build/_deps/minpack-build"
  "/Users/annikasalmi/funding_project_venus/venus_evolution/volc_gases/_skbuild/macosx-14.0-arm64-3.9/cmake-build/_deps/minpack-subbuild/minpack-populate-prefix"
  "/Users/annikasalmi/funding_project_venus/venus_evolution/volc_gases/_skbuild/macosx-14.0-arm64-3.9/cmake-build/_deps/minpack-subbuild/minpack-populate-prefix/tmp"
  "/Users/annikasalmi/funding_project_venus/venus_evolution/volc_gases/_skbuild/macosx-14.0-arm64-3.9/cmake-build/_deps/minpack-subbuild/minpack-populate-prefix/src/minpack-populate-stamp"
  "/Users/annikasalmi/funding_project_venus/venus_evolution/volc_gases/_skbuild/macosx-14.0-arm64-3.9/cmake-build/_deps/minpack-subbuild/minpack-populate-prefix/src"
  "/Users/annikasalmi/funding_project_venus/venus_evolution/volc_gases/_skbuild/macosx-14.0-arm64-3.9/cmake-build/_deps/minpack-subbuild/minpack-populate-prefix/src/minpack-populate-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/Users/annikasalmi/funding_project_venus/venus_evolution/volc_gases/_skbuild/macosx-14.0-arm64-3.9/cmake-build/_deps/minpack-subbuild/minpack-populate-prefix/src/minpack-populate-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/Users/annikasalmi/funding_project_venus/venus_evolution/volc_gases/_skbuild/macosx-14.0-arm64-3.9/cmake-build/_deps/minpack-subbuild/minpack-populate-prefix/src/minpack-populate-stamp${cfgdir}") # cfgdir has leading slash
endif()
