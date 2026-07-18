include(FetchContent)

# --- Eigen ---
FetchContent_Declare(
    eigen
    GIT_REPOSITORY https://github.com/eigen-mirror/eigen.git
    GIT_TAG        3.4.0
    GIT_SHALLOW    TRUE
)
set(EIGEN_BUILD_DOC OFF CACHE BOOL "" FORCE)
set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(EIGEN_BUILD_PKGCONFIG OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(eigen)

# --- Spectra ---
FetchContent_Declare(
    spectra
    GIT_REPOSITORY https://github.com/yixuan/spectra.git
    GIT_TAG        v1.0.1
    GIT_SHALLOW    TRUE
)
set(SPECTRA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(SPECTRA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(spectra)

# FetchContent only exposes the bare `Spectra` target; the namespaced
# alias exists only for an installed find_package(). Add it so links work.
if(NOT TARGET Spectra::Spectra)
    add_library(Spectra::Spectra ALIAS Spectra)
endif()

# --- raylib ---
FetchContent_Declare(
    raylib
    GIT_REPOSITORY https://github.com/raysan5/raylib.git
    GIT_TAG        5.5
    GIT_SHALLOW    TRUE
)
set(BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(BUILD_GAMES OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(raylib)

# --- raygui (header-only) ---
FetchContent_Declare(
    raygui
    GIT_REPOSITORY https://github.com/raysan5/raygui.git
    GIT_TAG        4.0
    GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(raygui)
add_library(raygui INTERFACE)
target_include_directories(raygui INTERFACE ${raygui_SOURCE_DIR}/src)


