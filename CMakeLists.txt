cmake_minimum_required(VERSION 3.16)
project(manifold LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

# dependencies
include(cmake/FetchDependencies.cmake)

# ---- solver library  ----
add_library(manifold_solver STATIC
    src/solver/rigid_body_system.cpp
    src/solver/rigid_body.cpp
    src/solver/system_state.cpp
    src/solver/ode_solver.cpp
    src/solver/rk4_ode_solver.cpp
    src/solver/sle_solver.cpp
    src/solver/conjugate_gradient_sle_solver.cpp
    src/solver/constraint.cpp
    src/solver/generic_body_system.cpp
)
target_include_directories(manifold_solver PUBLIC include)
target_link_libraries(manifold_solver PUBLIC Eigen3::Eigen)

# ---- control (header-only for now) ----
add_library(manifold_control INTERFACE)
target_include_directories(manifold_control INTERFACE include)

# ---- renderer (header-only, links raylib) ----
add_library(manifold_renderer INTERFACE)
target_include_directories(manifold_renderer INTERFACE include)
target_link_libraries(manifold_renderer INTERFACE raylib)

# ---- demos & tests ----
add_subdirectory(demos)
# add_subdirectory(tests)
