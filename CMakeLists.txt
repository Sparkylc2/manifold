cmake_minimum_required(VERSION 3.16)
project(manifold LANGUAGES C CXX)
add_compile_options($<$<CONFIG:Release>:-march=native>)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# dependencies
include(cmake/FetchDependencies.cmake)

find_package(OpenMP)

# ---- solver library  ----
add_library(manifold_solver STATIC
    src/solver/conjugate_gradient_sle_solver.cpp
    src/solver/constraint.cpp
    src/solver/euler_ode_solver.cpp
    src/solver/gauss_seidel_sle_solver.cpp
    src/solver/gaussian_elimination_sle_solver.cpp
    src/solver/generic_body_system.cpp
    src/solver/ode_solver.cpp
    src/solver/rigid_body_system.cpp
    src/solver/rigid_body.cpp
    src/solver/rk4_ode_solver.cpp
    src/solver/sle_solver.cpp
    src/solver/system_state.cpp
    src/solver/forces/barnes_hut_gravity.cpp
    src/solver/forces/constant_speed_motor.cpp
    src/solver/forces/control_torque.cpp
    src/solver/forces/damper.cpp
    src/solver/forces/direct_force.cpp
    src/solver/forces/exact_gravity.cpp
    src/solver/forces/impulse.cpp
    src/solver/forces/mouse_spring.cpp
    src/solver/forces/spring.cpp
    src/solver/forces/torsion_spring.cpp
    src/solver/forces/uniform_gravity.cpp
    src/solver/constraints/distance_constraint.cpp
    src/solver/constraints/fixed_position_constraint.cpp
    src/solver/constraints/fixed_rotation_constraint.cpp
    src/solver/constraints/gear_constraint.cpp
    src/solver/constraints/line_constraint.cpp
    src/solver/constraints/link_constraint.cpp
    src/solver/constraints/no_slip_constraint.cpp
    src/solver/constraints/rolling_constraint.cpp
    src/solver/constraints/rotation_friction_constraint.cpp
)
target_include_directories(manifold_solver PUBLIC include)
target_link_libraries(manifold_solver PUBLIC Eigen3::Eigen)


# ---- fluid (header-only for now) ----
add_library(manifold_fluid INTERFACE)
target_include_directories(manifold_fluid INTERFACE include)
target_link_libraries(manifold_fluid INTERFACE Eigen3::Eigen)

# ---- ai ----
add_library(manifold_ai STATIC
    src/ai/archive.cpp
    src/ai/autoencoder.cpp
    src/ai/dense_layer.cpp
    src/ai/conv_layer.cpp
    src/ai/conv_autoencoder.cpp
    src/ai/layer.cpp
    src/ai/history.cpp
    src/ai/io.cpp
    src/ai/pod.cpp
    src/ai/snapshot_recorder.cpp
    src/ai/svd.cpp
)
target_include_directories(manifold_ai PUBLIC include)
target_link_libraries(manifold_ai PUBLIC Eigen3::Eigen )
target_link_libraries(manifold_ai PUBLIC Spectra::Spectra)
if(OpenMP_CXX_FOUND)
    target_link_libraries(manifold_ai PUBLIC OpenMP::OpenMP_CXX)
endif()

# ---- electrical ----
add_library(manifold_electrical STATIC
    src/electrical/elements/capacitor.cpp
    src/electrical/elements/current_source.cpp
    src/electrical/elements/diode.cpp
    src/electrical/elements/inductor.cpp
    src/electrical/elements/op_amp.cpp
    src/electrical/elements/resistor.cpp
    src/electrical/elements/voltage_source.cpp
    src/electrical/circuit_assembler.cpp
    src/electrical/circuit_system.cpp
)
target_include_directories(manifold_electrical PUBLIC include)
target_link_libraries(manifold_electrical PUBLIC Eigen3::Eigen )
# ---- fea ----
add_library(manifold_fea STATIC
    src/fea/mesh.cpp
    src/fea/boundary.cpp
    src/fea/assembler.cpp
    src/fea/fea_solver.cpp
    src/fea/elements/cst_elastic.cpp
    src/fea/elements/quad_elastic.cpp
    src/fea/elements/tri_thermal.cpp
    src/fea/integrators/newmark.cpp
    src/fea/integrators/backward_euler.cpp
)
target_include_directories(manifold_fea PUBLIC include)
target_link_libraries(manifold_fea PUBLIC Eigen3::Eigen manifold_solver)





# ---- control (header-only for now) ----
add_library(manifold_control INTERFACE)
target_include_directories(manifold_control INTERFACE include)

# ---- pde (header-only) ----
add_library(manifold_pde INTERFACE)
target_include_directories(manifold_pde INTERFACE include)
target_link_libraries(manifold_pde INTERFACE Eigen3::Eigen)

# ---- renderer (links raylib) ----
add_library(manifold_renderer STATIC
    src/renderer/raylib_renderer.cpp
    src/renderer/layered_renderer.cpp
    src/renderer/camera_controller.cpp
    src/renderer/hud_panel.cpp
    src/renderer/scene3d.cpp
    src/renderer/plot3d.cpp
    src/renderer/field_view.cpp
    src/renderer/texture_view.cpp
    src/renderer/equation_cache.cpp
    src/renderer/demo_base.cpp
    src/renderer/body_visuals.cpp
    src/renderer/plot_widget.cpp
    src/renderer/annotation_visuals.cpp
    src/renderer/constraint_visuals.cpp
    src/renderer/circuit_visuals.cpp
)
target_include_directories(manifold_renderer PUBLIC include)
target_link_libraries(manifold_renderer PUBLIC raylib manifold_solver manifold_electrical Eigen3::Eigen)

# ---- demos & tests ----
add_subdirectory(demos)

# ---- headless trainers ----
add_executable(train_esn_demo tools/train_esn_demo.cpp)
target_link_libraries(train_esn_demo PRIVATE manifold_ai manifold_fluid)
target_include_directories(train_esn_demo PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# ---- profiling ----
add_subdirectory(src/profiling)

# add_subdirectory(tests)
