# profiling always builds with MANIFOLD_PROFILE so timing is available
# without needing to rebuild the whole project

# we need our own solver copy with the define baked in
add_library(manifold_solver_profiled STATIC
    ${CMAKE_SOURCE_DIR}/src/solver/conjugate_gradient_sle_solver.cpp
    ${CMAKE_SOURCE_DIR}/src/solver/constraint.cpp
    ${CMAKE_SOURCE_DIR}/src/solver/euler_ode_solver.cpp
    ${CMAKE_SOURCE_DIR}/src/solver/gauss_seidel_sle_solver.cpp
    ${CMAKE_SOURCE_DIR}/src/solver/gaussian_elimination_sle_solver.cpp
    ${CMAKE_SOURCE_DIR}/src/solver/generic_body_system.cpp
    ${CMAKE_SOURCE_DIR}/src/solver/ode_solver.cpp
    ${CMAKE_SOURCE_DIR}/src/solver/rigid_body_system.cpp
    ${CMAKE_SOURCE_DIR}/src/solver/rigid_body.cpp
    ${CMAKE_SOURCE_DIR}/src/solver/rk4_ode_solver.cpp
    ${CMAKE_SOURCE_DIR}/src/solver/sle_solver.cpp
    ${CMAKE_SOURCE_DIR}/src/solver/system_state.cpp
    ${CMAKE_SOURCE_DIR}/src/solver/forces/barnes_hut_gravity.cpp
    ${CMAKE_SOURCE_DIR}/src/solver/forces/constant_speed_motor.cpp
    ${CMAKE_SOURCE_DIR}/src/solver/forces/control_torque.cpp
    ${CMAKE_SOURCE_DIR}/src/solver/forces/damper.cpp
    ${CMAKE_SOURCE_DIR}/src/solver/forces/direct_force.cpp
    ${CMAKE_SOURCE_DIR}/src/solver/forces/exact_gravity.cpp
    ${CMAKE_SOURCE_DIR}/src/solver/forces/impulse.cpp
    ${CMAKE_SOURCE_DIR}/src/solver/forces/mouse_spring.cpp
    ${CMAKE_SOURCE_DIR}/src/solver/forces/spring.cpp
    ${CMAKE_SOURCE_DIR}/src/solver/forces/torsion_spring.cpp
    ${CMAKE_SOURCE_DIR}/src/solver/forces/uniform_gravity.cpp
    ${CMAKE_SOURCE_DIR}/src/solver/constraints/distance_constraint.cpp
    ${CMAKE_SOURCE_DIR}/src/solver/constraints/fixed_position_constraint.cpp
    ${CMAKE_SOURCE_DIR}/src/solver/constraints/fixed_rotation_constraint.cpp
    ${CMAKE_SOURCE_DIR}/src/solver/constraints/gear_constraint.cpp
    ${CMAKE_SOURCE_DIR}/src/solver/constraints/line_constraint.cpp
    ${CMAKE_SOURCE_DIR}/src/solver/constraints/link_constraint.cpp
    ${CMAKE_SOURCE_DIR}/src/solver/constraints/no_slip_constraint.cpp
    ${CMAKE_SOURCE_DIR}/src/solver/constraints/rolling_constraint.cpp
    ${CMAKE_SOURCE_DIR}/src/solver/constraints/rotation_friction_constraint.cpp
)

target_include_directories(manifold_solver_profiled PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(manifold_solver_profiled PUBLIC Eigen3::Eigen)
target_compile_definitions(manifold_solver_profiled PUBLIC MANIFOLD_PROFILE)

add_executable(solver_benchmark solver_benchmark.cpp)
target_link_libraries(solver_benchmark PRIVATE manifold_solver_profiled)
target_compile_definitions(solver_benchmark PRIVATE MANIFOLD_PROFILE)
