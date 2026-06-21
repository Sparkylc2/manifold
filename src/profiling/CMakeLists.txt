# profiling always builds with MANIFOLD_PROFILE so timing is available
# without needing to rebuild the whole project

# we need our own solver copy with the define baked in
add_library(manifold_solver_profiled STATIC
    ${CMAKE_SOURCE_DIR}/src/solver/rigid_body_system.cpp
    ${CMAKE_SOURCE_DIR}/src/solver/rigid_body.cpp
    ${CMAKE_SOURCE_DIR}/src/solver/system_state.cpp
    ${CMAKE_SOURCE_DIR}/src/solver/ode_solver.cpp
    ${CMAKE_SOURCE_DIR}/src/solver/rk4_ode_solver.cpp
    ${CMAKE_SOURCE_DIR}/src/solver/euler_ode_solver.cpp
    ${CMAKE_SOURCE_DIR}/src/solver/sle_solver.cpp
    ${CMAKE_SOURCE_DIR}/src/solver/conjugate_gradient_sle_solver.cpp
    ${CMAKE_SOURCE_DIR}/src/solver/constraint.cpp
    ${CMAKE_SOURCE_DIR}/src/solver/generic_body_system.cpp
)
target_include_directories(manifold_solver_profiled PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(manifold_solver_profiled PUBLIC Eigen3::Eigen)
target_compile_definitions(manifold_solver_profiled PUBLIC MANIFOLD_PROFILE)

add_executable(solver_benchmark solver_benchmark.cpp)
target_link_libraries(solver_benchmark PRIVATE manifold_solver_profiled)
target_compile_definitions(solver_benchmark PRIVATE MANIFOLD_PROFILE)
