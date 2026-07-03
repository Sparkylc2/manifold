add_executable(solver_tests solver/solver_tests.cpp)
target_link_libraries(solver_tests PRIVATE manifold_solver)

add_executable(pde_tests pde/pde_tests.cpp)
target_link_libraries(pde_tests PRIVATE manifold_pde)
