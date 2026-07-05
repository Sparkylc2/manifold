add_executable(solver_tests solver/solver_tests.cpp)
target_link_libraries(solver_tests PRIVATE manifold_solver)

add_executable(pde_tests pde/pde_tests.cpp)
target_link_libraries(pde_tests PRIVATE manifold_pde)

add_executable(ai_tests ai/ai_tests.cpp)
target_link_libraries(ai_tests PRIVATE manifold_ai)
