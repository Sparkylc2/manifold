add_executable(solver_tests solver/solver_tests.cpp)
target_link_libraries(solver_tests PRIVATE manifold_solver)

add_executable(pde_tests pde/pde_tests.cpp)
target_link_libraries(pde_tests PRIVATE manifold_pde)

add_executable(ai_tests ai/ai_tests.cpp)
target_link_libraries(ai_tests PRIVATE manifold_ai)

add_executable(ai_cae_ae_tests ai/cae_ae_tests_ai.cpp)
target_link_libraries(ai_cae_ae_tests PRIVATE manifold_ai)


add_executable(ai_serialize_tests ai/serialize_tests.cpp)
target_link_libraries(ai_serialize_tests PRIVATE manifold_ai)
