find_package(Threads REQUIRED)

# one TU per demo: each demo header is parsed by exactly one factory .cpp,
# so editing a demo rebuilds only that demo and the 37 build in parallel
add_library(manifold_demos STATIC
    factories/ae_karman_demo.cpp
    factories/aerofoil_elevator_demo.cpp
    factories/aerofoil_flutter_demo.cpp
    factories/cae_karman_demo.cpp
    factories/cart_double_pendulum_demo.cpp
    factories/cart_pendulum_demo.cpp
    factories/circuit_demo.cpp
    factories/crank_slider.cpp
    factories/cube_demo.cpp
    factories/deformation_jacobian_demo.cpp
    factories/diffuser_demo.cpp
    factories/double_pendulum_demo.cpp
    factories/engine_demo.cpp
    factories/esn_karman_demo.cpp
    factories/fea_flutter_demo.cpp
    factories/fluid_demo.cpp
    factories/flutter_demo.cpp
    factories/graphics_test_demo.cpp
    factories/heat_demo.cpp
    factories/info_demo.cpp
    factories/jansen_demo.cpp
    factories/karman_demo.cpp
    factories/nbody_demo.cpp
    factories/nozzle_demo.cpp
    factories/pde_demo.cpp
    factories/pendulum_demo.cpp
    factories/pod_karman_demo.cpp
    factories/radial_engine_demo.cpp
    factories/rocket_landing_demo.cpp
    factories/showcase2_demo.cpp
    factories/solar_system_demo.cpp
    factories/spring_demo.cpp
    factories/story_demo.cpp
    factories/string_art.cpp
    factories/supersonic_demo.cpp
    factories/truss_demo.cpp
    factories/wave_demo.cpp
)

target_include_directories(manifold_demos PUBLIC
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/include
)
target_link_libraries(manifold_demos PUBLIC
    manifold_solver
    manifold_control
    manifold_pde
    manifold_ai
    manifold_fea
    manifold_fluid
    manifold_electrical
    manifold_renderer
    raygui
    Threads::Threads
)

add_subdirectory(launcher)
