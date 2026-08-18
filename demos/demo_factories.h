#pragma once

#include <manifold/renderer/demo_base.h>
#include <memory>

namespace manifold::Demo {

using DemoFactory = std::unique_ptr<DemoBase> (*)();

// defined in the generated disabled_demos.cpp: true for factories compiled out
// by -DMANIFOLD_DEMOS, whose stub returns nullptr
bool demo_stubbed(DemoFactory factory);

// one factory per demo; each is defined in its own TU so a demo header is
// parsed by exactly one translation unit
std::unique_ptr<DemoBase> make_pendulum_demo();
std::unique_ptr<DemoBase> make_double_pendulum_demo();
std::unique_ptr<DemoBase> make_spring_demo();
std::unique_ptr<DemoBase> make_circuit_demo();
std::unique_ptr<DemoBase> make_cart_pendulum_demo();
std::unique_ptr<DemoBase> make_cart_double_pendulum_demo();
std::unique_ptr<DemoBase> make_nbody_demo();
std::unique_ptr<DemoBase> make_truss_demo();
std::unique_ptr<DemoBase> make_collapse_demo();
std::unique_ptr<DemoBase> make_huygens_demo();
std::unique_ptr<DemoBase> make_solar_system_demo();
std::unique_ptr<DemoBase> make_crank_slider();
std::unique_ptr<DemoBase> make_jansen_demo();
std::unique_ptr<DemoBase> make_story_demo();
std::unique_ptr<DemoBase> make_graphics_test_demo();
std::unique_ptr<DemoBase> make_cube_demo();
std::unique_ptr<DemoBase> make_engine_demo();
std::unique_ptr<DemoBase> make_radial_engine_demo();
std::unique_ptr<DemoBase> make_fluid_demo();
std::unique_ptr<DemoBase> make_rocket_landing_demo();
std::unique_ptr<DemoBase> make_karman_demo();
std::unique_ptr<DemoBase> make_pod_karman_demo();
std::unique_ptr<DemoBase> make_ae_karman_demo();
std::unique_ptr<DemoBase> make_esn_karman_demo();
std::unique_ptr<DemoBase> make_cae_karman_demo();
std::unique_ptr<DemoBase> make_forecast_karman_demo();
std::unique_ptr<DemoBase> make_ae_compress_demo();
std::unique_ptr<DemoBase> make_aerofoil_flutter_demo();
std::unique_ptr<DemoBase> make_fea_flutter_demo();
std::unique_ptr<DemoBase> make_aerofoil_elevator_demo();
std::unique_ptr<DemoBase> make_supersonic_demo();
std::unique_ptr<DemoBase> make_nozzle_demo();
std::unique_ptr<DemoBase> make_diffuser_demo();
std::unique_ptr<DemoBase> make_flutter_demo();
std::unique_ptr<DemoBase> make_pde_demo();
std::unique_ptr<DemoBase> make_heat_demo();
std::unique_ptr<DemoBase> make_wave_demo();
std::unique_ptr<DemoBase> make_string_art();
std::unique_ptr<DemoBase> make_info_demo();
std::unique_ptr<DemoBase> make_showcase2_demo();
std::unique_ptr<DemoBase> make_deformation_jacobian_demo();

} // namespace manifold::Demo
