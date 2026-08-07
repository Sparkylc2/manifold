#include "../solar_system_demo.h"

#include "../demo_factories.h"

namespace manifold::Demo {

std::unique_ptr<DemoBase> make_solar_system_demo() {
    return std::make_unique<SolarSystemDemo>();
}

} // namespace manifold::Demo
