#include "../aerofoil_elevator_demo.h"

#include "../demo_factories.h"

namespace manifold::Demo {

std::unique_ptr<DemoBase> make_aerofoil_elevator_demo() {
    return std::make_unique<AerofoilElevatorDemo>();
}

} // namespace manifold::Demo
