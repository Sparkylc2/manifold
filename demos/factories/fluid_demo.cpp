#include "../fluid_demo.h"

#include "../demo_factories.h"

namespace manifold::Demo {

std::unique_ptr<DemoBase> make_fluid_demo() {
    return std::make_unique<FluidDemo>();
}

} // namespace manifold::Demo
