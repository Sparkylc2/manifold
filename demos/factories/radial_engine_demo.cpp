#include "../radial_engine_demo.h"

#include "../demo_factories.h"

namespace manifold::Demo {

std::unique_ptr<DemoBase> make_radial_engine_demo() {
    return std::make_unique<RadialEngineDemo>();
}

} // namespace manifold::Demo
