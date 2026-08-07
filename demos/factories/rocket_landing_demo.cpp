#include "../rocket_landing_demo.h"

#include "../demo_factories.h"

namespace manifold::Demo {

std::unique_ptr<DemoBase> make_rocket_landing_demo() {
    return std::make_unique<RocketLandingDemo>();
}

} // namespace manifold::Demo
