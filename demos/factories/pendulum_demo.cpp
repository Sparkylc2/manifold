#include "../pendulum_demo.h"

#include "../demo_factories.h"

namespace manifold::Demo {

std::unique_ptr<DemoBase> make_pendulum_demo() {
    return std::make_unique<PendulumDemo>();
}

} // namespace manifold::Demo
