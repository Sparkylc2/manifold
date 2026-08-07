#include "../double_pendulum_demo.h"

#include "../demo_factories.h"

namespace manifold::Demo {

std::unique_ptr<DemoBase> make_double_pendulum_demo() {
    return std::make_unique<DoublePendulumDemo>();
}

} // namespace manifold::Demo
