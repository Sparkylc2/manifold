#include "../cart_double_pendulum_demo.h"

#include "../demo_factories.h"

namespace manifold::Demo {

std::unique_ptr<DemoBase> make_cart_double_pendulum_demo() {
    return std::make_unique<CartDoublePendulumDemo>();
}

} // namespace manifold::Demo
