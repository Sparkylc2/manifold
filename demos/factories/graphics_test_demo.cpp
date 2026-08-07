#include "../graphics_test_demo.h"

#include "../demo_factories.h"

namespace manifold::Demo {

std::unique_ptr<DemoBase> make_graphics_test_demo() {
    return std::make_unique<GraphicsTestDemo>();
}

} // namespace manifold::Demo
