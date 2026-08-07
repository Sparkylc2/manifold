#include "../showcase2_demo.h"

#include "../demo_factories.h"

namespace manifold::Demo {

std::unique_ptr<DemoBase> make_showcase2_demo() {
    return std::make_unique<Showcase2Demo>();
}

} // namespace manifold::Demo
