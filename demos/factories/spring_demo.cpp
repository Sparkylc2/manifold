#include "../spring_demo.h"

#include "../demo_factories.h"

namespace manifold::Demo {

std::unique_ptr<DemoBase> make_spring_demo() {
    return std::make_unique<SpringDemo>();
}

} // namespace manifold::Demo
