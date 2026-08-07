#include "../heat_demo.h"

#include "../demo_factories.h"

namespace manifold::Demo {

std::unique_ptr<DemoBase> make_heat_demo() {
    return std::make_unique<HeatDemo>();
}

} // namespace manifold::Demo
