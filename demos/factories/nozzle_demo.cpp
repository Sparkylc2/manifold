#include "../nozzle_demo.h"

#include "../demo_factories.h"

namespace manifold::Demo {

std::unique_ptr<DemoBase> make_nozzle_demo() {
    return std::make_unique<NozzleDemo>();
}

} // namespace manifold::Demo
