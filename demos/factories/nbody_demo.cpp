#include "../nbody_demo.h"

#include "../demo_factories.h"

namespace manifold::Demo {

std::unique_ptr<DemoBase> make_nbody_demo() {
    return std::make_unique<NBodyDemo>();
}

} // namespace manifold::Demo
