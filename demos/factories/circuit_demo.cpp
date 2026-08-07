#include "../circuit_demo.h"

#include "../demo_factories.h"

namespace manifold::Demo {

std::unique_ptr<DemoBase> make_circuit_demo() {
    return std::make_unique<CircuitDemo>();
}

} // namespace manifold::Demo
