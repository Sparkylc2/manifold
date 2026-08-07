#include "../truss_demo.h"

#include "../demo_factories.h"

namespace manifold::Demo {

std::unique_ptr<DemoBase> make_truss_demo() {
    return std::make_unique<TrussDemo>();
}

} // namespace manifold::Demo
