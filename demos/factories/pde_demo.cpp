#include "../pde_demo.h"

#include "../demo_factories.h"

namespace manifold::Demo {

std::unique_ptr<DemoBase> make_pde_demo() {
    return std::make_unique<PDEDemo>();
}

} // namespace manifold::Demo
