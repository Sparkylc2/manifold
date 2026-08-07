#include "../diffuser_demo.h"

#include "../demo_factories.h"

namespace manifold::Demo {

std::unique_ptr<DemoBase> make_diffuser_demo() {
    return std::make_unique<DiffuserDemo>();
}

} // namespace manifold::Demo
