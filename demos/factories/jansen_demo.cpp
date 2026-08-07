#include "../jansen_demo.h"

#include "../demo_factories.h"

namespace manifold::Demo {

std::unique_ptr<DemoBase> make_jansen_demo() {
    return std::make_unique<JansenDemo>();
}

} // namespace manifold::Demo
