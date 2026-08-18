#include "../huygens_demo.h"

#include "../demo_factories.h"

namespace manifold::Demo {

std::unique_ptr<DemoBase> make_huygens_demo() {
    return std::make_unique<HuygensDemo>();
}

} // namespace manifold::Demo
