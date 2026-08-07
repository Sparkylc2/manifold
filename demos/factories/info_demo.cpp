#include "../info_demo.h"

#include "../demo_factories.h"

namespace manifold::Demo {

std::unique_ptr<DemoBase> make_info_demo() {
    return std::make_unique<InfoDemo>();
}

} // namespace manifold::Demo
