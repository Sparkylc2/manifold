#include "../supersonic_demo.h"

#include "../demo_factories.h"

namespace manifold::Demo {

std::unique_ptr<DemoBase> make_supersonic_demo() {
    return std::make_unique<SupersonicDemo>();
}

} // namespace manifold::Demo
