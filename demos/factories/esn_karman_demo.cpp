#include "../esn_karman_demo.h"

#include "../demo_factories.h"

namespace manifold::Demo {

std::unique_ptr<DemoBase> make_esn_karman_demo() {
    return std::make_unique<ESNKarmanDemo>();
}

} // namespace manifold::Demo
