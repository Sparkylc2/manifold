#include "../cae_karman_demo.h"

#include "../demo_factories.h"

namespace manifold::Demo {

std::unique_ptr<DemoBase> make_cae_karman_demo() {
    return std::make_unique<CAEKarmanDemo>();
}

} // namespace manifold::Demo
