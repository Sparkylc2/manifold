#include "../ae_karman_demo.h"

#include "../demo_factories.h"

namespace manifold::Demo {

std::unique_ptr<DemoBase> make_ae_karman_demo() {
    return std::make_unique<AEKarmanDemo>();
}

} // namespace manifold::Demo
