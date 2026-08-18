#include "../ae_compress_demo.h"

#include "../demo_factories.h"

namespace manifold::Demo {

std::unique_ptr<DemoBase> make_ae_compress_demo() {
    return std::make_unique<AECompressDemo>();
}

} // namespace manifold::Demo
