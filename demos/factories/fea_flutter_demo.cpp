#include "../fea_flutter_demo.h"

#include "../demo_factories.h"

namespace manifold::Demo {

std::unique_ptr<DemoBase> make_fea_flutter_demo() {
    return std::make_unique<FeaFlutterDemo>();
}

} // namespace manifold::Demo
