#include "../flutter_demo.h"

#include "../demo_factories.h"

namespace manifold::Demo {

std::unique_ptr<DemoBase> make_flutter_demo() {
    return std::make_unique<FlutterDemo>();
}

} // namespace manifold::Demo
