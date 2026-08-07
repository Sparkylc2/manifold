#include "../cube_demo.h"

#include "../demo_factories.h"

namespace manifold::Demo {

std::unique_ptr<DemoBase> make_cube_demo() {
    return std::make_unique<CubeDemo>();
}

} // namespace manifold::Demo
