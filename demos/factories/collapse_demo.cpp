#include "../collapse_demo.h"

#include "../demo_factories.h"

namespace manifold::Demo {

std::unique_ptr<DemoBase> make_collapse_demo() {
    return std::make_unique<CollapseDemo>();
}

} // namespace manifold::Demo
