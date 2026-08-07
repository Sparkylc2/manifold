#include "../engine_demo.h"

#include "../demo_factories.h"

namespace manifold::Demo {

std::unique_ptr<DemoBase> make_engine_demo() {
    return std::make_unique<EngineDemo>();
}

} // namespace manifold::Demo
