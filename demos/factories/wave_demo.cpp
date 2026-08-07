#include "../wave_demo.h"

#include "../demo_factories.h"

namespace manifold::Demo {

std::unique_ptr<DemoBase> make_wave_demo() {
    return std::make_unique<WaveDemo>();
}

} // namespace manifold::Demo
