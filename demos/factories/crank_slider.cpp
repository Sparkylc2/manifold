#include "../crank_slider.h"

#include "../demo_factories.h"

namespace manifold::Demo {

std::unique_ptr<DemoBase> make_crank_slider() {
    return std::make_unique<CrankSliderDemo>();
}

} // namespace manifold::Demo
