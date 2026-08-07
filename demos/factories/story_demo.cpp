#include "../story_demo.h"

#include "../demo_factories.h"

namespace manifold::Demo {

std::unique_ptr<DemoBase> make_story_demo() {
    return std::make_unique<StoryDemo>();
}

} // namespace manifold::Demo
