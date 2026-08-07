#include "../string_art.h"

#include "../demo_factories.h"

namespace manifold::Demo {

std::unique_ptr<DemoBase> make_string_art() {
    return std::make_unique<StringArt>();
}

} // namespace manifold::Demo
