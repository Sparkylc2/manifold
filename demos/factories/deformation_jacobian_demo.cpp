#include "../deformation_jacobian_demo.h"

#include "../demo_factories.h"

namespace manifold::Demo {

std::unique_ptr<DemoBase> make_deformation_jacobian_demo() {
    return std::make_unique<DeformationJacobianDemo>();
}

} // namespace manifold::Demo
