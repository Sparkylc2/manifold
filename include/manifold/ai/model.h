
#include <Eigen/Core>

namespace manifold::AI {
using namespace Eigen;
class Model {
  public:
    virtual ~Model() = default;

    virtual double fit(const MatrixXd &X, int epochs);
    virtual VectorXd infer(const VectorXd &x) const = 0; // M -> r
};
} // namespace manifold::AI
