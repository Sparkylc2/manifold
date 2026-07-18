#include <format>
#include <stdexcept>

#include <manifold/electrical/circuit_assembler.h>

namespace manifold::Electrical {

void CircuitAssembler::add_G(int i, int j, double v) {
    // idx of -1 is the ground node
    if (i == -1 || j == -1)
        return;
    if (i < 0 || j < 0 || i >= m_s.G.rows() || j >= m_s.G.cols())
        throw std::runtime_error(std::format(
            "invalid i or j with: i={}, j={} ({} <= i < {}, {} <= j < {})", i,
            j, -1, m_s.G.rows(), -1, m_s.G.cols()));

    m_s.G(i, j) += v;
}

void CircuitAssembler::add_C(int i, int j, double v) {
    // idx of -1 is the ground node
    if (i == -1 || j == -1)
        return;

    if (i < 0 || j < 0 || i >= m_s.C.rows() || j >= m_s.C.cols())
        throw std::runtime_error(std::format(
            "invalid i or j with: i={}, j={} ({} <= i < {}, {} <= j < {})", i,
            j, -1, m_s.C.rows(), -1, m_s.C.cols()));
    m_s.C(i, j) += v;
}

void CircuitAssembler::add_b(int i, double v) {
    // idx of -1 is the ground node
    if (i == -1)
        return;
    if (i < 0 || i >= m_s.b.size())
        throw std::runtime_error(std::format(
            "invalid i with: i={},  {} <= i < {}", i, 0, m_s.b.size()));

    m_s.b(i) += v;
}

} // namespace manifold::Electrical
