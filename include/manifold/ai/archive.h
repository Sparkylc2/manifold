#pragma once
#include <Eigen/Dense>
#include <Eigen/SparseCore>
#include <fstream>
#include <string>
#include <string_view>
#include <unsupported/Eigen/CXX11/Tensor>

namespace manifold::AI {
using namespace Eigen;

// visitor for weight (de)serialization: one serialize() per type feeds both
// directions, mode picks read vs write. order-based binary, so save and load
// must visit fields in the same order.
struct Archive {
    enum Mode { Save, Load } m_mode;
    bool saving() const { return m_mode == Mode::Save; }

    explicit Archive(Mode m) : m_mode(m) {}
    virtual ~Archive() = default;
    virtual void operator()(std::string_view name, int &e_num) = 0;
    virtual void operator()(std::string_view name, double &v) = 0;
    virtual void operator()(std::string_view name, VectorXd &v) = 0;
    virtual void operator()(std::string_view name, MatrixXd &M) = 0;
    virtual void operator()(std::string_view name, SparseMatrix<double> &S) = 0;
    virtual void operator()(std::string_view name, Tensor<double, 3> &T3) = 0;
    virtual void operator()(std::string_view name, Tensor<double, 4> &T4) = 0;
};

struct SaveArchive : Archive {
    std::ofstream m_f;

    explicit SaveArchive(const std::string &path)
        : Archive(Mode::Save), m_f(path, std::ios::binary) {}

    void operator()(std::string_view name, int &e_num) override;
    void operator()(std::string_view name, double &v) override;
    void operator()(std::string_view name, VectorXd &v) override;
    void operator()(std::string_view name, MatrixXd &M) override;
    void operator()(std::string_view name, SparseMatrix<double> &S) override;
    void operator()(std::string_view name, Tensor<double, 3> &T3) override;
    void operator()(std::string_view name, Tensor<double, 4> &T4) override;
};

struct LoadArchive : Archive {
    std::ifstream m_f;

    explicit LoadArchive(const std::string &path)
        : Archive(Mode::Load), m_f(path, std::ios::binary) {}

    void operator()(std::string_view name, int &e_num) override;
    void operator()(std::string_view name, double &v) override;
    void operator()(std::string_view name, VectorXd &v) override;
    void operator()(std::string_view name, MatrixXd &M) override;
    void operator()(std::string_view name, SparseMatrix<double> &S) override;
    void operator()(std::string_view name, Tensor<double, 3> &T3) override;
    void operator()(std::string_view name, Tensor<double, 4> &T4) override;
};
} // namespace manifold::AI
