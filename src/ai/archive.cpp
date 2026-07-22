#include <cassert>
#include <cstdint>
#include <manifold/ai/archive.h>
#include <vector>

namespace manifold::AI {

namespace {
void put_name(std::ofstream &f, std::string_view name) {
    uint32_t n = (uint32_t)name.size();
    f.write((const char *)&n, sizeof n);
    f.write(name.data(), n);
}
void check_name(std::ifstream &f, std::string_view name) {
    uint32_t n = 0;
    f.read((char *)&n, sizeof n);
    std::string got(n, '\0');
    f.read(got.data(), n);
    assert(got == name && "checkpoint field mismatch");
    (void)name;
}

template <typename POD> void put(std::ofstream &f, POD x) {
    f.write((const char *)&x, sizeof x);
}
template <typename POD> POD get(std::ifstream &f) {
    POD x{};
    f.read((char *)&x, sizeof x);
    return x;
}

template <int N>
void save_tensor(std::ofstream &f, std::string_view name, Tensor<double, N> &T) {
    put_name(f, name);
    for (int i = 0; i < N; ++i)
        put<int64_t>(f, T.dimension(i));
    f.write((const char *)T.data(), (int64_t)sizeof(double) * T.size());
}
template <int N>
void load_tensor(std::ifstream &f, std::string_view name, Tensor<double, N> &T) {
    check_name(f, name);
    Eigen::array<Index, N> d;
    for (int i = 0; i < N; ++i)
        d[i] = (Index)get<int64_t>(f);
    T.resize(d);
    f.read((char *)T.data(), (int64_t)sizeof(double) * T.size());
}
} // namespace

void SaveArchive::operator()(std::string_view name, int &e_num) {
    put_name(m_f, name);
    put<int64_t>(m_f, e_num);
}
void SaveArchive::operator()(std::string_view name, double &v) {
    put_name(m_f, name);
    m_f.write((const char *)&v, sizeof v);
}
void SaveArchive::operator()(std::string_view name, VectorXd &v) {
    MatrixXd M = v;
    (*this)(name, M);
}
void SaveArchive::operator()(std::string_view name, MatrixXd &M) {
    put_name(m_f, name);
    int64_t r = M.rows(), c = M.cols();
    m_f.write((const char *)&r, sizeof r);
    m_f.write((const char *)&c, sizeof c);
    m_f.write((const char *)M.data(), sizeof(double) * r * c);
}
void SaveArchive::operator()(std::string_view name, SparseMatrix<double> &S) {
    put_name(m_f, name);
    S.makeCompressed();
    put<int64_t>(m_f, S.rows());
    put<int64_t>(m_f, S.cols());
    put<int64_t>(m_f, S.nonZeros());
    for (int k = 0; k < S.outerSize(); ++k)
        for (SparseMatrix<double>::InnerIterator it(S, k); it; ++it) {
            put<int64_t>(m_f, it.row());
            put<int64_t>(m_f, it.col());
            put<double>(m_f, it.value());
        }
}
void SaveArchive::operator()(std::string_view name, Tensor<double, 3> &T3) {
    save_tensor(m_f, name, T3);
}
void SaveArchive::operator()(std::string_view name, Tensor<double, 4> &T4) {
    save_tensor(m_f, name, T4);
}

void LoadArchive::operator()(std::string_view name, int &e_num) {
    check_name(m_f, name);
    e_num = (int)get<int64_t>(m_f);
}
void LoadArchive::operator()(std::string_view name, double &v) {
    check_name(m_f, name);
    m_f.read((char *)&v, sizeof v);
}
void LoadArchive::operator()(std::string_view name, VectorXd &v) {
    MatrixXd M;
    (*this)(name, M);
    v = M.col(0);
}
void LoadArchive::operator()(std::string_view name, MatrixXd &M) {
    check_name(m_f, name);
    int64_t r = 0, c = 0;
    m_f.read((char *)&r, sizeof r);
    m_f.read((char *)&c, sizeof c);
    M.resize(r, c);
    m_f.read((char *)M.data(), sizeof(double) * r * c);
}
void LoadArchive::operator()(std::string_view name, SparseMatrix<double> &S) {
    check_name(m_f, name);
    const int64_t r = get<int64_t>(m_f);
    const int64_t c = get<int64_t>(m_f);
    const int64_t nnz = get<int64_t>(m_f);
    std::vector<Triplet<double>> trips;
    trips.reserve(nnz);
    for (int64_t i = 0; i < nnz; ++i) {
        const int64_t row = get<int64_t>(m_f);
        const int64_t col = get<int64_t>(m_f);
        trips.emplace_back((int)row, (int)col, get<double>(m_f));
    }
    S.resize(r, c);
    S.setFromTriplets(trips.begin(), trips.end());
}
void LoadArchive::operator()(std::string_view name, Tensor<double, 3> &T3) {
    load_tensor(m_f, name, T3);
}
void LoadArchive::operator()(std::string_view name, Tensor<double, 4> &T4) {
    load_tensor(m_f, name, T4);
}

} // namespace manifold::AI
