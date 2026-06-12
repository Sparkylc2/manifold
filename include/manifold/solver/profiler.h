#pragma once

#include <array>
#include <chrono>

namespace manifold::Solver {

template <typename T, int N> class RingBuffer {
  public:
    RingBuffer() { m_data.fill(T{}); }

    void push(T value) {
        m_data[m_head] = value;
        m_head = (m_head + 1) % N;
        if (m_count < N)
            ++m_count;
    }

    T average() const {
        if (m_count == 0)
            return T{};
        T sum{};
        for (int i = 0; i < m_count; ++i)
            sum += m_data[i];
        return sum / m_count;
    }

    T last() const {
        if (m_count == 0)
            return T{};
        return m_data[(m_head + N - 1) % N];
    }

    void clear() {
        m_head = 0;
        m_count = 0;
        m_data.fill(T{});
    }

    int count() const { return m_count; }

  private:
    std::array<T, N> m_data;
    int m_head = 0;
    int m_count = 0;
};

#ifdef MANIFOLD_PROFILE

class ScopedTimer {
  public:
    ScopedTimer(long long &target)
        : m_target(target), m_start(std::chrono::steady_clock::now()) {}

    ~ScopedTimer() {
        m_target = std::chrono::duration_cast<std::chrono::microseconds>(
                       std::chrono::steady_clock::now() - m_start)
                       .count();
    }

    ScopedTimer(const ScopedTimer &) = delete;
    ScopedTimer &operator=(const ScopedTimer &) = delete;

  private:
    long long &m_target;
    std::chrono::steady_clock::time_point m_start;
};

#define MANIFOLD_CONCAT_IMPL(a, b) a##b
#define MANIFOLD_CONCAT(a, b) MANIFOLD_CONCAT_IMPL(a, b)
#define PROFILE_SCOPE(var)                                                     \
    ::manifold::Solver::ScopedTimer MANIFOLD_CONCAT(_profiler_, __LINE__)(var)

#else

#define PROFILE_SCOPE(var) ((void)0)

#endif

} // namespace manifold::Solver
