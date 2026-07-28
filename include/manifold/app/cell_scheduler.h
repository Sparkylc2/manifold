#pragma once

#include <manifold/renderer/showcase_cell.h>

#include <algorithm>
#include <vector>

namespace manifold::App {

// steps only the cells that are on screen or about to be
class CellScheduler {
  public:
    enum class State { Dormant, Warming, Live, Retired };

    // [t_on, t_off) is the window the cell is visible
    // it is stepped from t_on - cell->warmup() and stopped after t_off
    // a cell that should keep running to the end of the reel just gets a large
    // t_off
    void add(Demo::ShowcaseCell *cell, double t_on, double t_off) {
        m_entries.push_back({cell, t_on, t_off, State::Dormant, false});
    }

    void clear() { m_entries.clear(); }

    // deliberately does not initialize
    void reset() {
        m_time = 0.0;
        for (auto &e : m_entries) {
            e.state = State::Dormant;
            e.inited = false;
        }
    }

    void process(double dt) {
        m_time += dt;
        for (auto &e : m_entries) {
            const double wake = e.t_on - e.cell->warmup();

            if (m_time >= e.t_off) {
                e.state = State::Retired;
                continue;
            }
            if (m_time < wake) {
                e.state = State::Dormant;
                continue;
            }

            if (!e.inited) {
                e.cell->initialize();
                e.inited = true;
            }

            e.state = m_time < e.t_on ? State::Warming : State::Live;
            e.cell->process(dt);
        }
    }

    // a slot must not be drawn before its cell has been built
    bool ready(int i) const { return m_entries[i].inited; }

    // fast-forwards from scratch to t_target
    void seek(double t_target, double step = 1.0 / 240.0,
              int max_steps = 200000) {
        reset();
        const int n = std::min(max_steps, (int)std::max(0.0, t_target / step));
        for (int i = 0; i < n; ++i)
            process(step);
    }

    double time() const { return m_time; }

    State state(int i) const { return m_entries[i].state; }

    double warm_fraction(int i) const {
        const Entry &e = m_entries[i];
        const double w = e.cell->warmup();
        if (w <= 0.0)
            return 1.0;
        return std::clamp((m_time - (e.t_on - w)) / w, 0.0, 1.0);
    }

    int count() const { return (int)m_entries.size(); }

    int stepping_count() const {
        int n = 0;
        for (auto &e : m_entries)
            n += (e.state == State::Warming || e.state == State::Live);
        return n;
    }

  private:
    struct Entry {
        Demo::ShowcaseCell *cell;
        double t_on, t_off;
        State state;
        bool inited;
    };

    std::vector<Entry> m_entries;
    double m_time = 0.0;
};

} // namespace manifold::App
