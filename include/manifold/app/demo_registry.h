#pragma once

#include <algorithm>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include <manifold/renderer/demo_base.h>

namespace manifold::App {

struct DemoEntry {
    std::string id;
    std::string name;
    std::string category;
    std::string description;
    std::function<std::unique_ptr<Demo::DemoBase>()> factory;
};

class DemoRegistry {
  public:
    template <typename T>
    void add(const std::string &id, const std::string &name,
             const std::string &category, const std::string &description) {
        m_entries.push_back({id, name, category, description,
                             []() { return std::make_unique<T>(); }});
    }

    const std::vector<DemoEntry> &entries() const { return m_entries; }

    std::vector<std::string> categories() const {
        std::vector<std::string> cats;
        for (auto &e : m_entries) {
            if (std::find(cats.begin(), cats.end(), e.category) == cats.end())
                cats.push_back(e.category);
        }
        // "Sandbox" always sorts last
        std::stable_sort(cats.begin(), cats.end(),
                         [](const std::string &a, const std::string &b) {
                             if (a == "Sandbox")
                                 return false;
                             if (b == "Sandbox")
                                 return true;
                             return a < b;
                         });
        return cats;
    }

    std::vector<const DemoEntry *> by_category(const std::string &cat) const {
        std::vector<const DemoEntry *> result;
        for (auto &e : m_entries) {
            if (cat.empty() || e.category == cat)
                result.push_back(&e);
        }
        return result;
    }

    const DemoEntry *find(const std::string &id) const {
        for (auto &e : m_entries) {
            if (e.id == id)
                return &e;
        }
        return nullptr;
    }

    int count() const { return (int)m_entries.size(); }

  private:
    std::vector<DemoEntry> m_entries;
};

} // namespace manifold::App
