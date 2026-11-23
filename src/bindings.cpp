// pybind11 bindings exposing the C++ core as tickstream._core.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <stdexcept>
#include <string>

#include "tickstream/tickstream.hpp"

namespace py = pybind11;
using namespace tickstream;

namespace {

// Python ints hash as-is (masked to 64 bits); strings (tickers, order IDs) via FNV-1a.
uint64_t key_of(const py::handle& obj) {
    if (py::isinstance<py::str>(obj)) return hash_bytes(obj.cast<std::string>());
    if (py::isinstance<py::bytes>(obj)) return hash_bytes(obj.cast<std::string>());
    if (py::isinstance<py::int_>(obj)) {
        py::int_ mask(0xFFFFFFFFFFFFFFFFULL);
        return py::reinterpret_steal<py::int_>(PyNumber_And(obj.ptr(), mask.ptr())).cast<uint64_t>();
    }
    throw py::type_error("keys must be int, str or bytes");
}

}  // namespace

PYBIND11_MODULE(_core, m) {
    m.doc() = "Streaming sketches and pair statistics for market data (C++ core).";

    py::class_<CountMinSketch>(m, "CountMinSketch")
        .def(py::init<uint32_t, uint32_t, uint64_t>(), py::arg("width"), py::arg("depth"), py::arg("seed") = 0x5eed)
        .def_static("from_error", &CountMinSketch::from_error, py::arg("epsilon"), py::arg("delta"),
                    py::arg("seed") = 0x5eed)
        .def("add", [](CountMinSketch& s, py::handle k, uint64_t c) { s.add(key_of(k), c); },
             py::arg("key"), py::arg("count") = 1)
        .def("add_conservative", [](CountMinSketch& s, py::handle k, uint64_t c) { s.add_conservative(key_of(k), c); },
             py::arg("key"), py::arg("count") = 1)
        .def("add_many",
             [](CountMinSketch& s, py::array_t<uint64_t, py::array::c_style | py::array::forcecast> keys,
                py::object counts) {
                 auto k = keys.unchecked<1>();
                 if (counts.is_none()) {
                     py::gil_scoped_release nogil;
                     for (py::ssize_t i = 0; i < k.shape(0); ++i) s.add(k(i));
                     return;
                 }
                 auto arr = counts.cast<py::array_t<uint64_t, py::array::c_style | py::array::forcecast>>();
                 auto c = arr.unchecked<1>();
                 if (c.shape(0) != k.shape(0)) throw std::invalid_argument("keys and counts differ in length");
                 py::gil_scoped_release nogil;
                 for (py::ssize_t i = 0; i < k.shape(0); ++i) s.add(k(i), c(i));
             },
             py::arg("keys"), py::arg("counts") = py::none(),
             "Add a uint64 numpy array of keys (optionally weighted) without per-item Python overhead.")
        .def("estimate", [](const CountMinSketch& s, py::handle k) { return s.estimate(key_of(k)); })
        .def("__getitem__", [](const CountMinSketch& s, py::handle k) { return s.estimate(key_of(k)); })
        .def("merge", &CountMinSketch::merge)
        .def("clear", &CountMinSketch::clear)
        .def_property_readonly("width", &CountMinSketch::width)
        .def_property_readonly("depth", &CountMinSketch::depth)
        .def_property_readonly("total", &CountMinSketch::total)
        .def_property_readonly("memory_bytes", &CountMinSketch::memory_bytes);

    py::class_<HyperLogLog>(m, "HyperLogLog")
        .def(py::init<uint8_t, uint64_t>(), py::arg("precision") = 14, py::arg("seed") = 0x11)
        .def("add", [](HyperLogLog& h, py::handle k) { h.add(key_of(k)); })
        .def("add_many",
             [](HyperLogLog& h, py::array_t<uint64_t, py::array::c_style | py::array::forcecast> keys) {
                 auto k = keys.unchecked<1>();
                 py::gil_scoped_release nogil;
                 for (py::ssize_t i = 0; i < k.shape(0); ++i) h.add(k(i));
             })
        .def("estimate", &HyperLogLog::estimate)
        .def("__len__", [](const HyperLogLog& h) { return static_cast<size_t>(h.estimate() + 0.5); })
        .def("merge", &HyperLogLog::merge)
        .def("clear", &HyperLogLog::clear)
        .def_property_readonly("standard_error", &HyperLogLog::standard_error)
        .def_property_readonly("memory_bytes", &HyperLogLog::memory_bytes);

    py::class_<Reservoir<py::object>>(m, "Reservoir")
        .def(py::init<size_t, uint64_t>(), py::arg("k"), py::arg("seed") = 42)
        .def("add", &Reservoir<py::object>::add)
        .def("add_many", [](Reservoir<py::object>& r, py::iterable it) { for (auto x : it) r.add(py::reinterpret_borrow<py::object>(x)); })
        .def_property_readonly("sample", [](const Reservoir<py::object>& r) { return py::list(py::cast(r.sample())); })
        .def_property_readonly("seen", &Reservoir<py::object>::seen)
        .def_property_readonly("capacity", &Reservoir<py::object>::capacity);

    py::class_<RunningStats>(m, "RunningStats")
        .def(py::init<>())
        .def("add", &RunningStats::add)
        .def("add_many", [](RunningStats& s, py::array_t<double, py::array::c_style | py::array::forcecast> xs) {
            auto x = xs.unchecked<1>();
            for (py::ssize_t i = 0; i < x.shape(0); ++i) s.add(x(i));
        })
        .def_property_readonly("count", &RunningStats::count)
        .def_property_readonly("mean", &RunningStats::mean)
        .def_property_readonly("variance", &RunningStats::variance)
        .def_property_readonly("stddev", &RunningStats::stddev)
        .def_property_readonly("min", &RunningStats::min)
        .def_property_readonly("max", &RunningStats::max);

    py::class_<EwmStats>(m, "EwmStats")
        .def(py::init<double>(), py::arg("halflife"))
        .def("add", &EwmStats::add)
        .def("zscore", &EwmStats::zscore)
        .def_property_readonly("count", &EwmStats::count)
        .def_property_readonly("mean", &EwmStats::mean)
        .def_property_readonly("variance", &EwmStats::variance)
        .def_property_readonly("stddev", &EwmStats::stddev);

    py::class_<PairStats>(m, "PairStats")
        .def(py::init<double, double>(), py::arg("halflife"), py::arg("spread_halflife") = 0.0)
        .def("update",
             [](PairStats& p, double x, double y) {
                 auto s = p.update(x, y);
                 return py::make_tuple(s.beta, s.spread, s.z);
             },
             py::arg("x"), py::arg("y"), "Returns (beta, spread, z) scored against prior ticks only.")
        .def("update_many",
             [](PairStats& p, py::array_t<double, py::array::c_style | py::array::forcecast> xs,
                py::array_t<double, py::array::c_style | py::array::forcecast> ys) {
                 auto x = xs.unchecked<1>();
                 auto y = ys.unchecked<1>();
                 if (x.shape(0) != y.shape(0)) throw std::invalid_argument("x and y differ in length");
                 const py::ssize_t n = x.shape(0);
                 py::array_t<double> beta(n), spread(n), z(n);
                 auto b = beta.mutable_unchecked<1>();
                 auto sp = spread.mutable_unchecked<1>();
                 auto zz = z.mutable_unchecked<1>();
                 {
                     py::gil_scoped_release nogil;
                     for (py::ssize_t i = 0; i < n; ++i) {
                         auto s = p.update(x(i), y(i));
                         b(i) = s.beta; sp(i) = s.spread; zz(i) = s.z;
                     }
                 }
                 return py::make_tuple(beta, spread, z);
             },
             py::arg("x"), py::arg("y"),
             "Vectorized update; returns numpy arrays (beta, spread, z) with no look-ahead.")
        .def_property_readonly("count", &PairStats::count)
        .def_property_readonly("beta", &PairStats::beta)
        .def_property_readonly("alpha", &PairStats::alpha)
        .def_property_readonly("correlation", &PairStats::correlation);

    m.def("hash_key", [](py::handle k) { return key_of(k); }, "The 64-bit key a value maps to.");
}
