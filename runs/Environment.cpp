#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "Transition.h"

namespace py = pybind11;

PYBIND11_MODULE(transition_pybind, m) {
    py::class_<Transition>(m, "Transition")
        .def(py::init<const std::vector<std::vector<int>>&,
                      const std::vector<std::vector<std::vector<int>>>&,
                      const std::vector<std::vector<int>>&,
                      int, bool>(),
             py::arg("board"), py::arg("cutter"), py::arg("goal"),
             py::arg("frequ") = 1, py::arg("test") = false)

        .def("step", &Transition::step)
        .def("action_sample", &Transition::action_sample)
        .def("reset", &Transition::reset)
        .def("seed", &Transition::seed);
}
