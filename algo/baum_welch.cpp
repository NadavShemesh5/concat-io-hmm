#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cfenv>
#include <limits>

namespace py = pybind11;
using ssize_t = Py_ssize_t;

std::tuple<double, py::array_t<double>, py::array_t<double>> forward(
  py::array_t<double> startprob_,
  py::array_t<double> transmat_,
  py::array_t<double> frameprob_)
{
  auto min_sum = 1e-300;

  auto startprob = startprob_.unchecked<1>();
  auto transmat = transmat_.unchecked<2>();
  auto frameprob = frameprob_.unchecked<2>();
  auto ns = frameprob.shape(0), nc = frameprob.shape(1);
  if (startprob.shape(0) != nc
      || transmat.shape(0) != nc || transmat.shape(1) != nc) {
    throw std::invalid_argument{"shape mismatch"};
  }
  auto fwdlattice_ = py::array_t<double>{{ns, nc}};
  auto fwd = fwdlattice_.mutable_unchecked<2>();
  auto scaling_ = py::array_t<double>{{ns}};
  auto scaling = scaling_.mutable_unchecked<1>();
  auto log_prob = 0.;
  {
    py::gil_scoped_release nogil;
    std::fill_n(fwd.mutable_data(0, 0), fwd.size(), 0);
    for (auto i = 0; i < nc; ++i) {
      fwd(0, i) = startprob(i) * frameprob(0, i);
    }
    auto sum = std::accumulate(&fwd(0, 0), &fwd(0, nc), 0.);
    if (sum < min_sum) {
      throw std::range_error{"forward pass failed with underflow; "
                             "consider using implementation='log' instead"};
    }
    auto scale = scaling(0) = 1. / sum;
    log_prob -= std::log(scale);
    for (auto i = 0; i < nc; ++i) {
      fwd(0, i) *= scale;
    }
    for (auto t = 1; t < ns; ++t) {
      for (auto j = 0; j < nc; ++j) {
        for (auto i = 0; i < nc; ++i) {
          fwd(t, j) += fwd(t - 1, i) * transmat(i, j);
        }
        fwd(t, j) *= frameprob(t, j);
      }
      auto sum = std::accumulate(&fwd(t, 0), &fwd(t, nc), 0.);
      if (sum < min_sum) {
        throw std::range_error{"forward pass failed with underflow; "
                               "consider using implementation='log' instead"};
      }
      auto scale = scaling(t) = 1. / sum;
      log_prob -= std::log(scale);
      for (auto j = 0; j < nc; ++j) {
        fwd(t, j) *= scale;
      }
    }
  }
  return {log_prob, fwdlattice_, scaling_};
}

py::array_t<double> backward(
  py::array_t<double> startprob_,
  py::array_t<double> transmat_,
  py::array_t<double> frameprob_,
  py::array_t<double> scaling_)
{
  auto startprob = startprob_.unchecked<1>();
  auto transmat = transmat_.unchecked<2>();
  auto frameprob = frameprob_.unchecked<2>();
  auto scaling = scaling_.unchecked<1>();
  auto ns = frameprob.shape(0), nc = frameprob.shape(1);
  if (startprob.shape(0) != nc
      || transmat.shape(0) != nc || transmat.shape(1) != nc
      || scaling.shape(0) != ns) {
    throw std::invalid_argument{"shape mismatch"};
  }
  auto bwdlattice_ = py::array_t<double>{{ns, nc}};
  auto bwd = bwdlattice_.mutable_unchecked<2>();
  py::gil_scoped_release nogil;
  std::fill_n(bwd.mutable_data(0, 0), bwd.size(), 0);
  for (auto i = 0; i < nc; ++i) {
    bwd(ns - 1, i) = scaling(ns - 1);
  }
  for (auto t = ns - 2; t >= 0; --t) {
    for (auto i = 0; i < nc; ++i) {
      for (auto j = 0; j < nc; ++j) {
        bwd(t, i) += transmat(i, j) * frameprob(t + 1, j) * bwd(t + 1, j);
      }
      bwd(t, i) *= scaling(t);
    }
  }
  return bwdlattice_;
}

py::array_t<double> compute_xi_sum(
  py::array_t<double> fwdlattice_,
  py::array_t<double> transmat_,
  py::array_t<double> bwdlattice_,
  py::array_t<double> frameprob_)
{
  auto fwd = fwdlattice_.unchecked<2>();
  auto transmat = transmat_.unchecked<2>();
  auto bwd = bwdlattice_.unchecked<2>();
  auto frameprob = frameprob_.unchecked<2>();
  auto ns = frameprob.shape(0), nc = frameprob.shape(1);
  if (fwd.shape(0) != ns || fwd.shape(1) != nc
      || transmat.shape(0) != nc || transmat.shape(1) != nc
      || bwd.shape(0) != ns || bwd.shape(1) != nc) {
    throw std::invalid_argument{"shape mismatch"};
  }
  auto xi_sum_ = py::array_t<double>{{nc, nc}};
  auto xi_sum = xi_sum_.mutable_unchecked<2>();
  std::fill_n(xi_sum.mutable_data(0, 0), xi_sum.size(), 0);
  py::gil_scoped_release nogil;
  for (auto t = 0; t < ns - 1; ++t) {
    for (auto i = 0; i < nc; ++i) {
      for (auto j = 0; j < nc; ++j) {
        xi_sum(i, j) += fwd(t, i)
                        * transmat(i, j)
                        * frameprob(t + 1, j)
                        * bwd(t + 1, j);
      }
    }
  }
  return xi_sum_;
}

PYBIND11_MODULE(baum_welch, m) {
  m
    .def("forward", forward)
    .def("backward", backward)
    .def("compute_xi_sum", compute_xi_sum)
    ;
}