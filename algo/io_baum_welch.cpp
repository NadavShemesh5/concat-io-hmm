#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <cmath>
#include <numeric>
#include <algorithm>

namespace py = pybind11;
using ssize_t = Py_ssize_t;

const double MIN_SUM = 1e-300;

std::tuple<double, py::array_t<double>, py::array_t<double>> forward(
    py::array_t<double> trans_mat_,
    py::array_t<double> emit_mat_,
    py::array_t<double> start_mat_,
    py::array_t<int> inputs_,
    py::array_t<int> outputs_,
    py::array_t<int> lengths_
) {
    auto trans_mat = trans_mat_.unchecked<3>();
    auto emit_mat = emit_mat_.unchecked<3>();
    auto start_mat = start_mat_.unchecked<2>();
    auto inputs = inputs_.unchecked<2>();
    auto outputs = outputs_.unchecked<2>();
    auto lengths = lengths_.unchecked<1>();

    auto N = inputs.shape(0);
    auto T_max = inputs.shape(1);
    auto S = trans_mat.shape(1);

    auto fwdlattice_ = py::array_t<double>{{N, T_max, S}};
    auto fwd = fwdlattice_.mutable_unchecked<3>();

    auto scaling_ = py::array_t<double>{{N, T_max}};
    auto scaling = scaling_.mutable_unchecked<2>();

    double total_log_prob = 0.0;

    py::gil_scoped_release nogil;
    for (ssize_t n = 0; n < N; ++n) {
        auto T_n = lengths(n);
        double log_prob_n = 0.0;
        auto u_0 = inputs(n, 0);
        auto v_0 = outputs(n, 0);
        double sum = 0.0;
        for (ssize_t i = 0; i < S; ++i) {
            double p = start_mat(i, u_0) * emit_mat(u_0, i, v_0);
            fwd(n, 0, i) = p;
            sum += p;
        }

        if (sum < MIN_SUM) sum = 1.0;
        double scale = 1.0 / sum;
        scaling(n, 0) = scale;
        log_prob_n -= std::log(scale);
        for (ssize_t i = 0; i < S; ++i) fwd(n, 0, i) *= scale;
        for (ssize_t t = 1; t < T_n; ++t) {
            auto u_t = inputs(n, t);
            auto v_t = outputs(n, t);

            sum = 0.0;
            for (ssize_t j = 0; j < S; ++j) {
                double acc = 0.0;
                for (ssize_t i = 0; i < S; ++i) {
                    acc += fwd(n, t - 1, i) * trans_mat(u_t, i, j);
                }

                double val = acc * emit_mat(u_t, j, v_t);
                fwd(n, t, j) = val;
                sum += val;
            }
            if (sum < MIN_SUM) sum = 1.0;

            scale = 1.0 / sum;
            scaling(n, t) = scale;
            log_prob_n -= std::log(scale);
            for (ssize_t j = 0; j < S; ++j) fwd(n, t, j) *= scale;
        }
        total_log_prob += log_prob_n;
    }

    return {total_log_prob, fwdlattice_, scaling_};
}

py::array_t<double> backward(
    py::array_t<double> trans_mat_,
    py::array_t<double> emit_mat_,
    py::array_t<double> scaling_batch_,
    py::array_t<int> inputs_,
    py::array_t<int> outputs_,
    py::array_t<int> lengths_
) {
    auto trans_mat = trans_mat_.unchecked<3>();
    auto emit_mat = emit_mat_.unchecked<3>();
    auto scaling = scaling_batch_.unchecked<2>();
    auto inputs = inputs_.unchecked<2>();
    auto outputs = outputs_.unchecked<2>();
    auto lengths = lengths_.unchecked<1>();

    auto N = inputs.shape(0);
    auto T_max = inputs.shape(1);
    auto S = trans_mat.shape(1);

    auto bwdlattice_ = py::array_t<double>{{N, T_max, S}};
    auto bwd = bwdlattice_.mutable_unchecked<3>();

    py::gil_scoped_release nogil;
    std::fill_n(bwd.mutable_data(0, 0, 0), bwd.size(), 0.0);
    for (ssize_t n = 0; n < N; ++n) {
        auto T_n = lengths(n);
        if (T_n == 0) continue;
        for (ssize_t i = 0; i < S; ++i) bwd(n, T_n - 1, i) = scaling(n, T_n - 1);
        for (ssize_t t = T_n - 2; t >= 0; --t) {
            auto u_next = inputs(n, t + 1);
            auto v_next = outputs(n, t + 1);

            for (ssize_t i = 0; i < S; ++i) {
                double acc = 0.0;
                for (ssize_t j = 0; j < S; ++j) {
                    acc += trans_mat(u_next, i, j) * emit_mat(u_next, j, v_next) * bwd(n, t + 1, j);
                }
                bwd(n, t, i) = acc * scaling(n, t);
            }
        }
    }
    return bwdlattice_;
}

void compute_xi_sum(
    py::array_t<double> fwd_batch_,
    py::array_t<double> trans_mat_,
    py::array_t<double> bwd_batch_,
    py::array_t<double> emit_mat_,
    py::array_t<int> inputs_batch_,
    py::array_t<int> outputs_batch_,
    py::array_t<int> lengths_,
    py::array_t<double> out_stats_
) {
    auto fwd = fwd_batch_.unchecked<3>();
    auto trans_mat = trans_mat_.unchecked<3>();
    auto bwd = bwd_batch_.unchecked<3>();
    auto emit_mat = emit_mat_.unchecked<3>();
    auto inputs = inputs_batch_.unchecked<2>();
    auto outputs = outputs_batch_.unchecked<2>();
    auto lengths = lengths_.unchecked<1>();
    auto out = out_stats_.mutable_unchecked<3>();

    auto N = inputs.shape(0);
    auto S = trans_mat.shape(1);

    py::gil_scoped_release nogil;
    for (ssize_t n = 0; n < N; ++n) {
        auto T_n = lengths(n);
        for (ssize_t t = 0; t < T_n - 1; ++t) {
            auto u_next = inputs(n, t + 1);
            auto v_next = outputs(n, t + 1);
            for (ssize_t i = 0; i < S; ++i) {
                for (ssize_t j = 0; j < S; ++j) {
                    double val = fwd(n, t, i)
                               * trans_mat(u_next, i, j)
                               * emit_mat(u_next, j, v_next)
                               * bwd(n, t + 1, j);

                    out(u_next, i, j) += val;
                }
            }
        }
    }
}

py::array_t<int> predict_inputs_marginal(
    py::array_t<double> trans_mat_,
    py::array_t<double> emit_mat_,
    py::array_t<double> start_mat_,
    py::array_t<int> outputs_,
    py::array_t<int> lengths_
) {
    auto trans_mat = trans_mat_.unchecked<3>(); // (n_inputs, S, S)
    auto emit_mat = emit_mat_.unchecked<3>();   // (n_inputs, S, n_outputs)
    auto start_mat = start_mat_.unchecked<2>(); // (n_inputs, S)
    auto outputs = outputs_.unchecked<2>();
    auto lengths = lengths_.unchecked<1>();

    auto N = outputs.shape(0);
    auto T_max = outputs.shape(1);
    auto U = trans_mat.shape(0); // Number of Inputs
    auto S = trans_mat.shape(1);

    // FIX 1: Explicitly initialize with zeros to avoid garbage in padded regions
    auto inputs_seq_ = py::array_t<int>({N, T_max});
    std::fill(inputs_seq_.mutable_data(), inputs_seq_.mutable_data() + inputs_seq_.size(), 0);

    auto inputs_seq = inputs_seq_.mutable_unchecked<2>();

    py::gil_scoped_release nogil;

    // Pre-allocate buffers to avoid re-allocation in loops
    std::vector<double> alpha(S);
    std::vector<double> next_alpha(S);

    for (ssize_t n = 0; n < N; ++n) {
        auto T_n = lengths(n);
        if (T_n == 0) continue;

        // Reset Alpha (Belief) to Uniform or Start State distribution
        std::fill(alpha.begin(), alpha.end(), 1.0 / S);

        for (ssize_t t = 0; t < T_n; ++t) {
            auto v_curr = outputs(n, t);

            double best_u_prob = -1.0;
            int best_u = 0;

            // --- Step 1: Find Best Input 'u' for current step ---
            for (ssize_t u = 0; u < U; ++u) {
                double current_u_prob = 0.0;

                // Sum prob of seeing Output given Input 'u'
                // We marginalize over transitions from ALL previous states 'i' to ALL next states 'j'
                for (ssize_t i = 0; i < S; ++i) {
                    if (alpha[i] <= 1e-300) continue;

                    for (ssize_t j = 0; j < S; ++j) {
                        double trans_p = (t == 0) ? start_mat(u, j) : trans_mat(u, i, j);
                        double emit_p = emit_mat(u, j, v_curr);

                        // Accumulate joint probability for this specific input 'u'
                        current_u_prob += alpha[i] * trans_p * emit_p;
                    }
                }

                if (current_u_prob > best_u_prob) {
                    best_u_prob = current_u_prob;
                    best_u = u;
                }
            }

            inputs_seq(n, t) = best_u;

            // --- Step 2: Update Alpha (Belief) for next step ---
            // We must update our state belief based on the input we just "chose" (best_u)
            std::fill(next_alpha.begin(), next_alpha.end(), 0.0);
            double sum_norm = 0.0;

            for (ssize_t j = 0; j < S; ++j) {
                double val = 0.0;
                for (ssize_t i = 0; i < S; ++i) {
                     double trans_p = (t == 0) ? start_mat(best_u, j) : trans_mat(best_u, i, j);
                     val += alpha[i] * trans_p;
                }
                // Emission probability
                val *= emit_mat(best_u, j, v_curr);

                next_alpha[j] = val;
                sum_norm += val;
            }

            // Normalize to prevent underflow
            if (sum_norm < 1e-300) sum_norm = 1.0;
            double scale = 1.0 / sum_norm;
            for(int k=0; k < S; ++k) alpha[k] = next_alpha[k] * scale;
        }
    }
    return inputs_seq_;
}

py::array_t<int> posterior_predict_output(
    py::array_t<double> trans_mat_,
    py::array_t<double> emit_mat_,
    py::array_t<double> start_mat_,
    py::array_t<int> inputs_,
    py::array_t<int> lengths_
) {
    auto trans_mat = trans_mat_.unchecked<3>(); // (n_inputs, S, S)
    auto emit_mat = emit_mat_.unchecked<3>();   // (n_inputs, S, n_outputs)
    auto start_mat = start_mat_.unchecked<2>(); // (n_inputs, S)
    auto inputs = inputs_.unchecked<2>();       // (N, T)
    auto lengths = lengths_.unchecked<1>();     // (N)

    auto N = inputs.shape(0);
    auto T_max = inputs.shape(1);
    auto S = trans_mat.shape(1);
    auto O = emit_mat.shape(2);

    // FIX: Initialize with zeros to prevent garbage in padded regions
    auto outputs_seq_ = py::array_t<int>({N, T_max});
    std::fill(outputs_seq_.mutable_data(), outputs_seq_.mutable_data() + outputs_seq_.size(), 0);

    auto output_seq = outputs_seq_.mutable_unchecked<2>();

    // Pre-allocate buffers for Forward-Backward
    std::vector<std::vector<double>> alpha(T_max, std::vector<double>(S));
    std::vector<double> scaling(T_max);

    py::gil_scoped_release nogil;

    for (ssize_t n = 0; n < N; ++n) {
        auto T_n = lengths(n);
        if (T_n == 0) continue;

        // --- 1. Forward Pass (Alpha) ---
        // IGNORE Emission probabilities (Unit Emission Trick)
        // to infer states based purely on Input transitions.

        // t = 0
        auto u_0 = inputs(n, 0);
        double sum = 0.0;
        for (ssize_t i = 0; i < S; ++i) {
            double p = start_mat(u_0, i);
            alpha[0][i] = p;
            sum += p;
        }
        if (sum < 1e-300) sum = 1.0;
        double scale = 1.0 / sum;
        scaling[0] = scale;
        for (ssize_t i = 0; i < S; ++i) alpha[0][i] *= scale;

        // t = 1 ... T-1
        for (ssize_t t = 1; t < T_n; ++t) {
            auto u_t = inputs(n, t);
            sum = 0.0;
            for (ssize_t j = 0; j < S; ++j) {
                double acc = 0.0;
                for (ssize_t i = 0; i < S; ++i) {
                    acc += alpha[t - 1][i] * trans_mat(u_t, i, j);
                }
                alpha[t][j] = acc;
                sum += acc;
            }
            if (sum < 1e-300) sum = 1.0;
            scale = 1.0 / sum;
            scaling[t] = scale;
            for (ssize_t j = 0; j < S; ++j) alpha[t][j] *= scale;
        }

        // --- 2. Backward Pass + Decoding ---

        for (ssize_t t = T_n - 1; t >= 0; --t) {
            auto u_t = inputs(n, t);

            // A. Find Best State (argmax Gamma)
            // Gamma = Alpha * Beta
            double max_gamma_val = -1.0;
            int best_state = 0;

            for (ssize_t i = 0; i < S; ++i) {
                double gamma_val = alpha[t][i];
                if (gamma_val > max_gamma_val) {
                    max_gamma_val = gamma_val;
                    best_state = i;
                }
            }

            // B. Find Best Output (argmax Emission from Best State)
            // We select the output that maximizes P(Output | Input, Best_State)
            double max_emit_val = -1.0;
            int best_out = 0;
            for (ssize_t o = 0; o < O; ++o) {
                double val = emit_mat(u_t, best_state, o);
                if (val > max_emit_val) {
                    max_emit_val = val;
                    best_out = o;
                }
            }
            output_seq(n, t) = best_out;
        }
    }

    return outputs_seq_;
}


PYBIND11_MODULE(io_baum_welch, m) {
  m.def("forward", forward)
   .def("backward", backward)
   .def("compute_xi_sum", compute_xi_sum)
   .def("predict_inputs_marginal", predict_inputs_marginal)
   .def("posterior_predict_output", posterior_predict_output);
}