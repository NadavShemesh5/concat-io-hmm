import numpy as np
from functools import wraps
from time import time
from numba import njit, prange

EPS = 1e-12
LOG_EPS = -100.0


def normalize(a, axis=None):
    a_sum = a.sum(axis)
    if axis and a.ndim > 1:
        # Make sure we don't divide by zero.
        a_sum[a_sum == 0] = 1
        shape = list(a.shape)
        shape[axis] = 1
        a_sum.shape = shape

    a /= a_sum


def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        print('func:%r took: %2.4f sec' % \
          (f.__name__, te-ts))
        return result

    return wrap


@njit(parallel=True, fastmath=True)
def viterbi_forward(batch_data, start_mat, trans_mat, emit_mat):
    N, T = batch_data.shape
    n_inputs, n_states_prev, n_states = trans_mat.shape
    _, _, n_outputs = emit_mat.shape

    output_seq = np.zeros((N, T), dtype=np.uint16)

    # Pre-compute logs to avoid repeated calculations
    # Add epsilon to prevent log(0)
    log_start = np.log(start_mat + EPS)
    log_trans = np.log(trans_mat + EPS)
    log_emit = np.log(emit_mat + EPS)

    # Parallelize over the batch dimension
    for n in prange(N):
        # Viterbi Tables
        # v_probs[t, s] = max log prob of reaching state s at time t
        v_probs = np.full((T, n_states), LOG_EPS, dtype=np.float64)
        # backpointers[t, s] = previous state that maximized v_probs
        backpointers = np.zeros((T, n_states), dtype=np.uint16)

        # --- Initialization (t=0) ---
        inp0 = batch_data[n, 0]
        for s in range(n_states):
            v_probs[0, s] = log_start[inp0, s]

        # --- Recursion (t=1 to T-1) ---
        for t in range(1, T):
            inp = batch_data[n, t]
            for s_curr in range(n_states):
                max_p = LOG_EPS
                best_prev = 0

                # Iterate previous states to find best transition
                for s_prev in range(n_states_prev):
                    # Transition cost depends on Input[t] (structure of original code)
                    # p = P(path_to_prev) + P(transition | input)
                    curr_p = v_probs[t - 1, s_prev] + log_trans[inp, s_prev, s_curr]

                    if curr_p > max_p:
                        max_p = curr_p
                        best_prev = s_prev

                v_probs[t, s_curr] = max_p
                backpointers[t, s_curr] = best_prev

        # --- Backtracking ---
        # Find best final state
        best_state = np.argmax(v_probs[T - 1])

        # Traverse backwards
        # We can fill output_seq directly during backtracking
        for t in range(T - 1, -1, -1):
            curr_s = best_state
            inp = batch_data[n, t]

            # Deterministic output: Argmax P(Output | Input, State)
            # Find best output index for this (Input, State) pair
            best_out_idx = 0
            max_emit_val = LOG_EPS
            for o in range(n_outputs):
                if log_emit[inp, curr_s, o] > max_emit_val:
                    max_emit_val = log_emit[inp, curr_s, o]
                    best_out_idx = o

            output_seq[n, t] = best_out_idx

            # Move to previous state for next iteration
            if t > 0:
                best_state = backpointers[t, curr_s]

    return output_seq


@njit(parallel=True, fastmath=True)
def viterbi_backward(target_outputs, start_mat, trans_mat, emit_mat):
    N, T = target_outputs.shape
    n_inputs, n_states_prev, n_states = trans_mat.shape

    input_seq = np.zeros((N, T), dtype=np.uint16)

    log_start = np.log(start_mat + EPS)
    log_trans = np.log(trans_mat + EPS)
    log_emit = np.log(emit_mat + EPS)

    for n in prange(N):
        # Viterbi variables
        v_probs = np.full((T, n_states), LOG_EPS, dtype=np.float64)
        path_states = np.zeros((T, n_states), dtype=np.uint16)  # Stores best S_prev
        path_inputs = np.zeros(
            (T, n_states), dtype=np.uint16
        )  # Stores best Input for (S_curr)

        # --- Initialization (t=0) ---
        target_out0 = target_outputs[n, 0]
        for s in range(n_states):
            # Find best Input that could start this State and emit this Output
            # Cost = P(Start S | Input) * P(Output | Input, S)
            best_p = LOG_EPS
            best_inp = 0

            for i in range(n_inputs):
                p = log_start[i, s] + log_emit[i, s, target_out0]
                if p > best_p:
                    best_p = p
                    best_inp = i

            v_probs[0, s] = best_p
            path_inputs[0, s] = best_inp

        # --- Recursion ---
        for t in range(1, T):
            target_out = target_outputs[n, t]

            for s_curr in range(n_states):
                max_global_p = LOG_EPS
                best_prev_s = 0
                best_curr_i = 0

                # To reach S_curr, we must pick an Input[t] and come from S_prev
                # We maximize over both simultaneously.
                # Score = V_prob[t-1, S_prev] + P(S_curr | S_prev, Input) + P(Output | Input, S_curr)

                # Optimization: Inner loop over Input, Outer over S_prev
                # (or vice versa, typically inputs are fewer)

                for i in range(n_inputs):
                    # Compute emission cost for this input once
                    emit_cost = log_emit[i, s_curr, target_out]

                    if emit_cost <= LOG_EPS:
                        continue  # Skip impossible emissions

                    for s_prev in range(n_states_prev):
                        trans_cost = log_trans[i, s_prev, s_curr]
                        prev_score = v_probs[t - 1, s_prev]

                        total_p = prev_score + trans_cost + emit_cost

                        if total_p > max_global_p:
                            max_global_p = total_p
                            best_prev_s = s_prev
                            best_curr_i = i

                v_probs[t, s_curr] = max_global_p
                path_states[t, s_curr] = best_prev_s
                path_inputs[t, s_curr] = best_curr_i

        # --- Backtrack ---
        best_s = np.argmax(v_probs[T - 1])

        for t in range(T - 1, -1, -1):
            input_seq[n, t] = path_inputs[t, best_s]
            if t > 0:
                best_s = path_states[t, best_s]

    return input_seq


@njit(fastmath=True)
def sample_1d(probs):
    cumsum = np.cumsum(probs)
    total = cumsum[-1]
    rand_val = np.random.random() * total
    return np.searchsorted(cumsum, rand_val)


@njit(parallel=True, fastmath=True)
def greedy_forward(batch_data, start_mat, trans_mat, emit_mat):
    N, T = batch_data.shape
    output_seq = np.zeros((N, T), dtype=np.uint16)

    # Parallelize over batch dimension
    for n in prange(N):
        # Current hidden state
        curr_state = 0

        for t in range(T):
            input_token = batch_data[n, t]

            # --- 1. Determine Hidden State (Transition) ---
            if t == 0:
                # Use Start Matrix
                # probs: (n_states,)
                state_probs = start_mat[input_token]
                curr_state = sample_1d(state_probs)
            else:
                # Use Transition Matrix given previous state
                # probs: (n_states,)
                state_probs = trans_mat[input_token, curr_state]
                curr_state = sample_1d(state_probs)

            # --- 2. Determine Output (Emission) ---
            # probs: (n_outputs,)
            emit_probs = emit_mat[input_token, curr_state]
            output_token = sample_1d(emit_probs)

            output_seq[n, t] = output_token

    return output_seq


@njit(fastmath=True)
def random_choice_2d(probs_2d):
    flat_probs = probs_2d.ravel()
    cumsum = np.cumsum(flat_probs)
    cutoff = np.random.random() * cumsum[-1]

    # Binary search (searchsorted) is faster than linear scan for large arrays
    flat_idx = np.searchsorted(cumsum, cutoff)

    # Convert back to 2D indices
    rows, cols = probs_2d.shape
    r = flat_idx // cols
    c = flat_idx % cols
    return r, c


@njit(parallel=True, fastmath=True)
def greedy_backward(target_outputs, start_mat, trans_mat, emit_mat):
    N, T = target_outputs.shape
    n_inputs, n_states_prev, n_states_curr = trans_mat.shape

    input_seq = np.zeros((N, T), dtype=np.uint16)

    # Parallelize over batch
    for n in prange(N):
        # 1. Initialize Last State (at T-1)
        # We need a starting point. Since we don't have a 'future' to condition on,
        # we can sample uniformly or based on emission prob alone.
        # Approximation: Sample uniform random state to start the chain backward.
        curr_s = np.random.randint(0, n_states_curr)

        # 2. Iterate Backwards
        for t in range(T - 1, -1, -1):
            target_out = target_outputs[n, t]

            # We need to pick (Input, S_prev) that leads to 'curr_s'
            # Joint Prob = P(curr_s | S_prev, Input) * P(Output | Input, S_prev)
            # Note: The emission usually depends on (Input, S_prev) or (Input, S_curr)?
            # Based on standard HMM: Emission depends on Current State (hidden variable).
            # In your previous Viterbi, emission was emit[Input, State, Output].

            # Slicing trans_mat for the specific target 'curr_s'
            # Shape: (n_inputs, n_states_prev)
            # This represents P(S_next=curr_s | Input, S_prev)
            trans_slice = trans_mat[:, :, curr_s]

            # Emission slice: emit[Input, S_prev, Output]
            # Warning: Standard HMM emits from the state you ARE in, not the one you leave.
            # Assuming emit depends on the state *at time t* (which is S_prev in this backward view).
            emit_slice = emit_mat[:, :, target_out]

            # Combine probabilities (Scaling Method)
            # Shape: (n_inputs, n_states_prev)
            joint_probs = trans_slice * emit_slice

            # Handle t=0 (Initial Step uses start_mat instead of trans_mat)
            if t == 0:
                # Shape: (n_inputs, n_states_prev) -> here S_prev acts as S_0
                joint_probs = start_mat * emit_slice

            i_idx, s_prev = random_choice_2d(joint_probs)

            input_seq[n, t] = i_idx
            curr_s = s_prev

    return input_seq