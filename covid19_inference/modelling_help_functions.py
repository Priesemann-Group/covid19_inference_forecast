import theano
import theano.tensor as tt
import numpy as np


def _SIR_model(λ, μ, S_begin, I_begin, N):
    new_I_0 = tt.zeros_like(I_begin)
    def next_day(λ, S_t, I_t, _, μ, N):
        new_I_t = λ/N*I_t*S_t
        S_t = S_t - new_I_t
        I_t = I_t + new_I_t - μ * I_t
        I_t = tt.clip(I_t, 0, N) # for stability
        return S_t, I_t, new_I_t
    outputs , _  = theano.scan(fn=next_day, sequences=[λ],
                               outputs_info=[S_begin, I_begin, new_I_0],
                               non_sequences = [μ, N])
    S_all, I_all, new_I_all = outputs
    return S_all, I_all, new_I_all


def _SEIR_model_with_delay(λ, μ, S_begin, new_E_begin, I_begin, N, median_incubation = 5,
                           sigma_incubation = 0.418):
    """

    """
    x = np.arange(1,9)
    β = _tt_lognormal(x, tt.log(median_incubation), sigma_incubation)
    new_I_0 = tt.zeros_like(I_begin)
    def next_day(λ, S_t, nE1, nE2, nE3, nE4, nE5, nE6, nE7, nE8, I_t, _, μ, β, N):
        new_E_t = λ/N*I_t*S_t
        S_t = S_t - new_E_t
        new_I_t = β[0]*nE1 + β[1]*nE2 + β[2]*nE3 + β[3]*nE4 + β[4]*nE5 + β[5]*nE6 + β[6]*nE7 + β[7]*nE8
        I_t = I_t + new_I_t - μ * I_t
        I_t = tt.clip(I_t, 0.001, N) # for stability
        #I_t = tt.nnet.sigmoid(I_t/N)*N
        #S_t = tt.clip(S_t, -1, N+1)
        return S_t, new_E_t, I_t, new_I_t
    outputs , _  = theano.scan(fn=next_day, sequences=[λ],
                               outputs_info=[S_begin,
                                             dict(initial =  new_E_begin, taps = [-1, -2,-3,-4,-5,-6,-7,-8]),
                                             I_begin,
                                             new_I_0],
                               non_sequences = [μ, β, N])
    S_all, new_E_all, I_all, new_I_all = outputs
    return S_all, new_E_all, I_all, new_I_all


def _tt_lognormal(x, mu, sigma):
    distr = tt.exp(-(tt.log(x) - mu)**2/(2*sigma**2))
    return distr/tt.sum(distr, axis=0)

def _delay_cases(new_I_t, len_new_I_t, len_new_cases_obs, delay, delay_betw_input_output):
    """
    Delays the input new_I_t by delay and return and array with length len_new_cases_obs
    The initial delay of the output is set by delay_arr.

    Take care that delay is smaller or equal than delay_arr, otherwise zeros are
    returned, which could potentially lead to errors

    Also assure that len_new_I_t is larger then len(new_cases_obs)-delay, otherwise it
    means that the simulated data is not long enough to be fitted to the data.
    """
    delay_mat = _make_delay_matrix(n_rows=len_new_I_t,
                                  n_columns=len_new_cases_obs, initial_delay=delay_betw_input_output)
    inferred_cases = _interpolate(new_I_t, delay, delay_mat)
    return inferred_cases


def _make_delay_matrix(n_rows, n_columns, initial_delay=0):
    """
    Has in each entry the delay between the input with size n_rows and the output
    with size n_columns
    """
    size = max(n_rows, n_columns)
    mat = np.zeros((size, size))
    for i in range(size):
        diagonal = np.ones(size - i) * (initial_delay + i)
        mat += np.diag(diagonal, i)
    for i in range(1, size):
        diagonal = np.ones(size - i) * (initial_delay - i)
        mat += np.diag(diagonal, -i)
    return mat[:n_rows, :n_columns]

def _apply_delay(array, delay, sigma_delay, delay_mat):
    mat = _tt_lognormal(delay_mat, mu=np.log(delay), sigma=sigma_delay)
    return tt.dot(array, mat)

def _delay_cases_lognormal(input_arr, len_input_arr, len_output_arr,  median_delay, scale_delay, delay_betw_input_output):
    delay_mat = _make_delay_matrix(n_rows=len_input_arr,
                                  n_columns=len_output_arr, initial_delay=delay_betw_input_output)
    delay_mat[delay_mat<0.01] = 0.01 #needed because negative values lead to nans in the lognormal distribution.
    delayed_arr = _apply_delay(input_arr, median_delay, scale_delay, delay_mat)
    return delayed_arr


def _interpolate(array, delay, delay_matrix):
    interp_matrix = tt.maximum(1 - tt.abs_(delay_matrix - delay), 0)
    interpolation = tt.dot(array, interp_matrix)
    return interpolation


def _smooth_step_function(λ_begin, λ_end, t_begin, t_end, t_total):
    t = np.arange(t_total)
    return tt.clip((t - t_begin) / (t_end - t_begin), 0, 1) * (λ_end - λ_begin) + λ_begin