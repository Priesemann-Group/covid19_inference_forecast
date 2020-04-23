import theano
import theano.tensor as tt
import numpy as np


def tt_lognormal(x, mu, sigma):
    distr = 1/x* tt.exp(-((tt.log(x) - mu) ** 2) / (2 * sigma ** 2))
    return distr / tt.sum(distr, axis=0)


def delay_cases(new_I_t, len_new_I_t, len_out, delay, delay_diff):
    """
        Delays (time shifts) the input new_I_t by delay.

        Parameters
        ----------
        new_I_t : ~numpy.ndarray or theano vector
            Input to be delayed.

        len_new_I_t : integer
            Length of new_I_t. (Think len(new_I_t) ).
            Assure len_new_I_t is larger then len(new_cases_obs)-delay, otherwise it
            means that the simulated data is not long enough to be fitted to the data.

        len_out : integer
            Length of the output.

        delay : number
            If delay is an integer, the array will be exactly shifted. Else, the data
            will be shifted and intepolated (convolved with hat function of width one).
            Take care that delay is smaller than or equal to delay_diff,
            otherwise zeros are returned, which could potentially lead to errors.

        delay_diff: integer
            The difference in length between the new_I_t and the output.

        Returns
        -------
            an array with length len_out that was time-shifted by delay
    """

    # elementwise delay of input to output
    delay_mat = make_delay_matrix(
        n_rows=len_new_I_t, n_columns=len_out, initial_delay=delay_diff
    )
    inferred_cases = interpolate(new_I_t, delay, delay_mat)
    return inferred_cases


def make_delay_matrix(n_rows, n_columns, initial_delay=0):
    """
        Has in each entry the delay between the input with size n_rows and the output
        with size n_columns

        initial_delay is the top-left element.
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


def apply_delay(array, delay, sigma_delay, delay_mat):
    mat = tt_lognormal(delay_mat, mu=np.log(delay), sigma=sigma_delay)
    return tt.dot(array, mat)


def delay_cases_lognormal(
    input_arr,
    len_input_arr,
    len_output_arr,
    median_delay,
    scale_delay,
    delay_betw_input_output,
):
    delay_mat = make_delay_matrix(
        n_rows=len_input_arr,
        n_columns=len_output_arr,
        initial_delay=delay_betw_input_output,
    )
    delay_mat[
        delay_mat < 0.01
    ] = 0.01  # needed because negative values lead to nans in the lognormal distribution.
    delayed_arr = apply_delay(input_arr, median_delay, scale_delay, delay_mat)
    return delayed_arr


def interpolate(array, delay, delay_matrix):
    """
        smooth the array (if delay is no integer)
    """
    interp_matrix = tt.maximum(1 - tt.abs_(delay_matrix - delay), 0)
    interpolation = tt.dot(array, interp_matrix)
    return interpolation


def smooth_step_function(start_val, end_val, t_begin, t_end, t_total):
    """
        Instead of going from start_val to end_val in one step, make the change a
        gradual linear slope.

        Parameters
        ----------
            start_val : number
                Starting value

            end_val : number
                Target value

            t_begin : number or array (theano)
                Time point (inbetween 0 and t_total) where start_val is placed

            t_end : number or array (theano)
                Time point (inbetween 0 and t_total) where end_val is placed

            t_total : integer
                Total number of time points

        Returns
        -------
            : theano vector
                vector of length t_total with the values of the parameterised f(t)
    """
    t = np.arange(t_total)

    return (
        tt.clip((t - t_begin) / (t_end - t_begin), 0, 1) * (end_val - start_val)
        + start_val
    )
