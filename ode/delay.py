"""Functionality from the following file:

    https://github.com/Priesemann-Group/covid19_inference_forecast/blob/master/
        covid19_inference/model_helper.py

Reimplemented to remove dependency on theano and perform the same
operations using numpy.
"""

import numpy as np


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


def interpolate(array, delay, delay_matrix):
    """
    smooth the array (if delay is no integer)
    """
    interp_matrix = np.maximum(1 - np.abs(delay_matrix - delay), 0)
    interpolation = np.dot(array, interp_matrix)
    # interpolation = array @ interp_matrix
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
        np.clip((t - t_begin) / (t_end - t_begin), 0, 1) * (end_val - start_val)
        + start_val
    )

