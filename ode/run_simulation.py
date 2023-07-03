import datetime
import numpy as np
import pints
import pandas
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import scipy.integrate

import delay
import solvers


# Set Latex and fonts
import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = \
    r'\usepackage{{amsmath}}\renewcommand{\sfdefault}{phv}'


class ChangepointCovidSIR(pints.ForwardModel):
    """SIR model for COVID with changepoints in spreading rate.

    Parameters
    ----------
    num_changepoints : int
        How many changepoints
    solver : {'daily', 'RK45'}
        How to solve the differential equations
    """
    def __init__(self,
                 date_begin_sim,
                 num_days_sim,
                 diff_data_sim,
                 num_changepoints=3,
                 solver='daily'):
        super().__init__()
        self.num_changepoints = num_changepoints
        self.solver = solver

        # population of germany
        self.N = 83e6

        # Solver dates
        self.date_begin_sim = date_begin_sim
        self.num_days_sim = num_days_sim
        self.diff_data_sim = diff_data_sim

        self.forecast = False
        self.modulate = False

    def n_outputs(self):
        return 1

    def n_parameters(self):
        return 6 + 3 * self.num_changepoints

    def simulate(self, parameters, times):
        # times will be ignored
        # instead, it looks at the timing parameters given to the solver

        # Extract parameters from parameters vector
        N = self.N
        nc = self.num_changepoints
        lambdas = parameters[:nc+1]
        changepoints = parameters[nc+1:2*nc+1]
        dts = parameters[2*nc+1:3*nc+1]
        mu = parameters[-5]
        D = parameters[-4]
        I0 = parameters[-3]
        f_w = parameters[-2]
        phi_w = parameters[-1]

        changepoints = list(changepoints)

        for i, c in enumerate(changepoints):
            changepoints[i] += (datetime.datetime(2020, 3, 1) - self.date_begin_sim).days - 2

        # Build lambda profile
        lambda_t_list = [lambdas[0] * np.ones(self.num_days_sim)]
        lambda_before = lambdas[0]
        for tr_begin, tr_len, lambda_after in zip(changepoints,
                                                  dts,
                                                  lambdas[1:]):
            lambda_t = delay.smooth_step_function(
                start_val=0,
                end_val=1,
                t_begin=tr_begin,
                t_end=tr_begin + tr_len,
                t_total=self.num_days_sim,
            ) * (lambda_after - lambda_before)
            lambda_before = lambda_after
            lambda_t_list.append(lambda_t)
        lambda_t = sum(lambda_t_list)

        S = [N - I0]
        I = [I0]
        I_new = [0]

        if self.solver == 'daily':
            for lam in lambda_t:
                I_newt = lam / N * I[-1] * S[-1]
                St = S[-1] - I_newt
                It = I[-1] + I_newt - mu * I[-1]
                S.append(St)
                I.append(It)
                I_new.append(I_newt)
            I_new = I_new[1:]


        elif self.solver == 'RK45':
            l_times = np.arange(1, len(lambda_t)+1)
            lambda_func = interp1d(l_times, lambda_t, fill_value='extrapolate')
            def deriv(t, y):
                lam = lambda_func(t+1)
                dS = -lam * y[0] * y[1] / N
                dI = lam * y[0] * y[1] / N - mu * y[1]
                return [dS, dI]

            t_range = (0, len(lambda_t)+1)
            res = scipy.integrate.solve_ivp(
                deriv,
                t_range,
                [N - I0, I0],
                t_eval=np.arange(0, len(lambda_t) + 2),
                # rtol=1e-7,
                # atol=1e-7,
                # method='RK45',
                method=solvers.ForwardEuler,
                step_size=0.1,
            )

            S = res.y[0][1:]
            I_new = -np.diff(S)

        else:
            raise ValueError('solver not recognized')

        # Apply delay
        C = delay.delay_cases(
            I_new,
            self.num_days_sim,
            self.num_days_sim - self.diff_data_sim,
            D,
            self.diff_data_sim
        )

        # Apply modulation
        if self.modulate:
            t = np.arange(self.num_days_sim - self.diff_data_sim)
            date_begin = self.date_begin_sim + datetime.timedelta(days=self.diff_data_sim+1)
            weekday_begin = date_begin.weekday()
            t -= weekday_begin
            modulation = 1 - np.abs(np.sin(t/7 * np.pi + phi_w / 2))
            vec = np.ones(self.num_days_sim - self.diff_data_sim) - (1 - f_w) * modulation
            C = vec * C

        if self.forecast:
            return C
        else:
            return C[:len(times)]


def get_data():
    """Load the Germany data.

    Returns
    -------
    dates : list
        List of datetimes
    data : np.array
        Total cases at each day
    """
    data = pandas.read_csv('data/germany_data.csv')

    dates = data.columns
    data = data.to_numpy()[0]

    python_dates = []
    for x in dates:
        month, day, year = x.split('/')
        month = int(month)
        day = int(day)
        year = int(year) + 2000
        python_dates.append(datetime.datetime(year, month, day))

    return python_dates, data


def figure7():
    """Forward simulations of SIR.
    """
    dates, data = get_data()

    # Choose the data from 2020 March 1 to 2020 April 21
    dates = dates[39:-6]
    data = data[39:-6]

    # Convert the total cases to daily cases
    data = np.diff(data)
    dates = dates[1:]

    # Calculate timing parameters according to the paper
    num_days_data = len(data)  # Number of days of data points
    diff_data_sim = 16  # Must always be longer than delay D
    num_days_future = 28  # How long to forecast after data

    date_begin_sim = datetime.datetime(2020, 3, 1) - datetime.timedelta(days=diff_data_sim)
    date_end_sim = datetime.datetime(2020, 4, 21) + datetime.timedelta(days=num_days_future)
    num_days_sim = (date_end_sim - date_begin_sim).days

    # Make model and run it at parameter values from the paper
    m = ChangepointCovidSIR(date_begin_sim, num_days_sim, diff_data_sim)
    m.forecast = True
    true_params = [0.43, 0.25, 0.15, 0.09, 6.7, 16.2, 23.6, 3.0, 3.0, 4.0,
                   0.13, 11.4, 36.1, 0.6, 0.4]
    c = m.simulate(true_params, np.arange(len(data)))
    # plt.plot(np.arange(1, len(c)+1), c)
    # plt.scatter(np.arange(1, len(data)+1), data)
    # plt.plot(np.arange(1, len(c)+1), c)
    # plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(2, 2, 3)

    diff_data_sim = 14
    num_days_future = 0
    date_begin_sim = datetime.datetime(2020, 2, 14) \
                     - datetime.timedelta(days=diff_data_sim)
    date_end_sim = datetime.datetime(2020, 3, 21) \
                   + datetime.timedelta(days=num_days_future)
    num_days_sim = (date_end_sim - date_begin_sim).days
    m = ChangepointCovidSIR(
        date_begin_sim, num_days_sim, diff_data_sim, num_changepoints=1)
    m.forecast = True

    params_case1 = [0.41, 0.041, 0.0, 7.0, 0.12, 8.6, 19.2, 0.0, 0.0]
    c = m.simulate(params_case1, np.arange(-14, 22))
    ax.plot(np.arange(-14, 22), c, ls='-', color='green')

    m.solver = 'RK45'
    params_case1 = [0.41, 0.041, 0.0, 7.0, 0.12, 8.6, 10.25, 0.0, 0.0]
    c = m.simulate(params_case1, np.arange(-14, 22))
    ax.plot(np.arange(-14, 22), c, ls='-.', color='green')

    m.solver = 'daily'
    params_case2 = [0.41, 0.21, 0.0, 7.0, 0.12, 8.6, 19.2, 0.0, 0.0]
    c = m.simulate(params_case2, np.arange(-14, 22))
    ax.plot(np.arange(-14, 22), c, ls='-', color='orange', zorder=-1)

    m.solver = 'RK45'
    params_case1 = [0.41, 0.21, 0.0, 7.0, 0.12, 8.6, 10.25, 0.0, 0.0]
    c = m.simulate(params_case1, np.arange(-14, 22))
    ax.plot(np.arange(-14, 22), c, ls='-.', color='orange', zorder=-1)

    m.solver = 'daily'
    params_case2 = [0.41, 0.41, 0.0, 7.0, 0.12, 8.6, 19.2, 0.0, 0.0]
    c = m.simulate(params_case2, np.arange(-14, 22))
    ax.plot(np.arange(-14, 22), c, ls='-', color='red', zorder=-2)

    m.solver = 'RK45'
    params_case1 = [0.41, 0.41, 0.0, 7.0, 0.12, 8.6, 10.25, 0.0, 0.0]
    c = m.simulate(params_case1, np.arange(-14, 22))
    ax.plot(np.arange(-14, 22), c, ls='-.', color='red', zorder=-2)

    ax.plot([-100, -99], [0, 1], ls='-', color='k', label=r'$\Delta t=1.0$')
    ax.plot([-100, -99], [0, 1], ls='-.', color='k', label=r'$\Delta t=0.1$')

    ax.set_xlim(-14, 21)
    ax.set_xticks([-14, -7, 0, 7, 14, 21])
    ax.set_ylim(0, 50000)
    ax.set_xlabel('Day')
    ax.set_ylabel('Daily new cases')
    ax.legend()

    ax = fig.add_subplot(2, 2, 1)
    ax.plot([-14, 21], [0.41, 0.41], color='red', label='No social distancing')
    ax.plot([-14, 0, 7, 21], [0.41, 0.41, 0.21, 0.21], color='orange',
            label='Mild social distancing')
    ax.plot([-14, 0, 7, 21], [0.41, 0.41, 0.041, 0.041], color='green',
            label='Strong social distancing')

    ax.set_ylabel(r'$\lambda$')
    ax.set_xlim(-14, 21)
    ax.set_xticks([-14, -7, 0, 7, 14, 21])
    ax.legend()

    ax = fig.add_subplot(2, 2, 4)
    params_case1 = [0.41, 0.041, -5, 7.0, 0.12, 8.6, 19.2, 0.0, 0.0]
    c = m.simulate(params_case1, np.arange(-14, 22))
    ax.plot(np.arange(-14, 22), c, ls='-', color='green')

    m.solver = 'RK45'
    params_case1 = [0.41, 0.041, -5, 7.0, 0.12, 8.6, 10.25, 0.0, 0.0]
    c = m.simulate(params_case1, np.arange(-14, 22))
    ax.plot(np.arange(-14, 22), c, ls='-.', color='green')

    m.solver = 'daily'
    params_case2 = [0.41, 0.041, 0.0, 7.0, 0.12, 8.6, 19.2, 0.0, 0.0]
    c = m.simulate(params_case2, np.arange(-14, 22))
    ax.plot(np.arange(-14, 22), c, ls='-', color='orange', zorder=-1)

    m.solver = 'RK45'
    params_case1 = [0.41, 0.041, 0.0, 7.0, 0.12, 8.6, 10.25, 0.0, 0.0]
    c = m.simulate(params_case1, np.arange(-14, 22))
    ax.plot(np.arange(-14, 22), c, ls='-.', color='orange', zorder=-1)

    m.solver = 'daily'
    params_case2 = [0.41, 0.041, 5, 7.0, 0.12, 8.6, 19.2, 0.0, 0.0]
    c = m.simulate(params_case2, np.arange(-14, 22))
    ax.plot(np.arange(-14, 22), c, ls='-', color='red', zorder=-2)

    m.solver = 'RK45'
    params_case1 = [0.41, 0.041, 5, 7.0, 0.12, 8.6, 10.25, 0.0, 0.0]
    c = m.simulate(params_case1, np.arange(-14, 22))
    ax.plot(np.arange(-14, 22), c, ls='-.', color='red', zorder=-2)

    ax.plot([-100, -99], [0, 1], ls='-', color='k', label=r'$\Delta t=1.0$')
    ax.plot([-100, -99], [0, 1], ls='-.', color='k', label=r'$\Delta t=0.1$')

    ax.set_xlim(-14, 21)
    ax.set_xticks([-14, -7, 0, 7, 14, 21])
    ax.set_ylim(0, 70000)
    ax.set_xlabel('Day')
    ax.set_ylabel('Daily new cases')
    ax.legend()


    ax = fig.add_subplot(2, 2, 2)
    ax.plot([-14, 5, 12, 21], [0.41, 0.41, 0.041, 0.041], color='red',
            label='Start 5 days')
    ax.plot([-14, 0, 7, 21], [0.41, 0.41, 0.041, 0.041], color='orange',
            label='Start 0 days')
    ax.plot([-14, -5, 2, 21], [0.41, 0.41, 0.041, 0.041], color='green',
            label='Start -5 days')

    ax.set_ylabel(r'$\lambda$')
    ax.set_xlim(-14, 21)
    ax.set_xticks([-14, -7, 0, 7, 14, 21])
    ax.legend()

    fig.set_tight_layout(True)

    fig.text(.03, .95, 'a.', fontsize=15)
    fig.text(.53, .95, 'b.', fontsize=15)

    plt.show()


if __name__ == '__main__':
    figure7()

