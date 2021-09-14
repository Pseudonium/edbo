"""Gotten from https://scikit-learn.org/stable/auto_examples/gaussian_process/plot_gpr_noisy_targets.html

This produces 'Initial observations of test function',
as well as 'Test function surrogate model',
diagrams for illustrating bayesian optimisation.
"""


import numpy as np
from matplotlib import pyplot as plt

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

np.random.seed(1)

COLORS = ['#F72585', '#7209B7', '#3A0CA3', '#4361EE', '#4CC9F0']

def f(x):
    """The function to predict."""
    return x * np.sin(x)

X = np.atleast_2d([1., 5., 4., 7., 9.]).T
y = f(X).ravel()

x = np.atleast_2d(np.linspace(0, 10, 1000)).T

kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)

gp.fit(X, y)

y_pred, sigma = gp.predict(x, return_std=True)

fig_1, ax_1 = plt.subplots(1)

ax_1.plot(X, y, '.', markersize=10, label='Observations', color=COLORS[0])
ax_1.set_ylim(-10, 20)
ax_1.legend(loc='upper left')

fig_2, ax_2 = plt.subplots(1)
ax_2.plot(x, f(x), ':', label=r'$f(x) = x\,\sin(x)$', color='black')
ax_2.plot(X, y, '.', markersize=10, label='Observations', color=COLORS[0])
ax_2.plot(x, y_pred, '-', label='Prediction', color=COLORS[1])
max_i_exploit = np.argmax(y_pred)
max_i_explore = np.argmax(sigma)
max_i_balance = np.argmax(y_pred + 1.5 * sigma)
ax_2.fill(np.concatenate([x, x[::-1]]),
         np.concatenate([y_pred - 1.5 * sigma,
                        (y_pred + 1.5 * sigma)[::-1]]),
         alpha=.5, fc=COLORS[4], ec='None', label='1.5 standard deviation')
ax_2.set_ylim(-10, 20)
ax_2.set_xlim(0, 10)

for i, (max_i, array) in enumerate(zip(
    [max_i_explore, max_i_exploit, max_i_balance], # i meaning 'index of'
    # so e.g. max_i_explore is the index of the point with the max
    # 'explore' acquisition function value.
    # Here 'explore' is VarMax, 'exploit' is MeanMax, 'balance' is a version
    # of UCB (Upper Confidence Bound).
    [y_pred, y_pred, y_pred + 1.5 * sigma]
)):
    ax_2.plot(x[max_i], array[max_i], '.', color=COLORS[2])
    if i == 0:
        xy = (x[max_i], array[max_i] - 1.5)
    else:
        xy = (x[max_i], array[max_i] + 0.3)
    ax_2.annotate(str(i + 1), xy, fontweight='bold', fontsize=14)


ax_2.legend(loc='upper left')

ax_2.set_facecolor((240/255,232/255,237/255))
fig_2.set_facecolor((240/255,232/255,237/255))

plt.show()
