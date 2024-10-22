#!/usr/bin/env python3

import numpy as np
import xarray as xr
import scipy

import seaborn as sns
import matplotlib.pyplot as plt

prior = np.ones(10) / 10

# d = np.array((1, 1, 0, 0, 0, 0, 0, 0, 1, 0))
d = np.array((1, 1, 1, 0, 0, 0, 1, 1, 0, 0))

# P(Z=z | L)

def lh_hallway(doors, z=1, z_prob=.75):
    likelihood = np.empty(len(doors))
    likelihood[doors==z] = z_prob
    likelihood[doors!=z] = 1 - z_prob
    return likelihood


# first observation z
z = 1 # no door

# P(Z=z | L) = ℓ_L(z)

# P(L | Z=z) = P(Z=z | L) * P(L) / P(Z=z)
# P(Z=z) = sum_l P(Z=z | L=l) * P(L=l)
def update(likelihood, belief):
    x = likelihood * belief
    return x / x.sum()

def perfect_predict(v, belief):
    return np.roll(belief, v)

def predict(v, belief):
    prediction_prob = [.1, .8, .1]
    belief = scipy.ndimage.convolve(belief, prediction_prob, mode='wrap')
    return perfect_predict(v, belief)

beliefs = []
kinds = []

for n, z in enumerate(d):
    likelihood = lh_hallway(d, z)
    posterior = update(likelihood, prior) # belief
    beliefs.append(posterior); kinds.append(f'posterior{n}')
    # prior = perfect_predict(1, posterior) # also belief
    prior = predict(1, posterior) # also belief
    beliefs.append(prior); kinds.append(f'prior{n}')

print(posterior)

x = xr.Dataset(
    {"prob": (('kind', 'pos'), beliefs)},
    coords=dict(kind=kinds, pos=np.arange(len(d)))
).stack(desired=['kind', 'pos'])
df = x.to_dataframe()
grid = sns.FacetGrid(
    df, col='kind', palette="tab20c",
    col_wrap=2, height=.8, aspect=2
)
grid.map(plt.plot, "pos", "prob", marker="o")
plt.show()