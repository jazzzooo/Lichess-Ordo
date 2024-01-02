#!/usr/bin/env python3

import numpy as np
from prediction_errors import adjust, prediction_errors
from tqdm import tqdm

# The ratigns can get stuck in a local minima
# so we do a big jump sometimes to adjust them.
# These jump values are similar to the K-factor in Elo
SMALL_JUMP = 1.7
LARGE_JUMP = 75.0
LARGE_EVERY = 100

STEPS = 1000

# Load densely packed data
with open("data/packed-blitz", "rb") as f:
    n_players = int.from_bytes(f.read(4), "little")
    n_opponents = np.fromfile(f, dtype=np.uint16, count=n_players).astype(np.intc)
    indices = np.zeros(n_players + 1, dtype=np.intc)
    indices[1:] = np.cumsum(n_opponents)
    n_pairings = n_opponents.sum()
    total = np.fromfile(f, dtype=np.uint16, count=n_players).astype(np.float64)
    scored = np.fromfile(f, dtype=np.uint16, count=n_players).astype(np.float64) / 2
    opponents = np.ascontiguousarray(np.fromfile(f, dtype=np.uint32, count=n_pairings).astype(np.intc))
    opp_played = np.ascontiguousarray(np.fromfile(f, dtype=np.uint16, count=n_pairings).astype(np.intc))


def optimize(steps=STEPS, small_jump=SMALL_JUMP, large_jump=LARGE_JUMP, large_every=LARGE_EVERY, *, verbose=True):
    # Converting back to normal ratings can be done with:
    # log10(rating) * 400 + 1500
    # altho the median will turn out not to be 1500 usually.
    # maybe the median or even mode rating should be added instead?
    ratings = np.ones(n_players, dtype=np.float64, order="C")
    errors = np.zeros(n_players, dtype=np.float64, order="C")
    # also we dont actually need to start with 1500 ratings for everyone
    # seeding any kind of ratings will make it converge 100x faster
    iterator = tqdm(range(steps), smoothing=0) if verbose else range(steps)
    for i in iterator:
        prediction_errors(errors, ratings, 1 / ratings, scored, opponents, opp_played, indices, n_players)
        adjust(ratings, errors, total, n_players, large_jump if not i % int(large_every) else small_jump)
        if verbose:
            loss = np.abs(errors).sum()
            iterator.set_description(f"loss: {loss:16f}")
    # return ratings, np.abs(errors).sum()
    return ratings, np.abs(errors).sum()


if __name__ == "__main__":
    ratings, loss = optimize()
