#!/usr/bin/env python3

import numpy as np
from prediction_errors import adjust, prediction_errors
from tqdm import trange

# It helps to do big adjustments sometimes to make convergence faster.
# Especially when you start everyone off at 1500
SMALL_JUMP = 1.6
LARGE_JUMP = 33
LARGE_EVERY = 21

STEPS = 1200

# White advantage constant, based on the number of wins in lichess data up to November 2023
# Elo advantage can be calculated as 400 * (log10(W + D/2) - log10(B + D/2))
# TimeControl              W         B        D   ≈elo
# UltraBullet       14988707  14323116   477377  7.764
# Bullet           784325525 740605279 48396796  9.657
# Blitz           1054975859 978840645 92865717 12.444
# Rapid            299734348 275830958 27386294 13.781
# Classical         57414088  52809117  5140023 13.876
# Correspondence     4474613   4241503   629812  8.668
W, B, D = 1054975859, 978840645, 92865717
WA = (W + D / 2) / (B + D / 2)

# Load densely packed data
with open("data/blitz.packed", "rb") as f:
    # Number of players
    n_players = int.from_bytes(f.read(4), "little")

    # For each player, the number of different opponents that they faced
    n_opponents = np.fromfile(f, dtype=np.uint16, count=n_players).astype(np.intc)
    n_pairings = n_opponents.sum()

    # The offsets in the opponents and totals arrays to see how many games each player had
    # against a specific opponent
    indices = np.zeros(n_players + 1, dtype=np.intc)
    indices[1:] = np.cumsum(n_opponents)

    # The total number of games played by each player. We precompute the inverse to avoid dividing later
    totals_inv = 1 / (np.fromfile(f, dtype=np.uint16, count=n_players).astype(np.float64))

    # How many points total did the player score
    scored = np.fromfile(f, dtype=np.uint16, count=n_players).astype(np.float64) / 2

    # Now the big arrays:
    # Indexes of the opponents faced by each player.
    # player_i will have faced players with ids opponents[indices[i]:indices[i+1]]
    opponents = np.fromfile(f, dtype=np.uint32, count=n_pairings).astype(np.intc)
    # The total number of games played by each opponent of each player, as white and as black
    white_opp_totals = np.fromfile(f, dtype=np.uint16, count=n_pairings).astype(np.intc)
    black_opp_totals = np.fromfile(f, dtype=np.uint16, count=n_pairings).astype(np.intc)

    # Optionally, load the player ids
    ids = [f.read(26).rstrip(b"\0").decode() for b in range(n_players)]


def optimize(steps=STEPS, small_jump=SMALL_JUMP, large_jump=LARGE_JUMP, large_every=LARGE_EVERY, *, verbose=True):
    # Converting back to normal ratings can be done with:
    # log10(rating) * 400 + 1500
    # altho the median will turn out not to be 1500 usually.
    # maybe the median or even mode rating should be added instead?
    ratings = np.ones(n_players, dtype=np.float64)
    errors = np.zeros(n_players, dtype=np.float64)
    # also we dont actually need to start with 1500 ratings for everyone
    # seeding any kind of ratings will make it converge 100x faster
    iterator = trange(steps, smoothing=0) if verbose else range(steps)
    for i in iterator:
        prediction_errors(errors, ratings, scored, opponents, white_opp_totals, black_opp_totals, indices, WA)
        adjust(ratings, errors, totals_inv, small_jump if i % int(large_every) else large_jump)
        loss = np.abs(errors).sum()
        if verbose:
            iterator.set_description(f"loss: {loss:16.5f}")
        if loss < 1.0:
            break
    return ratings, np.abs(errors).sum()


if __name__ == "__main__":
    ratings, loss = optimize()

players = dict(zip(ids, np.log10(ratings) * 400 + 1500))
for i in sorted(players, key=players.get, reverse=True)[:10]:
    print(f"{i:26} {players[i]:.3f}")
