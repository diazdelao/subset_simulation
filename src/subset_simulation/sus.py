"""
Run the Subset Simulation driver and assemble per-level outputs.

This module advances the SuS levels, tracks intermediate thresholds and
failure counts, and returns the full sample history together with CCDF data.
"""

import numpy as np
from mma import MMA
from G import G_batch, _G_cached

def sus(
    x,
    n,
    p,
    d,
    YF,
    max_levels=None,
    threshold_tol=1e-8,
    stagnation_patience=5 # check whether threshold has stalled
):
    nc = int(n * p)  # number of chains (or top np samples)
    ns = int((1 - p) / p)  # number of states per chain
    Yi = []  # list for intermediate failure thresholds
    L = 0  # level number
    print('SuS Level = 0')
    nF = []  # counter for failure samples
    stop_reason = None
    stagnant_levels = 0
    # Use linspace so the percentile estimator always has exactly n entries.
    prob_ratio = np.linspace((n - 1) / n, 0, n)

    # Keep level data in Python lists and only stack at the end. This avoids
    # repeated array reallocation via vstack/concatenate on every level.
    current_x = np.array(x[:, :, 0], copy=True)
    current_y = G_batch(current_x.T)
    nF.append(int(np.count_nonzero(current_y > YF)))

    x_levels = [current_x.copy()]
    y_levels = [current_y.copy()]
    sorted_levels = [np.sort(current_y)]
    
    # Debug initial failure samples
    print(f"Initial number of failure samples at Level 0: {nF[0]} out of {n}")

    # Keep the algorithmic stopping rule separate from runtime safety guards.
    # We stop naturally when the current level reaches the target conditional
    # failure rate, and use optional safeguards only to prevent wasted work.
    while nF[-1] / n < p:
        if max_levels is not None and L >= max_levels:
            stop_reason = f"Reached max_levels={max_levels} before hitting the target failure rate."
            print(stop_reason)
            break

        L += 1
        print(f'SuS Level = {L}\n===============')
        if d > 2:
            print(_G_cached.cache_info()) 

        # Use partial selection to isolate the best nc seeds without fully
        # sorting the whole level. We only sort the selected seeds because MMA
        # needs the top samples at the front of the level array.
        cutoff = n - nc
        partitioned = np.partition(current_y, cutoff)
        Yi.append((partitioned[cutoff - 1] + partitioned[cutoff]) / 2)

        top_idx = np.argpartition(current_y, -nc)[-nc:]
        top_idx = top_idx[np.argsort(current_y[top_idx])[::-1]]
        
        # Print the new intermediate threshold
        print(f"Threshold for Level {L}: {Yi[L - 1]}")

        # If the threshold stops improving across several levels, the chains are
        # no longer moving the simulation toward the failure domain in a useful
        # way. This is a better safety stop than a hard-coded level cap.
        if len(Yi) > 1 and abs(Yi[-1] - Yi[-2]) < threshold_tol:
            stagnant_levels += 1
        else:
            stagnant_levels = 0

        if stagnant_levels >= stagnation_patience:
            # We are stopping before MMA generates level L, so roll back the
            # bookkeeping for this attempted level. This keeps L, Yi, nF, and
            # the stored level arrays consistent with the last completed level.
            Yi.pop()
            L -= 1
            stop_reason = (
                "\nSTOPPED EARLY BECAUSE THE INTERMEDIATE THRESHOLD STAGNATED "
                f"FOR {stagnation_patience} CONSECUTIVE LEVELS.\n"
            )
            print(stop_reason)
            break

        # MMA only needs the current level with the best nc samples in front.
        # Passing a single-level work array keeps the MMA interface compatible
        # while the full history is stored separately in Python lists.
        mma_input = np.array(current_x, copy=True)
        mma_input[:, :nc] = current_x[:, top_idx]

        next_level = MMA(nc, ns, d, mma_input[:, :, np.newaxis], [Yi[L - 1]], 0)

        # Batch evaluation removes the repeated scalar G() loop for each level.
        current_x = next_level[:, :, 1]
        current_y = G_batch(current_x.T)
        nF.append(int(np.count_nonzero(current_y > YF)))

        x_levels.append(current_x.copy())
        y_levels.append(current_y.copy())
        sorted_levels.append(np.sort(current_y))

        print(f"Number of failure samples at Level {L}: {nF[L]} out of {n}\n")

    if stop_reason is None:
        stop_reason = "Reached target conditional failure rate."

    # Build bi/ccdf once from per-level blocks to avoid repeated concatenation.
    retained_per_level = n - nc
    bi_blocks = []
    ccdf_blocks = []
    for level in range(L):
        bi_blocks.append(sorted_levels[level][:retained_per_level])
        ccdf_blocks.append((prob_ratio * (p ** level))[:retained_per_level])
    bi_blocks.append(sorted_levels[L])
    ccdf_blocks.append(prob_ratio * (p ** L))

    bi = np.concatenate(bi_blocks)
    ccdf = np.concatenate(ccdf_blocks)
    pF = (p ** L) * nF[L] / n
    N = n + n * (1 - p) * L

    x_out = np.stack(x_levels, axis=2)
    y_out = np.vstack(y_levels)

    return {
        'pF': pF,
        'L': L,
        'N': N,
        'Y': y_out,
        'Yi': Yi,
        'nF': nF,
        'x': x_out,
        'ccdf': ccdf,
        'bi': bi,
        'stop_reason': stop_reason
    }
