import numpy as np
from G import G


def MMA(nc, ns, d, x, Yi, L):
    z = np.zeros((d, nc, ns + 1))
    z[:, :, 0] = x[:, :nc, L]

    for j in range(nc):
        for m in range(ns):
            current = z[:, j, m]

            # Propose all d coordinates in one shot instead of looping over k.
            proposal = current + np.random.normal(0, 1, size=d)

            # Equivalent Metropolis ratio for the standard normal target.
            ratio = np.exp((current ** 2 - proposal ** 2) / 2)

            # One accept/reject decision per coordinate.
            accept = np.random.uniform(0, 1, size=d) < np.minimum(1.0, ratio)

            # Keep proposed coordinates where accepted; otherwise keep current ones.
            q = np.where(accept, proposal, current)

            if G(tuple(q)) > Yi[L]:
                z[:, j, m + 1] = q
            else:
                z[:, j, m + 1] = current

    new_x_layer = np.zeros((d, x.shape[1], 1))
    x = np.concatenate((x, new_x_layer), axis=2)

    # Flatten (chain, state) into the sample axis expected by x[:, :, L + 1].
    x[:, : nc * (ns + 1), L + 1] = z.transpose(0, 1, 2).reshape(d, nc * (ns + 1))

    return x
