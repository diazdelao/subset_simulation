"""
Estimate candidate failure thresholds from a pilot Monte Carlo sample.

This standalone helper draws Gaussian samples from the same input law used by
SuS and reports suggested YF values for target tail probabilities.
"""

import argparse

import numpy as np

from G import G_batch, get_available_test_functions, get_test_function_name, set_test_function


def pilot_thresholds(d, n_pilot=50000, pf_targets=(1e-1, 1e-2, 1e-3, 1e-4), seed=None):
    """
    Estimate sensible YF values from the same Gaussian input law used by SuS.
    """
    rng = np.random.default_rng(seed)
    X_pilot = rng.normal(size=(n_pilot, d))
    Y_pilot = G_batch(X_pilot)

    print("Pilot threshold helper")
    print(f"Test function = {get_test_function_name()}")
    print(f"Dimension = {d}")
    print(f"Pilot sample size = {n_pilot}")
    print(f"mean(G) = {Y_pilot.mean():.6f}")
    print(f"std(G) = {Y_pilot.std():.6f}")

    # Because failure is defined as G(x) > YF, the threshold for a target
    # probability pf is approximately the (1 - pf) quantile of G(X).
    suggested_yf = {}
    for pf in pf_targets:
        yf = np.quantile(Y_pilot, 1 - pf)
        suggested_yf[pf] = yf
        print(f"Suggested YF for P(G > YF) ~= {pf:.0e}: {yf:.6f}")

    print()
    return suggested_yf, Y_pilot


def _parse_pf_targets(raw_targets):
    return tuple(float(target) for target in raw_targets)


def main():
    parser = argparse.ArgumentParser(
        description="Run the pilot threshold helper independently from the main SuS driver."
    )
    parser.add_argument("--d", type=int, default=2, help="Problem dimension.")
    parser.add_argument(
        "--function",
        default=get_test_function_name(),
        choices=get_available_test_functions(),
        help="Registered test function to evaluate."
    )
    parser.add_argument(
        "--n-pilot",
        type=int,
        default=50000,
        help="Number of Gaussian pilot samples."
    )
    parser.add_argument(
        "--pf-target",
        dest="pf_targets",
        nargs="+",
        default=("1e-1", "1e-2", "1e-3", "1e-4"),
        help="One or more target failure probabilities."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for reproducibility."
    )
    args = parser.parse_args()

    set_test_function(args.function)
    pilot_thresholds(
        d=args.d,
        n_pilot=args.n_pilot,
        pf_targets=_parse_pf_targets(args.pf_targets),
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
