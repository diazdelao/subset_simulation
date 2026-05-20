"""
Run the main Subset Simulation workflow for a chosen test function.

Usage:
    python master_sus.py --f ackley --d 2 --yf -0.01 --n 5000 --p 0.1

Current command-line arguments:
    --f   Registered test function to evaluate. Default: ackley
    --d   Problem dimension. Default: 2
    --yf  Failure threshold. Default: -0.01
    --n   Number of samples. Default: 5000
    --p   Level probability. Default: 0.1

This script parses command-line arguments, generates the initial Gaussian
sample, runs SuS, reports the estimated failure probability, and plots the
level samples for 2D problems.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import time
#from functools import lru_cache

from G import (
    get_available_test_functions,
    get_test_function_name,
    set_test_function,
)
from sus import sus  
from plotsus2d import plotsus2d  


def _parse_explicit_int(value):
    try:
        return int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"invalid integer value '{value}'. Use explicit integer notation like 10000, not scientific notation like 1e4."
        ) from exc


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Subset Simulation with a selectable test function."
    )
    # Keep the parsed attribute name as args.function even though the CLI flag
    # is shortened to --f for convenience. dest="function" is required to
    # override the default argument name.
    parser.add_argument(
        "--f",
        dest="function",
        default="ackley",
        choices=get_available_test_functions(),
        help="Registered test function to evaluate.",
    )
    parser.add_argument("--d", type=int, default=2, help="Problem dimension.")
    parser.add_argument("--yf", type=float, default=-0.01, help="Failure threshold.")
    parser.add_argument(
        "--n",
        type=_parse_explicit_int,
        default=int(5e3),
        help="Number of samples.",
    )
    parser.add_argument("--p", type=float, default=0.1, help="Level probability.")
    return parser.parse_args()


def main():
    args = parse_args()

    set_test_function(args.function)
    print(f"Using test function: {get_test_function_name()}")

    # Initial Gaussian samples
    d = args.d
    YF = args.yf
    n = args.n
    p = args.p
    X = np.zeros((d, n, 1))
    X[:, :, 0] = multivariate_normal.rvs(mean=np.zeros(d), cov=np.eye(d), size=n).T   
    
    # Run SuS
    # ================================================================
    start_time = time.time()   
    S = sus(X, n, p, d, YF)  
    pF = S['pF']
    print(f"Probability of Failure = {pF}")
    elapsed_time = time.time() - start_time  
    print(f"Elapsed time: {elapsed_time:.2f} seconds")
    print(f"Function evaluations = {S['N']}\n")
    #print(G.cache_info())

    # Plot levels from 1 to L (exclude level 0)
    # ================================================================
    Xall = S['x']  # Get all levels of samples from SuS
    # Generate the plot for levels (EXCLUDE level 0)
    if d == 2 and S['L'] > 0:
        plotsus2d(Xall[:,:,1:], S['L'], S['Yi'], d)
        plt.show()  

    #plt.plot(S['bi'], S['ccdf'])
    #plt.show()

if __name__ == "__main__":
    main()
