import itertools

import numpy as np

from code_attacker_discounted.tests import *
from general_utils import calc_A_mu, neighbor


def calc_mu(M, pi, d_0):
    gamma = M[4]
    P = M[3]
    states_count = M[0]
    T = np.zeros((states_count, states_count))
    for s1 in range(states_count):
        for s2 in range(states_count):
            T[s1, s2] = P[s1, s2, pi[s1]]

    if gamma == 1:
        A = np.zeros((states_count, states_count))
        A[0:states_count - 1, :] = np.transpose(T - np.identity(states_count))[1:, :]
        A[-1, :] = np.ones(states_count)

        b = np.zeros(states_count)
        b[-1] = 1
    else:
        A = np.transpose(np.identity(states_count) - gamma * T)
        b = (1 - gamma) * d_0

    return np.linalg.solve(A, b)


def calc_reachtimes(M, pi):
    P = M[3]
    gamma = M[4]
    states_count = M[0]
    T = np.zeros((states_count, states_count))
    for x in range(states_count):
        for y in range(states_count):
            T[x, y] = P[x, y, pi[x]]

    reach_times = np.zeros((states_count, states_count))
    for s2 in range(states_count):
        A = np.delete(T, s2, 1)
        A = np.delete(A, s2, 0)
        h = np.linalg.inv(np.identity(states_count - 1) - gamma * A) @ np.ones(states_count - 1)
        h = np.insert(h, s2, 0)
        reach_times[:, s2] = h

    return reach_times


def calc_chi(M_0, pi_t, epsilon, d_0):
    states_count = M_0[0]
    actions_count = M_0[1]
    R_0 = M_0[2]
    R_0_vector = np.asarray(R_0).reshape(-1)
    rho_t = calc_A_mu(M_0, pi_t, calc_mu(M_0, pi_t, d_0)) @ R_0_vector
    chi = np.zeros((states_count, actions_count))
    for s in range(states_count):
        for a in range(actions_count):
            if a != pi_t[s]:
                pi = neighbor(pi_t, s, a)
                mu = calc_mu(M_0, pi, d_0)
                rho = calc_A_mu(M_0, pi, mu) @ R_0_vector
                chi[s, a] = max((rho - rho_t + epsilon) / mu[s], 0)
    return chi


def calc_V_values(M, pi, d_0):
    states_count = M[0]
    actions_count = M[1]
    gamma = M[4]
    R = M[2]
    P = M[3]
    T = np.zeros((states_count, states_count))
    for x in range(states_count):
        for y in range(states_count):
            T[x, y] = P[x, y, pi[x]]

    R_pi = np.zeros(states_count)
    for x in range(states_count):
        R_pi[x] = R[x, [pi[x]]]

    rho = calc_rho(M, pi, d_0)

    if gamma == 1:
        A = np.zeros((states_count, states_count))
        A[0:states_count - 1, :] = (np.identity(states_count) - T)[1:, :]
        A[states_count - 1, 0] = 1

        b = np.zeros(states_count)
        b[0:states_count - 1] = (R_pi - rho * np.ones(states_count))[1:]
    else:
        A = np.identity(states_count) - gamma * T
        b = R_pi - rho * np.ones(states_count)

    return np.linalg.solve(A, b)


def calc_rho(M, pi, d_0):
    R_0 = M[2]
    R_0_vector = np.asarray(R_0).reshape(-1)
    return calc_A_mu(M, pi, calc_mu(M, pi, d_0)) @ R_0_vector


if __name__ == "__main__":
    M_0, pi_t, epsilon, delta, p, costr, costp, d_0 = testMDPv1()
    P = M_0[3]
    states_count = M_0[0]
    T = np.zeros((states_count, states_count))
    for s1 in range(states_count):
        for s2 in range(states_count):
            T[s1, s2] = P[s1, s2, pi_t[s1]]

    print(T)
    print(d_0)
    print(M_0[4])
    print(calc_mu(M_0, pi_t, d_0))


def check_results(M, pi_t, epsilon, d_0, print_rhos=False):
    states_count = M[0]
    actions_count = M[1]
    rho_t = calc_rho(M, pi_t, d_0)
    all_policies = [np.array(x) for x in itertools.product(range(actions_count), repeat=states_count)]
    result = True
    for pi in all_policies:
        rho = calc_rho(M, pi, d_0)
        if print_rhos:
            if np.array_equal(pi, pi_t):
                print(pi, rho, "---> target")
            else:
                print(pi, rho)

        if rho > rho_t - epsilon + 0.001 and not np.array_equal(pi, pi_t):
            result = False
    return result
