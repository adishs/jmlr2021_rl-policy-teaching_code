import numpy as np


def calc_cost(R, R_0, P, P_0, p, costr, costp):
    states_count = R.shape[0]
    actions_count = R.shape[1]
    comps = []
    for s in range(states_count):
        for a in range(actions_count):
            comps.append(costp * np.linalg.norm(P[s, :, a] - P_0[s, :, a], ord=1) + costr * abs(R[s, a] - R_0[s, a]))
    l1diffs = np.array(comps)
    return np.linalg.norm(l1diffs, ord=p)


def neighbor(pi, s, a):
    n = np.copy(pi)
    n[s] = a
    return n


def calc_A_mu(M, pi, mu):
    states_count = M[0]
    actions_count = M[1]
    A_mu = np.zeros(states_count * actions_count)
    for s in range(states_count):
        for a in range(actions_count):
            if a == pi[s]:
                A_mu[s * actions_count + a] = mu[s]
    return A_mu


