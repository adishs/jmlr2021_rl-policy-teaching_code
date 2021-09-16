from general_utils import calc_cost
from code_attacker_discounted.utils import *
from code_attacker_discounted.tests import *
import cvxpy as cp


def joint_attack_nontargetonly_theoretical(M_0, pi_t, epsilon, delta, p, costr, costp, d_0):
    states_count = M_0[0]
    actions_count = M_0[1]
    R_0 = M_0[2]
    P_0 = M_0[3]
    gamma = M_0[4]

    V = calc_V_values(M_0, pi_t, d_0)  # V^\{pi_t}_0
    # print("V:", V)
    rho = calc_rho(M_0, pi_t, d_0)  # \rho^{\pi_t}_0
    # print("Rho:", rho)
    T = calc_reachtimes(M_0, pi_t)  # T^{pi_t}(s, s')
    # print("T:\n", T)
    U = np.zeros((states_count, states_count))
    eta = np.ones(states_count)
    chi = calc_chi(M_0, pi_t, epsilon, d_0)
    for s1 in range(states_count):
        for s2 in range(states_count):
            eta[s1] = 1 - (1 - gamma) * T[:, s1] @ d_0
            U[s1, s2] = gamma * V[s2] + epsilon * gamma / eta[s1] * T[s2, s1]
    # print("eta:", eta)
    # print("U:\n", U)
    feasible = True
    P = np.copy(P_0)
    R = np.copy(R_0)
    for s in range(states_count):
        for a in range(actions_count):
            if a != pi_t[s]:
                n_s = sorted([i for i in range(states_count)], key=lambda i: -U[s, i])
                S = [0]
                C = [0]
                for i in range(states_count):
                    S.append(S[i] + (1 - delta) * P_0[s, n_s[i], a] * (U[s, n_s[i]] - U[s, n_s[states_count - 1]]))
                    C.append(C[i] + 2 * (1 - delta) * P_0[s, n_s[i], a])
                S.append(np.inf)
                max_i = states_count - 2
                for i in range(states_count - 1):
                    if S[i] >= chi[s, a] or costr * (U[s, n_s[i]] - U[s, n_s[states_count - 1]]) <= 2 * costp:
                        max_i = i - 1
                        break
                # print(U[s,:], S, max_i)
                if max_i >= 0:
                    P[s, 0:max_i, a] = delta * P_0[s, 0:max_i, a]
                    if S[max_i + 1] >= chi[s, a]:
                        P[s, max_i, a] = P_0[s, max_i, a] - (chi[s, a] - S[max_i]) / (U[s, n_s[i]] - U[s, n_s[states_count - 1]])
                    else:
                        P[s, max_i, a] = delta * P_0[s, 0:max_i, a]
                        R[s, a] = R_0[s, a] - (chi[s, a] - S[max_i + 1])
                else:
                    R[s, a] = R_0[s, a] - chi[s, a]
                P[s, states_count - 1, a] = 1 - np.sum(P[s, 0:states_count - 1, a])


    cost = None if not feasible else calc_cost(R, R_0, P, P_0, p, costr, costp)

    return (M_0[0], M_0[1], R, P, M_0[4]), cost, feasible


if __name__ == "__main__":
    M_0, pi_t, epsilon, delta, p, costr, costp, d_0 = testMDPv4()
    print("Nontarget")
    M, cost, feasible = joint_attack_nontargetonly_theoretical(M_0, pi_t, epsilon, delta, p, costr, costp, d_0)
    if feasible:
        print("Cost:", cost)
        print("R:\n", M[2])
        print("P:\n", M[3])
        print("Is correct?", check_results(M, pi_t, epsilon, d_0, print_rhos=True))
    else:
        print("Attack Not Feasible")
