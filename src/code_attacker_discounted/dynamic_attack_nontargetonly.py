from general_utils import calc_cost
from code_attacker_discounted.utils import *
from code_attacker_discounted.tests import *
import cvxpy as cp


def dynamic_attack_nontargetonly(M_0, pi_t, epsilon, delta, p, costr, costp, d_0):
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
    for s1 in range(states_count):
        for s2 in range(states_count):
            eta[s1] = 1 - (1 - gamma) * T[:, s1] @ d_0
            U[s1, s2] = gamma * V[s2] + epsilon * gamma / eta[s1] * T[s2, s1]
    # print("eta:", eta)
    # print("U:\n", U)
    feasible = True
    P = np.copy(P_0)
    for s in range(states_count):
        for a in range(actions_count):
            if a != pi_t[s]:
                # P(s, a, .)
                d = cp.Variable(states_count)
                constraints = []
                constraints.append(d >= delta * P_0[s, :, a])
                constraints.append(V[s] + rho - R_0[s, a] - epsilon / eta[s] >= d @ U[s, :])
                # being distribution
                constraints.append(d >= 0)
                constraints.append(d @ np.ones(states_count) == 1)
                # print(s, a)
                # print(U[s, :])
                # print(R_0[s, pi_t[s]] + B[s] - R_0[s, a] - epsilon)
                # print(P_0[s, :, a])

                obj = cp.Minimize(cp.norm(d - P_0[s, :, a], 1))
                prob = cp.Problem(obj, constraints)
                prob.solve(solver=cp.ECOS)

                if d.value is None:
                    feasible = False

                P[s, :, a] = d.value

    cost = None if not feasible else calc_cost(R_0, R_0, P, P_0, p, costr, costp)

    return (M_0[0], M_0[1], M_0[2], P, M_0[4]), cost, feasible


if __name__ == "__main__":
    M_0, pi_t, epsilon, delta, p, costr, costp, d_0 = testMDPv4()
    print("Nontarget")
    M, cost, feasible = dynamic_attack_nontargetonly(M_0, pi_t, epsilon, delta, p, costr, costp, d_0)
    if feasible:
        print(cost)
        print(M[3])
        print("Is correct?", check_results(M, pi_t, epsilon, d_0, print_rhos=True))
    else:
        print("Attack Not Feasible")
