from general_utils import *
from code_attacker_average.utils import *
from code_attacker_average.tests import *
import cvxpy as cp


def joint_attack_nontargetonly(M_0, pi_t, epsilon, delta, p, costr, costp, d_0):
    states_count = M_0[0]
    actions_count = M_0[1]
    R_0 = M_0[2]
    P_0 = M_0[3]

    V = calc_V_values(M_0, pi_t)  # V^\{pi_t}_0
    rho = calc_rho(M_0, pi_t)  # \rho^{\pi_t}_0
    T = calc_reachtimes(M_0, pi_t)  # T^{pi_t}(s, s')
    B = np.array([V[s] - R_0[s, pi_t[s]] + rho for s in range(states_count)])  # B^{\pi_t}_0
    U = np.zeros((states_count, states_count))
    for s1 in range(states_count):
        for s2 in range(states_count):
            U[s1, s2] = V[s2] + (epsilon * T[s2, s1] if s2 != s1 else 0)

    feasible = True
    P = np.copy(P_0)
    R = np.copy(R_0)
    for s in range(states_count):
        for a in range(actions_count):
            if a != pi_t[s]:
                # d = P(s, a, .), r = R(s,a)
                d = cp.Variable(states_count)
                r = cp.Variable()

                constraints = []
                constraints.append(d >= delta * P_0[s, :, a])
                constraints.append(R_0[s, pi_t[s]] + B[s] - epsilon >= r + d @ U[s, :])
                # being distribution
                constraints.append(d >= 0)
                constraints.append(d @ np.ones(states_count) == 1)
                # print(s, a)
                # print(U[s, :])
                # print(R_0[s, pi_t[s]] + B[s] - R_0[s, a] - epsilon)
                # print(P_0[s, :, a])

                obj = cp.Minimize(costp * cp.norm(d - P_0[s, :, a], 1) + costr * cp.abs(r - R_0[s, a]))
                prob = cp.Problem(obj, constraints)
                prob.solve(solver=cp.ECOS, max_iters=5000)

                if d.value is None:
                    feasible = False

                P[s, :, a] = d.value
                R[s, a] = r.value

    cost = calc_cost(R, R_0, P, P_0, p, costr, costp)

    return (M_0[0], M_0[1], R, P, M_0[4]), cost, feasible


if __name__ == "__main__":
    M_0, pi_t, epsilon, delta, p, costr, costp, d_0 = testMDPv4()
    print("Nontarget")
    M, cost, feasible = joint_attack_nontargetonly(M_0, pi_t, epsilon, delta, p, costr, costp, d_0)
    if feasible:
        print("Cost:", cost)
        print(M[3])
        print("Is correct?", check_results(M, pi_t, epsilon, print_rhos=True))
    else:
        print("Attack Not Feasible")
