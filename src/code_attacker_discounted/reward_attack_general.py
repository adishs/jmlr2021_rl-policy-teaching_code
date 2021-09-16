import cvxpy as cp

from general_utils import *
from code_attacker_discounted.tests import *
from code_attacker_discounted.utils import *


def reward_attack_general(M_0, pi_t, epsilon, p, costr, costp, d_0):
    states_count = M_0[0]
    actions_count = M_0[1]
    R_0 = M_0[2]
    P_0 = M_0[3]
    A_pi_t_mu = calc_A_mu(M_0, pi_t, calc_mu(M_0, pi_t, d_0))
    R = cp.Variable(states_count * actions_count)
    constraints = []
    for s in range(states_count):
        for a in range(actions_count):
            if a != pi_t[s]:
                pi = neighbor(pi_t, s, a)
                mu = calc_mu(M_0, pi, d_0)
                A_mu = calc_A_mu(M_0, pi, mu)
                constraints.append(R @ A_mu <= R @ A_pi_t_mu - epsilon)

    R_0_vector = np.asarray(R_0).reshape(-1)
    obj = cp.Minimize(cp.norm(R - R_0_vector, p))
    prob = cp.Problem(obj, constraints)
    prob.solve()
    R = R.value.reshape((M_0[0], M_0[1]))

    feasible = True
    cost = calc_cost(R, R_0, P_0, P_0, p, costr, costp)
    return (M_0[0], M_0[1], R, M_0[3], M_0[4]), cost, feasible


if __name__ == "__main__":
    M_0, pi_t, epsilon, delta, p, costr, costp, d_0 = testMDPv4()

    M, cost, feasible = reward_attack_general(M_0, pi_t, epsilon, p, costr, costp, d_0)
    print("Cost:", cost)
    print("General R:\n", M[2])
    print("Is correct?", check_results(M, pi_t, epsilon, d_0, print_rhos=True))
