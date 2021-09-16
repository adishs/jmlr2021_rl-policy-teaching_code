from general_utils import *
from code_attacker_average.tests import *
from code_attacker_average.utils import *


def reward_attack_nontargetonly(M_0, pi_t, epsilon, p, costr, costp, d_0):
    R_0 = M_0[2]
    P_0 = M_0[3]
    R = R_0 - calc_chi(M_0, pi_t, epsilon)
    feasible = True
    cost = calc_cost(R, R_0, P_0, P_0, p, costr, costp)
    return (M_0[0], M_0[1], R, M_0[3], M_0[4]), cost, feasible


if __name__ == "__main__":
    M_0, pi_t, epsilon, delta, p, costr, costp, d_0 = testMDPv4()

    M, cost, feasible = reward_attack_nontargetonly(M_0, pi_t, epsilon, p, costr, costp, d_0)

    print("Cost", cost)
    print("Non-target R:\n", M[2])
    print("Is correct?", check_results(M, pi_t, epsilon, print_rhos=True))
