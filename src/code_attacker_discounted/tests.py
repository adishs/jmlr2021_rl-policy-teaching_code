import numpy as np


def testMDPv1():
    states_count = 2
    actions_count = 2

    P_0 = np.zeros((states_count, states_count, actions_count))
    # a0: change state
    P_0[:, :, 0] = np.matrix(
        [[0.5, 0.5],
         [0.5, 0.5]])
    # a1: wp 0.5 stay in place, wp 0.5 change state
    P_0[:, :, 1] = np.matrix(
        [[0, 1],
         [1, 0]])

    # s0 is a lot better
    R_0 = np.matrix([
        [100, 100],
        [0, 0]
    ])

    gamma = 0.5

    M_0 = (states_count, actions_count, R_0, P_0, gamma)

    d_0 = np.zeros(states_count)
    d_0[0] = 1

    pi_t = [1, 1]
    epsilon = 17
    delta = 0.0001
    p = 1
    costr = 1
    costp = 1

    return M_0, pi_t, epsilon, delta, p, costr, costp, d_0


def testMDPv2():
    states_count = 4
    actions_count = 2

    P_0 = np.zeros((states_count, states_count, actions_count))
    # a0: change state
    P_0[:, :, 0] = np.matrix(
        [[0.4, 0.2, 0.2, 0.2],
         [0.725, 0.025, 0.225, 0.025],
         [0.025, 0.725, 0.025, 0.225],
         [0.2, 0.2, 0.4, 0.2]])
    # a1: wp 0.5 stay in place, wp 0.5 change state
    P_0[:, :, 1] = np.matrix(
        [[0.2, 0.4, 0.2, 0.2],
         [0.225, 0.025, 0.725, 0.025],
         [0.025, 0.225, 0.025, 0.725],
         [0.2, 0.2, 0.2, 0.4]])

    # s0 is a lot better
    R_0 = np.matrix([
        [0.8, 0.8],
        [0, 0],
        [0, 0],
        [0.8, 0.8]
    ])

    gamma = 1

    M_0 = (states_count, actions_count, R_0, P_0, gamma)

    d_0 = np.zeros(states_count)
    d_0[0] = 1

    pi_t = [1, 1, 1, 1]
    epsilon = 0.01
    delta = 0.0001
    p = 1
    costr = 1
    costp = 1

    return M_0, pi_t, epsilon, delta, p, costr, costp, d_0


def testMDPv3():
    states_count = 3
    actions_count = 2

    P_0 = np.zeros((states_count, states_count, actions_count))

    P_0[:, :, 0] = np.matrix(
        [[0.466666666667, 0.266666666667, 0.266666666667],
         [0.733333333333, 0.033333333333, 0.233333333333],
         [0.266666666667, 0.466666666667, 0.266666666667]])

    P_0[:, :, 1] = np.matrix(
        [[0.266666666667, 0.466666666667, 0.266666666667],
         [0.233333333333, 0.033333333333, 0.733333333333],
         [0.266666666667, 0.266666666667, 0.466666666667]])

    # P_0_copy = copy.deepcopy(P_0)
    # P_0_copy[:, :, 0] = [0.05, 0.9, 0.05]

    R_0 = np.matrix([
        [0.8, 0.8],
        [-1, -1],
        [0.8, 0.8]
    ])

    gamma = 1

    M_0 = (states_count, actions_count, R_0, P_0, gamma)

    d_0 = np.zeros(states_count)
    d_0[0] = 1

    pi_t = [1, 1, 1]
    epsilon = 0.1
    delta = 0.0001
    p = 1
    costr = 1
    costp = 1

    return M_0, pi_t, epsilon, delta, p, costr, costp, d_0

def testMDPv4():
    states_count = 4
    actions_count = 2
    success_prob = 0.9
    unif_prob = (1 - success_prob) / (states_count - 1)
    R = np.array([[-2.5, -2.5],
                  [0.5, 0.5],
                  [0.5, 0.5],
                  [-0.5, -0.5]])
    P = np.zeros((states_count, states_count, actions_count))
    P[:, :, 0] = np.array(
        [[success_prob, unif_prob, unif_prob, unif_prob],
         [success_prob, unif_prob, unif_prob, unif_prob],
         [unif_prob, success_prob, unif_prob, unif_prob],
         [unif_prob, unif_prob, success_prob, unif_prob]]
    )
    P[:, :, 1] = np.array(
        [[unif_prob, success_prob, unif_prob, unif_prob],
         [unif_prob, unif_prob, success_prob, unif_prob],
         [unif_prob, unif_prob, unif_prob, success_prob],
         [unif_prob, unif_prob, unif_prob, success_prob]]
    )
    gamma = 0.8

    d_0 = np.zeros(states_count)
    d_0[0] = 1

    M_0 = (states_count, actions_count, R, P, gamma)

    pi_t = [1, 1, 1, 1]
    epsilon = 0.1
    delta = 0.0001
    p = 1
    costr = 1
    costp = 1

    return M_0, pi_t, epsilon, delta, p, costr, costp, d_0


if __name__ == "__main__":
    M_0, pi_t, epsilon, delta, p, costr, costp, d_0 = testMDPv4()

    print(M_0)
    print(pi_t)
    print(epsilon)
    print(delta)
    print(p)
    print(costr)
    print(costp)
    print(d_0)
