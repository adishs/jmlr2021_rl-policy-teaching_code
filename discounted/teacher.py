import numpy as np
import sys
sys.path.append('../')
import MDPSolver
sys.path.append('../src')
from code_attacker_discounted.reward_attack_general import *
from code_attacker_discounted.reward_attack_nontargetonly import *
from code_attacker_discounted.dynamic_attack_nontargetonly import *
from code_attacker_discounted.joint_attack_nontargetonly import *
from code_attacker_discounted.utils import *

import copy

class teacher:
    def __init__(self, env, target_pi, epsilon, p, delta, costr, costp, d_0, teacher_type, pool=None):
        self.env = env
        self.target_pi = target_pi
        self.epsilon = epsilon
        self.p = p
        self.delta = delta
        self.costr = costr
        self.costp = costp
        self.d_0 = d_0
        self.teacher_type = teacher_type
        self.pool = pool
        self.M_0 = (env.n_states, env.n_actions, env.reward, env.T)

    #enddef

    def get_target_M(self,  M_0):
        if self.teacher_type == "general_attack_on_reward":
            return self.general_attack_on_reward(M_0, self.target_pi, self.epsilon, self.delta, self.p, self.costr, self.costp, self.d_0)
        elif self.teacher_type == "non_target_attack_on_reward":
            return self.non_target_attack_on_reward(M_0, self.target_pi, self.epsilon, self.delta, self.p, self.costr, self.costp, self.d_0)
        elif self.teacher_type == "non_target_attack_on_dynamics":
            return self.non_target_attack_on_dynamics(M_0, self.target_pi, self.epsilon, self.delta, self.p, self.costr, self.costp, self.d_0)
        elif self.teacher_type == "general_attack_on_dynamics":
            return self.general_attack_on_dynamics(M_0, self.target_pi, self.epsilon, self.delta, self.p, self.costr, self.costp, self.d_0)
        elif self.teacher_type == "non_target_attack_joint":
            return self.non_target_attack_joint(M_0, self.target_pi, self.epsilon, self.delta, self.p, self.costr, self.costp, self.d_0)
        elif self.teacher_type == "general_attack_joint":
            return self.general_attack_joint(M_0, self.target_pi, self.epsilon, self.delta, self.p, self.costr, self.costp, self.d_0)

        else:
            print("Wrong teacher type!!---", self.teacher_type)
            print("Please choose one of the following:")
            print("{}\n{}".format("general_attack_on_reward", "non_target_attack_on_reward"
                                  "general_attack_on_dynamics", "non_target_attack_on_dynamics"))
            exit(0)
        #enddef

    def general_attack_on_reward(self, M_0, pi_t, epsilon, delta, p, costr, costp, d_0):
        (n_states, n_action, R_T, P, gamma), cost, feasible = reward_attack_general(M_0, pi_t, epsilon, p,  costr, costp, d_0)
        return (n_states, n_action, R_T, P, gamma), cost, feasible
    #enddef

    def non_target_attack_on_reward(self, M_0, pi_t, epsilon, delta, p, costr, costp, d_0):
        (n_states, n_action, R_T, P, gamma), cost, feasible = reward_attack_nontargetonly(M_0, pi_t, epsilon, p, costr, costp, d_0)
        return (n_states, n_action, R_T, P, gamma), cost, feasible
    #enddef

    def non_target_attack_on_dynamics(self, M_0, pi_t, epsilon, delta, p, costr, costp, d_0):
        M, cost, feasible = dynamic_attack_nontargetonly(M_0, pi_t, epsilon, delta, p, costr, costp, d_0)
        n_states, n_action, R, P_T, gamma = M[0], M[1], M[2], M[3], M[4]
        return (n_states, n_action, R, P_T, gamma), cost, feasible
    #enddef

    def general_attack_on_dynamics(self, M_0, pi_t, epsilon, delta, p, costr, costp, d_0):
        feasible = False
        num_states, num_actions, R, P_in, gamma = M_0[0], M_0[1], M_0[2], M_0[3], M_0[4]
        #
        # pool = self.generate_pool(num_states, num_actions, R, P_in, pi_t, alpha, beta, n_copies_of_N)
        pool = self.pool
        pool_of_solutions, _, _ = self.solve_pool_dynamics(num_states, num_actions,
                                                R, gamma, epsilon, delta, p, costr, costp, d_0,  pool)
        if len(pool_of_solutions) > 0:
            feasible = True
            closest_P = self.get_P_with_smallest_norm(pool_of_solutions, P_in, p)

            n_states, n_action, R, P_T, gamma = num_states, num_actions, R, closest_P, gamma
            M_out = (num_states, num_actions, R, closest_P, gamma)
            # cost = self.cost(M_0, M_out, p=np.inf)

            return (n_states, n_action, R, P_T, gamma), None, True
        else:
            return None, None, False

    # def non_target_attack_on_dynamics_upperbound(self,M_0, pi_t, epsilon, delta):
    #     M, feasible = dynamic_attack_nontargetonly_upperbound(M_0, pi_t, epsilon, delta)
    #     n_states, n_action, R, P_T = M[0], M[1], M[2], M[3]
    #     return n_states, n_action, R, P_T, feasible
    # # enddef

    def non_target_attack_joint(self, M_0, pi_t, epsilon, delta, p, costr, costp, d_0):
        M, cost, feasible = joint_attack_nontargetonly(M_0, pi_t, epsilon, delta, p, costr, costp, d_0)
        n_states, n_action, R, P_T, gamma = M[0], M[1], M[2], M[3], M[4]
        return (n_states, n_action, R, P_T, gamma), cost, feasible
    # enddef

    def general_attack_joint(self, M_0, pi_t, epsilon, delta, p, costr, costp, d_0):
        R = M_0[2]
        P = M_0[3]
        M_R_attack, cost_R_attack, feasible_R_attack = self.general_attack_on_reward(M_0, pi_t, epsilon, delta, p, costr, costp, d_0)
        M_P_attack, cost_P_attack, feasible_P_attack = self.general_attack_on_dynamics(M_0, pi_t, epsilon, delta, p, costr, costp, d_0)

        if feasible_R_attack:
            R_hat = M_R_attack[2]
        else:
            R_hat = M_0[2]

        if feasible_P_attack:
            P_hat = M_P_attack[3]
        else:
            P_hat = M_0[3]

        alpha_R_array = np.round(np.arange(0, 1.001, 0.1), 3)
        alpha_P_array = np.round(np.arange(0, 1.001, 0.1), 3)

        tilde_M_array = []
        for alpha_R in alpha_R_array:
            for alpha_P in alpha_P_array:
                tilde_R = alpha_R * R + (1 - alpha_R) * R_hat
                tilde_P = alpha_P * P + (1 - alpha_P) * P_hat
                M_tilde = (M_0[0], M_0[1], tilde_R, tilde_P, M_0[4])
                tilde_M_array.append(M_tilde)


        #######
        best_M_tilde, minimum_cost, feasible = self.get_M_tilde_using_join_attack_with_smallest_norm(M_0, pi_t, epsilon, delta,
                                                         p, costr, costp, d_0, tilde_M_array)
        return best_M_tilde, minimum_cost, feasible
    #enddef

    def get_M_tilde_using_join_attack_with_smallest_norm(self, M_0, pi_t, epsilon, delta,
                                                         p, costr, costp, d_0, tilde_M_array):
        minimum_cost = np.inf
        best_M_tilde = None
        feasible_final = None
        for tilde_M in tilde_M_array:
            M_out, _, feasible = self.non_target_attack_joint(tilde_M, pi_t, epsilon, delta,
                                                         p, costr, costp, d_0)
            cost = self.cost(M_0, M_out, p)
            if minimum_cost > cost and feasible:
                minimum_cost = copy.deepcopy(cost)
                best_M_tilde = copy.deepcopy(M_out)
                feasible_final = copy.deepcopy(feasible)
        return best_M_tilde, minimum_cost, feasible_final
    #enddef

    def solve_pool_dynamics(self, num_states, num_actions,
                                                R, gamma, epsilon, delta, p, costr, costp, d_0,  pool):
        pool_of_solved_P = []
        pool_of_infeasible_P = []
        pool_of_solutions = []
        target_pi = self.target_pi
        for P in pool:
            M = (num_states, num_actions, R, P, gamma)
            M_t, cost, feasible = self.non_target_attack_on_dynamics(M, target_pi, epsilon, delta, p=p,
                                                                     costr=costr, costp=costp, d_0=d_0)
            if feasible:
                pool_of_solved_P.append(P)
                pool_of_solutions.append(M_t[3])
            else:
                pool_of_infeasible_P.append(P)
        return pool_of_solutions, pool_of_solved_P, pool_of_infeasible_P
    #enddef

    def solve_pool_joint(self, num_states, num_actions,
                                                R, gamma, epsilon, delta, p, costr, costp, d_0,  pool):
        pool_of_solved_P = []
        pool_of_infeasible_P = []
        pool_of_solutions = []
        target_pi = self.target_pi
        for P in pool:
            M = (num_states, num_actions, R, P, gamma)
            M_t, cost, feasible = self.non_target_attack_joint(M, target_pi, epsilon, delta, p=p,
                                                                     costr=costr, costp=costp, d_0=d_0)
            if feasible:
                pool_of_solved_P.append(P)
                pool_of_solutions.append(M_t[3])
            else:
                pool_of_infeasible_P.append(P)
        return pool_of_solutions, pool_of_solved_P, pool_of_infeasible_P
    #enddef

    def get_P_with_smallest_norm(self, pool_of_solutions_P, P_0, p):
        minimum = np.inf
        P_closest = None
        for P in pool_of_solutions_P:
            value = self.norm_p(P, P_0, p)
            if value < minimum:
                minimum = copy.deepcopy(value)
                P_closest = copy.deepcopy(P)
        return P_closest
    #enddef



    def norm_p(self, P, P_0, p):
        P_s_a = np.zeros((P.shape[0], P.shape[2]))
        for s in range(P.shape[0]):
            for a in range(P.shape[2]):
                P_s_a[s, a] = np.sum(np.abs(P[s, :, a]-P_0[s, :, a]))
                # P_s_a[s, a] = np.max(np.abs(P[s, :, a] - P_0[s, :, a]))

        return np.linalg.norm(P_s_a.flatten(), ord=p)
    #enddef

    # def cost(self, M_0, M_t, p):
    #     return np.linalg.norm((M_0[2]-M_t[2]).flatten(), ord=p) + self.norm_p(M_0[3], M_t[3], p=p)
    # #enddef

    def cost(self, M_0, M_t, p):

        R_0 = M_0[2]
        P_0 = M_0[3]

        R_t = M_t[2]
        P_t = M_t[3]

        states_count = R_0.shape[0]
        actions_count = R_0.shape[1]
        comps = []
        for s in range(states_count):
            for a in range(actions_count):
                comps.append(
                    self.costp * np.linalg.norm(P_t[s, :, a] - P_0[s, :, a], ord=1) + self.costr * abs(R_t[s, a] - R_0[s, a]))
        l1diffs = np.array(comps)
        return np.linalg.norm(l1diffs, ord=p)
#enddef

def normalize(vector):
    return vector/sum(vector)
#enddef

def create_perturb_P_for_target(num_states, num_actions, R, P_in, gamma, d_0, pi, alpha, beta, N):
    P_out = copy.deepcopy(P_in)
    for i in range(N):
        s = np.random.choice(np.arange(0, num_states, dtype="int"), size=1)[0]
        s_prime = np.random.choice(np.arange(0, num_states, dtype="int"), size=1)[0]

        P_tmp = copy.deepcopy(P_out)
        P_tmp[s, s_prime, pi[s]] = P_tmp[s, s_prime, pi[s]] + alpha
        P_tmp[s, :, pi[s]] = normalize(P_tmp[s, :, pi[s]])

        M_tmp = (num_states, num_actions, R, P_tmp, gamma)
        rho_tmp = calc_rho(M_tmp, pi, d_0)
        M_out = (num_states, num_actions, R, P_out, gamma)
        rho_out = calc_rho(M_out, pi, d_0)

        if (rho_tmp - rho_out) > beta:
            P_out = copy.deepcopy(P_tmp)

    return P_out
#enddef

def generate_pool(num_states, num_actions, R, P_in, gamma, d_0, pi, alpha=0.1, beta=0.0001, n_copies_of_N=5):
    N_array = [0, 10, 20, 30, 40, 50, 60, 80, 100, 150, 200, 250, 300]
    pool = []
    for n_copy in range(n_copies_of_N):
        for N in N_array:
            P_out = create_perturb_P_for_target(num_states, num_actions, R, P_in, gamma, d_0, pi, alpha, beta, N)
            pool.append(P_out)
    return pool
#enddef



if __name__ == "__main__":
    pass