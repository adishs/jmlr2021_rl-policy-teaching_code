import numpy as np
import copy
import sys
sys.path.append('../')
import os
import plot_chain
import MDPSolver
import learner
import teacher
import env_chain
import matplotlib.pyplot as plt



class teaching:
    def __init__(self, env, target_pi, epsilon, p, delta, costr, costp, d_0,
                 T_UCRL, alpha_UCRL, attackers_cost_y_axis, teacher_type):
        self.env = env
        self.reward_no_attack = env.reward
        _, self.pi_no_attack, _ = MDPSolver.averaged_valueIteration(env, env.reward)
        self.target_pi = target_pi
        self.epsilon = epsilon
        self.p = p
        self.delta = delta
        self.costr = costr
        self.costp = costp
        self.d_0 = d_0
        self.T_UCRL = T_UCRL
        self.attackers_cost_y_axis = attackers_cost_y_axis
        self.alpha_UCRL = alpha_UCRL
        self.teacher_type = teacher_type

        self.M_0 = env.get_M_0()
        self.pool = teacher.generate_pool(self.M_0[0], self.M_0[1],self.M_0[2], self.M_0[3], M_0[4],
                             target_pi)

        self.teacher = teacher.teacher(env, target_pi, epsilon, p, delta, costr, costp, d_0,
                                       teacher_type, self.pool)
        self.learner = None

    #enddef

    def modify_env(self, env, M):
        env_t = copy.deepcopy(env)
        env_t.reward = M[2]
        env_t.T = M[3]
        env_t.T_sparse_list = env_t.get_transition_sparse_list()
        return env_t
    #enddef

    def end_to_end_teaching(self):
        M_0 = env.get_M_0()
        M_t, _, feasible = self.teacher.get_target_M(M_0)
        env_t = self.modify_env(self.env, M_t)
        print(env_t.reward)
        self.learner = learner.learner(env_t)
        self.learner.UCRL(alpha=self.alpha_UCRL, n_rounds=self.T_UCRL,
                          pi_no_attack=self.pi_no_attack,
                          M_0=M_0, M_t=M_t, cost_y_axis=self.attackers_cost_y_axis,
                          costr=self.costr, costp=self.costp)
    #enddef


#enddef

def write_into_file(accumulator, exp_iter, teacher_type="online_teaching"):
    directory = 'results/{}'.format(teacher_type)
    filename = "convergence" + '_' + str(exp_iter) + '.txt'
    if not os.path.isdir(directory):
        os.makedirs(directory)
    filepath = directory + '/' + filename
    print("output file name  ", filepath)
    f = open(filepath, 'w')
    for key in accumulator:
        f.write(key + '\t')
        temp = list(map(str, accumulator[key]))
        for j in temp:
            f.write(j + '\t')
        f.write('\n')
    f.close()
#enddef


def accumulator_function(learner, dict_accumulator, teacher_type, p):
    for key in learner.accumulator_dict:
        accumulator_key = str(key) + "_{}_p={}".format(teacher_type, p)
        if accumulator_key in dict_accumulator:
            dict_accumulator[accumulator_key] += np.array(learner.accumulator_dict[key])
        else:
            dict_accumulator[accumulator_key] = np.array(learner.accumulator_dict[key])
    return dict_accumulator
#enddef


def calculate_average(dict_accumulator, number_of_iterations):
    for key in dict_accumulator:
        dict_accumulator[key] = dict_accumulator[key]/number_of_iterations
    return dict_accumulator
#enddef


########################################
if __name__ == "__main__":

    number_of_iterations = 20
    accumulator_dict = {}

    for iter_num in range(1, number_of_iterations + 1):

        gamma = 1
        n_states = 4
        n_actions = 2
        success_prob = 0.9
        unif_prob = (1 - success_prob)/(n_states-1)
        R = np.array([[-2.5, -2.5],
                      [0.5, 0.5],
                      [0.5, 0.5],
                      [-0.5, -0.5]])
        P = np.zeros((n_states, n_states, n_actions))
        P[:,:,0] = np.array(
            [[success_prob, unif_prob, unif_prob, unif_prob],
                    [success_prob, unif_prob, unif_prob, unif_prob],
                    [unif_prob, success_prob, unif_prob, unif_prob],
                    [unif_prob, unif_prob, success_prob, unif_prob]]
                            )
        P[:,:,1] = np.array(
            [[unif_prob, success_prob, unif_prob, unif_prob],
             [unif_prob, unif_prob, success_prob, unif_prob],
             [unif_prob, unif_prob, unif_prob, success_prob],
             [unif_prob, unif_prob, unif_prob, success_prob]]
        )

        M_0 = (n_states, n_actions, R, P, gamma)


        env = env_chain.Environment(M_0)

        target_pi = np.ones(env.n_states, dtype="int")

        costr = 3
        costp = 1
        d_0 = np.ones(n_states)/n_states
        # d_0[0] = 1

        ################## General Joint p=inf #########################
        teaching_obj_general_reward = teaching(env, target_pi=target_pi, epsilon=0.1, p=np.inf, delta=0.0001,
                                               costr=costr, costp=costp, d_0=d_0, T_UCRL=1e+5,
                                               alpha_UCRL=0.5, attackers_cost_y_axis=1,
                                               teacher_type="general_attack_joint")
        teaching_obj_general_reward.end_to_end_teaching()
        accumulator_dict = accumulator_function(teaching_obj_general_reward.learner, accumulator_dict,
                                                teaching_obj_general_reward.teacher_type, p="inf")


        ################## Non Target Joint #########################
        teaching_obj_general_reward = teaching(env, target_pi=target_pi, epsilon=0.1, p=np.inf, delta=0.0001,
                                               costr=costr, costp=costp, d_0=d_0, T_UCRL=1e+5,
                                               alpha_UCRL=0.5, attackers_cost_y_axis=1,
                                               teacher_type="non_target_attack_joint")
        teaching_obj_general_reward.end_to_end_teaching()
        accumulator_dict = accumulator_function(teaching_obj_general_reward.learner, accumulator_dict,
                                                teaching_obj_general_reward.teacher_type, p="inf")

    accumulator_dict = calculate_average(accumulator_dict, number_of_iterations)
    plot_chain.plot_online_teaching(dict_file=accumulator_dict, each_number=5000, show_plots=False)
    #########################################
        #write_into_file(accumulator_dict, iter_num, teacher_type="online_teaching_costr={}_costp={}".format(costr, costp))

    exit()




