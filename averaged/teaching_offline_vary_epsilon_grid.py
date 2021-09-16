import numpy as np
import copy
import sys
sys.path.append('../')
import os
import plot_grid
import MDPSolver
import env_gridworld
import learner
import teacher
import matplotlib.pyplot as plt



class teaaching:
    def __init__(self, M_0, settings_to_run, teachers_to_run, target_pi):
        self.M_0 = M_0
        self.settings_to_run = settings_to_run
        self.teachers_to_run = teachers_to_run
        self.target_pi = target_pi
        self.accumulator = {}

    #enddef

    def offline_attack(self):
        for setting in self.settings_to_run:
            # R_c, success_prob = setting[0], setting[1]
            gamma = self.teachers_to_run[0][2]["gamma"]
            R_c = M_0[2][0][0]
            # print(R_c)
            # exit(0)
            epsilon, success_prob = setting[0], setting[1]
            M_in = get_M( R_c, gamma=gamma)
            env_in = env_gridworld.Environment(M_in)
            pool = teacher.generate_pool(M_in[0], M_in[1], M_in[2], M_in[3], M_in[4], self.target_pi)
            for tchr in self.teachers_to_run:
                target_pi = tchr[2]["target_pi"]
                p = tchr[1]
                # epsilon = tchr[2]["epsilon"]
                delta = tchr[2]["delta"]
                costr = tchr[2]["costr"]
                costp = tchr[2]["costp"]
                d_0 = tchr[2]["d_0"]
                teacher_type = tchr[0]
                cost_y_axis = tchr[2]["cost_y_axis"]
                teacher_obj = teacher.teacher(env=env_in, target_pi=target_pi, p=p, epsilon=epsilon,
                                              delta=delta, costr=costr, costp=costp, d_0=d_0,
                                              teacher_type=teacher_type, pool=pool) #Pool here

                try:
                    M_out, _, feasible = teacher_obj.get_target_M(M_in)
                except Exception as e:
                    print("====================NOT FEASIBLE=======================")
                    print("==================== Exception --{}".format(e))

                    print("--teacher_type={}--R_c={}--P_success={}--eps={}".format(teacher_type, R_c, success_prob, epsilon))
                    input()
                if not feasible:
                    print("====================NOT FEASIBLE=======================")
                    print("--teacher_type={}--R_c={}--P_success={}--eps={}".format(teacher_type, R_c, success_prob,
                                                                                   epsilon))
                    cost = self.max_cost_value_if_non_feasible(cost_y_axis)
                    self.append_cost_to_accumulator(cost, teacher_type, p, cost_y_axis, success_prob, R_c)
                    continue
                else:
                    print("====================FEASIBLE=======================")
                    print("--teacher_type={}--R_c={}--P_success={}--eps={}".format(teacher_type, R_c, success_prob,
                                                                                   epsilon))

                env_out = env_gridworld.Environment(M_out)
                _, pi_T, _ = MDPSolver.averaged_valueIteration(env_out, env_out.reward)
                print(teacher_type)
                print("cost=", teacher_obj.cost(M_in, M_out, cost_y_axis))
                print("Policy for R_T=", pi_T)
                print("Opt_expected_reward on modified = ",
                      MDPSolver.compute_averaged_reward_given_policy(env_out, env_out.reward, pi_T))
                cost = teacher_obj.cost(M_in, M_out, cost_y_axis)
                self.append_cost_to_accumulator(cost, teacher_type, p, cost_y_axis, success_prob, R_c)
        return self.accumulator
    #enddef

    def max_cost_value_if_non_feasible(self, cost_y_axis):
        if cost_y_axis == np.inf:
            return 100
        else:
            print("cost_y_axis should be inf")
            exit(0)
    #enddef

    def append_cost_to_accumulator(self, cost, teacher_type, p, cost_y_axis, success_prob, R_c):
        key = "{}_p={}_cost_y_axis={}_R_c={}_success_prob={}".format(teacher_type, p, cost_y_axis, R_c, success_prob)
        if key in self.accumulator:
            self.accumulator[key].append(cost)
        else:
            self.accumulator[key] = [cost]
        #enddef


def write_into_file(accumulator, exp_iter, teacher_type="offline_teaching_eps"):
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

def get_M(R_c=-2.5, gamma=1):
    env = env_gridworld.Environment()
    n_states = env.n_states
    n_actions = env.n_actions
    R = env.reward
    P_0 = env.T

    R[0, :] = R_c
    # R[9,:] = R_c
    M_0 = (n_states, n_actions, R, P_0, gamma)

    return M_0
#enddef


def accumulator_function(tmp_dict, dict_accumulator):
    for key in tmp_dict:
        if key in dict_accumulator:
            dict_accumulator[key] += np.array(tmp_dict[key])
        else:
            dict_accumulator[key] = np.array(tmp_dict[key])
    return dict_accumulator
#enddef

def calculate_average(dict_accumulator, number_of_iterations):
    for key in dict_accumulator:
        dict_accumulator[key] = dict_accumulator[key]/number_of_iterations
    return dict_accumulator
#enddef

########################################
if __name__ == "__main__":

    number_of_iterations = 10
    dict_accumulator = {}
    for iter_num in range(1, number_of_iterations + 1):


        # ====================
        gamma = 1

        M_0 = get_M(R_c=-2.5)
        target_pi = env_gridworld.get_target_pi()
        d_0 = np.ones(M_0[0]) / M_0[0]
        costr = 3
        costp = 1



        params = {
            "target_pi": target_pi,
            "gamma": gamma,
            "epsilon": 0.1,
            "delta": 0.0001,
            "costr": costr,
            "costp": costp,
            "d_0": d_0,
            "cost_y_axis": np.inf
        }

        teachers_to_run = [("general_attack_on_reward", np.inf, params),
                             ("general_attack_on_dynamics", np.inf, params),
                            ("non_target_attack_joint", np.inf, params),
                            ("general_attack_joint", np.inf, params)
                           ]


        settings_to_run_init_1 = []
        p = 0.9
        for epsilon in np.round(np.arange(0, 1.21, 0.05), 2):
            settings_to_run_init_1.append((epsilon, p))
        print(settings_to_run_init_1)




        #=========================================

        settings_to_run_init = settings_to_run_init_1

        #=========================================

        #general_attack_on_reward #non_target_attack_on_reward
        teaaching_obj = teaaching(M_0, settings_to_run_init, teachers_to_run, target_pi) #settings_to_run_init

        acc_dict = teaaching_obj.offline_attack()
        dict_accumulator = accumulator_function(acc_dict, dict_accumulator)

    dict_accumulator = calculate_average(dict_accumulator, number_of_iterations)
    plot_grid.plot_offline_teaching_vary_eps(dict_file=dict_accumulator, each_number=1, show_plots=False)

    # print(acc_dict)
    # write_into_file(acc_dict, exp_iter=iter_num, teacher_type="offline_teaching_vary_eps_costr={}_costp={}_grid".format(costr, costp))
    exit(0)

    ####################################################