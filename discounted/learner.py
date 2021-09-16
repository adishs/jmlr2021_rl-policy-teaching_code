import numpy as np
import copy
import matplotlib.pyplot as plt
import sys
import os
sys.path.append('../')
import MDPSolver

class learner:
    def __init__(self, env, type="offline"):
        self.env = env
        self.type = type
        self.accumulator_dict = {}
        if type == "offline":
            # _, self.pi_d, pi_s = MDPSolver.valueIteration(env, env.reward)
            pass
        elif type == "online":
            pass
        else:
            print("unknown learner type: ", type)
            exit(0)
    #enddef

    def get_conf_r_t(self, N_t, alpha, t):
        env = copy.deepcopy(self.env)
        conf_r_t = np.zeros((env.n_states, env.n_actions))
        # conf_r_t = min(1, np.sqrt((np.log2(4*(t**alpha)*(env.n_states**2)*env.n_actions))
        #                               /(2*N_t)))

        for s in range(env.n_states):
            for a in range(env.n_actions):
                if N_t[s, a] == 0 or t == 0:
                    conf_r_t[s, a] = 1
                else:
                    conf_r_t[s, a]= min(1, np.sqrt((np.log2(4*(t**alpha)*(env.n_states**2)*env.n_actions))
                                      /(2*N_t[s, a])))
        return conf_r_t
    #enddef

    def get_conf_p_t(self, N_t, alpha, t):
        env = copy.deepcopy(self.env)
        conf_p_t = np.zeros((env.n_states, env.n_actions))
        for s in range(env.n_states):
            for a in range(env.n_actions):
                if N_t[s, a] == 0 or t == 0:
                    conf_p_t[s, a] = 1
                else:
                    conf_p_t[s, a] = min(1, np.sqrt((np.log2(2 * (t ** alpha) * env.n_states * env.n_actions))
                                                  / (2 * N_t[s, a])))
        return conf_p_t
    #enddef

    def get_r_hat(self, N_t, R_t):
        r_hat = np.zeros((self.env.n_states, self.env.n_actions))
        # r_hat = R_t / N_t
        # r_hat[np.isnan(r_hat)] = 1
        for s in range(self.env.n_states):
            for a in range(self.env.n_actions):
                r_hat[s, a] = R_t[s, a] / N_t[s, a] if N_t[s, a] > 0 else 1
        return r_hat
    #enddef
    def get_p_hat(self, N_t, P_t):
        p_hat = np.zeros((self.env.n_states, self.env.n_states, self.env.n_actions))
        for s in range(self.env.n_states):
            for a in range(self.env.n_actions):
                for s_n in range(self.env.n_states):
                    p_hat[s, s_n, a] = P_t[s, s_n, a] / N_t[s, a] if N_t[s, a] > 0 else 1 / self.env.n_states
        return p_hat
    #enddef
    # def compute_regret_given_pi(self, pi, s_t, t):
    #     for i in range(t):
    #         a_t = pi[s_t]
    #         r_t = self.env.reward[s_t, a_t]
    #     return r_t

    def compute_expected_action_diff_array(self, pi_star, s_t, a_t):
        accumulator = 0
        if pi_star[s_t] != a_t:
            accumulator += 1
        return accumulator
    #enddef

    def compute_accumulated_cost(self, s_t, a_t, M_0, M_t, cost_y_axis, costr, costp):

        TV_r = np.abs((M_0[2][s_t, a_t] - M_t[2][s_t, a_t]))
        TV_P = np.sum(np.abs(M_0[3][s_t, :, a_t] - M_t[3][s_t, :, a_t]))
        # TV_P = np.max(np.abs(M_0[3][s_t, :, a_t] - M_t[3][s_t, :, a_t]))
        if cost_y_axis == 1:
            accumulator = costr * TV_r + costp * TV_P
        # elif cost_y_axis == 2:
        #     accumulator = np.power(TV_r, cost_y_axis) + np.power(TV_P, cost_y_axis)
        else:
            print("cost_y_axis should be 1")
            exit(0)

        return accumulator

    def compute_accumulated_cost_no_attack(self, s_t, a_t, M_0, M_t, cost_y_axis, costr, costp):

        TV_r = np.abs((M_0[2][s_t, a_t] - M_0[2][s_t, a_t]))
        TV_P = np.sum(np.abs(M_0[3][s_t, :, a_t] - M_0[3][s_t, :, a_t]))
        # TV_P = np.max(np.abs(M_0[3][s_t, :, a_t] - M_0[3][s_t, :, a_t]))
        if cost_y_axis == 1:
            accumulator = costr * TV_r + costp * TV_P
        # elif cost_y_axis == 2:
        #     accumulator = np.power(TV_r, cost_y_axis) + np.power(TV_P, cost_y_axis)
        else:
            print("cost_y_axis should be 1")
            exit(0)

        return accumulator






    def make_epsilon_greedy_policy(self, Q, state, epsilon, nA):
        """
        Creates an epsilon-greedy policy based on a given Q-function and epsilon.

        Args:
            Q: A dictionary that maps from state -> action-values.
                Each value is a numpy array of length nA (see below)
            epsilon: The probability to select a random action . float between 0 and 1.
            nA: Number of actions in the environment.

        Returns:
            A function that takes the observation as an argument and returns
            the probabilities for each action in the form of a numpy array of length nA.

        """

        A = np.ones(nA, dtype=float) * epsilon / nA
        best_action = np.argmax(Q[state])
        A[best_action] += (1.0 - epsilon)

        return A

    # enddef

    def q_learning(self, n_rounds, pi_no_attack,
                   M_0, M_t, cost_y_axis, costr, costp,
                   alpha=0.5, epsilon=0.01):
        """
        Q-Learning algorithm: Off-policy TD control. Finds the optimal greedy policy
        while following an epsilon-greedy policy

        Args:
            env: environment.
            num_episodes: Number of episodes to run for.
            discount_factor: Gamma discount factor.
            alpha: TD learning rate.
            epsilon: Chance the sample a random action. Float betwen 0 and 1.

        Returns:
            Q is the optimal action-value function, a dictionary mapping state -> action values.
        """
        regret_array = []
        regret_learner_array = []
        expected_action_diff_array = []
        expected_action_diff_array_no_attack = []
        accumulated_cost_array = []
        accumulated_cost_array_no_attack = []
        regret_r_accumulator_learner = 0
        regret_r_accumulator_learner_no_attack = 0
        diff_action_count = 0
        accumulated_cost_count = 0
        accumulated_cost_no_attack_count = 0

        original_epsilon = copy.deepcopy(epsilon)
        env = copy.deepcopy(self.env)
        # The final action-value function.
        Q = np.zeros((env.n_states, env.n_actions))
        #pick first state
        s_t = np.random.choice(np.arange(0, env.n_states))
        # s_t = env.n_states-1

        _, _, pi_star, _ = MDPSolver.valueIteration(env, env.reward)
        # regret_r_accumulator_opt = MDPSolver.compute_averaged_reward_given_policy(env, env.reward, pi_star)
        for t in range(1, int(n_rounds)+1):
            # epsilon = original_epsilon / np.sqrt(t)
            if (t) % 1000 == 0:
                print("\rEpisode {}/{}.".format(t, int(n_rounds)))
                _, _, pi, _ = self.compute_policy(Q)
                env.draw_policy(pi)
            # Take a step
            action_probs = self.make_epsilon_greedy_policy(Q, s_t, epsilon, env.n_actions)
            a_t = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            s_t_plus_1 = env.get_next_state(s_t, a_t)
            r_t = env.reward[s_t, a_t]

            regret_r_accumulator_learner += r_t
            regret_learner_array.append(regret_r_accumulator_learner / t)
            # regret_array.append(t * regret_r_accumulator_opt - regret_r_accumulator_learner)
            diff_action_count += self.compute_expected_action_diff_array(pi_star, s_t, a_t)
            expected_action_diff_array.append((diff_action_count) / t)
            regret_r_accumulator_learner_no_attack += self.compute_expected_action_diff_array(pi_no_attack, s_t, a_t)
            expected_action_diff_array_no_attack.append(regret_r_accumulator_learner_no_attack / t)
            accumulated_cost_count += self.compute_accumulated_cost(s_t, a_t, M_t, M_0, cost_y_axis, costr, costp)
            accumulated_cost_array.append(np.power(accumulated_cost_count, 1 / cost_y_axis) / t)
            accumulated_cost_no_attack_count += self.compute_accumulated_cost_no_attack(s_t, a_t, M_t, M_0, cost_y_axis,
                                                                                        costr, costp)
            accumulated_cost_array_no_attack.append(np.power(accumulated_cost_no_attack_count, 1 / cost_y_axis) / t)

            # TD Update
            best_next_action = np.argmax(Q[s_t_plus_1])
            td_target = r_t + env.gamma * Q[s_t_plus_1][best_next_action]
            td_delta = td_target - Q[s_t][a_t]
            Q[s_t][a_t] = Q[s_t][a_t] + alpha * td_delta
            #update state
            s_t = copy.deepcopy(s_t_plus_1)

      # plt.plot(regret_array)
        self.accumulator_dict["regret_array"] = regret_array
        self.accumulator_dict["expected_action_diff_array"] = expected_action_diff_array
        self.accumulator_dict["regret_learner_array"] = regret_learner_array
        self.accumulator_dict["expected_action_diff_array_no_attack"] = expected_action_diff_array_no_attack
        self.accumulator_dict["accumulated_cost_array"] = accumulated_cost_array
        self.accumulator_dict["accumulated_cost_array_no_attack"] = accumulated_cost_array_no_attack
        # print(self.accumulator_dict["accumulated_cost_array_no_attack"])
    # enddef

    def compute_policy(self, Q):
        # calculate Value Function
        V = np.max(Q, axis=1)

        # DETERMINISTIC
        pi_d = np.argmax(Q, axis=1)
        # Stochastic
        pi_s = Q - np.max(Q, axis=1)[:, None]
        pi_s[np.where((-1e-2 <= pi_s) & (pi_s <= 1e-2))] = 1
        pi_s[np.where(pi_s <= 0)] = 0
        pi_s = pi_s / pi_s.sum(axis=1)[:, None]

        return Q, V, pi_d, pi_s
    # enddef


#enddef


if __name__ == "__main__":
    pass