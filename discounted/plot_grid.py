import numpy as np
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['font.serif'] = ['times new roman']
mpl.rcParams['text.latex.preamble'] = [r'\usepackage{newtxmath}']
import os

################ plot settings #################################
mpl.rc('font',**{'family':'serif','serif':['Times'], 'size': 30})
mpl.rc('legend', **{'fontsize': 22})
mpl.rc('text', usetex=True)
# fig_size = (5.5 / 2.54, 4 / 2.54)
fig_size = [6.5, 4.8]

file_path_out = 'plots/env_grid/'

def plot_online_teaching(dict_file, each_number=20000, show_plots=False):
    if not os.path.isdir(file_path_out):
        os.makedirs(file_path_out)

    starting_number = 10000

    plt.plot(
        np.arange(len(dict_file["accumulated_cost_array_no_attack_general_attack_joint_p=inf"][starting_number:]))[
        ::each_number],
        dict_file['accumulated_cost_array_no_attack_general_attack_joint_p=inf'][starting_number:][::each_number],
        label=r"None", color='#00ffff', marker="s")

    plt.plot(
        np.arange(len(dict_file["accumulated_cost_array_general_attack_joint_p=inf"][starting_number:]))[::each_number],
        dict_file['accumulated_cost_array_general_attack_joint_p=inf'][starting_number:][::each_number],
        label=r"JAttack $(\ell_\infty)$", color='#8B0000', marker="^", ls=":")

    plt.plot(np.arange(len(dict_file["accumulated_cost_array_non_target_attack_joint_p=inf"][starting_number:]))[
             ::each_number],
             dict_file['accumulated_cost_array_non_target_attack_joint_p=inf'][starting_number:][::each_number],
             label=r"NT-JAttack", color='g', marker="o", ls="--")

    plt.plot()
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.35), ncol=2, fancybox=False, shadow=False)
    plt.ylabel("Average attack cost")
    plt.xlabel(r'Time t (x$10^4$)')
    # plt.yticks([0, 0.1, 0.3])
    # plt.ylim(ymax=0.3)
    plt.xticks(np.arange(len(dict_file["expected_action_diff_array_non_target_attack_joint_p=inf"][starting_number:]))[
               ::each_number],
               ["1", "", "", "", "10", "", "", "", "", "20", "", "", "", "", "30",
                "", "", "", "", "40", "", "", "", "", "50"])
    plt.savefig(file_path_out + "online_attack-cost" + '.pdf', bbox_inches='tight')


    #===========================

    plt.figure(2, figsize=fig_size)

    plt.plot(np.arange(
        len(dict_file["expected_action_diff_array_no_attack_non_target_attack_joint_p=inf"][starting_number:]))[
             ::each_number],
             dict_file['expected_action_diff_array_no_attack_non_target_attack_joint_p=inf'][starting_number:][
             ::each_number],
             label=r"None", color='#00ffff', marker="s")

    plt.plot(np.arange(len(dict_file["expected_action_diff_array_general_attack_joint_p=inf"][starting_number:]))[
             ::each_number],
             dict_file['expected_action_diff_array_general_attack_joint_p=inf'][starting_number:][::each_number],
             label=r"JAttack $(\ell_\infty)$", color='#8B0000', marker="^", ls=":")

    plt.plot(np.arange(len(dict_file["expected_action_diff_array_non_target_attack_joint_p=inf"][starting_number:]))[
             ::each_number],
             dict_file['expected_action_diff_array_non_target_attack_joint_p=inf'][starting_number:][::each_number],
             label=r"NT-JAttack", color='g', marker="o", ls="--")

    plt.plot()
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.35), ncol=2, fancybox=False, shadow=False)
    plt.ylabel(r'Average mismatch')
    plt.xlabel(r'Time t (x$10^4$)')
    plt.yticks([0, 0.5, 1])
    plt.ylim(ymax=1.01)
    plt.xticks(np.arange(len(dict_file["expected_action_diff_array_non_target_attack_joint_p=inf"][starting_number:]))[
               ::each_number],
               ["1", "", "", "", "10", "", "", "", "", "20", "", "", "", "", "30",
                "", "", "", "", "40", "", "", "", "", "50"])
    plt.savefig(file_path_out + "online_mismatch" + '.pdf', bbox_inches='tight')

#enddef


def plot_offline_teaching_vary_c(dict_file, each_number=1, show_plots=False):

    if not os.path.isdir(file_path_out):
        os.makedirs(file_path_out)

    plt.figure(1, figsize=fig_size)

    plt.plot(
        np.arange(len(dict_file["general_attack_on_reward_p=inf_cost_y_axis=inf_success_prob=0.9"]))[::each_number],
        dict_file['general_attack_on_reward_p=inf_cost_y_axis=inf_success_prob=0.9'][::each_number],
        label=r"RAttack $(\ell_\infty)$", color='#4933FF', marker="*", ls=":")
    plt.plot(
        np.arange(len(dict_file["general_attack_on_dynamics_p=inf_cost_y_axis=inf_success_prob=0.9"]))[::each_number],
        dict_file['general_attack_on_dynamics_p=inf_cost_y_axis=inf_success_prob=0.9'][::each_number],
        label=r"DAttack $(\ell_\infty)$", color='#339CFF', marker="s", ls=":")
    plt.plot(np.arange(len(dict_file["general_attack_joint_p=inf_cost_y_axis=inf_success_prob=0.9"]))[::each_number],
             dict_file['general_attack_joint_p=inf_cost_y_axis=inf_success_prob=0.9'][::each_number],
             label=r"JAttack $(\ell_\infty)$", color='#8B0000', marker="^", ls=":")
    plt.plot(np.arange(len(dict_file["non_target_attack_joint_p=inf_cost_y_axis=inf_success_prob=0.9"]))[::each_number],
             dict_file['non_target_attack_joint_p=inf_cost_y_axis=inf_success_prob=0.9'][::each_number],
             label=r"NT-JAttack", color='g', ls="--", marker="o")

    plt.plot()
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.35), ncol=2, fancybox=False, shadow=False)
    plt.ylabel("Attack cost $(\ell_\infty)$")
    plt.xlabel(r'Reward for $s_0$ state ')
    plt.yticks(np.arange(0, 15.1, 3), ["0.0", "3.0", "6.0", "9.0", "12.0", "15.0"])
    # ["0.0", "1.0", "2.0", "3.0", "4.0", "5.0", "6.0", "7.0", "8.0", "9.0", "10"]
    plt.xticks(
        np.arange(len(dict_file["non_target_attack_joint_p=inf_cost_y_axis=inf_success_prob=0.9"]))[::each_number],
        np.arange(-5, 5.01, 1, dtype="int")[::each_number])
    plt.ylim(ymax=15.1)
    print(plt.rcParams.get('figure.figsize'))
    # exit(0)
    plt.savefig(file_path_out + "offline_vary-c" + '.pdf', bbox_inches='tight')
#enddef


def plot_offline_teaching_vary_eps(dict_file, each_number=1, show_plots=False):

    if not os.path.isdir(file_path_out):
        os.makedirs(file_path_out)

    n_points = 21
    for key in dict_file:
        dict_file[key] = dict_file[key][:n_points]

    plt.figure(1, figsize=fig_size)
    infeasible_scalar = 15

    plt.annotate('Infeasible', xy=(8.05, infeasible_scalar), xycoords='data',
                 xytext=(0.3, 0.87), textcoords='axes fraction',
                 arrowprops=dict(facecolor='gray', shrink=0.0001),
                 horizontalalignment='left', verticalalignment='top', size="20"
                 )

    plt.plot(np.arange(len(dict_file["general_attack_on_reward_p=inf_cost_y_axis=inf_R_c=-2.5_success_prob=0.9"]))[
             ::each_number], ([infeasible_scalar - 0.025] * len(
        dict_file["general_attack_on_reward_p=inf_cost_y_axis=inf_R_c=-2.5_success_prob=0.9"]))[::each_number],
             color='gray', lw=3)

    plt.plot(np.arange(len(dict_file["general_attack_on_reward_p=inf_cost_y_axis=inf_R_c=-2.5_success_prob=0.9"]))[
             ::each_number],
             dict_file['general_attack_on_reward_p=inf_cost_y_axis=inf_R_c=-2.5_success_prob=0.9'][::each_number],
             label=r"RAttack $(\ell_\infty)$", color='#4933FF', marker="*", ls=":")

    # plt.plot(np.arange(len(dict_file["general_attack_on_dynamics_p=inf_cost_y_axis=inf_R_c=-2.5_success_prob=0.9"]))[::each_number],dict_file['general_attack_on_dynamics_p=inf_cost_y_axis=inf_R_c=-2.5_success_prob=0.9'][::each_number],
    #          label=r"DAttack $(\ell_\infty)$", color='r', marker="s", ls=":")
    array_of_dynamics = dict_file['general_attack_on_dynamics_p=inf_cost_y_axis=inf_R_c=-2.5_success_prob=0.9'][
                        ::each_number]
    feasible_upper_index = np.argwhere(array_of_dynamics > infeasible_scalar)[0][0]
    plt.plot(np.arange(len(dict_file["general_attack_on_dynamics_p=inf_cost_y_axis=inf_R_c=-2.5_success_prob=0.9"]))[
             ::each_number][:feasible_upper_index], np.minimum(
        dict_file['general_attack_on_dynamics_p=inf_cost_y_axis=inf_R_c=-2.5_success_prob=0.9'][::each_number][
        :feasible_upper_index], infeasible_scalar),
             label=r"DAttack $(\ell_\infty)$",
             color='#339CFF', marker="s", ls=":")
    plt.plot(np.arange(len(dict_file["general_attack_on_dynamics_p=inf_cost_y_axis=inf_R_c=-2.5_success_prob=0.9"]))[
             ::each_number], np.minimum(
        dict_file['general_attack_on_dynamics_p=inf_cost_y_axis=inf_R_c=-2.5_success_prob=0.9'][::each_number],
        infeasible_scalar),
             color='#339CFF', ls=":")

    plt.plot(np.arange(len(dict_file["general_attack_joint_p=inf_cost_y_axis=inf_R_c=-2.5_success_prob=0.9"]))[
             ::each_number],
             dict_file['general_attack_joint_p=inf_cost_y_axis=inf_R_c=-2.5_success_prob=0.9'][::each_number],
             label=r"JAttack $(\ell_\infty)$", color='#8B0000', marker="^", ls=":")
    plt.plot(np.arange(len(dict_file["non_target_attack_joint_p=inf_cost_y_axis=inf_R_c=-2.5_success_prob=0.9"]))[
             ::each_number],
             dict_file['non_target_attack_joint_p=inf_cost_y_axis=inf_R_c=-2.5_success_prob=0.9'][::each_number],
             label=r"NT-JAttack", color='g', ls="--", marker="o")

    plt.plot()
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.35), ncol=2, fancybox=False, shadow=False)
    plt.ylabel("Attack cost $(\ell_\infty)$")
    plt.xlabel(r'$\epsilon$ margin')
    plt.yticks(np.arange(0, 15.1, 3), ["0.0", "3.0", "6.0", "9.0", "12.0", "15.0"])
    plt.xticks(np.arange(len(dict_file["general_attack_on_dynamics_p=inf_cost_y_axis=inf_R_c=-2.5_success_prob=0.9"]))[
               ::each_number],
               ["0", "", "", "", "0.2", "", "", "", "0.4", "", "", "", "0.6", "", "", "", "0.8", "", "", "", "1.0"])
    plt.ylim(ymax=15.1)
    print(plt.rcParams.get('figure.figsize'))
    # exit(0)
    plt.savefig(file_path_out + "offline_vary-eps" + '.pdf', bbox_inches='tight')

#enddef

