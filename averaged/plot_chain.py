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

file_path_out = 'plots/env_chain/'

def plot_online_teaching(dict_file, each_number=5000, show_plots=False):
    if not os.path.isdir(file_path_out):
        os.makedirs(file_path_out)

    starting_number = 1000

    plt.figure(1, figsize=fig_size)

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
    plt.xlabel(r'Time t (x$10^3$)')
    # plt.yticks([0, 0.5,1])
    # plt.ylim(ymax=1.2)
    plt.xticks(np.arange(len(dict_file["accumulated_cost_array_non_target_attack_joint_p=inf"][starting_number:]))[
               ::each_number],
               ["1", "", "", "", "20", "", "", "", "40", "", "", "", "60", "", "", "", "80", "", "", "100"])
    plt.savefig(file_path_out + "online_attack-cost" + '.pdf', bbox_inches='tight')


    #=============================


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
    plt.xlabel(r'Time t (x$10^3$)')
    plt.yticks([0, 0.5, 1])
    plt.ylim(ymax=1.2)
    plt.xticks(np.arange(len(dict_file["accumulated_cost_array_non_target_attack_joint_p=inf"][starting_number:]))[
               ::each_number],
               ["1", "", "", "", "20", "", "", "", "40", "", "", "", "60", "", "", "", "80", "", "", "100"])
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
    plt.yticks(np.arange(0, 9.1, 1), ["0.0", "1.0", "2.0", "3.0", "4.0", "5.0", "6.0", "7.0", "8.0", "9.0"])
    # ["0.0", "1.0", "2.0", "3.0", "4.0", "5.0", "6.0", "7.0", "8.0", "9.0", "10"]
    plt.xticks(
        np.arange(len(dict_file["non_target_attack_joint_p=inf_cost_y_axis=inf_success_prob=0.9"]))[::each_number],
        np.arange(-5, 9.01, 1, dtype="int")[::each_number])
    plt.ylim(ymax=9.1)
    print(plt.rcParams.get('figure.figsize'))
    # exit(0)
    plt.savefig(file_path_out + "offline_vary-c" + '.pdf', bbox_inches='tight')
#enddef


def plot_offline_teaching_vary_eps( dict_file, each_number=1, show_plots=False):

    if not os.path.isdir(file_path_out):
        os.makedirs(file_path_out)

    n_points = 21
    for key in dict_file:
        dict_file[key] = dict_file[key][:n_points]

    plt.figure(1, figsize=fig_size)
    infeasible_scalar = 9.0

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
    plt.yticks(np.arange(0, 9.1, 1), ["0.0", "1.0", "2.0", "3.0", "4.0", "5.0", "6.0", "7.0", "8.0", "9.0"])
    plt.xticks(np.arange(len(dict_file["general_attack_on_dynamics_p=inf_cost_y_axis=inf_R_c=-2.5_success_prob=0.9"]))[
               ::each_number],
               ["0", "", "", "", "0.2", "", "", "", "0.4", "", "", "", "0.6", "", "", "", "0.8", "", "", "", "1.0"])
    plt.ylim(ymax=9.1)
    print(plt.rcParams.get('figure.figsize'))
    # exit(0)
    plt.savefig(file_path_out + "offline_vary-eps" + '.pdf', bbox_inches='tight')

#enddef


##########  Plot Table ############

def do_average_over_n_states(dictionary):
    s_4_general_attack_on_reward = 0
    s_10_general_attack_on_reward = 0
    s_20_general_attack_on_reward = 0
    s_30_general_attack_on_reward =  0
    s_50_general_attack_on_reward = 0
    s_70_general_attack_on_reward = 0
    s_100_general_attack_on_reward = 0

    s_4_general_attack_on_dynamics = 0
    s_10_general_attack_on_dynamics = 0
    s_20_general_attack_on_dynamics = 0
    s_30_general_attack_on_dynamics = 0
    s_50_general_attack_on_dynamics = 0
    s_70_general_attack_on_dynamics = 0
    s_100_general_attack_on_dynamics = 0

    s_4_non_target_attack_joint = 0
    s_10_non_target_attack_joint = 0
    s_20_non_target_attack_joint = 0
    s_30_non_target_attack_joint = 0
    s_50_non_target_attack_joint = 0
    s_70_non_target_attack_joint = 0
    s_100_non_target_attack_joint = 0

    s_4_general_attack_joint = 0
    s_10_general_attack_joint = 0
    s_20_general_attack_joint = 0
    s_30_general_attack_joint = 0
    s_50_general_attack_joint = 0
    s_70_general_attack_joint = 0
    s_100_general_attack_joint = 0




    for key in dictionary:

        ####### general_attack_on_reward ####
        if "general_attack_on_reward" in key:
            if "4_nstates" in key:
                s_4_general_attack_on_reward += dictionary[key][0]

            if "10_nstates" in key:
                s_10_general_attack_on_reward += dictionary[key][0]

            if "20_nstates" in key:
                s_20_general_attack_on_reward += dictionary[key][0]

            if "30_nstates" in key:
                s_30_general_attack_on_reward += dictionary[key][0]

            if "50_nstates" in key:
                s_50_general_attack_on_reward += dictionary[key][0]

            if "70_nstates" in key:
                s_70_general_attack_on_reward += dictionary[key][0]


            if "100_nstates" in key:
                s_100_general_attack_on_reward += dictionary[key][0]

        ###### general_attack_on_dynamics #####
        if "general_attack_on_dynamics" in key:
            if "4_nstates" in key:
                s_4_general_attack_on_dynamics += dictionary[key][0]

            if "10_nstates" in key:
                s_10_general_attack_on_dynamics += dictionary[key][0]

            if "20_nstates" in key:
                s_20_general_attack_on_dynamics += dictionary[key][0]

            if "30_nstates" in key:
                s_30_general_attack_on_dynamics += dictionary[key][0]

            if "50_nstates" in key:
                s_50_general_attack_on_dynamics += dictionary[key][0]

            if "70_nstates" in key:
                s_70_general_attack_on_dynamics += dictionary[key][0]

            if "100_nstates" in key:
                s_100_general_attack_on_dynamics += dictionary[key][0]

        #### non_target_attack_joint ######
        if "non_target_attack_joint" in key:
            if "4_nstates" in key:
                s_4_non_target_attack_joint += dictionary[key][0]

            if "10_nstates" in key:
                s_10_non_target_attack_joint += dictionary[key][0]

            if "20_nstates" in key:
                s_20_non_target_attack_joint += dictionary[key][0]

            if "30_nstates" in key:
                s_30_non_target_attack_joint += dictionary[key][0]

            if "50_nstates" in key:
                s_50_non_target_attack_joint += dictionary[key][0]

            if "70_nstates" in key:
                s_70_non_target_attack_joint += dictionary[key][0]

            if "100_nstates" in key:
                s_100_non_target_attack_joint += dictionary[key][0]

        ### general_attack_joint #######
        if "general_attack_joint" in key:
            if "4_nstates" in key:
                s_4_general_attack_joint += dictionary[key][0]

            if "10_nstates" in key:
                s_10_general_attack_joint += dictionary[key][0]

            if "20_nstates" in key:
                s_20_general_attack_joint += dictionary[key][0]

            if "30_nstates" in key:
                s_30_general_attack_joint += dictionary[key][0]

            if "50_nstates" in key:
                s_50_general_attack_joint += dictionary[key][0]

            if "70_nstates" in key:
                s_70_general_attack_joint += dictionary[key][0]

            if "100_nstates" in key:
                s_100_general_attack_joint += dictionary[key][0]

    return s_4_general_attack_on_reward, s_10_general_attack_on_reward, s_20_general_attack_on_reward, \
           s_30_general_attack_on_reward, s_50_general_attack_on_reward, s_70_general_attack_on_reward ,s_100_general_attack_on_reward,\
            s_4_general_attack_on_dynamics, s_10_general_attack_on_dynamics, s_20_general_attack_on_dynamics, \
           s_30_general_attack_on_dynamics, s_50_general_attack_on_dynamics, s_70_general_attack_on_dynamics, s_100_general_attack_on_dynamics, \
           s_4_non_target_attack_joint, s_10_non_target_attack_joint, s_20_non_target_attack_joint, s_30_non_target_attack_joint, \
           s_50_non_target_attack_joint, s_70_non_target_attack_joint, s_100_non_target_attack_joint, \
           s_4_general_attack_joint, s_10_general_attack_joint, s_20_general_attack_joint, s_30_general_attack_joint, s_50_general_attack_joint, \
           s_70_general_attack_joint, s_100_general_attack_joint








def plot_table(dictionary):

    columns = (r'$|S|=4$', r'$|S|=10$', r'$|S|=20$',r'$|S|=30$', r'$|S|=50$', r'$|S|=70$', r'$|S|=100$')
    rows = [r'\textsc{RAttack}', r'\textsc{DAttack}', r'\textsc{NT-JAttack}', r'\textsc{JAttack}']

    s_4_general_attack_on_reward, s_10_general_attack_on_reward, s_20_general_attack_on_reward, \
    s_30_general_attack_on_reward, s_50_general_attack_on_reward, s_70_general_attack_on_reward, s_100_general_attack_on_reward, \
    s_4_general_attack_on_dynamics, s_10_general_attack_on_dynamics, s_20_general_attack_on_dynamics, \
    s_30_general_attack_on_dynamics, s_50_general_attack_on_dynamics, s_70_general_attack_on_dynamics, s_100_general_attack_on_dynamics, \
    s_4_non_target_attack_joint, s_10_non_target_attack_joint, s_20_non_target_attack_joint, s_30_non_target_attack_joint, \
    s_50_non_target_attack_joint, s_70_non_target_attack_joint, s_100_non_target_attack_joint, \
    s_4_general_attack_joint, s_10_general_attack_joint, s_20_general_attack_joint, s_30_general_attack_joint, s_50_general_attack_joint, \
    s_70_general_attack_joint, s_100_general_attack_joint = \
        do_average_over_n_states(dictionary)

    data = np.round(np.array([[s_4_general_attack_on_reward, s_10_general_attack_on_reward, s_20_general_attack_on_reward, \
    s_30_general_attack_on_reward, s_50_general_attack_on_reward, s_70_general_attack_on_reward, s_100_general_attack_on_reward],
                     [s_4_general_attack_on_dynamics, s_10_general_attack_on_dynamics, s_20_general_attack_on_dynamics, \
    s_30_general_attack_on_dynamics, s_50_general_attack_on_dynamics, s_70_general_attack_on_dynamics, s_100_general_attack_on_dynamics],
                     [s_4_non_target_attack_joint, s_10_non_target_attack_joint, s_20_non_target_attack_joint, s_30_non_target_attack_joint, \
    s_50_non_target_attack_joint, s_70_non_target_attack_joint, s_100_non_target_attack_joint],
                     [s_4_general_attack_joint, s_10_general_attack_joint, s_20_general_attack_joint, s_30_general_attack_joint, s_50_general_attack_joint, \
    s_70_general_attack_joint, s_100_general_attack_joint]]), 4)

    print(data)

    fig = plt.figure(1)
    fig.subplots_adjust(left=0.1,top=0.8, wspace=2, hspace=2)
    ax = plt.subplot2grid((4, 4), (0, 0), colspan=2, rowspan=2)
    ax.table(cellText=data,
              rowLabels=rows,
              colLabels=columns, loc="center")

    ax.axis("off")


    fig.set_size_inches(w=10, h=5)
    plt.savefig(file_path_out + "table.pdf", bbox_inches='tight')
    # plt.show()
#enddef


