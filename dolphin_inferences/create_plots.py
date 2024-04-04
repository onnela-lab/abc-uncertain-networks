
import numpy as np
import matplotlib.pyplot as plt
import pickle
import scipy.stats
import pandas
import seaborn as sns
import random
import matplotlib.patches as mpatches

plt.rc('axes', titlesize=15)     # fontsize of the axes title
plt.rc('axes', labelsize=15)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=15)    # fontsize of the tick labels
plt.rc('ytick', labelsize=15)    # fontsize of the tick labels
plt.rc('legend', fontsize=14)    # legend fontsize
plt.rc('figure', titlesize=12)  # fontsize of the figure title

base_file = "C:/dolphin_analysis/finished_dolphin_data/"

theta_file = open(base_file + "abc/abc_thetas.npy", "rb")
abc_thetas = np.load(theta_file)
theta_file.close()

n_params_file = open(base_file + "abc/abc_n_params.npy", "rb")
abc_n_params = np.load(n_params_file)
n_params_file.close()

fig,ax = plt.subplots(2,3,gridspec_kw={"height_ratios":[1,1]})
fig.subplots_adjust(hspace=0.1, bottom=0.1, wspace = 0.05)
fig.set_size_inches(8,4)

def single_plot(axis, letter, hist_data, color, alpha, bounds, x_label, x_label_position, ticks):
    axis.hist(hist_data, color = color, alpha = alpha, bins = 20)
    axis.set_xlim(bounds[0], bounds[1])
    axis.set_yticks([])
    if x_label_position == "top":
        axis.xaxis.set_label_position('top') 
        axis.xaxis.tick_top()
    axis.set_xlabel(x_label)
    axis.set_xticks(ticks)
    axis.set_xticklabels([str(ticks[0]), str(ticks[1])]) 
    caption_ypos = axis.get_ylim()[0] + 0.8 * (axis.get_ylim()[1] - axis.get_ylim()[0])
    caption_xpos = bounds[0] + 0.78*(bounds[1] - bounds[0])
    axis.text(caption_xpos, caption_ypos, "(" + letter +")", size = 14)
    
single_plot(ax[0,0], "a", abc_thetas[:,0], "dodgerblue", 0.3, [0,0.5], r"$\beta_c$", "top", [0,0.3])
single_plot(ax[0,1], "b", abc_thetas[:,1], "green", 0.3, [0,0.005], r"$\beta_j$", "top", [0,0.003])
single_plot(ax[0,2], "c", abc_thetas[:,2], "red", 0.3, [0,0.001], r"$\beta_a$", "top", [0,0.0006])
single_plot(ax[1,0], "d", abc_thetas[:,3], "grey", 0.5, [0,0.0005], r"$\epsilon$", "bottom", [0,0.0003])
single_plot(ax[1,1], "e", abc_thetas[:,4], "violet", 0.5, [0,3], r"$\gamma_a$", "bottom", [0,2])
single_plot(ax[1,2], "f", abc_thetas[:,5], "orange", 0.5, [0,120], r"$\gamma_b$", "bottom", [0,80])

plt.savefig(base_file + "/dolphin_histograms.pdf", bbox_inches='tight')

# Now, plot the infectious periods.
def unpack_results(results, test_times):
    # Unpacks a results vector into a dictionary, based on test_times.
    results_dict = {i: [] for i in list(test_times.keys())}
    counter = 0
    for k in list(test_times.keys()):
        for sample in test_times[k]:
            results_dict[k].append(results[counter])
            counter += 1
    return results_dict

def get_i_diffs(results_dict, test_times):
    i_diffs = []
    singletons = 0
    for k in results_dict.keys():
        if np.sum(results_dict[k]) == 1:
            singletons += 1
        if np.sum(results_dict[k]) < 2:
            continue
        else:
            first_i = -1
            first_r = -1
            for i in range(len(results_dict[k])):
                if results_dict[k][i] == 1 and first_i == -1: # Found our first infection.
                    first_i = test_times[k][i]
                    continue
                if results_dict[k][i] == 0 and first_i > 0 and first_r == -1: # We see a 0, but we know infection has happened and we haven't recovered.
                    first_r = test_times[k][i]
                    last_i = test_times[k][i-1]
                    i_diffs.append(last_i - first_i)
                    break
    #print("Singletons: " + str(singletons))
    return {"i_diffs": i_diffs, "singletons": singletons}

true_epidemic_loc = base_file + "/true_epidemic/"
test_times_fh = open(true_epidemic_loc + "test_times.pkl","rb")
test_times = pickle.load(test_times_fh)
test_times_fh.close()
true_results_fh = open(true_epidemic_loc + "true_epidemic.pkl", "rb")
true_results = pickle.load(true_results_fh)
true_results_fh.close()

params_fh = open(true_epidemic_loc + "params.pkl","rb")
params_dic = pickle.load(params_fh)


fh = open(base_file + "/abc/full_pp_check.pkl", "rb")
pp_check = pickle.load(fh)
fh.close()

true_results_dict = unpack_results(true_results, test_times)
true_diffs = get_i_diffs(true_results_dict, test_times)["i_diffs"]
true_singletons = get_i_diffs(true_results_dict, test_times)["singletons"]
mean_true_diffs = np.mean(true_diffs)
reps = len(pp_check["results"])
i_diff_means = []
all_i_diffs = []
all_singletons = []


for i in range(reps):
    results_dict = unpack_results(pp_check["results"][i], test_times)
    res = get_i_diffs(results_dict, test_times)
    i_diffs = res["i_diffs"]
    all_singletons.append(res["singletons"])
    if len(i_diffs) ==0:
        print("ERROR: i_diffs len is 0")
        continue
    i_diff_means.append(np.mean(i_diffs))
    all_i_diffs.extend(i_diffs)
    
    
"""
Recovery period pp-checks
Note that this opens a gaps.pkl and a first_times.pkl; these are generated via i_diff_pp_check_with_fill.py, which generates these items.
"""
fig,ax = plt.subplots()
gs = fig.add_gridspec(2,6,width_ratios=[1,1,1,1,1,1],height_ratios=[1.5,1])
ax0 = plt.subplot(gs[0,0:2])
ax1 = plt.subplot(gs[0,2:6])
ax2 = plt.subplot(gs[1,0:3])
ax3 = plt.subplot(gs[1,3:6])
fig.subplots_adjust(hspace=0.1, bottom=0.05, wspace = 0.05)
fig.set_size_inches(9,7)

ax1.hist(i_diff_means, alpha = 0.5, color = "dodgerblue", bins = 35, density = True)
ax1.set_xlim(0,70)
ax1.axvline(19.6, color = "purple", label = "Powell, 2020")
ax1.axvline(mean_true_diffs, color = "blue", label = "Data used in ABC")
ax1.legend()
ax1.set_yticks([])
ax1.set_xticks([0,60])
ax1.set_xlabel("Weeks", size = 16)
caption_ypos = ax1.get_ylim()[0] + 0.87 * (ax1.get_ylim()[1] - ax1.get_ylim()[0])
caption_xpos = 4
ax1.text(caption_xpos, caption_ypos, "(b)", size = 16)
#plt.title("Mean difference between first and last infected sighting")
ax1.xaxis.tick_top()
ax1.xaxis.set_label_position("top")
ax1.xaxis.labelpad = -10

i_period_means = []
# Now draw the recovery times.
for i in range(abc_thetas.shape[0]):
    r_shape = abc_thetas[i,4]
    r_scale = abc_thetas[i,5]
    i_period_means.append(r_scale * scipy.special.gamma(1 + 1/r_shape))
ax0.hist(i_period_means, color = "pink", bins = 30)
ax0.set_yticks([])
ax0.set_xlabel("Weeks", size = 16)
ax0.set_xlim(0,110)
ax0.set_xticks([0,90])
caption_ypos = ax0.get_ylim()[0] + 0.87 * (ax0.get_ylim()[1] - ax0.get_ylim()[0])
caption_xpos = 12
ax0.text(caption_xpos, caption_ypos, "(a)", size = 16)
ax0.xaxis.tick_top()
ax0.xaxis.set_label_position("top")
ax0.xaxis.labelpad = -10


fh = open(base_file + "true_epidemic/params.pkl", "rb")
params_dic = pickle.load(fh)
print(params_dic["prior_params"])


gaps_fh = open(base_file +"abc/gaps.pkl", "rb")
gaps_dict = pickle.load(gaps_fh)
gv = gaps_dict["true_gaps"]
all_gaps = gaps_dict["all_gaps"]
gaps_fh.close()
ft_fh = open(base_file + "abc/first_times.pkl", "rb")
first_times_dict = pickle.load(ft_fh)
ftv = first_times_dict["true_first_times"]
all_first_times = first_times_dict["all_first_times"]
ft_fh.close()

# Plot CDFs with 95% curves.
resolution = 150
xs = np.linspace(0,250,resolution)
intervals = [95, 80, 65]
alphas = [0.15, 0.4, 0.65]
ft_bounds = np.zeros((len(intervals)*2,resolution))
for k in range(len(xs)):
    x = xs[k]
    cdfs = []
    for i in range(len(all_first_times)):
        cdfs.append(np.sum(np.array(all_first_times[i]) < x)/len(all_first_times[i]))
    for i in range(len(intervals)):
        lower = np.percentile(cdfs, (100-intervals[i])/2)
        upper = np.percentile(cdfs, intervals[i] + (100-intervals[i])/2)
        ft_bounds[(i*2),k] = lower
        ft_bounds[(i*2) + 1, k] = upper
ax2.plot(np.sort(ftv), np.linspace(0, 1, len(ftv), endpoint = False), color = "red")
ax2.set_ylim(0,1)
ax2.set_xlim(0,250)
ax2.set_xticks([0,200])
ax2.set_yticks([0,1])
ax2.set_xlabel("Weeks", size = 16)
ax2.xaxis.labelpad = -10
ax2.yaxis.labelpad = -10
ax2.set_ylabel("Probability", size = 16)
caption_ypos = ax2.get_ylim()[0] + 0.8 * (ax2.get_ylim()[1] - ax2.get_ylim()[0])
caption_xpos = 20
ax2.text(caption_xpos, caption_ypos, "(c)", size = 16)
for i in range(len(intervals)):
         ax2.fill_between(xs, ft_bounds[(i*2), :], ft_bounds[(i*2) + 1, :], color = "grey", alpha = alphas[i])
g_bounds = np.zeros((len(intervals)*2,resolution))
for k in range(len(xs)):
    x = xs[k]
    cdfs = []
    for i in range(len(all_gaps)):
        cdfs.append(np.sum(np.array(all_gaps[i]) < x)/len(all_gaps[i]))
    for i in range(len(intervals)):
        lower = np.percentile(cdfs, (100-intervals[i])/2)
        upper = np.percentile(cdfs, intervals[i] + (100-intervals[i])/2)
        g_bounds[(i*2),k] = lower
        g_bounds[(i*2) + 1, k] = upper
ax3.plot(np.sort(gv), np.linspace(0, 1, len(gv), endpoint = False), color = "red")
for i in range(len(intervals)):
         ax3.fill_between(xs, g_bounds[(i*2), :], g_bounds[(i*2) + 1, :], color = "grey", alpha = alphas[i])
ax3.set_ylim(0,1)
ax3.set_xlim(0,250)
ax3.set_xticks([0,200])
ax3.set_xlabel("Weeks", size = 16)
ax3.set_yticks([])

ax3.xaxis.labelpad = -10
caption_ypos = ax3.get_ylim()[0] + 0.8 * (ax3.get_ylim()[1] - ax3.get_ylim()[0])
caption_xpos = 20
ax3.text(caption_xpos, caption_ypos, "(d)", size = 16)


intervals = [95, 80, 65]
alphas = [0.15, 0.4, 0.65]
patches = []
for i in range(len(intervals)):
    patches.append(mpatches.Patch(color = "grey", alpha = alphas[len(intervals)-1-i], label = str(intervals[len(intervals)-1-i]) + "% PI"))
ax3.legend(handles = patches)
    
plt.savefig(base_file + "/dolphin_pp_checks.pdf", bbox_inches='tight')

"""
Check on network 
"""
location = "C:/dolphin_analysis/finished_dolphin_data/"
fig = plt.figure(figsize=(8, 7)) 
gs = fig.add_gridspec(4,2,width_ratios=[1,1.4])
ax0 = plt.subplot(gs[:,0])
ax1 = plt.subplot(gs[0,1])
ax2 = plt.subplot(gs[1,1])
ax3 = plt.subplot(gs[2,1])
ax4 = plt.subplot(gs[3,1])
fig.subplots_adjust(hspace=0.1, bottom=0.1, wspace = 0.05)


dfh = open(location + "model_testing/ordered_nbin/discrepancies.pkl","rb")
discs = pickle.load(dfh)
dfh.close()

t_d_data = discs["t_d_data"]
t_d_artificial = discs["t_d_artificial"]

colors = ["dodgerblue", "green", "maroon", "violet", "orange"]
for t in range(5):
    d_data = t_d_data[:,t]
    d_artificial = t_d_artificial[:,t]
    ax0.scatter(d_data[d_data<d_artificial], d_artificial[d_data<d_artificial], s=15, alpha = 0.3, edgecolor='#333333', linewidth=1, color=colors[t])
    ax0.scatter(d_data[d_data>d_artificial], d_artificial[d_data>d_artificial], s=15, alpha = 0.3, edgecolor='#333333', linewidth=1, color='w')
min_val = np.min(np.array([np.min(t_d_data), np.min(t_d_artificial)]))
max_val = np.max(np.array([np.max(t_d_data), np.max(t_d_artificial)]))
ax0.plot([0, max_val], [0, max_val], c='k', ls='--', lw=1)
ax0.set_xlim(5000,25000)
ax0.set_ylim(10000,38000)
ax0.set_xlabel(r"$D(X_w,A_{tw}^{'},\phi_{tw}^{'})$", size = 16)
ax0.set_ylabel(r"$D(\tilde{X}_w,A_{tw}^{'},\phi_{tw}^{'})$", size = 16)
ax0.set_xticks([10000,20000])
ax0.set_yticks([15000,35000])
caption_ypos = ax0.get_ylim()[0] + 0.92 * (ax0.get_ylim()[1] - ax0.get_ylim()[0])
caption_xpos = ax0.get_xlim()[0] + 0.06 * (ax0.get_xlim()[1] - ax0.get_xlim()[0])
ax0.yaxis.labelpad = -20
ax0.text(caption_xpos, caption_ypos, "(a)", size = 16)




chains_location = location + "/ordered_nbin/"
colors = ["dodgerblue", "green", "red", "violet", "orange", "black", "pink", "gold", "brown", "indigo", "hotpink", "darkolivegreen", "deepskyblue","turquoise","slategrey","magenta"]
def plot_chains(axis, letter, name, num_chains, chain_length, ylim, yticks):
    fh = open(chains_location + "diagnostic_chains_ordered_nbin_" + str(name) + "_array.npy","rb")
    chain = np.load(fh)
    fh.close()
    axis.yaxis.tick_right()
    axis.set_ylim(ylim[0],ylim[1])
    axis.set_yticks([yticks[0], yticks[1]])
    axis.set_xlim(0,chain_length)
    x = np.array(range(chain_length))
    for i in range(num_chains):
        rand_sample = random.choices(x,k=500)
        axis.scatter(x[rand_sample],chain[i,rand_sample], alpha = 0.4, s = 0.9, color = colors[int(i%len(colors))])
        
    caption_ypos = axis.get_ylim()[0] + 0.70 * (axis.get_ylim()[1] - axis.get_ylim()[0])
    caption_xpos = 0.06*(chain_length)
    axis.text(caption_xpos, caption_ypos, letter, size = 16)
ax1.set_xticks([])
ax2.set_xticks([])
ax3.set_xticks([])
ax4.set_xticks([0,1000])
plot_chains(ax1, "(b)", "n_0_0", 10,1000,[0.01,0.017], [0.01,0.015])
plot_chains(ax2, "(c)", "n_0_1", 10,1000, [0, 1.2], [0,0.9])
plot_chains(ax3, "(d)", "p_t_0", 10,1000, [0,0.45], [0,0.35])
plot_chains(ax4, "(e)", "rho_t_0", 10,1000, [0,0.007], [0,0.005])
ax1.set_ylabel(r"$n_0$", rotation = 270, size = 16)
ax1.yaxis.set_label_position('right') 
ax1.yaxis.labelpad = -20
ax2.set_ylabel(r"$n_1$", rotation = 270, size = 16)
ax2.yaxis.set_label_position('right') 
ax2.yaxis.labelpad = -5
ax3.set_ylabel(r"$p_1$", rotation = 270, size = 16)
ax3.yaxis.set_label_position('right') 
ax3.yaxis.labelpad = -5
ax4.set_ylabel(r"$\rho_1$", rotation = 270, size = 16)
ax4.yaxis.set_label_position('right') 
ax4.yaxis.labelpad = -20
ax4.set_xticklabels(["0","0.005"])
plt.savefig(base_file + "/dolphin_network_diagnostics.pdf", bbox_inches='tight')



"""
Make trace plots for all variables, all times.
"""

def plot_chains_no_letter(axis, name, num_chains, chain_length, ylim, yticks):
    fh = open(chains_location + "diagnostic_chains_ordered_nbin_" + str(name) + "_array.npy","rb")
    chain = np.load(fh)
    fh.close()
    axis.yaxis.tick_right()
    axis.set_ylim(ylim[0],ylim[1])
    axis.set_yticks([])
    axis.set_xticks([])
    axis.set_xlim(0,chain_length)
    axis.tick_params(axis='x', labelsize=10)
    axis.tick_params(axis='y', labelsize=10)
    x = np.array(range(chain_length))
    for i in range(num_chains):
        rand_sample = random.choices(x,k=200) # Thin the sample a bit for plotting, or its hard to see what's going on.
        axis.scatter(x[rand_sample],chain[i,rand_sample], alpha = 0.4, s = 0.9, color = colors[int(i%len(colors))])
    
   
names = ["n_0", "n_1", "p_t", "rho_t"]
ax_dict = {"n_0": [[0.01,0.04], [0.01,0.03]], "n_1":[[0,1.2], [0,0.9]], "p_t": [[0,0.6],[0, 0.45]], "rho_t": [[0,0.08],[0,0.06]]}
fig,ax = plt.subplots()
gs = fig.add_gridspec(4,5)
fig.subplots_adjust(hspace=0.05, bottom=0.05, wspace = 0.05)
fig.set_size_inches(8,7)
for t in range(5):
    for i in range(len(names)):
        curr_axis = plt.subplot(gs[i,t])
        variable = names[i]
        if variable == "n_0":
            new_name = "n_" + str(t) + "_0"
        elif variable == "n_1":
            new_name = "n_" + str(t) + "_1"
        else:
            new_name = variable + "_"+ str(t)
        plot_chains_no_letter(curr_axis, new_name, 10, 1000, ax_dict[variable][0], ax_dict[variable][1])

for i in range(len(names)):
    curr_axis = plt.subplot(gs[i,4])
    ylim = ax_dict[names[i]][1]
    curr_axis.set_yticks([ylim[0],ylim[1]])
for t in range(5):
    curr_axis = plt.subplot(gs[3,t])
    curr_axis.set_xticks([0,800])
    
bottom_axis= plt.subplot(gs[3,2])
bottom_axis.set_xlabel("Iterations", size = 14)

ra0 = plt.subplot(gs[0,4])
ra1 = plt.subplot(gs[1,4])
ra2 = plt.subplot(gs[2,4])
ra3 = plt.subplot(gs[3,4])
ra0.set_ylabel(r"$n_0$", rotation = 270, size = 10)
ra0.yaxis.set_label_position('right') 
ra0.yaxis.labelpad = -7
ra1.set_ylabel(r"$n_1$", rotation = 270, size = 10)
ra1.yaxis.set_label_position('right') 
ra1.yaxis.labelpad = -7
ra2.set_ylabel(r"$p$", rotation = 270, size = 10)
ra2.yaxis.set_label_position('right') 
ra2.yaxis.labelpad = -7
ra3.set_ylabel(r"$\rho$", rotation = 270, size = 10)
ra3.yaxis.set_label_position('right') 
ra3.yaxis.labelpad = -7

ax0_t = plt.subplot(gs[0,0])
ax1_t = plt.subplot(gs[0,1])
ax2_t = plt.subplot(gs[0,2])
ax3_t = plt.subplot(gs[0,3])
ax4_t = plt.subplot(gs[0,4])
fig.subplots_adjust(hspace=0.1, bottom=0.1, wspace = 0.05)

ax0_t.set_xlabel(r"Year 1", size=10)
ax0_t.xaxis.set_label_position('top')
ax1_t.set_xlabel(r"Year 2", size=10)
ax1_t.xaxis.set_label_position('top')
ax2_t.set_xlabel(r"Year 3", size=10)
ax2_t.xaxis.set_label_position('top')
ax3_t.set_xlabel(r"Year 4", size=10)
ax3_t.xaxis.set_label_position('top')
ax4_t.set_xlabel(r"Year 5", size=10)
ax4_t.xaxis.set_label_position('top')

plt.savefig(base_file + "/dolphin_trace_plots.pdf", bbox_inches = "tight")
