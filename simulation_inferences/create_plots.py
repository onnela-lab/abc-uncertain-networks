# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 16:27:07 2024

@author: mhw20
"""


import numpy as np
import matplotlib.pyplot as plt
import pickle
import scipy.stats
import pandas
import seaborn as sns

plt.rc('axes', titlesize=12)     # fontsize of the axes title
plt.rc('axes', labelsize=12)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=12)    # fontsize of the tick labels
plt.rc('ytick', labelsize=12)    # fontsize of the tick labels
plt.rc('legend', fontsize=11)    # legend fontsize
plt.rc('figure', titlesize=12)  # fontsize of the figure title

base_file = "C:/simulation_abc/"

poisson_file = "C:simulation_abc/abc_poisson/"
ln_file = "C:/simulation_abc/abc_ln/"

poisson_abc_file = open(poisson_file + "abc_draws.pkl", "rb")
poisson_abc_draws = pickle.load(poisson_abc_file)
poisson_accepted_thetas = poisson_abc_draws["thetas"]

ln_abc_file = open(ln_file + "abc_draws.pkl", "rb")
ln_abc_draws = pickle.load(ln_abc_file)
ln_accepted_thetas = ln_abc_draws["thetas"]

true_beta = 0.15
true_gamma = 0.10

beta_violin_file = open(poisson_file + "/violin_data_beta.pkl", "rb")
beta_violin_df = pickle.load(beta_violin_file)
beta_violin_file.close()

gamma_violin_file = open(poisson_file + "/violin_data_gamma.pkl", "rb")
gamma_violin_df = pickle.load(gamma_violin_file)
gamma_violin_file.close()

true_beta = 0.15
prior_beta_a = 2
prior_beta_b = 4

fig,ax = plt.subplots(2,2,gridspec_kw={"height_ratios":[1,1]})
fig.subplots_adjust(hspace=0.03, bottom=0.1, wspace = 0.01)
fig.set_size_inches(8,3)

sns.violinplot(data = beta_violin_df, ax = ax[0,0], x = "Instance", y = "Beta", legend = False, color = "dodgerblue", alpha = 0.2)
sns.violinplot(data = gamma_violin_df, ax = ax[1,0], x = "Instance", y = "Gamma", legend = False, color = "green", alpha = 0.2)

ax[0,0].axhline(true_beta, color = "black", linestyle = "-.")
ax[1,0].axhline(true_gamma, color = "black", linestyle = "-.")

ax[0,0].set_ylabel(r"$\beta$")
ax[1,0].set_ylabel(r"$\gamma$")

ax[0,0].set_ylim(0,0.5)
ax[0,0].set_yticks([0,0.35])
ax[0,0].set_yticklabels(["0", "0.35"], rotation = 90)
ax[0,0].set_xticks([])

ax[1,0].set_ylim(0,0.28)
ax[1,0].set_yticks([0,0.2])
ax[1,0].set_yticklabels(["0", "0.2"], rotation = 90)

abc_file = open(ln_file + "abc_draws.pkl", "rb")
abc_draws = pickle.load(abc_file)
accepted_thetas = abc_draws["thetas"]
true_beta = 0.15
true_gamma = 0.10

beta_violin_file = open(ln_file + "/violin_data_beta.pkl", "rb")
beta_violin_df = pickle.load(beta_violin_file)
beta_violin_file.close()

gamma_violin_file = open(ln_file + "/violin_data_gamma.pkl", "rb")
gamma_violin_df = pickle.load(gamma_violin_file)
gamma_violin_file.close()


sns.violinplot(data = beta_violin_df, ax = ax[0,1], x = "Instance", y = "Beta", legend = False, color = "dodgerblue", alpha = 0.2)
sns.violinplot(data = gamma_violin_df, ax = ax[1,1], x = "Instance", y = "Gamma", legend = False, color = "green", alpha = 0.2)

ax[0,1].axhline(true_beta, color = "black", linestyle = "-.")
ax[1,1].axhline(true_gamma, color = "black", linestyle = "-.")

ax[0,1].set_ylim(0,0.5)
ax[1,1].set_ylim(0,0.28)
ax[0,1].set_yticks([])
ax[1,1].set_yticks([])
ax[0,1].set_ylabel("")
ax[1,1].set_ylabel("")

ax[0,1].set_xticks([])
ax[1,1].set_xlabel("")
ax[1,0].set_xlabel("                                                                     Instance")


beta_ypos = ax[0,1].get_ylim()[0] + 0.75 * (ax[0,1].get_ylim()[1] - ax[0,1].get_ylim()[0])
gamma_ypos = ax[1,1].get_ylim()[0] + 0.75 * (ax[1,1].get_ylim()[1] - ax[1,1].get_ylim()[0])
xpos = ax[0,1].get_xlim()[0] + 0.07 *(ax[0,1].get_xlim()[1] - ax[0,1].get_ylim()[0])

ax[0,0].text(xpos, beta_ypos,"(a)", size = 15)
ax[0,1].text(xpos, beta_ypos, "(b)", size = 15)
ax[1,0].text(xpos, gamma_ypos, "(c)", size = 15)
ax[1,1].text(xpos, gamma_ypos, "(d)", size = 15)

plt.savefig(base_file + "/combined_violin_plots.pdf", bbox_inches='tight')


# Now do the histograms.
fig,ax = plt.subplots(2,5,gridspec_kw={"height_ratios":[1,1]})
fig.set_size_inches(9,3.5)

def single_plot(axis, letter, hist_data, color, alpha, bounds, true_value, x_label = "", ticks = []):
    axis.hist(hist_data, color = color, alpha = alpha)
    axis.set_xlim(bounds[0], bounds[1])
    axis.set_yticks([])
    axis.axvline(true_value, color = "black", alpha = 0.3, linestyle = "-.")
    axis.xaxis.set_label_position('top') 
    if x_label == "":
        axis.set_xlabel("")
    else:
        axis.set_xlabel(x_label)
    if len(ticks) == 0:
        axis.set_xticks([])
    else:
        axis.set_xticks(ticks)
        axis.set_xticklabels([str(ticks[0]), str(ticks[1])]) 
    caption_ypos = axis.get_ylim()[0] + 0.8 * (axis.get_ylim()[1] - axis.get_ylim()[0])
    caption_xpos = bounds[0] + 0.78*(bounds[1] - bounds[0])
    axis.text(caption_xpos, caption_ypos, "(" + letter +")", size = 12)
        
single_plot(ax[0,0], "a", poisson_accepted_thetas[:,0], "dodgerblue", 0.2, [0,0.3], 0.15, r"$\beta$" )
single_plot(ax[0,1], "b", poisson_accepted_thetas[:,1], "green", 0.2, [0,0.2], 0.10, r"$\gamma$")
single_plot(ax[0,2], "c", poisson_abc_draws["network_params"][:,0], "red", 0.4, [0.9,1.1],1, r"$\lambda_0$")
single_plot(ax[0,3], "d", poisson_abc_draws["network_params"][:,1], "orange", 0.4, [6,9.5],8, r"$\lambda_1$")
single_plot(ax[0,4], "e", poisson_abc_draws["network_params"][:,2], "violet", 0.3, [0,0.08], 100, r"$\rho$")

single_plot(ax[1,0], "f", ln_accepted_thetas[:,0], "dodgerblue", 0.2, [0,0.3], 0.15, "", [0, 0.20])
single_plot(ax[1,1], "g", ln_accepted_thetas[:,1], "green", 0.2, [0,0.2], 0.10, "", [0, 0.15])
single_plot(ax[1,2], "h", ln_abc_draws["network_params"][:,0], "red", 0.4, [0.9,1.1], 1, "", [0.9, 1])
single_plot(ax[1,3], "i", ln_abc_draws["network_params"][:,1], "orange", 0.4, [6,9.5], 8, "", [6, 9])
single_plot(ax[1,4], "j", ln_abc_draws["network_params"][:,2], "violet", 0.3, [0,0.08], 100, "", [0,0.05])

plt.savefig(base_file + "/histograms.pdf", bbox_inches='tight')


poisson_coverage_file = open(poisson_file + "coverages.pkl", "rb")
poisson_coverages = pickle.load(poisson_coverage_file)
poisson_coverage_file.close()
poisson_np_file = open(poisson_file + "network_param_coverages.pkl","rb")
poisson_np_coverages = pickle.load(poisson_np_file)
poisson_np_file.close()

ln_coverage_file = open(ln_file + "coverages.pkl", "rb")
ln_coverages = pickle.load(ln_coverage_file)
ln_coverage_file.close()
ln_np_file = open(ln_file + "network_param_coverages.pkl","rb")
ln_np_coverages = pickle.load(ln_np_file)
ln_np_file.close()

def plot_coverages(letter, axis, name, coverages, coverage_dict, color, alpha, x_labels = False, y_labels = False):
    axis.plot([0,1],[0,1], linestyle = "-.", alpha = 0.7, color = "black")
    axis.scatter(coverages, coverage_dict[name].values(), color = color, alpha = alpha)
    if x_labels:
        axis.set_xticks([0,0.5])
        axis.set_xticklabels(["0", "0.5"])
    else:
        axis.set_xticks([])
    if y_labels:
        axis.set_yticks([0,0.5])
        axis.set_yticklabels(["0", "0.5"], rotation = 90)
    else:
        axis.set_yticks([])
    axis.set_xlim(0,1)
    axis.set_ylim(0,1)
    axis.text(0.75,0.13, "(" + letter + ")", size = 18)
   
fig,ax = plt.subplots(2,2,gridspec_kw={"height_ratios":[1,1]})
fig.set_size_inches(5,5)
fig.subplots_adjust(hspace=0.03, bottom=0.1, wspace = 0.1)

coverages = [0,0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95]
plot_coverages("a",ax[0,0], "beta", coverages, poisson_coverages, "dodgerblue", 0.4, x_labels = False, y_labels = True)
plot_coverages("b",ax[1,0], "gamma", coverages, poisson_coverages, "green", 0.4, x_labels = True, y_labels = True)

plot_coverages("c",ax[0,1], "beta", coverages, ln_coverages, "dodgerblue", 0.4, x_labels = False, y_labels = False)
plot_coverages("d",ax[1,1], "gamma", coverages, ln_coverages, "green", 0.4, x_labels = True, y_labels = False)
ax[1,0].set_ylabel("                              Empirical Coverage", size = 14)
ax[1,0].set_xlabel("                                  Nominal Coverage", size = 14)

plt.savefig(base_file + "/coverages.pdf", bbox_inches='tight')
