import pickle, os
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sbrn

# load data in
cur_dir = os.getcwd()
# with open("sens_run_no_immune_moi_10.pickle", "rb") as f:
# with open("sens_run_no_immune_synth.pickle", "rb") as f:
with open("sens_v5_run.pickle", "rb") as f:
    sdat = pickle.load(f)
# unpack the dictionary
data    = sdat['data']
obs     = sdat['obs']
samples = sdat['samples']
# remove 0 obs
obs_inds = np.where(np.array(obs)>0)
samples = np.array(samples)[obs_inds[0],:]
obs = np.array(obs)[obs_inds]
# observable function
key = "burst_size"
def get_burst_size(d):
    return d[key]
# get observables
# nobs = np.log10([get_burst_size(dat) for dat in data])
nobs = np.log10(obs)
# find indices
nobs1_ind = np.where(nobs<2)
nobs2_ind = np.where(nobs>=2)
# get samples and observables
s1 = samples[nobs1_ind[0],:]
o1 = nobs[nobs1_ind]
s2 = samples[nobs2_ind[0],:]
o2 = nobs[nobs2_ind]
# fitting
ft1 = LinearRegression().fit(s1, o1)
ft2 = LinearRegression().fit(s2, o2)
# score our fit
cs1 = ft1.score(s1, o1)
cs2 = ft2.score(s2, o2)
# get unit vectors
uv1 = ft1.coef_/(ft1.coef_.dot(ft1.coef_))
uv2 = ft2.coef_/(ft2.coef_.dot(ft2.coef_))
# projections onto coefficient vector
p1 = np.dot(s1, ft1.coef_)
p2 = np.dot(s2, ft2.coef_)
# predictions by the linear fit
pr1 = ft1.predict(s1)
pr2 = ft2.predict(s2)
# write some stuff out so we can see
print("#### Region 1 ####")
print("Fitting to log10 of {}".format(key))
print("Score: {}".format(cs1))
print("rate cts are k_deg, k_rep_half, k_max_rep and export rate")
print("Coef:  {}".format(ft1.coef_))
print("uv:  {}".format(uv1))
print("#### Region 2 ####")
print("Score: {}".format(cs2))
print("Coef:  {}".format(ft2.coef_))
print("uv:  {}".format(uv2))
# do plotting for region 1 
sbrn.scatterplot(p1, o1, label="region 1 data")
plt.plot(p1, pr1, color="r", label="r1 pred, score: {:.3f}".format(cs1))
plt.legend(frameon=False)
plt.title("$k_{deg}$: %.3f, $k_{rep\ half}$: %.3f, $k_{max\ rep}$: %.3f, $k_{export}$: %.3f, $k_{psyn}$: %.3f"%(ft1.coef_[0],ft1.coef_[1],ft1.coef_[2],ft1.coef_[3],ft1.coef_[4]))
plt.xlabel("projections")
plt.ylabel("$log_{10}$(burst size)")
plt.savefig("sens_scatter_region_1_psyn.png")
plt.close()
# do plotting for region 2
sbrn.scatterplot(p2, o2, label="region 2 data", color="orange")
plt.plot(p2, pr2, color="k", label="r2 pred, score: {:.3f}".format(cs2))
plt.legend(frameon=False)
plt.title("$k_{deg}$: %.3f, $k_{rep\ half}$: %.3f, $k_{max\ rep}$: %.3f, $k_{export}$: %.3f, $k_{psyn}$: %.3f"%(ft2.coef_[0],ft2.coef_[1],ft2.coef_[2],ft2.coef_[3], ft2.coef_[4]))
plt.xlabel("projections")
plt.ylabel("$log_{10}$(burst size)")
plt.savefig("sens_scatter_region_2_psyn.png")
plt.close()
