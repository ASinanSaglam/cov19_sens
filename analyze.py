import pickle, os, sys, warnings
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sbrn
warnings.filterwarnings("ignore")

def get_observable(sim_data):
    '''
    Given a simulation data, how you calculate 
    an observable 
    '''
    # e.g. this returns the final cell count
    return sim_data["cell_data"][-1]

cur_dir = os.getcwd()
with open(sys.argv[1], "rb") as f:
    sdat = pickle.load(f)

# IPython.embed();sys.exit()

data    = sdat['data']
samples = np.array(sdat['samples'])
obs     = np.array(sdat['obs'])
print("pre removing <0")
print(samples.shape)
print(obs.shape)
# inds_to_use_oz = np.logical_and(obs > 10, obs <= 1500)
inds_to_use_oz = obs>0
obs = obs[inds_to_use_oz]
samples = samples[inds_to_use_oz]
# inds_to_use = obs==0
# obs[inds_to_use] = obs[inds_to_use_oz].min()
# z = obs == 0
# nz = obs > 0
# obs[z] = obs[nz].min()
print("post removing <0")
print(samples.shape)
print(obs.shape)

# key = "survival"
# key = "log10(final cell count)"
key = "final cell count"
# nobs = obs
nobs = np.log10(obs)
# samples = np.power(10, samples)

ft = LinearRegression().fit(samples, nobs)
# ft = LinearRegression().fit(samples, np.log10(nobs))
corr_score = ft.score(samples, nobs)
uv = ft.coef_/np.sqrt(ft.coef_.dot(ft.coef_))

projs = np.dot(samples, ft.coef_)
# inds_to_redo_with = np.logical_and(projs >= -7, projs <= -6.5)
inds_to_redo_with = projs > -3.5
pred = ft.predict(samples)
c = ft.coef_
c = c/np.sqrt(c.dot(c))

sbrn.scatterplot(projs, nobs, label="data")
sbrn.scatterplot(projs[-1], [nobs[-1]], color="k", label="initial value")
plt.plot(projs, pred, color="r", label="prediction, score: {:.3f}".format(corr_score))
plt.legend(frameon=False)
plt.title("IFN para: {:.3f}, viral diff: {:.3f}, ACE2 bind: {:.3f}".format(c[0],c[1],c[2]))
plt.xlabel("projections")
plt.ylabel(key)
plt.savefig("sens_scatter.png")
plt.close()

print(c.dot(c))
print("Fit to log10 of {}".format(key))
print("Score: {}".format(corr_score))
print("IFN para: {:.3f}, viral diff: {:.3f}, ACE2 bind: {:.3f}".format(c[0],c[1],c[2]))
print("normalized coefficients:  {}".format(uv))
print("sample values - init vec: {}, proj: {:.3f}, obs: {}, log10 obs: {:.3f}".format(10**samples[0], projs[-1], int(nobs[-1]), np.log10(nobs[-1])))


samples = samples[inds_to_redo_with]
nobs = nobs[inds_to_redo_with]

# let's plot a couple other things separately
ace_vals = samples[:,2].reshape(-1,1)

ace_ft = LinearRegression().fit(ace_vals, nobs)
corr_score = ace_ft.score(ace_vals, nobs)
pred = ace_ft.predict(ace_vals)
plt.plot(ace_vals, pred, color="r", label="prediction, score: {:.3f}".format(corr_score))
sbrn.scatterplot(ace_vals[:,0], nobs)
plt.legend(frameon=False)
plt.xlabel("log10(ACE2 binding)")
plt.ylabel(key)
plt.savefig("ace2_binding.png")
# plt.savefig("ace2_binding.pdf")
plt.close()

ft = LinearRegression().fit(samples, nobs)
corr_score = ft.score(samples, nobs)
uv = ft.coef_/np.sqrt(ft.coef_.dot(ft.coef_))

projs = np.dot(samples, ft.coef_)
pred = ft.predict(samples)
c = ft.coef_
c = c/np.sqrt(c.dot(c))

sbrn.scatterplot(projs, nobs, label="data")
sbrn.scatterplot(projs[-1], [nobs[-1]], color="k", label="initial value")
plt.plot(projs, pred, color="r", label="prediction R2: {:.3f}".format(corr_score))
plt.legend(frameon=False)
plt.title("IFN para: {:.3f}, viral diff: {:.3f}, ACE2 bind: {:.3f}".format(c[0],c[1],c[2]))
plt.xlabel("projections")
plt.ylabel(key)
plt.savefig("sens_scatter_limited.png")
plt.close()

print("Fit to log10 of {}".format(key))
print("Score: {}".format(corr_score))
print("IFN para: {:.3f}, viral diff: {:.3f}, ACE2 bind: {:.3f}".format(c[0],c[1],c[2]))
print("normalized coefficients:  {}".format(uv))
print("sample values - init vec: {}, proj: {:.3f}, obs: {}, log10 obs: {:.3f}".format(10**samples[0], projs[-1], int(nobs[-1]), nobs[-1]))

sys.exit() 

# uv = ft.coef_
examples_id = nobs < 3.0
examples = samples[examples_id]
ex_projs = np.dot(examples, ft.coef_)
print(ex_projs.min(), ex_projs.max())
scales = np.linspace(-0.5, 2.0, num=15)

if os.path.exists("scaling.pickle"):
# if False:
    with open("scaling.pickle", "rb") as f:
        scaling_res = pickle.load(f)
    psampl = scaling_res["samples"]
    gobs = scaling_res["obs"]
    ndata = scaling_res["data"]
    scl_used = scaling_res["scl_used"]
else:
    from run_sens import run_repl_sens as rrs

    psampl = []
    ndata = []
    gobs = []
    scl_used = []
    ivec = np.log10([1.97707601e+00, 2.05523573e+00, 1.03156345e-03, 4.90820699e-05])
    print(f"initial vector {ivec}")
    cnt = 0
    for isc, scl in enumerate(scales):
        scaled_vector = scl*uv
        samp = ivec + scaled_vector
        samp_proj = np.dot(samp, ft.coef_)
        print(f"sample vector {samp}")
        print(f"projection {np.dot(samp, ft.coef_)}")
        dat = rrs(samp)
        psampl.append(samp)
        ndata.append(dat)
        gobs.append(get_observable(dat))
        scl_used.append(scl)
        print("#### Sample {} end ####".format(isc))
        os.chdir(cur_dir)
        cnt += 1

    scaling_res = {}
    scaling_res["samples"] = psampl
    scaling_res["obs"] = gobs
    scaling_res["data"] = ndata
    scaling_res["scl_used"] = scl_used
    
    with open("scaling.pickle", "wb") as f:
        pickle.dump(scaling_res, f)
    
# print(scales)
#print(gobs)
# import IPython;IPython.embed()
#import ipdb;ipdb.set_trace()
new_proj = np.array(psampl).dot(ft.coef_)
gobs = np.array(gobs)
zinds = gobs != 0
gobs = gobs[zinds]
new_proj = new_proj[zinds]
scl_used = np.array(scl_used)
scl_used = scl_used[zinds]
# gobs[zinds] = np.nan
# gobs = np.log10(gobs)
#gobs[zinds] = 0
# plt.plot(new_proj[2:], np.log10(gobs)[2:], lw=0, marker="+", markersize=15, color="k")
plt.plot(scl_used[:5], np.log10(gobs)[:5], lw=0, marker="+", markersize=15, color="k")
# plt.plot(scl_used, gobs, lw=0, marker="+", markersize=15, color="k")
plt.xlabel("scalar factor")
# plt.ylabel("finall live cell count")
plt.ylabel("$log_{10}$(final alive cell count)")
plt.savefig("scaling_psyn.png")
# plt.savefig("scaling_psyn.pdf")
