import os, pyDOE, pickle, sys, xmltodict
import covid_lib as cv
import numpy as np
from scipy.stats.distributions import uniform

##### previously used keys/configs
# template_config = os.path.abspath("scripts/viral_replication/viral_sanity_check/Viral_replication_no_virion_uptake_no_apop.xml")
# template_config = os.path.abspath("scripts/viral_replication/no_immune_mois/Viral_replication_no_immune.xml")
# deg_key = ['PhysiCell_settings','cell_definitions','cell_definition',0,'custom_data','basal_RNA_degradation_rate','#text']
# rep_half_key = ['PhysiCell_settings','cell_definitions','cell_definition',0,'custom_data','RNA_replication_half','#text']
# max_rep_key = ['PhysiCell_settings','cell_definitions','cell_definition',0,'custom_data','max_RNA_replication_rate','#text']
# export_key = ['PhysiCell_settings','cell_definitions','cell_definition',0,'custom_data','virion_export_rate','#text']
# synth_key = ['PhysiCell_settings','cell_definitions','cell_definition',0,'custom_data','protein_synthesis_rate','#text']
# dc_leave_key = ['PhysiCell_settings','user_parameters', 'DC_leave_prob','#text']
# ig_req_key = ['PhysiCell_settings','user_parameters', 'Ig_recuitment','#text']
#####

cur_dir = os.getcwd()
template_config = os.path.abspath("PhysiCell_settings.xml")

IFN_para_key = ['PhysiCell_settings','cell_definitions','cell_definition',0,'custom_data','max_interferon_secretion_rate_via_paracrine','#text']
viral_diff_key = ['PhysiCell_settings','microenvironment_setup','variable',0,'physical_parameter_set','diffusion_coefficient','#text'] 
ACE2_viral_binding_key = ['PhysiCell_settings','cell_definitions','cell_definition',0,'custom_data','ACE2_binding_rate','#text']
# tcell_req_key = ['PhysiCell_settings','user_parameters', 'T_Cell_Recruitment','#text']

keys = [IFN_para_key, viral_diff_key, ACE2_viral_binding_key] # , tcell_req_key]

def get_val(cdict, key_list):
    if len(key_list) != 1:
        return get_val(cdict[key_list[0]], key_list[1:])
    return cdict[key_list[0]]


def get_observable(sim_data):
    '''
    Given a simulation data, how you calculate 
    an observable 
    '''
    # e.g. this returns the final cell count
    return sim_data["cell_data"][-1]

def run_repl_sens(param_arr):
    '''
    This an example of how you to run a PhysiCell simulation
    from a given parameter array and get the results 
    '''
    chs = []
    for ip, p in enumerate(param_arr):
        chs.append((keys[ip], 10**p))
    # instantiate simulation
    sim = cv.COVIDSimulation(template_config,
            os.path.abspath("COVID19"), 
            changes=chs)
    # simulate 
    sim.simulate()
    # return data
    return sim.data

if __name__ == "__main__":
    main_dir = os.getcwd()
    run_pickle_name = "sens_v5.pickle"
    # the actual run
    if os.path.exists(run_pickle_name):
        print(f"## FOUND EXISTING RUN, CONTINUING: {run_pickle_name}")
        # continue run
        with open(run_pickle_name, "rb") as f:
            sens_run = pickle.load(f)
        # load to continue 
        data = sens_run["data"]
        obs = sens_run["obs"]
        valid_samples = sens_run["samples"]
        ctr = len(valid_samples)
        nsims = sens_run["nsims"]
        lhs_samples = sens_run["all_samples"]
        isamp = sens_run["current_sample_idx"]
    else:
        # initialize run
        print(f"## INITIALIZING RUN: {run_pickle_name}")
        # important values for the run
        ndim = len(keys)
        nsamples = 2000
        nsims = 5
        scl = 1
        lhs_samples = pyDOE.lhs(ndim, samples=nsamples)
        # get initial values
        # read config
        with open(template_config, "r") as f:
            lines = f.read()
            config_dict = xmltodict.parse(lines)
        # pull initial values
        init_arr = []
        for key in keys:
            init_arr.append(get_val(config_dict, key))
        # initial array, convert to doubles
        init_arr = np.array(init_arr, dtype=np.float64)
        # log10 of initial values 
        means = np.log10(init_arr)
        # get samples
        for i in range(ndim):
            lhs_samples[:,i] *= 2*scl
            lhs_samples[:,i] += means[i] - np.average(lhs_samples,axis=0)[i]
        # keep an eye on the samples
        print("init values: ", init_arr)
        print("log(init values): ", np.log10(init_arr))
        print("means: ", means)
        print("avg: ", np.average(lhs_samples, axis=0))
        print("max: ", np.max(lhs_samples, axis=0))
        print("min: " , np.min(lhs_samples, axis=0))
        print("width: " , np.max(lhs_samples, axis=0)-np.min(lhs_samples, axis=0))
        # run
        sens_run = {}
        data = []
        obs = []
        valid_samples = []
        ctr = 0
        isamp = 0
        sens_run["nsims"] = nsims
        sens_run["all_samples"] = lhs_samples

    # done initializing run, now start 
    while ctr < nsims:
        samp = lhs_samples[isamp]
        try:
            dat = run_repl_sens(samp)
            data.append(dat)
            obs.append(get_observable(dat))
            valid_samples.append(samp)
            print("#### Sample {} done ####".format(isamp))
            ctr += 1
        except:
            print("#### Sample {} failed ####".format(isamp))
            # import IPython;IPython.embed()
            pass
        isamp += 1
        os.chdir(main_dir)
    
        # store them in an object and save, for checkpointing
        sens_run["data"] = data
        sens_run["obs"] = obs
        sens_run["samples"] = valid_samples
        sens_run["current_sample_idx"] = isamp
        
        with open(run_pickle_name, "wb") as f:
            pickle.dump(sens_run, f)
