import os, pyDOE, pickle, xmltodict, shutil
import covid_lib as cv
import numpy as np

cur_dir = os.getcwd()
template_config = os.path.abspath("PhysiCell_settings.xml")

IFN_para_key = ['PhysiCell_settings','cell_definitions','cell_definition',0,'custom_data','max_interferon_secretion_rate_via_paracrine','#text']
viral_diff_key = ['PhysiCell_settings','microenvironment_setup','variable',0,'physical_parameter_set','diffusion_coefficient','#text'] 
ACE2_viral_binding_key = ['PhysiCell_settings','cell_definitions','cell_definition',0,'custom_data','ACE2_binding_rate','#text']

keys = [IFN_para_key, viral_diff_key, ACE2_viral_binding_key]

def get_val(cdict, key_list):
    if len(key_list) != 1:
        return get_val(cdict[key_list[0]], key_list[1:])
    return cdict[key_list[0]]

def write_repl_sens(param_arr, config_path=None):
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
    sim.write_config(config_name=config_path)

if __name__ == "__main__":
    main_dir = os.getcwd()
    run_pickle_name = "sens_v5_folders.pickle"
    bin_path = os.path.abspath("COVID19")
    # initialize run
    print(f"## INITIALIZING RUN")
    # important values for the run
    ndim = len(keys)
    nsamples = 2000
    nsims = 3
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
    # store them in an object and save, for checkpointing    
    sens_run = {}
    ctr = 0
    sens_run["data"] = []
    sens_run["obs"] = []
    sens_run["nsims"] = nsims
    sens_run["all_samples"] = lhs_samples
    
    with open(run_pickle_name, "wb") as f:
        pickle.dump(sens_run, f)
        
    # done initializing run, now start 
    while ctr < nsims:
        samp = lhs_samples[ctr]
        fold = f"repl_{ctr:04d}"
        fold_path = os.path.join(main_dir, fold)
        if not os.path.exists(fold_path):
            os.mkdir(fold_path)
        conf = "config.xml"
        full_fold = os.path.abspath(os.path.join(*[main_dir, fold, conf]))
        write_repl_sens(samp, full_fold)
        shutil.copy(bin_path, fold_path)
        ctr += 1    
