import tempfile, subprocess, xmltodict, glob
import shutil, sys, os, pickle
# sys.path.append('../../../')
from pyMCDS import pyMCDS
from scipy.stats import sem
import numpy as np

def make_change(xdict, cpath, change_to):
    if len(cpath) > 1:
        make_change(xdict[cpath[0]], cpath[1:], change_to)
    else:
        xdict[cpath[0]] = str(change_to)

class runFolder:
    def __init__(self, path):
        self.path = path
        self.old_dir = os.getcwd() 

    def __enter__(self):
        os.chdir(self.path)

    def __exit__(self, t, v, tr):
        # print(t,v,tr)
        os.chdir(self.path)

class COVIDSimulation:
    def __init__(self, config, simulator, output=None, changes=[]):
        # we need at least one base config file
        with open(config, "r") as f:
            lines = f.read()
            self.xmldict = xmltodict.parse(lines)
        self.base_config_file = config
        # we need the path to simulator
        self.simulator = simulator
        self.simulator_name = os.path.basename(self.simulator)
        # optinally we'll have changes to 
        # things so we can scan values
        # changes are formatted as a tuple
        # (path_to_change, change_value_to) 
        # and path to change is a list where the change 
        # needs to occur, e.g. dict["A"]["B"]["C"] change
        # list is ["A", "B", "C"]
        self.changes = changes
        # set output
        self.output = output

    def write_config(self, config_name="config.xml"):
        for change in self.changes:
            cpath, change_to = change
            make_change(self.xmldict, cpath, change_to)
        with open(config_name, "w") as f:
            f.write(xmltodict.unparse(self.xmldict))
        return config_name

    def simulate(self):
        if self.output is None:
            temp_fld = True
            self.output = tempfile.mkdtemp()
        else:
            temp_fld = False
            if os.path.isdir(self.output):
                shutil.rmtree(self.output)
            os.mkdir(self.output)
        cur_dir = os.getcwd()
        os.chdir(self.output)
        shutil.copy(self.simulator, os.getcwd())
        config_name = self.write_config()
        os.mkdir("output")
        try:
            rc = subprocess.run([os.path.abspath(self.simulator_name), config_name])
        except:
            print("run failed")
            os.chdir(cur_dir)
            if temp_fld:
                shutil.rmtree(self.output)
        try:
            self.data = self.analyze()
        except:
            print("analysis failed")
            os.chdir(cur_dir)
            if temp_fld:
                shutil.rmtree(self.output)
        os.chdir(cur_dir)
        # now that we are done, clean up
        # if it was a temporary directory
        if temp_fld:
            shutil.rmtree(self.output)

    def analyze(self, path=None):
        if path is None:
            path = os.getcwd()
        with runFolder(path) as rf:
            # process output
            data = {}
            virion_dat = []
            cell_dat = []
            disc_vir = []
            immune_dat = []
            cont_immune = []
            frac_inf = []
            inf_cells = []
            assembled_vir = []
            uncoated_virion = []
            viral_RNA = []
            viral_protein = []
            # export_virion = []
            # burst_size = []
            tlen = len(glob.glob("output/output*.xml"))
            # import ipdb;ipdb.set_trace()
            # import IPython;IPython.embed()
            for i in range(0,tlen-1):
                mcds = pyMCDS("output{:08d}.xml".format(i), "output")
                total_volume = 8000 
                virion_dens = mcds.data['continuum_variables']['virion']['data']
                virions = (virion_dens*total_volume).sum()
                virion_dat.append(virions)

                vir_arr = mcds.data['discrete_cells']['virion']

                infected = vir_arr >= 1
                frac_inf_tp = len(np.where(infected)[0])/len(vir_arr) if len(vir_arr) != 0 else 0
                frac_inf.append(frac_inf_tp)
                inf_cells.append(len(np.where(infected)[0]))

                disc_vir_cnt = vir_arr.sum()
                disc_vir.append(disc_vir_cnt)

                assembl_vir_cnt = mcds.data['discrete_cells']['assembled_virion'].sum()
                assembled_vir.append(assembl_vir_cnt)

                uncoated_virion.append(mcds.data['discrete_cells']['uncoated_virion'].sum())
                viral_RNA.append(mcds.data['discrete_cells']['viral_RNA'].sum())
                viral_protein.append(mcds.data['discrete_cells']['viral_protein'].sum())
                # export_virion.append(mcds.data['discrete_cells']['exported_virions'].sum())

                live_cell_cnt = (mcds.data['discrete_cells']['cell_type']==1).sum()
                cell_dat.append(live_cell_cnt)

                immune_cnt = (mcds.data['discrete_cells']['cell_type']>1).sum()
                immune_dat.append(immune_cnt)

                cont_immune_cnt = mcds.data['continuum_variables']['interferon 1']['data'].sum()
                cont_immune_cnt += mcds.data['continuum_variables']['pro-inflammatory cytokine']['data'].sum()
                cont_immune_cnt += mcds.data['continuum_variables']['chemokine']['data'].sum()
                cont_immune_cnt += mcds.data['continuum_variables']['debris']['data'].sum()
                cont_immune_cnt = cont_immune_cnt * total_volume
                cont_immune.append(cont_immune_cnt)

            # burst_size = float(np.loadtxt("exported_virions.txt"))

            data["virion_data"] = virion_dat
            data["discrete_virions"] = disc_vir 
            data["cell_data"] = cell_dat 
            data["immune_data"] = immune_dat 
            data["cont_immune"] = cont_immune
            data["frac_inf"] = frac_inf
            data["assembled_virions"] = assembled_vir
            data["infected_cells"] = inf_cells
            data["uncoated_virions"] = uncoated_virion 
            data["viral_RNA"] = viral_RNA 
            data["viral_protein"] = viral_protein
            # data["exported_virions"] = export_virion 
            # data['burst_size'] = burst_size/cell_dat[0]
        return data

    def save_data(self, fpath):
        with open(fpath, "wb") as f:
            pickle.dump(self.data, f)

if __name__ == "__main__":
    cur_dir = os.getcwd()
    template_config = os.path.abspath("Viral_replication_no_immune_no_uptake.xml")
    vir_key = ['PhysiCell_settings','cell_definitions','cell_definition',0,'custom_data','virion','#text']
    # vir_key = ['PhysiCell_settings','cell_definitions','cell_definition',0,'custom_data','basal_RNA_degradation_rate','#text']
    # vir_key = ['PhysiCell_settings','user_parameters','multiplicity_of_infection','#text']
    vir_dat = {}
    # for vir in ["0.012", "0.01", "0.008"]:
    for vir in ["10", "1", "0.1", "0.01"]:
    # for vir in ["10"]:
        os.chdir(cur_dir)
        # this needs path to a template configuration xml and a full
        # path to the COVID19 binary. Optionally it can take in 
        # potential parameter changes
        sim = COVIDSimulation(template_config, "/home/monoid/PROJECTS/Physicell_viral/own_fork/dev-mg/COVID19/PhysiCell/COVID19", 
                changes=[(vir_key, vir)])
                # output="/home/monoid/PROJECTS/Physicell_viral/own_fork/dev-mg/COVID19/PhysiCell/scripts/viral_replication/no_uptake_apoptosis/test_{}".format(vir),
        sim.simulate()
        vir_dat[vir] = sim.data

    os.chdir(cur_dir)
    with open("sim_res.pickle", "wb") as f:
        pickle.dump(vir_dat, f)
