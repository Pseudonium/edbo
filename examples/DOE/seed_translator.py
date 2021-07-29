# Imports

import pandas as pd
from edbo.utils import Data
import pdb
import random

END_MATCH = "_50.csv"

FOLDER_DICT = {
    'suzuki': 'data/suzuki/',
    'aryl': 'data/aryl_amination/',
    'direct': 'data/direct_arylation/'
}



INDEX_STR = 'experiment_index.csv'

class Reaction:
    def set_reactants(self):
        pass
    
    def drop_labels(self):
        for data in self.reactants:
            data.drop(['file_name', 'vibration', 'correlation', 'Rydberg', 
               'correction', 'atom_number', 'E-M_angle', 'MEAN', 'MAXG', 
               'STDEV'])
    
    def set_reaction_params(self):
        pass
    
    def __init__(self):
        self.set_reactants()
        self.drop_labels()
        self.set_reaction_params()
        self.get_results()
    
    def get_results(self):
        with open(FOLDER_DICT[self.name] + INDEX_STR) as f:
            self.FULL_RESULT_DICT = {",".join(line.split(",")[1:-1]): float(line.split(",")[-1][:-1]) for line in f.readlines()[1:]}
        
    

class Suzuki(Reaction):
    def set_reactants(self):
        self.electrophiles = Data(pd.read_csv('data/suzuki/electrophile_dft.csv'))
        self.nucleophiles = Data(pd.read_csv('data/suzuki/nucleophile_dft.csv'))
        self.ligands = Data(pd.read_csv('data/suzuki/ligand-random_dft.csv'))
        self.bases = Data(pd.read_csv('data/suzuki/base_dft.csv'))
        self.solvents = Data(pd.read_csv('data/suzuki/solvent_dft.csv'))
        self.reactants = [
            self.electrophiles,
            self.nucleophiles,
            self.ligands,
            self.bases,
            self.solvents
        ]
    
    def set_reaction_params(self):
        self.components = {
            'electrophile':'DFT',
            'nucleophile':'DFT',
            'ligand':'DFT',
            'base':'DFT',
            'solvent':'DFT'
        }
        self.dft = {
            'electrophile':self.electrophiles.data,
            'nucleophile':self.nucleophiles.data,
            'ligand':self.ligands.data,
            'base':self.bases.data,
            'solvent':self.solvents.data
        }
        self.encoding = {}
        self.FOLDER_PATH = "test_bo_suzuki/"
        self.name = 'suzuki'

    
    
class Aryl_Amination(Reaction):
    def set_reactants(self):
        self.aryl_halides = Data(pd.read_csv('data/aryl_amination/aryl_halide_dft.csv'))
        self.additives = Data(pd.read_csv('data/aryl_amination/additive_dft.csv'))
        self.bases = Data(pd.read_csv('data/aryl_amination/base_dft.csv'))
        self.ligands = Data(pd.read_csv('data/aryl_amination/ligand_ohe.csv'))
        self.reactants = [self.aryl_halides, self.additives, self.bases, self.ligands]
    
    def set_reaction_params(self):
        self.components = {
            'aryl_halide':'DFT',
            'additive':'DFT',
            'base':'DFT',
            'ligand':[
                'CC(C)C1=CC(C(C)C)=CC(C(C)C)=C1C2=C(P(C3CCCCC3)C4CCCCC4)C=CC=C2', 'CC(C)C(C=C(C(C)C)C=C1C(C)C)=C1C2=CC=CC=C2P(C(C)(C)C)C(C)(C)C', 'CC(C)C1=CC(C(C)C)=CC(C(C)C)=C1C2=C(P(C(C)(C)C)C(C)(C)C)C(OC)=CC=C2OC', 'CC(C1=C(C2=C(OC)C=CC(OC)=C2P(C34CC5CC(C4)CC(C5)C3)C67CC8CC(C7)CC(C8)C6)C(C(C)C)=CC(C(C)C)=C1)C'
            ]
        }
        self.dft = {
            'aryl_halide':self.aryl_halides.data,
            'additive':self.additives.data,
            'base':self.bases.data
        }
        self.encoding = {}
        self.FOLDER_PATH = "test_bo_aryl_amination/"
        self.name = 'aryl'
        

class Direct_Arylation(Reaction):
    def set_reactants(self):
        self.bases = Data(pd.read_csv('data/direct_arylation/base_dft.csv'))
        self.ligands = Data(pd.read_csv('data/direct_arylation/ligand-boltzmann_dft.csv'))
        self.solvents = Data(pd.read_csv('data/direct_arylation/solvent_dft.csv'))
        self.reactants = [self.bases, self.ligands, self.solvents]    

    def set_reaction_params(self):
        self.components = {
            'base':'DFT',
            'ligand':'DFT',
            'solvent':'DFT',
            'Concentration':[0.057, 0.1, 0.153],
            'Temp_C':[90, 105, 120]
        }
        self.dft = {
            'base':self.bases.data,
            'ligand':self.ligands.data,
            'solvent':self.solvents.data
        }
        self.encoding = {
            'Concentration':'numeric',
            'Temp_C':'numeric'
        }
        self.FOLDER_PATH = "test_bo_direct_arylation/"
        self.name = 'direct'
        

class_dict = {
    'suzuki': Suzuki,
    'aryl': Aryl_Amination,
    'direct': Direct_Arylation
}




def translate_seed_random(bo, reaction, seed=0, batch_size=0):
    bo.init_sample(seed=seed)
    bo.export_proposed('init.csv')
    yields = []
    with open('init.csv') as f:
        first_line = True
        for line in f:
            if first_line:
                first_line = False
                continue
            line = line.split(",")
            search_string = ",".join(line[1:-1])
            input_yield = reaction.FULL_RESULT_DICT[search_string]
            yields.append(str(input_yield))
    return yields

def get_worst_percentile(full_result_dict, centile=10):
    number_of_results = len(full_result_dict)
    sorted_by_yield = sorted(full_result_dict.items(), key=lambda item: item[1])
    bottom_centile = sorted_by_yield[:int(0.01 * centile * number_of_results)]
    return dict(bottom_centile)

def translate_seed_worst(bo, reaction, seed=0, batch_size=0):
    random.seed(seed)
    worst_result_dict = get_worst_percentile(reaction.FULL_RESULT_DICT, centile=10)
    yields = []
    for result in random.sample(list(worst_result_dict.keys()), batch_size):
        yields.append(str(worst_result_dict[result]))
    
    return yields

def translate_seed_file(filepath):
    name = filepath.split("_")[0]
    folder = FOLDER_DICT.get(name, "")
    if not folder:
        return None
    # Following line gives first element that matches this condition
    batch_size = int(next((item for item in filepath.split("_") if item.isnumeric()), "0"))
    if not batch_size:
        return None
    cls = class_dict[name]
    reaction = cls()
    if 'worst' in filepath:
        translate_seed_func = translate_seed_worst
    else:
        translate_seed_func = translate_seed_random
    from edbo.bro import BO_express, BO
    # BO object
    bo = BO_express(reaction.components,
                    encoding=reaction.encoding,
                    descriptor_matrices=reaction.dft,
                    acquisition_function='EI',
                    init_method='rand',
                    batch_size=batch_size,
                    target='yield')
    print("Instantiated BO object...")
    from gpytorch.priors import GammaPrior
    bo.lengthscale_prior = [GammaPrior(2.0, 0.2), 5.0]
    bo.outputscale_prior = [GammaPrior(5.0, 0.5), 8.0]
    bo.noise_prior = [GammaPrior(1.5, 0.5), 1.0]
    print("Constructed priors")
    newfile = ""
    new_filepath = filepath[:-4] + "_translated.csv" # Slicing removes the .csv from the end
    first_line = True
    with open(filepath) as f:
        for line in f:
            if first_line:
                newfile += line
                first_line = False
                continue
            seed = int(line.split(",")[0])
            line = line[:-1] # Strips the \n from the end
            line += "," + ",".join(translate_seed_func(bo, reaction, seed=seed, batch_size=batch_size)) + "\n"
            newfile += line
    
    with open(new_filepath, 'w') as f:
        f.write(newfile)

import os

for filename in os.listdir():
    if filename.endswith(END_MATCH):
        translate_seed_file(filename)
    