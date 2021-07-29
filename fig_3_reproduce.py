# Imports

import pandas as pd
from edbo.utils import Data
import pdb

#############################
#############################
##### REACTION ENCODING #####
#############################
#############################

print("Starting Reaction Encoding!")

# Load DFT descriptor CSV files computed with auto-qchem using pandas
# Instantiate a Data object


# Suzuki here
electrophiles = Data(pd.read_csv('data/suzuki/electrophile_dft.csv'))
nucleophiles = Data(pd.read_csv('data/suzuki/nucleophile_dft.csv'))
ligands = Data(pd.read_csv('data/suzuki/ligand-random_dft.csv'))
bases = Data(pd.read_csv('data/suzuki/base_dft.csv'))
solvents = Data(pd.read_csv('data/suzuki/solvent_dft.csv'))
reactants = [electrophiles, nucleophiles, ligands, bases, solvents]


print("Loaded csv files...")

# Use Data.drop method to drop descriptors containing some unwanted keywords

for data in reactants:
    data.drop(['file_name', 'vibration', 'correlation', 'Rydberg', 
               'correction', 'atom_number', 'E-M_angle', 'MEAN', 'MAXG', 
               'STDEV'])

print("Dropped unnecessary data...")

# Parameters in reaction space

# Suzuki here
components = {
    'electrophile':'DFT',
    'nucleophile':'DFT',
    'ligand':'DFT',
    'base':'DFT',
    'solvent':'DFT'
}


# External descriptor matrices override specified encoding


dft = {
    'electrophile':electrophiles.data,
    'nucleophile':nucleophiles.data,
    'ligand':ligands.data,
    'base':bases.data,
    'solvent':solvents.data
}

encoding = {}

############################
############################
#### Instantiating EDBO ####
############################
############################

FOLDER_PATH = "test_bo_suzuki/"

from edbo.bro import BO_express, BO
# BO object
bo = BO_express(components,                                 # Reaction parameters
                encoding=encoding,                          # Encoding specification
                descriptor_matrices=dft,                    # DFT descriptors
                acquisition_function='EI',                  # Use expectation value of improvement
                init_method='rand',                         # Use random initialization
                batch_size=5,                              # 10 experiments per round
                target='yield')                             # Optimize yield
print("Instantiated BO object...")
# BO_express actually automatically chooses priors
# We can reset them manually to make sure they match the ones from our paper
from gpytorch.priors import GammaPrior
bo.lengthscale_prior = [GammaPrior(2.0, 0.2), 5.0]
bo.outputscale_prior = [GammaPrior(5.0, 0.5), 8.0]
bo.noise_prior = [GammaPrior(1.5, 0.5), 1.0]
print("Constructed priors")

print(reaction.data)