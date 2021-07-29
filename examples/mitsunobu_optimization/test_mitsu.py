# Imports

import pandas as pd
from edbo.utils import Data

#############################
#############################
##### REACTION ENCODING #####
#############################
#############################

print("Starting Reaction Encoding!")

# Load DFT descriptor CSV files computed with auto-qchem using pandas
# Instantiate a Data object

azadicarbs = Data(pd.read_csv('descriptors/azadicarboxylate_boltzmann_dft.csv'))
phosphines = Data(pd.read_csv('descriptors/phosphine_boltzmann_dft.csv'))
solvents = Data(pd.read_csv('descriptors/solvent_dft.csv'))

print("Loaded csv files...")

# Use Data.drop method to drop descriptors containing some unwanted keywords

for data in [azadicarbs, phosphines, solvents]:
    data.drop(['file_name', 'entry', 'vibration', 'correlation', 'Rydberg', 
               'correction', 'atom_number', 'E-M_angle', 'MEAN', 'MAXG', 
               'STDEV'])
               
print("Dropped unnecessary data...")

# Parameters in reaction space

# Parameters in reaction space

components = {'azadicarboxylate':'DFT',                             # DFT descriptors
              'phosphine':'DFT',                                    # DFT descriptors
              'solvent':'DFT',                                      # DFT descriptors
              'substrate_concentration':[0.05, 0.10, 0.15, 0.20],   # Discrete grid of concentrations
              'azadicarb_equiv':[1.1, 1.3, 1.5, 1.7, 1.9],          # Discrete grid of equiv.
              'phos_equiv':[1.1, 1.3, 1.5, 1.7, 1.9],               # Discrete grid of equiv.
              'temperature':[5, 15, 25, 35, 45]}                    # Discrete grid of temperatures

# Encodings - if not specified EDBO will automatically use OHE

encoding = {'substrate_concentration':'numeric',                    # Numerical encoding
            'azadicarb_equiv':'numeric',                            # Numerical encoding
            'phos_equiv':'numeric',                                 # Numerical encoding
            'temperature':'numeric'}                                # Numerical encoding

# External descriptor matrices override specified encoding

dft = {'azadicarboxylate':azadicarbs.data,                          # Unprocessed descriptor DataFrame
       'phosphine':phosphines.data,                                 # Unprocessed descriptor DataFrame
       'solvent':solvents.data}                                     # Unprocessed descriptor DataFrame

############################
############################
#### Instantiating EDBO ####
############################
############################


from edbo.bro import BO_express

# BO object

bo = BO_express(components,                                 # Reaction parameters
                encoding=encoding,                          # Encoding specification
                descriptor_matrices=dft,                    # DFT descriptors
                acquisition_function='EI',                  # Use expectation value of improvement
                init_method='rand',                         # Use random initialization
                batch_size=10,                              # 10 experiments per round
                target='yield')                             # Optimize yield

print("Instantiated bo object")

# BO_express actually automatically chooses priors
# We can reset them manually to make sure they match the ones from our paper

from gpytorch.priors import GammaPrior

bo.lengthscale_prior = [GammaPrior(2.0, 0.2), 5.0]
bo.outputscale_prior = [GammaPrior(5.0, 0.5), 8.0]
bo.noise_prior = [GammaPrior(1.5, 0.5), 1.0]

########################
########################
#### Initialization ####
########################
########################

bo.init_sample(seed=0)             # Initialize
bo.export_proposed('init.csv')     # Export design to a CSV file
bo.get_experiments()               # Print selected experiments

####################################
####################################
#### Bayesian Optimization Loop ####
####################################
####################################

bo.add_results('results/init.csv')

import numpy as np
import matplotlib.pyplot as plt

def plot_kb_projections(n=2):
    """
    Plot 1D projection of Kriging believer parallel batch selection algorithm.
    """

    fig, ax = plt.subplots(len(bo.acq.function.projections[:n]),1, figsize=(12, n * 12 / 5))
    for i, p in enumerate(bo.acq.function.projections[:n]):
        ax[i].plot(range(len(p)), p, color='C' + str(i))
        ax[i].plot([np.argmax(p)], p[np.argmax(p)], 'X', markersize=10, color='black')
        ax[i].set_xlabel('X')
        ax[i].set_ylabel('EI')
        ax[i].set_title('Kriging Believer Projection:' + str(i))
    
    plt.tight_layout()
    plt.show()

def workflow(export_path):
    """
    Function for our BO pipeline.
    """
    
    bo.run()
    bo.plot_convergence()
    bo.model.regression()
    plot_kb_projections()
    bo.export_proposed(export_path)


for num in range(3):
    print("Starting round ", num)
    bo.add_results("results/round" + str(num) + ".csv")
    workflow('round' + str(num + 1) + '.csv')
    print("Finished round ", num)


results = pd.DataFrame(columns=bo.reaction.index_headers + ['yield'])
for path in ['init', 'round0', 'round1', 'round2']:
    results = pd.concat([results, pd.read_csv('results/' + path + '.csv', index_col=0)], sort=False)

results = results.sort_values('yield', ascending=False)
results.head()

fig, ax = plt.subplots(1, len(results.columns.values[:-1]), figsize=(30,5))

for i, feature  in enumerate(results.columns.values[:-1]):
    results[feature].iloc[:5].value_counts().plot(kind="bar", ax=ax[i]).set_title(feature)
plt.show()


from edbo.chem_utils import ChemDraw

for col in results.iloc[:,:3].columns.values:
    print('\nComponent:', col, '\n')
    cdx = ChemDraw(results[col].iloc[:5].drop_duplicates())
    cdx.show()