# Imports

import pandas as pd
from edbo.utils import Data
import pdb
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib
from sklearn.cluster import KMeans
import math
import numpy as np

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
                acquisition_function='VarMax',                  # Use expectation value of improvement
                init_method='rand',                         # Use random initialization
                batch_size=1,                              # 10 experiments per round
                target='yield')                             # Optimize yield
print("Instantiated BO object...")
# BO_express actually automatically chooses priors
# We can reset them manually to make sure they match the ones from our paper
from gpytorch.priors import GammaPrior
bo.lengthscale_prior = [GammaPrior(2.0, 0.2), 5.0]
bo.outputscale_prior = [GammaPrior(5.0, 0.5), 8.0]
bo.noise_prior = [GammaPrior(1.5, 0.5), 1.0]
print("Constructed priors")

data_embedded = TSNE(init='pca').fit_transform(bo.reaction.data)

N_CLUSTERS=5
kmeans = KMeans(n_clusters=N_CLUSTERS).fit_predict(bo.reaction.data)
cm = matplotlib.cm.get_cmap(name='viridis')
norm = matplotlib.colors.Normalize(vmin=0.0, vmax=N_CLUSTERS)
colors = [cm(norm(thing)) for thing in kmeans]

"""
fig, axs = plt.subplots(3)


for axes in axs:
    axes.scatter([item[0] for item in data_embedded], [item[1] for item in data_embedded], c=colors)
"""

fig, axs = plt.subplots(1)
axs.scatter([item[0] for item in data_embedded], [item[1] for item in data_embedded], c=colors)


####################################
####################################
#### Bayesian Optimization Loop ####
####################################
####################################

RESULT_PATH='data/suzuki/experiment_index.csv'
NUM_ROUNDS=10

path_cm = matplotlib.cm.get_cmap(name='Reds')
path_norm = matplotlib.colors.Normalize(vmin=0.0, vmax=NUM_ROUNDS)

with open(RESULT_PATH) as f:
    FULL_RESULT_DICT = {",".join(line.split(",")[1:-1]): float(line.split(",")[-1][:-1]) for line in f.readlines()[1:]}

def fill_in_experiment_values(input_path):
    # Reading in values
    newfile = ""
    with open(input_path) as f:
        # In this case f is a csv file
        first_line = True
        for line in f:
            original_line = line
            if first_line:
                newfile += line
                first_line = False
                continue
            line = line.split(",")
            search_string = ",".join(line[1:-1])
            input_yield = FULL_RESULT_DICT[search_string]
            line = ",".join(original_line.split(",")[:-1]) + "," + str(input_yield) + "\n"
            newfile += line
    with open(input_path, 'w') as f:
        f.write(newfile)


def workflow(export_path, count=0, indices=None, fig=0):
    #Function for our BO pipeline.
    
    if indices is None:
        indices = []
    
    bo.run()
    new_experiment_index = bo.get_experiments().index[0]
    indices.append(new_experiment_index)
    #axs[fig].scatter([data_embedded[new_experiment_index][0]], [data_embedded[new_experiment_index][1]], color=path_cm(path_norm(count)), s=9)
    axs.scatter([data_embedded[new_experiment_index][0]], [data_embedded[new_experiment_index][1]], color=path_cm(path_norm(count)), s=9)
    if len(indices) > 1:
        x, y = data_embedded[indices[count - 1]]
        x_new, y_new = data_embedded[indices[count]]
        dx, dy = x_new - x, y_new - y
        """
        axs[fig].arrow(
            x, y, dx, dy,
            width=0.1,
            length_includes_head=True,
            head_width = 3,
            head_length = 3,
            linestyle='--',
            color='black'
        )
        """
        colors = ['black', 'red', 'yellow'] 
        axs.arrow(
            x, y, dx, dy,
            width=0.1,
            length_includes_head=True,
            head_width = 3,
            head_length = 3,
            linestyle='--',
            color=colors[fig]
        )
    #plt.annotate(str(count), xy=data_embedded[new_experiment_index], xycoords='data')
    bo.export_proposed(export_path)
    return indices

indices = None

print(bo.init_sample(seed=213090120))             # Initialize
bo.export_proposed(FOLDER_PATH + 'init.csv')     # Export design to a CSV file
print(bo.get_experiments())               # Print selected experiments
fill_in_experiment_values(FOLDER_PATH + 'init.csv')
bo.add_results(FOLDER_PATH + 'init.csv')

for num in range(NUM_ROUNDS):
    print("Starting round ", num)
    #pdb.set_trace()
    try:
        indices = workflow(FOLDER_PATH + 'round' + str(num) + '.csv', count=num, indices=indices, fig=0)
    except RuntimeError as e:
        print(e)
        print("No idea how to fix this, seems to occur randomly for different seeds...")
        break
    fill_in_experiment_values(FOLDER_PATH + 'round' + str(num) + '.csv')
    bo.add_results(FOLDER_PATH + "round" + str(num) + ".csv")
    print("Finished round ", num)


indices = None

bo = BO_express(components,                                 # Reaction parameters
                encoding=encoding,                          # Encoding specification
                descriptor_matrices=dft,                    # DFT descriptors
                acquisition_function='MeanMax',                  # Use expectation value of improvement
                init_method='rand',                         # Use random initialization
                batch_size=1,                              # 10 experiments per round
                target='yield')                             # Optimize yield
print("Instantiated BO object...")
# BO_express actually automatically chooses priors
# We can reset them manually to make sure they match the ones from our paper
from gpytorch.priors import GammaPrior
bo.lengthscale_prior = [GammaPrior(2.0, 0.2), 5.0]
bo.outputscale_prior = [GammaPrior(5.0, 0.5), 8.0]
bo.noise_prior = [GammaPrior(1.5, 0.5), 1.0]
print("Constructed priors")

print(bo.init_sample(seed=213090120))             # Initialize
bo.export_proposed(FOLDER_PATH + 'init.csv')     # Export design to a CSV file
print(bo.get_experiments())               # Print selected experiments
fill_in_experiment_values(FOLDER_PATH + 'init.csv')
bo.add_results(FOLDER_PATH + 'init.csv')

for num in range(NUM_ROUNDS):
    print("Starting round ", num)
    #pdb.set_trace()
    try:
        indices = workflow(FOLDER_PATH + 'round' + str(num) + '.csv', count=num, indices=indices, fig=1)
    except RuntimeError as e:
        print(e)
        print("No idea how to fix this, seems to occur randomly for different seeds...")
        break
    fill_in_experiment_values(FOLDER_PATH + 'round' + str(num) + '.csv')
    bo.add_results(FOLDER_PATH + "round" + str(num) + ".csv")
    print("Finished round ", num)

indices = None

bo = BO_express(components,                                 # Reaction parameters
                encoding=encoding,                          # Encoding specification
                descriptor_matrices=dft,                    # DFT descriptors
                acquisition_function='EI',                  # Use expectation value of improvement
                init_method='rand',                         # Use random initialization
                batch_size=1,                              # 10 experiments per round
                target='yield')                             # Optimize yield
print("Instantiated BO object...")
# BO_express actually automatically chooses priors
# We can reset them manually to make sure they match the ones from our paper
from gpytorch.priors import GammaPrior
bo.lengthscale_prior = [GammaPrior(2.0, 0.2), 5.0]
bo.outputscale_prior = [GammaPrior(5.0, 0.5), 8.0]
bo.noise_prior = [GammaPrior(1.5, 0.5), 1.0]
print("Constructed priors")

print(bo.init_sample(seed=213090120))             # Initialize
bo.export_proposed(FOLDER_PATH + 'init.csv')     # Export design to a CSV file
print(bo.get_experiments())               # Print selected experiments
fill_in_experiment_values(FOLDER_PATH + 'init.csv')
bo.add_results(FOLDER_PATH + 'init.csv')

for num in range(NUM_ROUNDS):
    print("Starting round ", num)
    #pdb.set_trace()
    try:
        indices = workflow(FOLDER_PATH + 'round' + str(num) + '.csv', count=num, indices=indices, fig=2)
    except RuntimeError as e:
        print(e)
        print("No idea how to fix this, seems to occur randomly for different seeds...")
        break
    fill_in_experiment_values(FOLDER_PATH + 'round' + str(num) + '.csv')
    bo.add_results(FOLDER_PATH + "round" + str(num) + ".csv")
    print("Finished round ", num)


plt.show()