# Imports

### Going to use custom edbo implementation (changed the acq_func.py file to add my custom functions)
## So, need to point to that folder
# Answer via https://stackoverflow.com/questions/4383571/importing-files-from-different-folder

import sys
sys.path.insert(1, '../..')

import pandas as pd
#from edbo.utils import Data
#from edbo.bro import BO_express, BO

## This ensures we import from the custom files

from edbo.utils import Data
from edbo.bro import BO_express, BO

from gpytorch.priors import GammaPrior
import numpy as np
import matplotlib.pyplot as plt
import random
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

def simulate(seed=1, RESULT_PATH="", BATCH_SIZE=5, NUM_ROUNDS=5):

    # BO object
    bo = BO_express(components,                                 # Reaction parameters
                    encoding=encoding,                          # Encoding specification
                    descriptor_matrices=dft,                    # DFT descriptors
                    acquisition_function='TS-EI',                  # Use expectation value of improvement
                    ########################
                    #######################
                    ##### USING THOMPSON SAMPLING with EI
                    #######################
                    init_method='rand',                         # Use random initialization
                    batch_size=BATCH_SIZE,                              # 10 experiments per round
                    target='yield')                             # Optimize yield
    print("Instantiated BO object...")
    # BO_express actually automatically chooses priors
    # We can reset them manually to make sure they match the ones from our paper
    bo.lengthscale_prior = [GammaPrior(2.0, 0.2), 5.0]
    bo.outputscale_prior = [GammaPrior(5.0, 0.5), 8.0]
    bo.noise_prior = [GammaPrior(1.5, 0.5), 1.0]
    print("Constructed priors")
    ########################
    ########################
    #### Initialization ####
    ########################
    ########################
    print(bo.init_sample(seed=seed + 1))
    print("HEY WAIT A MINUTE")
    print(bo.init_sample(seed=seed))             # Initialize
    
    bo.export_proposed(FOLDER_PATH + 'init.csv')     # Export design to a CSV file
    print(bo.get_experiments())               # Print selected experiments
    ####################################
    ####################################
    #### Bayesian Optimization Loop ####
    ####################################
    ####################################
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

    fill_in_experiment_values(FOLDER_PATH + 'init.csv')
    bo.add_results(FOLDER_PATH + 'init.csv')
    def plot_kb_projections(n=2):
        #Plot 1D projection of Kriging believer parallel batch selection algorithm.

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
        #Function for our BO pipeline.
        
        bo.run()
        #bo.plot_convergence()
        #bo.model.regression()
        #plot_kb_projections()
        bo.export_proposed(export_path)

    for num in range(NUM_ROUNDS):
        print("Starting round ", num)
        #pdb.set_trace()
        try:
            workflow(FOLDER_PATH + 'round' + str(num) + '.csv')
        except RuntimeError as e:
            print(e)
            print("No idea how to fix this, seems to occur randomly for different seeds...")
            break
        fill_in_experiment_values(FOLDER_PATH + 'round' + str(num) + '.csv')
        bo.add_results(FOLDER_PATH + "round" + str(num) + ".csv")
        print("Finished round ", num)


    results = pd.DataFrame(columns=bo.reaction.index_headers + ['yield'])
    for path in [FOLDER_PATH + 'init'] + [FOLDER_PATH + 'round' + str(num) for num in range(NUM_ROUNDS)]:
        results = pd.concat([results, pd.read_csv(path + '.csv', index_col=0)], sort=False)

    results = results.sort_values('yield', ascending=False)

    top_yields = results.head()['yield'].tolist()

    #bo.plot_convergence()
    
    return top_yields[0]




#Format is reaction_choosingmethod_batchsize_experimentbudget_numberofrunsdone
# Key of choosingmethods:
# random - chosen at random using expected improvement as acquisition function
# worst - randomly chosen from bottom 10% of experiments using expected improvement
# randomts - chosen at random using thompson sampling
# randomtsei - chosen at random using hybrid thompson sampling and expected improvement (my own modification, not the ei-ts builtin


results_file_path = "suzuki_randomtsei_5_50_50.csv"
results_file = "seed,maximum observed yield" + "\n"

count = 0

for num in random.sample(range(10 ** 6), 50):
    print("On number ", count)
    count += 1
    print("SEED HERE IS ", num)
    simulation_result = simulate(seed=num, RESULT_PATH='data/suzuki/experiment_index.csv', BATCH_SIZE=5, NUM_ROUNDS=10)
    results_file += str(num) + "," + str(simulation_result) + "\n"


#Format is reaction_choosingmethod_batchsize_experimentbudget_numberofrunsdone

results_file_path = "suzuki_randomtsei_10_50_50.csv"
results_file = "seed,maximum observed yield" + "\n"

count = 0

for num in random.sample(range(10 ** 6), 50):
    print("On number ", count)
    count += 1
    print("SEED HERE IS ", num)
    simulation_result = simulate(seed=num, RESULT_PATH='data/suzuki/experiment_index.csv', BATCH_SIZE=10, NUM_ROUNDS=5)
    results_file += str(num) + "," + str(simulation_result) + "\n"


#Format is reaction_choosingmethod_batchsize_experimentbudget_numberofrunsdone

results_file_path = "suzuki_randomtsei_3_51_50.csv"
results_file = "seed,maximum observed yield" + "\n"

count = 0

for num in random.sample(range(10 ** 6), 50):
    print("On number ", count)
    count += 1
    print("SEED HERE IS ", num)
    simulation_result = simulate(seed=num, RESULT_PATH='data/suzuki/experiment_index.csv', BATCH_SIZE=3, NUM_ROUNDS=17)
    results_file += str(num) + "," + str(simulation_result) + "\n"