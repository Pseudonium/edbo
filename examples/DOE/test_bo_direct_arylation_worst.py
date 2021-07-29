# Imports

import pandas as pd
from edbo.utils import Data
import pdb
import random

#############################
#############################
##### REACTION ENCODING #####
#############################
#############################

print("Starting Reaction Encoding!")

# Load DFT descriptor CSV files computed with auto-qchem using pandas
# Instantiate a Data object


# direct arylation here
bases = Data(pd.read_csv('data/direct_arylation/base_dft.csv'))
ligands = Data(pd.read_csv('data/direct_arylation/ligand-boltzmann_dft.csv'))
solvents = Data(pd.read_csv('data/direct_arylation/solvent_dft.csv'))
reactants = [bases, ligands, solvents]

print("Loaded csv files...")

# Use Data.drop method to drop descriptors containing some unwanted keywords

for data in reactants:
    data.drop(['file_name', 'vibration', 'correlation', 'Rydberg', 
               'correction', 'atom_number', 'E-M_angle', 'MEAN', 'MAXG', 
               'STDEV'])

print("Dropped unnecessary data...")

# Parameters in reaction space

# direct arylation here
components = {
    'base':'DFT',
    'ligand':'DFT',
    'solvent':'DFT',
    'Concentration':[0.057, 0.1, 0.153],
    'Temp_C':[90, 105, 120]
}

# External descriptor matrices override specified encoding

# direct arylation here
dft = {
    'base':bases.data,
    'ligand':ligands.data,
    'solvent':solvents.data
}

encoding = {
    'Concentration':'numeric',
    'Temp_C':'numeric'
}

############################
############################
#### Instantiating EDBO ####
############################
############################

FOLDER_PATH = "test_bo_direct_arylation_worst/"

def fill_in_experiment_values(input_path, result_dict):
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
                # Now, unfortunately, it seems like for this experiment certain combinations were not listed on the index
                # I guess I'll assume these experiments failed in some way, and so assign a yield of 0?
                input_yield = result_dict.get(search_string, 0)
                line = ",".join(original_line.split(",")[:-1]) + "," + str(input_yield) + "\n"
                newfile += line
        with open(input_path, 'w') as f:
            f.write(newfile)

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

def workflow(export_path, bo):
    #Function for our BO pipeline.
    
    bo.run()
    #bo.plot_convergence()
    #bo.model.regression()
    #plot_kb_projections()
    bo.export_proposed(export_path)

def get_worst_percentile(full_result_dict, centile=10):
    number_of_results = len(full_result_dict)
    sorted_by_yield = sorted(full_result_dict.items(), key=lambda item: item[1])
    bottom_centile = sorted_by_yield[:int(0.01 * centile * number_of_results)]
    return dict(bottom_centile)


def simulate(seed=1, RESULT_PATH="", BATCH_SIZE=5, NUM_ROUNDS=5):

    from edbo.bro import BO_express, BO
    # BO object
    bo = BO_express(components,                                 # Reaction parameters
                    encoding=encoding,                          # Encoding specification
                    descriptor_matrices=dft,                    # DFT descriptors
                    acquisition_function='EI',                  # Use expectation value of improvement
                    init_method='external',                         # Use external initialization
                    batch_size=BATCH_SIZE,                              # 10 experiments per round
                    target='yield')                             # Optimize yield
    print("Instantiated BO object...")
    # BO_express actually automatically chooses priors
    # We can reset them manually to make sure they match the ones from our paper
    from gpytorch.priors import GammaPrior
    bo.lengthscale_prior = [GammaPrior(2.0, 0.2), 5.0]
    bo.outputscale_prior = [GammaPrior(5.0, 0.5), 8.0]
    bo.noise_prior = [GammaPrior(1.5, 0.5), 1.0]
    print("Constructed priors")
    ########################
    ########################
    #### Initialization ####
    ########################
    ########################
    
    """
    print(bo.init_sample(seed=seed))             # Initialize
    bo.export_proposed('init.csv')     # Export design to a CSV file
    print(bo.get_experiments())               # Print selected experiments
    """
    
    ####################################
    ####################################
    #### Bayesian Optimization Loop ####
    ####################################
    ####################################
    with open(RESULT_PATH) as f:
        FULL_RESULT_DICT = {",".join(line.split(",")[1:-1]): float(line.split(",")[-1][:-1]) for line in f.readlines()[1:]}
    
    worst_result_dict = get_worst_percentile(FULL_RESULT_DICT, centile=10)
    
    #print(list(worst_result_dict.items())[:5])
    
    init_file = ",base_SMILES_index,ligand_SMILES_index,solvent_SMILES_index,Concentration_index,Temp_C_index,yield\n"
    
    count = 1
    
    random.seed(seed) # Ensures repeatability
    
    for result in random.sample(list(worst_result_dict.keys()), BATCH_SIZE):
        init_file += str(count) + "," + result + "," + str(worst_result_dict[result]) + "\n"
        count += 1
    
    with open(FOLDER_PATH + 'init.csv', 'w') as f:
        f.write(init_file)
    bo.add_results(FOLDER_PATH + 'init.csv')
    import numpy as np
    import matplotlib.pyplot as plt

    for num in range(NUM_ROUNDS):
        print("Starting round ", num)
        #pdb.set_trace()
        try:
            workflow(FOLDER_PATH + 'round' + str(num) + '.csv', bo)
        except RuntimeError as e:
            print(e)
            print("No idea how to fix this, seems to occur randomly for different seeds...")
            break
        fill_in_experiment_values(FOLDER_PATH + 'round' + str(num) + '.csv', FULL_RESULT_DICT)
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

results_file_path = "direct_arylation_worst_10_50_50.csv"
results_file = "seed,maximum observed yield" + "\n"

count = 0

for num in random.sample(range(10 ** 6), 50):
    print("On number ", count)
    count += 1
    print("SEED HERE IS ", num)
    simulation_result = simulate(seed=num, RESULT_PATH='data/direct_arylation/experiment_index.csv', BATCH_SIZE=10, NUM_ROUNDS=5)
    results_file += str(num) + "," + str(simulation_result) + "\n"

with open(results_file_path, 'w') as f:
    f.write(results_file)

#Format is reaction_choosingmethod_batchsize_experimentbudget_numberofrunsdone

results_file_path = "direct_arylation_worst_5_50_50.csv"
results_file = "seed,maximum observed yield" + "\n"

count = 0

for num in random.sample(range(10 ** 6), 50):
    print("On number ", count)
    count += 1
    print("SEED HERE IS ", num)
    simulation_result = simulate(seed=num, RESULT_PATH='data/direct_arylation/experiment_index.csv', BATCH_SIZE=5, NUM_ROUNDS=10)
    results_file += str(num) + "," + str(simulation_result) + "\n"

with open(results_file_path, 'w') as f:
    f.write(results_file)


#Format is reaction_choosingmethod_batchsize_experimentbudget_numberofrunsdone

results_file_path = "direct_arylation_worst_3_51_50.csv"
results_file = "seed,maximum observed yield" + "\n"

count = 0

for num in random.sample(range(10 ** 6), 50):
    print("On number ", count)
    count += 1
    print("SEED HERE IS ", num)
    simulation_result = simulate(seed=num, RESULT_PATH='data/direct_arylation/experiment_index.csv', BATCH_SIZE=3, NUM_ROUNDS=17)
    results_file += str(num) + "," + str(simulation_result) + "\n"

with open(results_file_path, 'w') as f:
    f.write(results_file)