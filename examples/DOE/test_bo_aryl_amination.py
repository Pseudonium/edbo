# Imports

### Going to use custom edbo implementation (changed the acq_func.py file to add my custom functions)
## So, need to point to that folder
# Answer via https://stackoverflow.com/questions/4383571/importing-files-from-different-folder

import sys
sys.path.insert(1, '../..')
import pandas as pd
from edbo.utils import Data
from edbo.bro import BO_express
from gpytorch.priors import GammaPrior
import random
import os.path

RESULT_PATH = 'data/aryl_amination/experiment_index.csv'
FOLDER_PATH = "test_bo_aryl_amination/"
MASTER_SEED = 42
N_EXPERIMENTS = 50
METHODS = ['EI', 'TS', 'rand']
BATCH_ROUNDS = [(1, 50), (2, 25), (3, 17), (4, 12), (5, 10), (6, 8), (7, 7), (8, 6), (9, 5), (9, 6), (10, 5)]

random.seed(MASTER_SEED)  # Ensures repeatability
SEEDS = random.sample(range(10 ** 6), N_EXPERIMENTS)

#############################
#############################
##### REACTION ENCODING #####
#############################
#############################

print("Starting Reaction Encoding!")

# Load DFT descriptor CSV files computed with auto-qchem using pandas
# Instantiate a Data object


# aryl amination here
aryl_halides = Data(pd.read_csv('data/aryl_amination/aryl_halide_dft.csv'))
additives = Data(pd.read_csv('data/aryl_amination/additive_dft.csv'))
bases = Data(pd.read_csv('data/aryl_amination/base_dft.csv'))
ligands = Data(pd.read_csv('data/aryl_amination/ligand_ohe.csv'))
reactants = [aryl_halides, additives, bases, ligands]


print("Loaded csv files...")

# Use Data.drop method to drop descriptors containing some unwanted keywords

for data in reactants:
    data.drop(['file_name', 'vibration', 'correlation', 'Rydberg',
               'correction', 'atom_number', 'E-M_angle', 'MEAN', 'MAXG',
               'STDEV'])

print("Dropped unnecessary data...")

# Parameters in reaction space

# aryl amination here
components = {'aryl_halide':'DFT',                              # DFT descriptors
              'additive':'DFT',                                           # DFT descriptors
              'base':'DFT',
              'ligand':['CC(C)C1=CC(C(C)C)=CC(C(C)C)=C1C2=C(P(C3CCCCC3)C4CCCCC4)C=CC=C2', 'CC(C)C(C=C(C(C)C)C=C1C(C)C)=C1C2=CC=CC=C2P(C(C)(C)C)C(C)(C)C', 'CC(C)C1=CC(C(C)C)=CC(C(C)C)=C1C2=C(P(C(C)(C)C)C(C)(C)C)C(OC)=CC=C2OC', 'CC(C1=C(C2=C(OC)C=CC(OC)=C2P(C34CC5CC(C4)CC(C5)C3)C67CC8CC(C7)CC(C8)C6)C(C(C)C)=CC(C(C)C)=C1)C']}                     # Discrete grid of temperatures


# External descriptor matrices override specified encoding


# aryl amination here
dft = {'aryl_halide':aryl_halides.data,                   # Unprocessed descriptor DataFrame
       'additive':additives.data,                                             # Unprocessed descriptor DataFrame
       'base':bases.data}                                       # Unprocessed descriptor DataFrame

encoding = {}

############################
############################
#### Instantiating EDBO ####
############################
############################


with open(RESULT_PATH) as f:
    FULL_RESULT_DICT = {
        ",".join(
            line.split(",")[1:-1]
        ): float(
            line.split(",")[-1][:-1]
        ) for line in f.readlines()[1:]
    }

def instantiate_bo(acquisition_func: str, batch_size: int):
    bo = BO_express(
        components,
        encoding=encoding,
        descriptor_matrices=dft,
        acquisition_function=acquisition_func,
        init_method='rand',
        batch_size=batch_size,
        target='yield'
    )
    bo.lengthscale_prior = [GammaPrior(2.0, 0.2), 5.0]
    bo.outputscale_prior = [GammaPrior(5.0, 0.5), 8.0]
    bo.noise_prior = [GammaPrior(1.5, 0.5), 1.0]
    return bo


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
            input_yield = FULL_RESULT_DICT.get(search_string, 0) # Turns out some combinations disallowed
            line = ",".join(
                original_line.split(",")[:-1]
            ) + "," + str(input_yield) + "\n"
            newfile += line
    with open(input_path, 'w') as f:
        f.write(newfile)

def write_prop_read_run(bo, export_path):
    bo.export_proposed(export_path)
    fill_in_experiment_values(export_path)
    bo.add_results(export_path)
    bo.run()

def get_max_yield(bo, num_rounds):
    results = pd.DataFrame(columns=bo.reaction.index_headers + ['yield'])
    for path in [
        FOLDER_PATH + 'init'
    ] + [
        FOLDER_PATH + f'round{num}' for num in range(num_rounds)
    ]:
        results = pd.concat(
            [results, pd.read_csv(path + '.csv', index_col=0)],
            sort=False
        )
    print(results)
    return results['yield'].max()

def simulate_bo(seed, acquisition_func, batch_size, num_rounds):
    bo = instantiate_bo(acquisition_func, batch_size)
    bo.init_sample(seed=seed)
    print(bo.get_experiments())
    write_prop_read_run(bo, FOLDER_PATH + 'init.csv')

    for num in range(num_rounds):
        print(f"Starting round {num}")
        write_prop_read_run(bo, FOLDER_PATH + f"round{num}.csv")
        print(f"Finished round {num}")
    return get_max_yield(bo, num_rounds)



# Format is reaction_choosingmethod_batchsize_experimentbudget_numberofrunsdone
# Key of choosingmethods:
# random - chosen at random using expected improvement as acquisition function
# worst - randomly chosen from bottom 10% of experiments using expected improvement
# randomts - chosen at random using thompson sampling
# randomtsei - chosen at random using hybrid thompson sampling and expected improvement (my own modification, not the ei-ts builtin


# New key of choosing methods (no longer doing worst)
# EI - expected improvement
# TS - Thompson sampling
# TS-EI - hybrid

for method in METHODS:
    for batch_size, num_rounds in BATCH_ROUNDS:
        print(
            f"Testing bo with acquisition function {method}",
            f"\n with a batch size of {batch_size}",
            f"\n and doing {num_rounds} rounds",
        )
        results_file = "seed,maximum observed yield" + "\n"
        path = f"arylamination_{method}_{batch_size}_{batch_size * num_rounds}_{N_EXPERIMENTS}"
        path += "_new.csv"  # To differentiate with old files
        if os.path.isfile(path):
            # So we've already written data to it
            # No need to overwrite
            continue
        for index, seed in enumerate(SEEDS):
            print(f"On number {index} of {N_EXPERIMENTS}")
            result = simulate_bo(
                seed, method, batch_size, num_rounds
            )
            results_file += f"{seed},{result}\n"

        with open(path, 'w') as f:
            f.write(results_file)
