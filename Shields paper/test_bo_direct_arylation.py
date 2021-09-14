'''
Testing the optimiser on the direct arylation dataset

Some additional acquisition functions were added to the optimiser algorithm
in the acq_func.py file, so to import these we need to point to
the location of said file.

(Answer via https://stackoverflow.com/questions/4383571)
'''

import sys
sys.path.insert(1, '..')

# Then, we'll introduce the imports necessary to run the optimiser itself.

import pandas as pd
from edbo.utils import Data
from edbo.bro import BO_express
from gpytorch.priors import GammaPrior

'''
And finally, some standard libraries
to help with collecting data on optimiser performance.
'''

import random
import os.path

# Next, we'll define constants specific to the collection of performance data.

# Location of known yield results data
RESULT_PATH = 'data/direct_arylation/experiment_index.csv'

# Location used to store temp files and performance data
FOLDER_PATH = "test_bo_direct_arylation/"
DATA_PATH = FOLDER_PATH + "data/"
TEMP_PATH = FOLDER_PATH + "temp/"

# What we're trying to optimise
TARGET = 'yield'

# Seed used to ensure repeatability
MASTER_SEED = 42

# Number of times you want to run the optimisier on each 'configuration',
# where a configuration is a combination of acquisition function,
# batch size and a number of rounds.
N_EXPERIMENTS = 50

# For each test of the optimiser, the number of top values you'd like
# to store in the performance data file.
TOP_N = 1

# The acquisition functions you'd like to test
METHODS = ['EI', 'TS', 'rand']


# (batch_size, round) pairs. Idea is to approximate an experiment budget
# of 50, though of course you can define your own pairs for different budgets.
BATCH_ROUNDS = [
    (1, 50),
    (2, 25),
    (3, 17),
    (4, 12),
    (5, 10),
    (6, 8),
    (7, 7),
    (8, 6),
    (9, 5),
    (9, 6),
    (10, 5)
]

random.seed(MASTER_SEED)  # Ensures repeatability

# The seeds used to ensure the optimiser selects the same set of
# Initial experiments for each configuration.
# This generates an array of 50 integers from 0 to 10 ** 6 - 1.
# Each time you run the optimiser, you select one of these integers.
# The integer determines which initial experiments are chosen by the optimiser.
# For the same integer and batch size, you're guaranteed the same
# initial experiments. Of course, with different batch sizes,
# the set of initial experiments will be different anyway.
SEEDS = random.sample(range(10 ** 6), N_EXPERIMENTS)

#############################
#############################
##### REACTION ENCODING #####
#############################
#############################

print("Starting Reaction Encoding!")

# Load DFT descriptor CSV files computed with auto-qchem using pandas
# Instantiate a Data object
bases = Data(pd.read_csv('data/direct_arylation/base_dft.csv'))
ligands = Data(pd.read_csv('data/direct_arylation/ligand-boltzmann_dft.csv'))
solvents = Data(pd.read_csv('data/direct_arylation/solvent_dft.csv'))
reactants = [bases, ligands, solvents]

print("Loaded csv files...")

# Use Data.drop method to drop descriptors containing some unwanted keywords

for data in reactants:
    data.drop(
        [
            'file_name',
            'vibration',
            'correlation',
            'Rydberg',
            'correction',
            'atom_number',
            'E-M_angle',
            'MEAN',
            'MAXG',
            'STDEV'
        ]
    )

print("Dropped unnecessary data...")

# Parameters in reaction space

components = {
    'base':'DFT',
    'ligand':'DFT',
    'solvent':'DFT',
    'Concentration':[0.057, 0.1, 0.153],
    'Temp_C':[90, 105, 120]
}


# External descriptor matrices override specified encoding

dft = {
    'base':bases.data,
    'ligand':ligands.data,
    'solvent':solvents.data
}

encoding = {}

############################
############################
#### Instantiating EDBO ####
############################
############################

# Yield data is provided at RESULT_PATH
# We read in the csv file, and create a python dictionary
# The 'key' is the part of the line in the csv file just before the yield
# The 'value' is then the yield, as a float
# E.g. if the configuration was halide1,additive3,base4,ligand2,56.78
# The dictionary entry would be "halide1,additive3,base4,ligand2": 56.78

with open(RESULT_PATH) as f:
    FULL_RESULT_DICT = {
        ",".join(
            line.split(",")[1:-1]
        ): float(
            line.split(",")[-1][:-1]
        ) for line in f.readlines()[1:]  # First line has headers,
        # so ignore it.
    }

# This function is specifically for 'worst' initial selection
# To use it, set worst=True in simulate_bo.
# You'll need to manually alter the naming scheme.

def get_worst_percentile(full_result_dict, centile=10):
    # Get the bottom centile% of experiments from the results table.
    number_of_results = len(full_result_dict)
    sorted_by_yield = sorted(full_result_dict.items(), key=lambda item: item[1])
    bottom_centile = sorted_by_yield[:int(0.01 * centile * number_of_results)]
    return dict(bottom_centile)




def instantiate_bo(acquisition_func: str, batch_size: int):
    bo = BO_express(
        components,
        encoding=encoding,
        descriptor_matrices=dft,
        acquisition_function=acquisition_func,
        init_method='rand', # Allows control of initial experiments,
        # via seeds.
        batch_size=batch_size,
        target=TARGET
    )

    # The priors are set to the ones in the paper, for consistency.
    bo.lengthscale_prior = [GammaPrior(2.0, 0.2), 5.0]
    bo.outputscale_prior = [GammaPrior(5.0, 0.5), 8.0]
    bo.noise_prior = [GammaPrior(1.5, 0.5), 1.0]
    return bo

'''
When running a BO_express instance, the proposed experiments are output to a
csv file, with the experimenter meant to manually input the resulting yields
from running those conditions. Of course, for our purposes this would ideally
be automatic - however, the BO_express instance doesn't support
a results table as input.

There is a parent class BO that does accept a results table as input, but
it requires manual construction of the domain
as well as manual numerical encoding, which is a hassle.

Instead, we let the proposed experiments be output to a csv file,
and then programmatically read in the csv file
and fill in the resulting yield values from our known results table.

It's likely slower than the program having access to the results table
in memory, but it works!
'''

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
            search_string = ",".join(line[1:-1]) # Everything except the
            # input for the yield value, i.e. the combination.
            input_yield = FULL_RESULT_DICT.get(search_string, 0)
            # Here, the 'get' method returns 0 if the combination
            # is not listed in the yield results table.
            # In other files, this is unnecessary, since we construct
            # the reaction domain from the yield results table.
            # However, here we simply multiply together
            # the possible combinations of base, ligand etc to form
            # our reaction domain, which may lead to combinations
            # that don't have entries in the yield results table.
            line = ",".join(
                original_line.split(",")[:-1]
            ) + "," + str(input_yield) + "\n" # 'Filling in' the yield value.
            newfile += line
    with open(input_path, 'w') as f:
        f.write(newfile)

def write_prop_read_run(bo, export_path):
    'Helper function for running a single round of optimisation.'
    bo.export_proposed(export_path)
    fill_in_experiment_values(export_path)
    bo.add_results(export_path)
    bo.run()


def get_max_yields(bo, num_rounds):
    results = pd.DataFrame(columns=bo.reaction.index_headers + [TARGET])
    for path in [
        TEMP_PATH
    ] + [
        TEMP_PATH + f'round{num}' for num in range(num_rounds)
    ]:
        results = pd.concat(
            [results, pd.read_csv(path + '.csv', index_col=0)],
            sort=False
        )
    return sorted(results[TARGET].tolist(), reverse=True)[:TOP_N]

def simulate_bo(seed, acquisition_func, batch_size, num_rounds, worst=False):
    bo = instantiate_bo(acquisition_func, batch_size)
    if worst:

        # Choose from bottom 10% of experiments
        worst_result_dict = get_worst_percentile(FULL_RESULT_DICT, centile=10)
        init_file = ",base_SMILES_index,ligand_SMILES_index,solvent_SMILES_index,Concentration_index,Temp_C_index,yield\n"
        count = 1
        random.seed(seed)
        for result in random.sample(list(worst_result_dict.keys()), batch_size):
            init_file += str(count) + "," + result + "," + str(
                worst_result_dict[result]
            ) + "\n"
            count += 1
        with open(FOLDER_PATH + 'init.csv', 'w') as f:
            f.write(init_file)
        bo.add_results(FOLDER_PATH + 'init.csv')
        bo.run()
    else:
        bo.init_sample(seed=seed)
        print(bo.get_experiments())
        write_prop_read_run(bo, TEMP_PATH + 'init.csv')

    for num in range(num_rounds):
        print(f"Starting round {num}")
        write_prop_read_run(bo, TEMP_PATH + f"round{num}.csv")
        print(f"Finished round {num}")
    return get_max_yields(bo, num_rounds)


# Key of acquisition functions
# EI - expected improvement
# TS - Thompson sampling
# TS-EI - hybrid (custom implementation, 1 TS and n - 1 EI for batch size n)
# EI-TS - hybrid (default implementation, 1 EI and n - 1 TS for batch size n)

for method in METHODS:
    for batch_size, num_rounds in BATCH_ROUNDS:
        print(
            f"Testing bo with acquisition function {method}",
            f"\n with a batch size of {batch_size}",
            f"\n and doing {num_rounds} rounds",
        )
        results_file = "seed,maximum observed yield" + "\n"
        name = f"direct_arylation_{method}_{batch_size}_{batch_size * num_rounds}_{N_EXPERIMENTS}"
        path = DATA_PATH + name + ".csv"
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
