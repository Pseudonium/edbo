'''
Testing the optimiser on the harvard CEP dataset

Some additional acquisition functions were added to the optimiser algorithm
in the acq_func.py file, so to import these we need to point to
the location of said file.

(Answer via https://stackoverflow.com/questions/4383571)
'''

import sys
sys.path.insert(1, '..')

# Then, we'll introduce the imports necessary to run the optimiser itself.
import pandas as pd
from edbo.bro import BO_express
from gpytorch.priors import GammaPrior
from edbo.feature_utils import mordred
from rdkit import Chem
from rdkit.Chem import AllChem

'''
And finally, some standard libraries
to help with collecting data on optimiser performance.
'''

import random
import os.path

# Next, we'll define constants specific to the collection of performance data.

# Location of pce data
RESULT_PATH = 'moldata_clean.csv'

# Location used to store temp files and performance data
FOLDER_PATH = "test_bo_clean/"
DATA_PATH = FOLDER_PATH + "data/"
TEMP_PATH = FOLDER_PATH + "temp/"

# What we're trying to optimise
TARGET = 'pce'

# Seed used to ensure repeatability
MASTER_SEED = 69

# Number of times you want to run the optimisier on each 'configuration',
# where a configuration is a combination of acquisition function,
# batch size and a number of rounds.
N_EXPERIMENTS = 50

# For each test of the optimiser, the number of top values you'd like
# to store in the performance data file.
TOP_N = 1

# The acquisition functions you'd like to test
METHODS = ['EI', 'rand', 'TS']

# (batch_size, round) pairs. Idea is to approximate an experiment budget
# of 50, though of course you can define your own pairs for different budgets.
BATCH_ROUNDS = [(1, 50), (3, 17), (5, 10), (10, 5)]

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


# Can't load all 2 million+ data points into optimiser at once,
# so instead take a random sample of size 10,000.
SAMPLE_SIZE = 10 ** 4

#############################
#############################
##### REACTION ENCODING #####
#############################
#############################

print("Starting Reaction Encoding!")

components = {
    'chemical': '<defined in descriptor_matrices>'
}

full_clean_df = pd.read_csv('moldata_clean.csv')

random.seed(MASTER_SEED)  # Ensures repeatability
sample_clean_df = full_clean_df.iloc[
    random.sample(range(len(full_clean_df)), SAMPLE_SIZE)
]
sample_clean_df.set_index('SMILES_str', inplace=True)
scd = sample_clean_df

print("Now generating the mordred descriptors...")

encoded_df = pd.DataFrame.from_records(
    [
        AllChem.GetMorganFingerprintAsBitVect(
            Chem.MolFromSmiles(item), 2, nBits=512
        ) for item in scd.index
    ]
)

encoded_df.insert(0, 'chemical_SMILES', scd.index)

# This prevents errors with the bo
encoded_df.columns = encoded_df.columns.astype(str)



############################
############################
#### Instantiating EDBO ####
############################
############################

# In our case, we can use the sample_clean_df itself as our
# 'results dict', so we don't need a separate one.

def instantiate_bo(acquisition_func: str, batch_size: int):
    bo = BO_express(
        components,
        encoding={},
        acquisition_function=acquisition_func,
        descriptor_matrices={'chemical': encoded_df},
        init_method='rand',
        batch_size=batch_size,
        target=TARGET
    )
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
            smiles_string = ",".join(line[1:-1])
            pce_value = sample_clean_df.loc[smiles_string][TARGET]
            line = "".join(
                [
                    ",".join(original_line.split(",")[:-1]),
                    ",",
                    str(pce_value),
                    "\n"
                ]
            )
            newfile += line
    with open(input_path, 'w') as f:
        f.write(newfile)

def write_prop_read_run(bo, export_path):
    bo.export_proposed(export_path)
    fill_in_experiment_values(export_path)
    bo.add_results(export_path)
    bo.run()

def get_max_yields(bo, num_rounds):
    results = pd.DataFrame(columns=bo.reaction.index_headers + [TARGET])
    for path in [
        TEMP_PATH + 'init'
    ] + [
        TEMP_PATH + f'round{num}' for num in range(num_rounds)
    ]:
        results = pd.concat(
            [results, pd.read_csv(path + '.csv', index_col=0)],
            sort=False
        )
    print(results)
    return sorted(results[TARGET].tolist(), reverse=True)[:TOP_N]

def simulate_bo(seed, acquisition_func, batch_size, num_rounds):
    bo = instantiate_bo(acquisition_func, batch_size)
    bo.init_sample(seed=seed)
    print(bo.get_experiments())
    write_prop_read_run(bo, TEMP_PATH + 'init.csv')

    for num in range(num_rounds):
        print(f"Starting round {num}")
        write_prop_read_run(bo, TEMP_PATH + f"round{num}.csv")
        print(f"Finished round {num}")
    return get_max_yields(bo, num_rounds)

for method in METHODS:
    for batch_size, num_rounds in BATCH_ROUNDS:
        print(
            f"Testing bo with acquisition function {method}",
            f"\n with a batch size of {batch_size}",
            f"\n and doing {num_rounds} rounds",
        )
        results_file = "seed,maximum observed yield" + ",".join(
            str(num) for num in range(2, TOP_N + 1)
        ) + "\n"
        name = f"harvard_{method}_{batch_size}_{batch_size * num_rounds}_{N_EXPERIMENTS}"
        path = DATA_PATH + name + '.csv'
        if os.path.isfile(path):
            # So we've already written data to it
            # No need to overwrite
            continue
        for index, seed in enumerate(SEEDS):
            print(f"On number {index} of {N_EXPERIMENTS}")
            result = simulate_bo(
                seed, method, batch_size, num_rounds
            )
            results_file += f"{seed},{','.join(str(res) for res in result)}\n"
            print(f"{seed},{','.join(str(res) for res in result)}\n")

        with open(path, 'w') as f:
            f.write(results_file)
