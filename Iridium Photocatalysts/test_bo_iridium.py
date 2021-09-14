'''
Testing the optimiser on the iridium photocatalyst dataset

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

'''
And finally, some standard libraries
to help with collecting data on optimiser performance.
'''

import random
import os.path

# Next, we'll define constants specific to the collection of performance data.

# Location of known rate constant data
RESULT_PATH = 'Iridium_data.csv'

# Location used to store temp files and performance data
FOLDER_PATH = "test_bo_iridium/"
DATA_PATH = FOLDER_PATH + "data/"
TEMP_PATH = FOLDER_PATH + "temp/"

# What we're trying to optimise
TARGET = 'Rate Constants'

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

#############################
#############################
##### REACTION ENCODING #####
#############################
#############################

print("Starting Reaction Encoding!")

CN_ligands = [
    'c2ccc(c1ccccn1)cc2',
    'Cc2ccc(c1ccccc1)nc2',
    'Cc2ccc(c1ccc(F)cc1)nc2',
    'Cc2ccc(c1ccc(Cl)cc1)nc2',
    'Cc2ccc(c1ccc(C#N)cc1)nc2',
    'COc2ccc(c1ccc(C)cn1)cc2',
    'Cc2ccc(c1ccc(F)cc1F)nc2',
    'c2ccc(n1cccn1)cc2',
    'c1ccc3c(c1)ccc2cccnc23',
    'Cc3ccc(c2ccc(c1ccccc1)cc2)nc3',
    'c3ccc(c2nc1ccccc1s2)cc3',
    'c3ccc(c2nc1ccccc1o2)cc3',
    'Cc3ccc(c2nc1ccccc1o2)cc3',
    'COc3ccc(c2nc1ccccc1s2)cc3',
    'COc3ccc(c2nc1ccccc1o2)cc3',
    'Fc3ccc(c2nc1ccccc1o2)cc3',
    'Clc3ccc(c2nc1ccccc1s2)cc3',
    'Clc3ccc(c2nc1ccccc1o2)cc3',
    'FC(F)(F)c3ccc(c2nc1ccccc1s2)cc3',
    'FC(F)(F)c3ccc(c2nc1ccccc1o2)cc3',
    'Cc2ccc(c1cccc(C#N)c1)nc2',
    'COc2cccc(c1ccc(C)cn1)c2',
    'Cc2ccc(c1ccc(S(C)(=O)=O)cc1)nc2',
    'Cc2ccc(c1ccc(OC(F)(F)F)cc1)nc2',
    'Cc2ccc(c1ccc(C)cn1)cc2',
    'Cc4ccc(c3ccc(N(c1ccccc1)c2ccccc2)cc3)nc4',
    'c4ccc(c3ccc(c2cc(c1ccccc1)ccn2)cc3)cc4',
    'Fc2ccc(c1ccccn1)cc2',
    'Cc2ccc(c1cccc(F)c1)nc2',
    'Cc2ccc(c1ccccc1F)nc2',
    'Cc2ccc(c1ccc(F)c(C(F)(F)F)c1)nc2',
    'Cc2ccc(c1cccc(Cl)c1)nc2',
    'c2ccc(Cc1ccccn1)cc2',
    'CCc2ccc(c1ccccc1)nc2',
    'Cc2cnc(c1ccccc1)cc2c3ccccc3',
    'Fc3ccc(c2cc(c1ccccc1)ccn2)cc3',
    'CCc2ccc(c1ccc(Cl)cc1)nc2',
    'Clc3ccc(c2cc(c1ccccc1)ccn2)cc3',
    'c4ccc(c3ccnc(c2ccc(N1CCCC1)cc2)c3)cc4',
    'COc3ccc(c2cc(c1ccccc1)ccn2)cc3',
    'COc3ccc(c2cc(c1ccccc1)c(C)cn2)cc3',
    'CCc2ccc(c1cccc(OC)c1)nc2',
    'COc3cccc(c2cc(c1ccccc1)ccn2)c3',
    'COc3cccc(c2cc(c1ccccc1)c(C)cn2)c3',
    'c4ccc(c3ccnc(c2ccc1ccccc1c2)c3)cc4',
    'FC(F)(F)Oc3ccc(c2nc1ccccc1s2)cc3',
    'c4ccc3cc(c2nc1ccccc1o2)ccc3c4',
    'Cc3ccc(c2ccc1ccccc1c2)nc3',
]

NN_ligands = [
    'c2ccc(c1ccccn1)nc2',
    'Cc2ccnc(c1cc(C)ccn1)c2',
    'Cc2ccc(c1ccc(C)cn1)nc2',
    'Cc2cccc(c1cccc(C)n1)n2',
    'c4ccc(c3ccnc(c2cc(c1ccccc1)ccn2)c3)cc4',
    'CC(C)(C)c2ccnc(c1cc(C(C)(C)C)ccn1)c2',
    'FC(F)(F)c2ccc(c1ccc(C(F)(F)F)cn1)nc2',
    'COc2ccnc(c1cc(OC)ccn1)c2',
    'O=C(O)c2ccnc(c1cc(C(=O)O)ccn1)c2',
    'O=c2c(=O)c1cccnc1c3ncccc23',
    'c1cnc3c(c1)ccc2cccnc23',
    'O=c2c1cccnc1c3ncccc23',
    'Cc3cnc2c(ccc1c(C)c(C)cnc12)c3C',
    'c4ccc3nc(c2ccc1ccccc1n2)ccc3c4',
    'Cc5cc(c1ccccc1)c4ccc3c(c2ccccc2)cc(C)nc3c4n5',
    'c5ccc(c1ccnc4c1ccc3c(c2ccccc2)ccnc34)cc5',
    'Cc2cc1cccnc1c3ncccc23',
    'CC1(C)C(=O)C(C)(C)c3c1c2cccnc2c4ncccc34',
    'Cc1ccnc3c1ccc2c(C)ccnc23',
    'Cc3ccc2ccc1ccc(C)nc1c2n3',
    'CCCCCCCCCc2ccnc(c1cc(CCCCCCCCC)ccn1)c2',
    'Cc1ccnc3c1ccc2cccnc23',
    'CS(C)=O',
    'CC(C)(O)Cc2ccc(c1ccc(CC(C)(C)O)cn1)nc2'
]

# Parameters in reaction space

components = {
    'CN_ligand': CN_ligands,
    'NN_ligand': NN_ligands,
}


encoding = {
    'CN_ligand':'smiles',
    'NN_ligand':'smiles',
}

############################
############################
#### Instantiating EDBO ####
############################
############################

# TARGET data is provided at RESULT_PATH
# We read in the csv file, and create a python dictionary
# The 'key' is the part of the line in the csv file just before the yield
# The 'value' is then the yield, as a float
# E.g. if the configuration was halide1,additive3,base4,ligand2,56.78
# The dictionary entry would be "halide1,additive3,base4,ligand2": 56.78

with open(RESULT_PATH) as f:
    FULL_RESULT_DICT = {
        ",".join(
            line.split(",")[:-1]
        ): float(
            line.split(",")[-1][:-1]
        ) for line in f.readlines()[1:]
    }

def instantiate_bo(acquisition_func: str, batch_size: int):
    bo = BO_express(
        components,
        encoding=encoding,
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
        name = f"iridium_{method}_{batch_size}_{batch_size * num_rounds}_{N_EXPERIMENTS}"
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
