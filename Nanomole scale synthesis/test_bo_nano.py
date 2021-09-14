'''
Testing the optimiser on the nanomole dataset

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

'''
And finally, some standard libraries
to help with collecting data on optimiser performance.
'''

import random
import os.path

# Next, we'll define constants specific to the collection of performance data.

# Location of known rate constant data
RESULT_PATH = 'S3_Data.csv'

# Location used to store temp files and performance data
FOLDER_PATH = "test_bo_nano/"
DATA_PATH = FOLDER_PATH + "data/"
TEMP_PATH = FOLDER_PATH + "temp/"

# What we're trying to optimise
TARGET = 'MISER Area Count'

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

electrophiles = [
    'O=C2OC(Cn1ccnn1)CN2c3ccc(I)c(F)c3',
    'COCCCc4cc(CN(C(=O)C1CN(C(=O)OC(C)(C)C)CCC1c2ccn(C)c(=O)c2)C3CC3)cc(Br)c4C',
    'CN/2C(=O)CC(C)(c1cc(Br)cs1)NC2=N/C(=O)OC(C)(C)C',
    'CC(C)n3c(C(=O)N(C)C)c2CCN(Cc1ccc(F)c(Cl)c1)C(=O)c2c(O)c3=O',
    'CN4CCN(C(=O)O[C@H]2c1nccnc1C(=O)N2c3ccc(Cl)cn3)CC4',
    'COC(=O)C(C)(C)Cc4c(SC(C)(C)C)c3cc(OCc2cc1ccccc1cn2)ccc3n4Cc5ccc(Cl)cc5'
]
nucleophiles = [
    'CCOC(=O)N1CCNCC1',
    'COC(=O)C1(CN)CC1',
    'CC(C)(C)OC(N)=O',
    'Nc1cc(F)ccn1',
    'NS(=O)(=O)c1cccs1',
    'CN(C)CC(N)=O',
    'N=C(N)C1CC1',
    'Cn1nccc1CCO',
    'O',
    'Cn1nccc1[B]2OC(C)(C)C(C)(C)O2',
    'C#Cc1cnccn1'
]
catalysts = [
    'CC(C)c4cc(C(C)C)c(c1ccccc1P(C2CCCCC2)C3CCCCC3)c(C(C)C)c4',
    'CC(C)(C)P(c1cc[cH-]c1)C(C)(C)C.CC(C)(C)P(c1cc[cH-]c1)C(C)(C)C.[Fe+2]',
    'CC(C)c2cc(C(C)C)c(c1ccccc1P(C(C)(C)C)C(C)(C)C)c(C(C)C)c2',
    'COc1ccc(OC)c(P(C(C)(C)C)C(C)(C)C)c1c2c(C(C)C)cc(C(C)C)cc2C(C)C',
    'COc2ccc(C)c(c1c(C(C)C)cc(C(C)C)cc1C(C)C)c2P(C(C)(C)C)C(C)(C)C',
    'COc7ccc(OC)c(P(C23CC1CC(CC(C1)C2)C3)C56CC4CC(CC(C4)C5)C6)c7c8c(C(C)C)cc(C(C)C)cc8C(C)C'
]

bases = [
    'CN(C)P(=NC(C)(C)CC(C)(C)C)(N(C)C)N(C)C',
    'CN(C)P(=NC(C)(C)C)(N=P(N(C)C)(N(C)C)N(C)C)N(C)C',
    'C2CCC1=NCCCN1CC2',
    'CN1CCCN2CCCN=C12',
    'CN(C)/C(=N\C(C)(C)C)N(C)C',
    'CCN(CC)P1(=NC(C)(C)C)NCCCN1C',
    'CC(C)(C)N=P(N1CCCC1)(N2CCCC2)N3CCCC3',
    'CCN=P(N=P(N(C)C)(N(C)C)N(C)C)(N(C)C)N(C)C'
]



# Now, due to the 1500 vs 3168 problem, I need to make a descriptor
# matrix of ~1500 rows that explicitly contains each configuration,
# if you want to use the 'fixed' method.

# Now, gotta get mordreds for all of these

mordred_dfs = [
    mordred(reactants).set_index('_SMILES')
    for reactants in [electrophiles, nucleophiles, catalysts, bases]
]

mordred_df = mordred(electrophiles + nucleophiles + catalysts + bases, dropna=True).set_index('_SMILES')

smile_to_mordred = {
    key: mordred_df.loc[key].tolist()
    for key in mordred_df.index
}

# Then, gotta make the descriptor matrix...



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

def search_term_to_mordred(search_term):
    """
    Takes in a search term of the form
    Electrophile,Nucleophile,Catalyst,Base
    And returns a single list
    By concatenating the mordred descriptors of each
    Of the components
    """

    full_descriptor = []
    for component in search_term.split(","):
        full_descriptor.extend(smile_to_mordred[component])
    return full_descriptor


descriptor_matrix = pd.DataFrame.from_records(
    [
        search_term_to_mordred(key)
        for key in FULL_RESULT_DICT.keys()
    ]
)

# ;;; is used as a separator as it doesn't interfere with csv or SMILES.
descriptor_matrix.insert(0, 'Configuration', list(";;;".join(key.split(",")) for key in FULL_RESULT_DICT.keys()))

# This prevents errors with the bo
descriptor_matrix.columns = descriptor_matrix.columns.astype(str)


components_3168 = {
    'electrophile': electrophiles,
    'nucleophile': nucleophiles,
    'catalyst': catalysts,
    'base': bases,
}

components_1500 = {
    'Configuration': '<defined in descriptor_matrices>'
}

encoding_3168 = {
    'electrophile':'smiles',
    'nucleophile':'smiles',
    'catalyst':'smiles',
    'base':'smiles'
}

encoding_1500 = {}

############################
############################
#### Instantiating EDBO ####
############################
############################

def instantiate_bo(acquisition_func: str, batch_size: int, fixed=True):
    if fixed:
        bo = BO_express(
            components_1500,
            encoding=encoding_1500,
            acquisition_func=acquisition_func,
            descriptor_matrices = {'Configuration': descriptor_matrix},
            init_method='rand', # Allows control of initial experiments,
            # via seeds.
            batch_size=batch_size,
            target=TARGET
        )
    else:
        bo = BO_express(
            components_3168, # second by default
            encoding=encoding_3168,
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

FIXED = True

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
        if FIXED:
            initial = 'nanofixed_'
        else:
            initial = 'nano_'
        name = initial + f"{method}_{batch_size}_{batch_size * num_rounds}_{N_EXPERIMENTS}"
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
