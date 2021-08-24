import sys
sys.path.insert(1, '../..')
import pandas as pd
from edbo.bro import BO_express
from gpytorch.priors import GammaPrior
import random
import os.path
from edbo.feature_utils import mordred

RESULT_PATH = 'S3_Data.csv'
FOLDER_PATH = "test_bo_nano/"
MASTER_SEED = 42
N_EXPERIMENTS = 50
METHODS = ['EI', 'TS', 'TS-EI']
METHODS = ['EI']
METHODS = ['EI', 'rand', 'TS']
BATCH_ROUNDS = [(1, 50), (2, 25), (3, 17), (4, 12), (5, 10), (6, 8), (7, 7), (8, 6), (9, 5), (9, 6), (10, 5)]
BATCH_ROUNDS = [(1, 50), (3, 17), (5, 10), (10, 5)]

random.seed(MASTER_SEED)  # Ensures repeatability
SEEDS = random.sample(range(10 ** 6), N_EXPERIMENTS)

# Going to use custom edbo implementation
# (changed the acq_func.py file to add my custom functions)
# So, need to point to that folder
# Answer via
# https://stackoverflow.com/questions/
# 4383571/importing-files-from-different-folder

#############################
#############################
##### REACTION ENCODING #####
#############################
#############################

print("Starting Reaction Encoding!")

# Load DFT descriptor CSV files computed with auto-qchem using pandas
# Instantiate a Data object


# Suzuki here
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

descriptor_matrix.insert(0, 'Configuration', list(";;;".join(key.split(",")) for key in FULL_RESULT_DICT.keys()))

# This prevents errors with the bo
descriptor_matrix.columns = descriptor_matrix.columns.astype(str)

print(descriptor_matrix)


# Parameters in reaction space

# Suzuki here
components = {
    'electrophile': electrophiles,
    'nucleophile': nucleophiles,
    'catalyst': catalysts,
    'base': bases,
}


components = {
    'Configuration': '<defined in descriptor_matrices>'
}


# External descriptor matrices override specified encoding

encoding = {
    'electrophile':'smiles',
    'nucleophile':'smiles',
    'catalyst':'smiles',
    'base':'smiles'
}

############################
############################
#### Instantiating EDBO ####
############################
############################

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
        encoding={},
        acquisition_function=acquisition_func,
        #descriptor_matrices = {'Configuration': descriptor_matrix},
        init_method='rand',
        batch_size=batch_size,
        target='MISER Area Count'
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
            search_string = ",".join(search_string.split(";;;"))
            #input_area = FULL_RESULT_DICT[search_string]
            input_area = FULL_RESULT_DICT.get(search_string, 0)
            line = ",".join(
                original_line.split(",")[:-1]
            ) + "," + str(input_area) + "\n"
            newfile += line
    with open(input_path, 'w') as f:
        f.write(newfile)

def write_prop_read_run(bo, export_path):
    bo.export_proposed(export_path)
    fill_in_experiment_values(export_path)
    bo.add_results(export_path)
    bo.run()

def get_max_yield(bo, num_rounds):
    results = pd.DataFrame(columns=bo.reaction.index_headers + ['MISER Area Count'])
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
    return results['MISER Area Count'].max()

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
        path = f"nanofixed_{method}_{batch_size}_{batch_size * num_rounds}_{N_EXPERIMENTS}"
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
