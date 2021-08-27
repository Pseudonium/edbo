import sys
sys.path.insert(1, '../..')
import pandas as pd
from edbo.bro import BO_express
from gpytorch.priors import GammaPrior
import random
import os.path
from edbo.feature_utils import mordred

RESULT_PATH = 'Iridium_data.csv'
FOLDER_PATH = "test_bo_iridium/"
MASTER_SEED = 42
N_EXPERIMENTS = 50
METHODS = ['EI', 'rand', 'TS']
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

'''
# Now, gotta get mordreds for all of these

mordred_dfs = [
    mordred(reactants).set_index('_SMILES')
    for reactants in [CN_ligands, NN_ligands]
]

mordred_df = mordred(CN_ligands + NN_ligands, dropna=True).set_index('_SMILES')

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
'''

# Parameters in reaction space

# Suzuki here
components = {
    'CN_ligand': CN_ligands,
    'NN_ligand': NN_ligands,
}


# External descriptor matrices override specified encoding

encoding = {
    'CN_ligand':'smiles',
    'NN_ligand':'smiles',
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
        encoding=encoding,
        acquisition_function=acquisition_func,
        init_method='rand',
        batch_size=batch_size,
        target='Rate Constants'
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
            input_area = FULL_RESULT_DICT[search_string]
            #input_area = FULL_RESULT_DICT.get(search_string, 0)
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
    results = pd.DataFrame(columns=bo.reaction.index_headers + ['Rate Constants'])
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
    return results['Rate Constants'].max()

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
        path = f"iridium_{method}_{batch_size}_{batch_size * num_rounds}_{N_EXPERIMENTS}"
        #path += "_new.csv"  # To differentiate with old files
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
