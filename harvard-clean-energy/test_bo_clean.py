import pandas as pd
from edbo.bro import BO_express

import random
from edbo.feature_utils import mordred
from rdkit import Chem
from rdkit.Chem import AllChem
import time

MASTER_SEED = 69                      # Nice
#MASTER_SEED = 39  #  Potential bad algorithm performance
SAMPLE_SIZE = 5 * (10 ** 3)
SAMPLE_SIZE = 10 ** 4

print("Starting encoding!")

components = {
    'chemical': '<defined in descriptor_matrices>'
}

full_clean_df = pd.read_csv('moldata_clean.csv')[[
    "SMILES_str",
    "e_homo_alpha",
    "e_lumo_alpha",
    "pce"
]]

random.seed(MASTER_SEED)  # Ensures repeatability
sample_clean_df = full_clean_df.iloc[
    random.sample(range(len(full_clean_df)), SAMPLE_SIZE)
]
sample_clean_df.set_index('SMILES_str', inplace=True)
scd = sample_clean_df

print(scd['pce'].max())

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
            pce_value = sample_clean_df.loc[smiles_string]['pce']
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


def workflow(bo, export_path):
    bo.run()
    bo.export_proposed(export_path)
    fill_in_experiment_values(export_path)
    bo.add_results(export_path)


def simulate_bo(seed, batch_size, num_rounds):
    print("Initialising bo")
    bo = BO_express(
        reaction_components=components,
        encoding={},
        descriptor_matrices={'chemical': encoded_df},
        acquisition_function='TS',
        init_method='rand',
        target='pce',
        batch_size=batch_size,
    )
    print("Finished setting up bo!")

    bo.init_sample(seed)
    print(bo.get_experiments())
    bo.export_proposed('init.csv')
    fill_in_experiment_values('init.csv')
    bo.add_results('init.csv')

    for num in range(num_rounds):
        print("Starting round ", num)
        workflow(bo, f"round{num}.csv")
        print("Finished round ", num)

    results = pd.DataFrame(columns=bo.reaction.index_headers + ['pce'])
    for path in ['init'] + ['round' + str(num) for num in range(num_rounds)]:
        results = pd.concat([results, pd.read_csv(path + '.csv', index_col=0)], sort=False)

    results = results.sort_values('pce', ascending=False)

    top_yields = results.head()['pce'].tolist()
    return top_yields

random.seed(MASTER_SEED)
seeds = random.sample(range(10 ** 6), 50)

results_file_path = "harvardtop5_randomts_10_100_50.csv"
results_file = "seed,maximum observed pce, 2, 3, 4, 5" + "\n"

for index, seed in enumerate(seeds):
    print("On number ", index)
    simulation_result = simulate_bo(seed, 10, 10)
    print(simulation_result)
    results_file += str(seed) + "," + ",".join(str(num) for num in simulation_result) + "\n"

with open(results_file_path, 'w') as f:
    f.write(results_file)

"""
bad_subset = False

seed = 2

worst_result = 12 # Higher than max
worst_seed = 0

while not bad_subset:
    random.seed(seed)
    print("On seed: ", seed)
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

    simulation_result = simulate_bo(0, 10, 10)
    if simulation_result < 9:
        bad_subset = True
        print("GOT ONE!")
    else:
        if simulation_result < worst_result:
            worst_result = simulation_result
            worst_seed = seed
        print("Worst so far is ", worst_result, " with seed ", worst_seed)
        seed += 1

print(seed)
"""
