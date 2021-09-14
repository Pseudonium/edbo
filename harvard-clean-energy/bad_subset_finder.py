import pandas as pd
from edbo.bro import BO_express
import random
from edbo.feature_utils import mordred
from gpytorch.priors import GammaPrior
from rdkit import Chem
from rdkit.Chem import AllChem
import os.path
import time

bad_subset = False
SAMPLE_SIZE = 10 ** 4

seed = 2

worst_result = 12 # Higher than max
worst_seed = 0

full_clean_df = pd.read_csv('moldata_clean.csv')


def instantiate_bo(acquisition_func: str, batch_size: int):
    bo = BO_express(
        components,
        encoding={},
        acquisition_function=acquisition_func,
        descriptor_matrices={'chemical': encoded_df},
        init_method='rand',
        batch_size=batch_size,
        target='pce'
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

def write_prop_read_run(bo, export_path):
    bo.export_proposed(export_path)
    fill_in_experiment_values(export_path)
    bo.add_results(export_path)
    bo.run()

def get_max_yields(bo, num_rounds):
    results = pd.DataFrame(columns=bo.reaction.index_headers + ['pce'])
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
    return sorted(results['pce'].tolist(), reverse=True)[:TOP_N]

def simulate_bo(seed, acquisition_func, batch_size, num_rounds):
    bo = instantiate_bo(acquisition_func, batch_size)
    bo.init_sample(seed=seed)
    print(bo.get_experiments())
    write_prop_read_run(bo, FOLDER_PATH + 'init.csv')

    for num in range(num_rounds):
        print(f"Starting round {num}")
        write_prop_read_run(bo, FOLDER_PATH + f"round{num}.csv")
        print(f"Finished round {num}")
    return get_max_yields(bo, num_rounds)

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
