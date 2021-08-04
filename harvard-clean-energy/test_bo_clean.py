import pandas as pd
from edbo.bro import BO_express

import random
from edbo.feature_utils import mordred

MASTER_SEED = 69                      # Nice
SAMPLE_SIZE = 5 * (10 ** 3)

print("Starting encoding!")

components = {
    'chemical': '<defined in descriptor_matrices>'
}

full_clean_df = pd.read_csv('moldata.csv')[[
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


print("Now generating the mordred descriptors...")
descriptors = pd.concat(
    [
        mordred(
            scd.index, name='chemical'
        ).set_index(scd.index),  # Mordred encoding from SMILES
        scd[["e_homo_alpha", "e_lumo_alpha"]]  # From harvard data
    ],
    axis=1
)
#print(descriptors)

print("Initialising bo")
bo = BO_express(
    reaction_components=components,
    encoding={},
    descriptor_matrices={'chemical': descriptors},
    acquisition_function='TS',
    init_method='rand',
    target='pce',
    batch_size=5,
)
print("Finished setting up bo!")


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

bo.init_sample(MASTER_SEED)
print(bo.get_experiments())
bo.export_proposed('init.csv')
fill_in_experiment_values('init.csv')
bo.add_results('init.csv')

def workflow(export_path):
    bo.run()
    bo.export_proposed(export_path)
    fill_in_experiment_values(export_path)
    bo.add_results(export_path)

for num in range(10):
    print("Starting round ", num)
    workflow(f"round{num}.csv")
    print("Finished round ", num)

bo.plot_convergence()
