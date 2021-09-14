from rdkit import Chem
import pandas as pd

full_df = pd.read_csv('moldata.csv')

partial_df = full_df[['SMILES_str', 'pce', 'tmp_smiles_str']]
del full_df  # frees up memory


for index in range(len(partial_df)):
    if Chem.MolFromSmiles(partial_df.iloc[index]["SMILES_str"]) is None:
        # Sometimes this occurs. However, I've checked that in every case,
        # the tmp_smiles_str does perform correctly.
        # However, there are instances where the tmp_smiles_str fails,
        # but the regular SMILES_str succeeds, which is why we can't
        # simply use the tmp_smiles_str. Instead, we replace
        # SMILES_str with tmp_smiles_str anytime this happens.
        partial_df.iloc.at[index, "SMILES_str"] = partial_df.iloc[index]['tmp_smiles_str']


# Now it's all fixed, we can drop the tmp_smiles_str column
partial_df = partial_df.drop('tmp_smiles_str')

# Finally, we write the resulting dataframe
partial_df.to_csv('moldata_clean.csv', index=False)