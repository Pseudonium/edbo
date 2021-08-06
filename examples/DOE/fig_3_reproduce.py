# Imports

import pandas as pd
from edbo.utils import Data
# import pdb
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib
from sklearn.cluster import KMeans
import numpy as np
from sklearn import metrics
from edbo.bro import BO_express
from gpytorch.priors import GammaPrior
import random

###
# Constants
###

MASTER_SEED = 213090120
COLORS = ['black', 'red', 'yellow']


#############################
#############################
##### REACTION ENCODING #####
#############################
#############################

print("Starting Reaction Encoding!")

# Load DFT descriptor CSV files computed with auto-qchem using pandas
# Instantiate a Data object


# Suzuki here
electrophiles = Data(pd.read_csv('data/suzuki/electrophile_dft.csv'))
nucleophiles = Data(pd.read_csv('data/suzuki/nucleophile_dft.csv'))
ligands = Data(pd.read_csv('data/suzuki/ligand-random_dft.csv'))
bases = Data(pd.read_csv('data/suzuki/base_dft.csv'))
solvents = Data(pd.read_csv('data/suzuki/solvent_dft.csv'))
reactants = [electrophiles, nucleophiles, ligands, bases, solvents]


print("Loaded csv files...")

# Use Data.drop method to drop descriptors containing some unwanted keywords

for data in reactants:
    data.drop(['file_name', 'vibration', 'correlation', 'Rydberg',
               'correction', 'atom_number', 'E-M_angle', 'MEAN', 'MAXG',
               'STDEV'])

print("Dropped unnecessary data...")

# Parameters in reaction space

# Suzuki here
components = {
    'electrophile': 'DFT',
    'nucleophile': 'DFT',
    'ligand': 'DFT',
    'base': 'DFT',
    'solvent': 'DFT'
}


# External descriptor matrices override specified encoding


dft = {
    'electrophile':electrophiles.data,
    'nucleophile':nucleophiles.data,
    'ligand':ligands.data,
    'base':bases.data,
    'solvent':solvents.data
}

encoding = {}

############################
############################
#### Instantiating EDBO ####
############################
############################

def instantiate_bo(acquisition_func: str, batch_size: int, init_method='rand'):
    bo = BO_express(
        components,
        encoding=encoding,
        descriptor_matrices=dft,
        acquisition_function=acquisition_func,
        init_method=init_method,
        batch_size=batch_size,
        target='yield'
    )
    # BO_express actually automatically chooses priors
    # We can reset them manually to make sure they match the ones from our paper
    bo.lengthscale_prior = [GammaPrior(2.0, 0.2), 5.0]
    bo.outputscale_prior = [GammaPrior(5.0, 0.5), 8.0]
    bo.noise_prior = [GammaPrior(1.5, 0.5), 1.0]
    return bo

"""
bo = instantiate_bo('VarMax', 1)

FOLDER_PATH = "test_bo_suzuki/"
#####
# Constructing kmeans clusters
####

data_embedded = TSNE(init='pca').fit_transform(bo.reaction.data)

N_CLUSTERS = 5
kmeans = KMeans(n_clusters=N_CLUSTERS).fit_predict(bo.reaction.data)
cm = matplotlib.cm.get_cmap(name='viridis')
norm = matplotlib.colors.Normalize(vmin=0.0, vmax=N_CLUSTERS)
colors = [cm(norm(thing)) for thing in kmeans]

fig_cluster, axs_cluster = plt.subplots(1)
axs_cluster.scatter([item[0] for item in data_embedded], [item[1] for item in data_embedded], c=colors)

axs_cluster.set_xlabel('t-SNE1')
axs_cluster.set_ylabel('t-SNE2')
axs_cluster.set_title('Paths taken in reaction space')

fig_3b, (axs_r2, axs_yield) = plt.subplots(nrows=2, ncols=1, sharex=True)

axs_yield.set_xlabel('Experiment')

axs_r2.set_ylabel('Model fit score')
axs_yield.set_ylabel('Observed yield')

####################################
####################################
#### Bayesian Optimization Loop ####
####################################
####################################

RESULT_PATH = 'data/suzuki/experiment_index.csv'
NUM_ROUNDS = 50

path_cm = matplotlib.cm.get_cmap(name='Reds')
path_norm = matplotlib.colors.Normalize(vmin=0.0, vmax=NUM_ROUNDS)

with open(RESULT_PATH) as f:
    FULL_RESULT_DICT = {",".join(line.split(",")[1:-1]): float(line.split(",")[-1][:-1]) for line in f.readlines()[1:]}

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
            input_yield = FULL_RESULT_DICT[search_string]
            line = ",".join(original_line.split(",")[:-1]) + "," + str(input_yield) + "\n"
            newfile += line
    with open(input_path, 'w') as f:
        f.write(newfile)

    return input_yield


def workflow(export_path, count=0, indices=None, fig=0, plot=True):
    #Function for our BO pipeline.

    if indices is None:
        indices = []

    bo.run()
    new_experiment_index = bo.get_experiments().index[0]
    indices.append(new_experiment_index)
    if len(indices) > 1 and plot:
        axs_cluster.scatter([data_embedded[new_experiment_index][0]], [data_embedded[new_experiment_index][1]], color=path_cm(path_norm(count)), s=9)
        x, y = data_embedded[indices[count - 1]]
        x_new, y_new = data_embedded[indices[count]]
        dx, dy = x_new - x, y_new - y
        axs_cluster.arrow(
            x, y, dx, dy,
            width=0.1,
            length_includes_head=True,
            head_width = 3,
            head_length = 3,
            linestyle='--',
            color=COLORS[fig]
        )
    bo.export_proposed(export_path)
    return indices


human_readable_domain_data = bo.reaction.base_data[bo.reaction.index_headers]
results_array = np.array([FULL_RESULT_DICT[",".join(human_readable_domain_data.iloc[i].tolist())] for i in range(len(human_readable_domain_data))])
# The point of the ",".join is that the .tolist() returns all the descriptors in order as a list
# And then we join them with commas to form the search key for the results dict


def simulate_bo(bo, fig_num):
    indices = None

    obs_yields = []

    bo.init_sample(seed=MASTER_SEED)             # Initialize
    bo.export_proposed(FOLDER_PATH + 'init.csv')     # Export design to a CSV file
    obs_yields.append(fill_in_experiment_values(FOLDER_PATH + 'init.csv'))
    bo.add_results(FOLDER_PATH + 'init.csv')

    r2_values = list()

    for num in range(NUM_ROUNDS):
        print("Starting round ", num)
        try:
            indices = workflow(
                FOLDER_PATH + 'round' + str(num) + '.csv',
                count=num,
                indices=indices,
                fig=fig_num,
                plot = num < 10 # So don't plot subsequent 10 experiments
            )
        except RuntimeError as e:
            print(e)
            print("No idea how to fix this, seems to occur randomly for different seeds...")
            break
        obs_yields.append(fill_in_experiment_values(FOLDER_PATH + 'round' + str(num) + '.csv'))
        bo.add_results(FOLDER_PATH + "round" + str(num) + ".csv")
        print("Finished round ", num)
        pred = np.array(bo.obj.scaler.unstandardize(bo.model.predict(bo.obj.domain.values)))
        print(f"Current R^2 value is {metrics.r2_score(results_array, pred)}")
        r2_values.append(metrics.r2_score(results_array, pred))

    # The very first score tends to be very negative, so instead
    # we will ignore the first one
    axs_r2.plot(list(range(NUM_ROUNDS))[1:], r2_values[1:], color=COLORS[fig_num])

    axs_yield.plot(list(range(NUM_ROUNDS + 1)), obs_yields, color=COLORS[fig_num])


"""

"""
simulate_bo(bo, 0)

bo = instantiate_bo('MeanMax', 1)
print("Instantiated BO object...")
simulate_bo(bo, 1)

bo = instantiate_bo('EI', 2)
print("Instantiated BO object...")
simulate_bo(bo, 2)
"""

NUM_ROUNDS = 10
NUM_AVG = 50

fig_max_yield, axs_max_yield = plt.subplots(1)

"""
random.seed(MASTER_SEED)
seeds = random.sample(range(10 ** 6), NUM_AVG)

def simulate_bo_2(method, fig_num):
    # Doing an average of ~ 5 here, so use the master seed to make a random sample


    full_yields = []
    for seed in seeds:
        bo = instantiate_bo(method, 5)
        if method == 'greedy':
            bo.eps = 0.1   # To match the value used in the paper
        bo.init_sample(seed=seed)             # Initialize
        bo.export_proposed(FOLDER_PATH + 'init.csv')     # Export design to a CSV file
        fill_in_experiment_values(FOLDER_PATH + 'init.csv')
        bo.add_results(FOLDER_PATH + 'init.csv')

        for num in range(NUM_ROUNDS):
            print("Starting round ", num)
            bo.run()
            bo.export_proposed(FOLDER_PATH + 'round' + str(num) + '.csv')
            fill_in_experiment_values(FOLDER_PATH + 'round' + str(num) + '.csv')
            bo.add_results(FOLDER_PATH + "round" + str(num) + ".csv")
            print("Finished round ", num)


        max_yields = []
        results = pd.DataFrame(columns=bo.reaction.index_headers + ['yield'])
        for index, path in enumerate([FOLDER_PATH + 'init'] + [FOLDER_PATH + 'round' + str(num) for num in range(NUM_ROUNDS)]):
            results = pd.concat([results, pd.read_csv(path + '.csv', index_col=0)], sort=False)
            results = results.sort_values('yield', ascending=False)
            max_yields.append(results['yield'].tolist()[0])

        full_yields.append(max_yields)
    return pd.DataFrame.from_records(full_yields)


methods = ['EI', 'TS', 'greedy', 'MeanMax', 'VarMax']

yield_df = pd.DataFrame(columns=['method'].extend(range(11)))
yield_df['method'] = methods

yield_dict = {}

for index, method in enumerate(methods):
    print("TRYING OUT METHOD ", method)
    result = simulate_bo_2(method, index)
    #yield_df = pd.concat(yield_df, )
    yield_dict[method] = result

for key, value in yield_dict.items():
    value.insert(0, 'method', [key for num in range(NUM_AVG)], allow_duplicates=True)

full_yield_df = pd.DataFrame()

for value in yield_dict.values():
    full_yield_df = pd.concat([full_yield_df, value])

full_yield_df.to_csv('fig3c.csv')
"""

# Have already computed the above
# Now just need to read in the file and produce the plot

full_yield_df = pd.read_csv('fig3c.csv')

# Start with plotting the graph, which needs the average, as well as the
# standard deviation for EI specifically

average_df = pd.DataFrame(columns=['method'].extend(range(11)))

result_df = full_yield_df[full_yield_df['method'] == 'EI']
print('Initial result dataframe: ', result_df)

print(r'Now aggregated into mean: \n', result_df.agg(func='mean').loc('4'))
print('New result dataframe: ', result_df)
print('Full one is still untouched: ', full_yield_df)
