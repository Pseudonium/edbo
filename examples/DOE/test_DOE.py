#############################
#############################
#### Build Search Spaces ####
#############################
#############################

# Imports

import pandas as pd
from edbo.utils import Data
from edbo.feature_utils import reaction_space
from data_loader import aryl_amination, suzuki

# Suzuki reaction data

class reaction1:
    """Suzuki data loader and component shuffling"""

    def __init__(self):

        # Load OHE data
        self.data = suzuki()
        
        # Components required for DOE
        self.components = {'electrophile':0,
                           'nucleophile':1,
                           'base':2,
                           'ligand':4,
                           'solvent':5}
    
    def shuffle(self, seed=0):
        """Suffle DataFrame using sampling."""
        
        rxn = Data(self.data.copy())
        
        # Shuffle to change ordering of variables    
        rxn.base_data = rxn.base_data.sample(len(rxn.base_data), random_state=seed).copy().reset_index(drop=True)
        rxn.data = rxn.base_data.copy()
    
        # Clean data
        rxn.clean()
        rxn.drop(['entry'])
    
        # Required for DOE function
        rxn.index_headers = ['electrophile_SMILES', 
                             'nucleophile_SMILES',
                             'base_SMILES',
                             'ligand_SMILES',
                             'solvent_SMILES']
    
        return rxn

# Aryl amination reaction data

class reaction2:
    """Aryl amination data loader and component shuffling"""

    def __init__(self, subset):

        # Load OHE data
        self.data = aryl_amination(subset=subset)
        
        # Components required for DOE
        self.components = {'aryl_halide':0,
                           'additive':1,
                           'base':2,
                           'ligand':4}
    
    def shuffle(self, seed=0):
        """Suffle DataFrame using sampling."""
        
        rxn = Data(self.data.copy())
        
        # Shuffle to change ordering of variables    
        rxn.base_data = rxn.base_data.sample(len(rxn.base_data), random_state=seed).copy().reset_index(drop=True)
        rxn.data = rxn.base_data.copy()
    
        # Clean data
        rxn.clean()
        rxn.drop(['entry'])
    
        # Required for DOE function
        rxn.index_headers = ['aryl_halide_SMILES', 
                             'additive_SMILES',
                             'base_SMILES',
                             'ligand_SMILES']
    
        return rxn

##################
##################
#### Modeling ####
##################
##################


# Imports

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import ARDRegression
from sklearn.model_selection import GridSearchCV
import numpy as np

from edbo.bro import BO
from edbo.acq_func import top_predicted

from DOE import generalized_subset_design, external_design

# Polynomial regression model using ARD and grid search CV for training

class Polynomial_ARD_Model:
    """Bayesian linear polynomial regression object compatible with the BO framework."""
    
    def __init__(self, X, y, **kwargs):
        
        # CV set gamma prior parameters - no GS for now
        self.alphas = np.logspace(-6, 0.5, 7)
        
        # Initialize model
        self.model = ARDRegression(n_iter=50)
        
        # Make sure X and y are numpy arrays
        self.X = np.array(X)
        self.y = np.array(y)
        
        # Polynomial expansion
        self.expander = PolynomialFeatures(2)
        self.X = self.expander.fit_transform(self.X)
        
    # Fit    
    def fit(self):
        """Train the model using grid search CV.""" 
        
        parameters = [{'alpha_1': self.alphas, 'alpha_2': self.alphas}]
        
        # Set the number of folds
        if len(self.X) < 5:
            n_folds = len(self.X)
        else:
            n_folds = 5
        
        # Run grid search
        if n_folds > 1:
        
            # Select l1 term via grid search
            self.grid_search = GridSearchCV(self.model, 
                                       parameters, 
                                       cv=n_folds, 
                                       refit=True,
                                       n_jobs=-1)
        
            self.grid_search.fit(self.X, self.y)
        
            # Set model to trained model
            self.model = self.grid_search.best_estimator_
        
        # Just fit model
        else:
            self.model.fit(self.X, self.y)
            
    def get_scores(self):
        """Get grid search cross validation results.""" 
        
        # Plot results
        scores = self.grid_search.cv_results_['mean_test_score']
        scores_std = self.grid_search.cv_results_['std_test_score']
        
        return scores, scores_std
        
    # Predict   
    def predict(self, points):
        """Model predictions.""" 
        
        # Make sure points in a numpy array
        points = np.array(points)
        
        # Expand
        points = self.expander.fit_transform(points)
        
        # Make predicitons
        pred = self.model.predict(points)
        
        return pred
        
    # Regression   
    def regression(self, return_data=False, export_path=None, return_scores=False):
        """Helper method for visualizing the models regression performance."""

        pred = self.predict(self.X)
        obs = self.y        
        return pred_obs(pred, 
                        obs, 
                        return_data=return_data, 
                        export_path=export_path,
                        return_scores=return_scores) 
    
    # Estimate variance
    def variance(self, points):
        """Estimated variance of Bayesian linear model."""
        
        # Make sure points in a numpy array
        points = np.array(points)
        
        # Expand
        points = self.expander.fit_transform(points)
        
        # Make predicitons
        pred, std = self.model.predict(points, return_std=True)
        
        return std**2

# DOE testing function utilizing edbo framework

def doe_optimization(reaction, seed, reduction=21, budget=50, design='gsd', external=None):
    """
    Run a design, fit a quadratic model and use the remainder of the 
    experimental budget to evaluate the top predicted experiments.
    """

    # Data
    space = reaction.shuffle(seed)

    # DOE
    if design == 'gsd':
        # Choose a reduction which gets closest to but is <= a budget of 50 experiments
        designer = generalized_subset_design(space, reaction.components)
        designer.build(reduction=reduction)
    elif design == 'external':
        designer = external_design(space, reaction.components, external)
    
    designer.get_experiments()              # Fetch corresponding experiments
    designer.encoded()                      # Fetch encoded experiments
    
    # Use BO to model
    bo = BO(domain=space.data.drop('yield', axis=1),       # Domain
            exindex=space.data,                            # Result index
            results=designer.encoded_design,               # DOE results
            acquisition_function='rand',                   # Random acquisition just to train the model
            batch_size=50 - len(designer.encoded_design),  # Number of points to choose using model                            
            fast_comp=True,                                # Use fast computation features
            init_method='external',                        # Initialization from results
            model=Polynomial_ARD_Model)                    # Polynomial model
    
    # Train model
    bo.run()
    
    # Get top predicted points
    top = top_predicted(bo.batch_size, False)
    best = top.run(bo.model, bo.obj)
    bo.obj.get_results(best, append=True)
    
    # Get max observed yield
    maximum = bo.obj.results_input()['yield'].max()
    
    # Model fit
    r2 = bo.model.model.score(bo.model.X, bo.model.y)
    
    # Reporting
    print('Seed:', seed, '|',
          'Design size:', len(designer.design), '|',
          'Fit R^2:', r2, '|',
          'DOE max:', bo.obj.results_input().iloc[:-bo.batch_size]['yield'].max(), '|',
          'Max:', bo.obj.results_input()['yield'].max(), '|',
          'Ground truth:', bo.obj.exindex['yield'].max(),
          )
    
    return designer, bo, maximum



"""

######################################
### Generalized Subset Design Test ###
######################################


# Data
reaction = reaction2(1)

# DOE + optimization
designer, bo, maximum = doe_optimization(reaction, 1, reduction=21, budget=50)
print(designer.design.head())
print(designer.experiment_design.head())
print(designer.encoded_design.head())


from edbo.plot_utils import pred_obs

pred = bo.obj.scaler.unstandardize(bo.model.predict(bo.obj.results.drop('yield', axis=1).values))
obs = bo.obj.results_input()['yield'].values
pred_obs(pred, obs, title='Training')

pred = bo.obj.scaler.unstandardize(bo.model.predict(bo.obj.domain.values))
obs = bo.obj.exindex['yield'].values
pred_obs(pred, obs, title='Full Space Fit')

"""

"""

#############################
### D-Optimal Design Test ###
#############################


# Data
reaction = reaction1()

# D-optimal design from R
external = pd.read_csv('R_doptimal/suzuki_d-optimal.csv').iloc[:,1:] - 1 # Adjust for python indices

# DOE + optimization
designer, bo, maximum = doe_optimization(reaction, 3, budget=50, design='external', external=external)

#print(designer.design.head())
#print(designer.experiment_design.head())
#print(designer.encoded_design.head())

from edbo.plot_utils import pred_obs

pred = bo.obj.scaler.unstandardize(bo.model.predict(bo.obj.results.drop('yield', axis=1).values))
obs = bo.obj.results_input()['yield'].values
pred_obs(pred, obs, title='Training')

pred = bo.obj.scaler.unstandardize(bo.model.predict(bo.obj.domain.values))
obs = bo.obj.exindex['yield'].values
pred_obs(pred, obs, title='Full Space Fit')

"""

####################################
####################################
#### Simulations and Statistics ####
####################################
####################################


# Simulation function

def simulation(reaction, rounds, reduction=21, budget=50, design='gsd', external=None):

    i = 0
    count = 0
    maxs = []
    results = []
    
    while count < rounds:
        # Some data sets are missing a few values which leads to an exception when fetching results
        try:
            designer, bo, maximum = doe_optimization(reaction, i, reduction=reduction, budget=budget, design=design, external=external)
            result = list(bo.obj.results_input()['yield'].values)
            
            # Try not to end up with the same experimental design twice
            if result not in results:
                results.append(result)
                maxs.append(maximum)
                count += 1
                i += 1
            else:
                i += 1
        except:
            i += 1
            pass
        
    return results, maxs


import matplotlib.pyplot as plt

# Statistics functions

import statsmodels.stats.api as sms

# Welsh's t-test    

def t_test(X1, X2, sample_vars='equal', print_out=True):
    """
    t-test for the null hypothesis of identical means. The unpaired 
    t-test should not be used if there is a significant difference 
    between the variances of the two samples. Here, if the variance
    of the samples is not assumed to be the same, then Welsh t-test 
    with Satterthwait degrees of freedom is used.
    """
    
    if sample_vars == 'equal':
        usevar = 'pooled'
    else:
        usevar = 'unequal'
        
    cm = sms.CompareMeans(sms.DescrStatsW(X1), sms.DescrStatsW(X2))
    ci = cm.tconfint_diff(usevar=usevar)
    t_value = cm.ttest_ind(usevar=usevar)
    
    if print_out == True:
        print('95% CI:', ci)
        print('DOF:', t_value[2])
        print('t-stat:', t_value[0])
        print('p-value:', t_value[1])
        
    return [t_value[2], t_value[0], t_value[1], ci[0], ci[1]]

# Report summary statistics    

def statistic_summary(bo_maxs, doe_maxs, export_path=None):
    """Distributional and statistical summary."""
    
    # Disributions
    plt.figure(figsize=(5,5))
    
    plt.hist(bo_maxs, alpha=0.5, label='Bayesian Optimization')
    plt.hist(doe_maxs, alpha=0.5, label='DOE Optimization')
    plt.legend(loc='upper left')
    plt.xlabel('Max Observed Yield')
    plt.ylabel('Count')
    
    if export_path is not None:
        plt.savefig(export_path + '.svg', format='svg', dpi=1200, bbox_inches='tight')
        plt.show()
    else:
        plt.show()
    
    # Statistics
    print('\n') 
    print('---------------------------------------')
    print('-------------Distributions-------------')
    print('---------------------------------------')
    print('BO Mean:', sum(bo_maxs)/len(bo_maxs), '|', 'DOE Mean:', sum(doe_maxs)/len(doe_maxs))
    print('BO STD:',  np.array(bo_maxs).std(),   '|', 'DOE STD:',  np.array(doe_maxs).std())
    print('BO Min:',  min(bo_maxs), '|', 'DOE Min:',  min(doe_maxs))
    print('\n') 
    print('---------------------------------------')
    print('---Hypothesis Test: MeanBO = MeanDOE---')
    print('---------------------------------------')
    print('BO Mean - DOE Mean:', sum(bo_maxs)/len(bo_maxs) - sum(doe_maxs)/len(doe_maxs))
    t = t_test(bo_maxs, doe_maxs, sample_vars='unequal')
    
    dist = [sum(bo_maxs)/len(bo_maxs) - sum(doe_maxs)/len(doe_maxs),
            np.array(bo_maxs).std() - np.array(doe_maxs).std(),
            min(bo_maxs) - min(doe_maxs)]
    
    columns = ['BO Mean - DOE Mean', 'BO STD - DOE STD', 'BO Min - DOE Min', 'DOF', 't-stat', 'p-value', 'CI95 Lower', 'CI95 Upper']
    
    return pd.DataFrame([dist + t], columns=columns)


# BO results

paths = ['simulation_results/Suz_bs=5_rand_GP-EI.csv',
         'simulation_results/CNar1_bs=5_rand_GP-EI.csv',
         'simulation_results/CNar2_bs=5_rand_GP-EI.csv',
         'simulation_results/CNar3_bs=5_rand_GP-EI.csv',
         'simulation_results/CNar4_bs=5_rand_GP-EI.csv',
         'simulation_results/CNar5_bs=5_rand_GP-EI.csv']

bo_results = {}
bo_maxs = {}
for path, key in zip(paths,['reaction1', 'reaction2a', 'reaction2b', 'reaction2c', 'reaction2d', 'reaction2e']):
    result = pd.read_csv(path, index_col=0)
    bo_results[key] = result
    bo_maxs[key] = result.max(axis=1).values


#############################################
### Generalized Subset Design Simulations ###
#############################################


PATH = 'simulation_results/'

# Data
reaction = reaction1()
name = PATH + 'Suzuki_GSD_Poly_ARD_Model'

# Simulation
results1, maxs1 = simulation(reaction, 20, reduction=19, budget=50)
pd.DataFrame(results1).to_csv(name + '.csv')

r1 = statistic_summary(bo_maxs['reaction1'], maxs1, export_path='simulation_results/Suzuki_hist_GSD')


# Data
reaction = reaction2(1)
name = PATH + 'CNar1_GSD_Poly_ARD_Model'

# Simulation
results2a, maxs2a = simulation(reaction, 20, reduction=21, budget=50)
pd.DataFrame(results2a).to_csv(name + '.csv')

r2a = statistic_summary(bo_maxs['reaction2a'], maxs2a, export_path='simulation_results/CNar1_hist_GSD')


# Data
reaction = reaction2(2)
name = PATH + 'CNar2_GSD_Poly_ARD_Model'

# Simulation
results2b, maxs2b = simulation(reaction, 20, reduction=21, budget=50)
pd.DataFrame(results2b).to_csv(name + '.csv')

r2b = statistic_summary(bo_maxs['reaction2b'], maxs2b, export_path='simulation_results/CNar2_hist_GSD')


# Data
reaction = reaction2(3)
name = PATH + 'CNar3_GSD_Poly_ARD_Model'

# Simulation
results2c, maxs2c = simulation(reaction, 20, reduction=21, budget=50)
pd.DataFrame(results2c).to_csv(name + '.csv')

r2c = statistic_summary(bo_maxs['reaction2c'], maxs2c, export_path='simulation_results/CNar3_hist_GSD')


# Data
reaction = reaction2(4)
name = PATH + 'CNar4_GSD_Poly_ARD_Model'

# Simulation
results2d, maxs2d = simulation(reaction, 20, reduction=21, budget=50)
pd.DataFrame(results2d).to_csv(name + '.csv')

r2d = statistic_summary(bo_maxs['reaction2d'], maxs2d, export_path='simulation_results/CNar4_hist_GSD')


# Data
reaction = reaction2(5)
name = PATH + 'CNar5_GSD_Poly_ARD_Model'

# Simulation
results2e, maxs2e = simulation(reaction, 20, reduction=21, budget=50)
pd.DataFrame(results2e).to_csv(name + '.csv')

r2e = statistic_summary(bo_maxs['reaction2e'], maxs2e, export_path='simulation_results/CNar5_hist_GSD')


summary = pd.concat([r1, r2a, r2b, r2c, r2d, r2e]).reset_index(drop=True)
summary.insert(0, 'Reaction', ['1', '2a', '2b', '2c', '2d', '2e'])
summary.to_csv('simulation_results/gsd_summary.csv', index=False)
summary