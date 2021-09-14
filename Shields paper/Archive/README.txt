This directory contains old performance data files I generated.

In order to start optimising, the EDBO optimiser must choose a set of initial experiments to evaluate the target value. In every performance test, these initial experiments are chosen randomly.
In order to ensure reproducibility, I manually set the random seed each time before the optimiser chose the initial experiment set, and recorded this seed next to the yield value.

However, when I initially ran the performance experiments, I didn't ensure that the set of seeds chosen was consistent across all files. Thus, in these old data files, the set of seeds is different per file. While the results are still reproducible (by reading in the set of seeds and manually setting them for each different iteration), it means the data collected here isn't as comparable (e.g. perhaps the optimiser only performed better on EI compared to TS because of the starting experiments being different).


In the new data files, this is fixed by having a 'master seed' (set to 42) which is used to select 50 random numbers from 0 to 999999, which form the set of seeds for every data file. Thus, the data is properly comparable.


However, I have decided to keep the old collected data in this archive, since it is still reproducible. The naming scheme is slightly different however:

- The first set of characters is either suzuki_, aryl_amination_, or direct_arylation_, which indicates the reaction type. I've also separated these into folders to make this clearer.
- The next word between underscores gives information about the configuration.
    - 'random' = initial experiments chosen randomly, acquisition function of expected improvement
    - 'randomts' = initial experiments chosen randomly, acquisition function of thompson sampling
    - 'randomtsei' = initial experiments chosen randomly, acquisition function of TS-EI (check a test_bo_ file in the Shields Paper folder for explanation)
    - 'worst' = initial experiments chosen randomly *from the bottom 10% of overall experiments, sorted by yield* (hence choosing the 'worst' experiments as a starting point), acquisition function of expected improvement.
- Afterwards, the name is always of the form x_y_50 - x indicates the batch size used, y indicates the experiment budget (batch size*rounds of optimising done), and 50 indicates that there were 50 optimiser instances run under these conditions, with different starting conditions determined by the seeds.


Some files also have a '_translated' at the end - these contain additional columns which give information about the yield values of the initially selected experiments. This was to see whether there was any correlation between the initial yield values and resulting optimiser performance - for example, one hypothesis was that having a high initial yield value could actually hurt performance, since the optimiser would be more likely to get trapped in that local maximum. Nothing of note was observed, though, so this has not been done on the new data files. These were generated using a variation of the 'seed_translator' script (in Shields Paper/Utilities), which should be simple to replicate for other files if desired.