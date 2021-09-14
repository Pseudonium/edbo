Data here is gotten from 'https://figshare.com/articles/dataset/moldata_csv/9640427', although the folder itself doesn't contain the file, as it's too large (more than half a gigabyte!). So, if you wish to run 'test_bo_clean', you must first do the following:

1) Download 'moldata.csv' from the link above
2) Place the file into this folder (specifically 'harvard-clean-energy')
3) Run 'moldata_cleaner.py', which formats the file to make it usable for the script, and removes unnecessary information, to produce a new file 'moldata_clean.csv'

You may then safely delete 'moldata.csv' from the folder if you wish.



This folder also contains a script called 'bad_subset_finder'. Due to the large size of the harvard dataset, I couldn't load everything into an optimiser instance. Instead, I took a random sample of size 10000 from the subset, and performed experiments with that. The optimiser seemed to perform very well, which was promising - however, we wanted to see if we could find a subset where the optimiser performed poorly.

So, the 'bad_subset_finder' aims to find a seed for which a single run of the optimiser performs poorly. I tested it for seeds well into the hundreds, and got the 'worst' seed as 39 (for a size of 10000), which is why some of the files have 'harvard39' as a prefix - these indicate a 'MASTER_SEED' of 39.

Some files also have 'top5' - these come from experiments where we were looking at the top5 results of the optimiser, rather than just the top result. You can replicate this by setting 'TOP_N' to 5, though you will need to manually change the naming scheme. This is not present in the newer, non-archived data files.

Key for archived data files:

- 'randomei' = initial selection is random, acquisition function of expected improvement
- 'randomeits' = initial selection is random, acquisition function of EI-TS (see a test_bo file for explanation)
- 'randomrand' = initial selection is random, acquisition function is also random (no real bayesian optimisation here, just choosing points randomly, acts as a control)
- 'randomts' = initial selection is random, acquisition function is thompson sampling



In the non-archived data files, there are some additional acquisition functions tested:

- 'EI-1' - regular expected improvement
- 'EI-2' - expected improvement where the second highest observed value is taken as the 'highest' for purposes of calculation
- 'EI-5' - same as EI-2, except 5th highest value is taken

I've modified the acq_func.py file in the edbo folder so that you can implement 'EI-k' for 'k'th highest value if necessary.



- 'E3I' - exploration enhanced expected improvement, alternative expected improvement algorithm.