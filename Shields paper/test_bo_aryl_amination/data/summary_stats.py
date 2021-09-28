import os
import numpy as np

def process_file(filename):
    """Want to return tuple of the form:
    (reaction, acqusition function, batch size, experiment budget, mean final yield, standard deviation)"""
    final_yields = list()
    with open(filename) as f:
        for line in f.readlines()[1:]:
            final_yields.append(float(line.split(",")[1]))
    
    reaction, acqusition_func, batch_size, budget, _ = filename.split("_")
    
    mean_yield = sum(final_yields) / len(final_yields)
    standard_dev = np.std(final_yields)
    
    return (reaction, acqusition_func, batch_size, budget, mean_yield, standard_dev)



OUTPUT_NAME = "Optimiser Summmary Statistics.csv"

def main():
    summary_stats = list()
    for filename in os.listdir():
        if filename.split(".")[1] != "csv" or filename == OUTPUT_NAME:
            continue
        summary_stats.append(process_file(filename))
    
    output_file = "reaction,acqusition function,batch size,experiment budget,mean final yield,standard deviation\n"
    for stat in summary_stats:
        output_file += ",".join(str(item) for item in stat) + "\n"
    
    with open(OUTPUT_NAME, "w") as f:
        f.write(output_file)
    
    return 0


main()