# Experiments for Don't Waste Your Time: Early Stopping for Cross-Validation
The repository contains the code for the experiments in the paper
"Don't Waste Your Time: Early Stopping for Cross-Validation".


## Installation
To install the package, first clone/unpack the repository, `cd` into it and
then run the following command:
```bash
pip install -r requirements.txt  # Ensure the same versions of the packages
pip install -e .  # Allow the package to be edited and used
```

You can test that the installation was successful by running the following
which lists the available commands:
```bash
python e1.py --help
```

### Example data
We provide example data at the following link: https://figshare.com/s/b1653f7976f459de47d8
It includes data the the MLP experiments for all datasets and all outer folds, with 10 fold inner cross-validation.
This data can be used for plotting as described below (TODO:)

## Running the experiments
Each experiment is given a certain `--expname` and is defined inside of `e1.py`.
We've set up a small `"reproduce"` experiment to show a minamal working reconstruction
of the workflows used in the paper. To run the experiment, use the following command:
```bash
python e1.py run --expname reproduce
```

If you're using a slurm cluster, you can use the following instead:
```bash
python e1.py submit --expname reproduce
```

If you're going to be running many experiments in parallel on the cluster, we advise
you to use the `download.py` script to first download the openml data to prevent races and
data corruption.

Please see `python e1.py --help` for more.

## Getting the status of the experiments
To get the status of the experiments, use the following command:
```bash
python e1.py status --expname reproduce --out status.parquet
```

This will print the status of the experiments and output a dataframe to `status.parquet`
than can be used if required.

## Collecting results of experiments
To collect the results of the experiments, use the following command:
```bash
python e1.py collect --expname reproduce --out reproduce-results.parquet
```

## Plotting the results
To plot the results of the experiments, use the following command:
```bash
python e1.py plot-stacked \
    --methods "disabled" "current_average_worse_than_mean_best" "current_average_worse_than_best_worst_split" \
    --outpath . \
    --prefix reproduce \
    --n-splits 10 \
    --model mlp \
    --metric roc_auc_ovr \
    --time-limit 30 \
    repr-results.parquet
```

TODO: Add more on plotting
