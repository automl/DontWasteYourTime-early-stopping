# Experiments for Don't Waste Your Time: Early Stopping for Cross-Validation
The repository contains the code for the experiments in the paper
"Don't Waste Your Time: Early Stopping for Cross-Validation".


## Installation
To install the package, first clone/unpack the repository, `cd` into it and
then run the following command:
```bash
#! In a virtual env with python 3.10, other versions may or may not work.

pip install -r requirements.txt  # Ensure the same versions of the packages
pip install -e .  # Allow the package to be edited and used
```

You can test that the installation was successful by running the following
which lists the available commands:
```bash
python e1.py --help
```

If things were unsuccessful, please ensure the output of `pip list` matches that
of the `requirements.txt` file.

### Experiment Data
We provide the data generated from the experiments at the following link:
* https://figshare.com/s/413b730e2d943190d580

Please create a folder `./data` and unzip the archive into that directory, you should set it up as the following:
```
data
├── data.zip
├── mlp-nsplits-10.parquet.gzip
├── mlp-nsplits-20.parquet.gzip
├── mlp-nsplits-20-unseeded.parquet.gzip
├── mlp-nsplits-2-5.parquet.gzip
├── mlp-nsplits-3.parquet.gzip
├── mlp-nsplits-5.parquet.gzip
├── opt-mlp-nsplits-10.parquet.gzip
├── opt-rf-nsplits-10.parquet.gzip
├── rf-nsplits-10.parquet.gzip
├── rf-nsplits-20.parquet.gzip
├── rf-nsplits-20-unseeded.parquet.gzip
├── rf-nsplits-2-5.parquet.gzip
├── rf-nsplits-3.parquet.gzip
└── rf-nsplits-5.parquet.gzip
```

This data can be used for plotting, for which we provide a `plots.sh` script.
This will produce all plots and place them in the `./plots` directory.
It produces a `.pdf` and a `.png` for each plot, however the `.pdf` should be prefered if
something looks wrong with the `.png`, for example legend scaling for the footprint plots.

Please refer to the `plots.sh` script for examples of generating plots, use any of the following:
```bash
e1.py plot --help
e1.py plot-stacked --help
e1.py src/exps/footprint.py --help
```

The footprint plots can take a long time to generate due to iterative MDS embedding which
is a non-linear scaling with the number of data points.
The provided dataset is not too long, however exploring a dataset with only `3` folds can likely take
up to an hour, depending on the dataset size and the number of trials that were evaluated.

We provide the command to do so seperatly which you can adapt as required:
```bash
# --borders How many random configurations to use for informing MDS about the borders of the space
# --support How many random configurations to use for informing MDS about random locations in the space
# --outpath Where to save the plots
# --method Which methods to plot (only really supports the main 3)
# --dataset Which openml task id to retrieve and plot
# --fold Which fold to plot
# --max-iter How many iterations to run the MDS for
# --seed Random seed for the MDS
# --njobs How many jobs to use for the MDS
# --ignore-too-many-configs Ignore the error when there are too many configurations (i.e. you acknowledge it takes time)
python src/exps/footprint.py \
    --borders 300 \
    --support 400 \
    --outpath plots/footprint-main \
    --method current_average_worse_than_mean_best current_average_worse_than_best_worst_split \
    --dataset 168350 \
    --fold 7 \
    --max-iter 100 \
    --seed 0 \
    --njobs -1 \
    --ignore-too-many-configs \
    data/mlp-nsplits-10.parquet.gzip
```

## Running the experiments
Each experiment is given a certain `--expname` and is defined inside of `e1.py`.
We've set up a small `"reproduce"` experiment to show a minamal working reconstruction
of the workflows used in the paper. To run the experiment, use the following command:
```bash
python e1.py run --expname reproduce
```

This will run the follow set of experiments, 2 datasets, 10 fold cross-validation,
2 outer folds, 3 methods, 30 seconds time limit, totalling 12 experiments which
should take roughly `12 * 30 = 360 ` seconds to run.
```python
n_splits = [10]
pipeline = "mlp_classifier"
n_cpu = 4
suite = [146818, 146820]
time_seconds = 30
folds = [0, 1]
methods = [
    "disabled",
    "current_average_worse_than_best_worst_split",
    "current_average_worse_than_mean_best",
]
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
python e1.py collect --expname reproduce --out data/reproduce-results.parquet
```

The results of these experiments can be used to the various plotting commands by
passing in the `data/reproduce-results.parquet` file.
