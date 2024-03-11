# Footprint
#python src/exps/footprint.py --borders 300 --support 400 --max-iter 100 --outpath plots/footprint-main --method current_average_worse_than_mean_best current_average_worse_than_best_worst_split --fold 7 data/mlp-nsplits-10.parquet.gzip
#
# Inc traces (no test)
python e1.py plot-stacked --methods "disabled" "current_average_worse_than_mean_best" "current_average_worse_than_best_worst_split" --outpath plots --prefix results-main --n-splits 3 5 10 --model mlp rf --ax-height 3 --ax-width 5 --metric roc_auc_ovr --time-limit 3600 data/mlp-nsplits-3.parquet.gzip data/mlp-nsplits-5.parquet.gzip data/mlp-nsplits-10.parquet.gzip data/rf-nsplits-3.parquet.gzip data/rf-nsplits-5.parquet.gzip data/rf-nsplits-10.parquet.gzip
# Inc traces (with test)
python e1.py plot-stacked --with-test --methods "disabled" "current_average_worse_than_mean_best" "current_average_worse_than_best_worst_split" --outpath plots --prefix results-2-5 --n-splits -5 --model mlp rf --ax-height 3 --ax-width 7 --metric roc_auc_ovr --time-limit 3600 data/mlp-nsplits-2-5.parquet.gzip data/rf-nsplits-2-5.parquet.gzip
python e1.py plot-stacked --with-test --methods "disabled" "current_average_worse_than_mean_best" "current_average_worse_than_best_worst_split" --outpath plots --prefix results-2-10 --n-splits 20 --model mlp rf --ax-height 3 --ax-width 7 --metric roc_auc_ovr --time-limit 3600 data/mlp-nsplits-20.parquet.gzip data/rf-nsplits-20.parquet.gzip
python e1.py plot-stacked --with-test --methods "disabled" "current_average_worse_than_mean_best" "current_average_worse_than_best_worst_split" --outpath plots --prefix results-test-3 --n-splits 3 --model mlp rf --ax-height 3 --ax-width 7 --metric roc_auc_ovr --time-limit 3600 data/mlp-nsplits-3.parquet.gzip data/rf-nsplits-3.parquet.gzip
python e1.py plot-stacked --with-test --methods "disabled" "current_average_worse_than_mean_best" "current_average_worse_than_best_worst_split" --outpath plots --prefix results-test-5 --n-splits 5 --model mlp rf --ax-height 3 --ax-width 7 --metric roc_auc_ovr --time-limit 3600 data/mlp-nsplits-5.parquet.gzip data/rf-nsplits-5.parquet.gzip
#
# Speedups MLPS
python e1.py plot --methods "disabled" "current_average_worse_than_mean_best" "current_average_worse_than_best_worst_split" --outpath plots --prefix results-main-mlp-10 --n-splits 10 --model mlp --metric roc_auc_ovr --time-limit 3600 --kind speedups data/mlp-nsplits-10.parquet.gzip
python e1.py plot --methods "disabled" "current_average_worse_than_mean_best" "current_average_worse_than_best_worst_split" --outpath plots --prefix results-main-mlp-5 --n-splits 5 --model mlp --metric roc_auc_ovr --time-limit 3600 --kind speedups data/mlp-nsplits-5.parquet.gzip
python e1.py plot --methods "disabled" "current_average_worse_than_mean_best" "current_average_worse_than_best_worst_split" --outpath plots --prefix results-main-mlp-3 --n-splits 3 --model mlp --metric roc_auc_ovr --time-limit 3600 --kind speedups data/mlp-nsplits-3.parquet.gzip
python e1.py plot --methods "disabled" "current_average_worse_than_mean_best" "current_average_worse_than_best_worst_split" --outpath plots --prefix results-main-mlp-2-10 --n-splits 20 --model mlp --metric roc_auc_ovr --time-limit 3600 --kind speedups data/mlp-nsplits-20.parquet.gzip
#
# Speedups RFS
python e1.py plot --methods "disabled" "current_average_worse_than_mean_best" "current_average_worse_than_best_worst_split" --outpath plots --prefix results-main-rf-10 --n-splits 10 --model rf --metric roc_auc_ovr --time-limit 3600 --kind speedups data/rf-nsplits-10.parquet.gzip
python e1.py plot --methods "disabled" "current_average_worse_than_mean_best" "current_average_worse_than_best_worst_split" --outpath plots --prefix results-main-rf-5 --n-splits 5 --model rf --metric roc_auc_ovr --time-limit 3600 --kind speedups data/rf-nsplits-5.parquet.gzip
python e1.py plot --methods "disabled" "current_average_worse_than_mean_best" "current_average_worse_than_best_worst_split" --outpath plots --prefix results-main-rf-3 --n-splits 3 --model rf --metric roc_auc_ovr --time-limit 3600 --kind speedups data/rf-nsplits-3.parquet.gzip
python e1.py plot --methods "disabled" "current_average_worse_than_mean_best" "current_average_worse_than_best_worst_split" --outpath plots --prefix results-main-rf-2-10 --n-splits 20 --model rf --metric roc_auc_ovr --time-limit 3600 --kind speedups data/rf-nsplits-20.parquet.gzip
#
# Optimized MLP, RF 10 splits
# ---
# Inc
python e1.py plot-stacked --merge-opt-into-method --methods "disabled" "current_average_worse_than_mean_best" "current_average_worse_than_best_worst_split" --outpath plots --prefix results-main-opt --n-splits 10 --model mlp rf --metric roc_auc_ovr --ax-width 6 --ax-height 4 --ncols-legend 3 --time-limit 3600 data/opt-mlp-nsplits-10.parquet.gzip data/opt-rf-nsplits-10.parquet.gzip
# Speedups
python e1.py plot --merge-opt-into-method --methods "disabled" "current_average_worse_than_mean_best" "current_average_worse_than_best_worst_split" --outpath plots --prefix app-opt-mlp-10 --n-splits 10 --model mlp --metric roc_auc_ovr --time-limit 3600 --kind speedups data/opt-mlp-nsplits-10.parquet.gzip
python e1.py plot --merge-opt-into-method --methods "disabled" "current_average_worse_than_mean_best" "current_average_worse_than_best_worst_split" --outpath plots --prefix app-opt-rf-10 --n-splits 10 --model rf --metric roc_auc_ovr --time-limit 3600 --kind speedups data/opt-rf-nsplits-10.parquet.gzip
