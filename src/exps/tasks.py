# AMLB classification
from __future__ import annotations

AMLB_CLASSIFICATION_FULL = [
    2073,  # Fails to stratify split correctly due to not enough instances of a class
    3945,  # Failed with SMAC due to timeout on all folds
    7593,  # No configuration evaluated within 10 minutes
    10090,  # Partial failure to evaluate all folds with SMAC
    146818,
    146820,
    167120,
    168350,
    168757,
    168784,
    168868,  # Failed with SMAC due to timeout on all folds
    168909,  # Failed with SMAC due to timeout on all folds
    168910,
    168911,
    189354,  # Failed with SMAC due to timeout on all folds
    189355,  # Failed with all due to timeout on all folds
    189356,  # Failed with all due to timeout on all folds
    189922,
    190137,
    190146,
    190392,
    190410,
    190411,
    190412,
    211979,  # Failed with SMAC due to timeout on all folds
    211986,  # Failed with SMAC due to timeout on all folds
    359953,  # Fold 7 fails (too few classes for 10 fold cv)
    359954,
    359955,
    359956,
    359957,
    359958,
    359959,
    359960,
    359961,
    359962,
    359963,
    359964,
    359965,
    359966,
    359967,
    359968,
    359969,
    359970,
    359971,
    359972,
    359973,
    359974,  # Failed with all due to timeout on all folds
    359975,
    359976,  # Failed with SMAC due to timeout on all folds
    359977,  # Partial failures with SMAC due to timeout on folds
    359979,
    359980,
    359981,
    359982,
    359983,
    359984,  # Failed with SMAC due to timeout on all folds
    359985,  # Failed with SMAC due to timeout on all folds
    359986,  # TODO: Got weird process termination errors, should rerun with SMAC
    359987,  # All folds fails (too few classes for 10 fold cv)
    359988,  # TODO: Rerun, weird killing process error
    359989,  # TODO Same as above
    359990,  # Failed with SMAC due to timeout on all folds
    359991,  # Failed with SMAC due to timeout on all folds
    359992,
    359993,
    359994,  # No configuration evaluated within 10 minutes
    360112,  # No configuration evaluated within 10 minutes
    360113,  # No configuration evaluated within 10 minutes
    360114,  # No configuration evaluated within 10 minutes
    360975,  # No configuration evaluated within 10 minutes
]

AMLB_CLASSIFICATION_SMALL_LESS_THAN_50K = [
    # 2073,  # Fails to stratify split correctly due to not enough instances of a class
    146818,
    146820,
    168350,
    168757,
    190146,
    359954,
    359955,
    359956,
    359959,
    359960,
    359962,
    359963,
]

TASKS = {
    "amlb_classification_full": AMLB_CLASSIFICATION_FULL,
    "amlb_classification_less_than_50k": AMLB_CLASSIFICATION_SMALL_LESS_THAN_50K,
    "debug": [31],
}
