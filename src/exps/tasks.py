# AMLB classification
from __future__ import annotations

AMLB_CLASSIFICATION_FULL = [
    # 2073,  # Fails to stratify split correctly due to not enough instances of a class
    3945,
    # 7593,  # No configuration evaluated within 10 minutes
    10090,
    146818,
    146820,
    167120,
    168350,
    168757,
    168784,
    168868,
    168909,
    168910,
    168911,
    189354,
    # 189355,  # No configuration evaluated within 10 minutes
    # 189356,  # No configuration evaluated within 10 minutes
    189922,
    190137,
    190146,
    190392,
    190410,
    190411,
    190412,
    211979,
    211986,
    359953,
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
    359974,
    359975,
    359976,
    359977,
    359979,
    359980,
    359981,
    359982,
    359983,
    359984,
    359985,
    359986,
    359987,
    359988,
    359989,
    359990,
    359991,
    359992,
    359993,
    # 359994,  # No configuration evaluated within 10 minutes
    # 360112,  # No configuration evaluated within 10 minutes
    # 360113,  # No configuration evaluated within 10 minutes
    # 360114,  # No configuration evaluated within 10 minutes
    # 360975,  # No configuration evaluated within 10 minutes
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
