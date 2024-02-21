from __future__ import annotations

from pathlib import Path

import pandas as pd
from amltk.data import reduce_dtypes


def path_col_to_str(_df: pd.DataFrame) -> pd.DataFrame:
    path_dtypes = _df.select_dtypes(Path).columns
    return _df.astype({k: pd.StringDtype() for k in path_dtypes})


def shrink_dataframe(_df: pd.DataFrame) -> pd.DataFrame:
    string_cols = _df.dtypes[_df.dtypes == "string"].index
    _df[string_cols] = _df[string_cols].astype("category")
    _df = path_col_to_str(_df)
    _df = _df.convert_dtypes()
    return reduce_dtypes(_df, reduce_int=True, reduce_float=True)
