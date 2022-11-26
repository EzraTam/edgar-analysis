"""
Module for analyzing financial data
"""

import os

import json

from typing import List, Optional, Dict, Union

from functools import reduce

# from icecream import ic

import seaborn as sns

import pandas as pd

from config import df_config

from ..edgar_data.edgar_data import EdgarDataComp 


interval_conf={
        "yearly":{
            "key":"year",
            "ascending":False
        },
        "quarterly":{
            "key":["year","fp"],
            "ascending":[False,False]
        }
}



class FinancialAnalyze(EdgarDataComp):
    """Class for analysing financial data from Edgar dict"""

    def __init__(
        self,
        interval:str,
        comp_nm: Optional[str]=None,
        data_comp:Optional[Dict]=None,
        path_data:Optional[str]=None,
        path_folder:Optional[str]=None,
        cik:Optional[str]=None,
        norm_mill: Optional[bool] = True,
        table: Optional[Dict[str, pd.DataFrame]] = None
    ) -> None:

        super().__init__(interval,data_comp,path_data,path_folder,cik,norm_mill)

        self.comp_nm = comp_nm if comp_nm else self.data_comp["entityName"]

        self.table = table if table else {}

    @classmethod
    def _concat_data(cls,li_df_concat: List[pd.DataFrame]):
        return (
            pd.concat(li_df_concat, ignore_index=True)
            .drop_duplicates(ignore_index=True)
            .sort_values(interval_conf[cls.interval]["key"], ascending=interval_conf[cls.interval]["ascending"], ignore_index=True)
        )

    @classmethod
    def _merge_data(cls,li_df_merge: List[pd.DataFrame]):
        return reduce(
            lambda df_old, df_to_add: pd.merge(
                df_old, df_to_add, how="outer", on=interval_conf[cls.interval]["key"]
            ),
            li_df_merge,
        ).sort_values(interval_conf[cls.interval]["key"], ascending=interval_conf[cls.interval]["ascending"], ignore_index=True)

