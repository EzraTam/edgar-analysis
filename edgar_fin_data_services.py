"""
Module for analyzing financial data
"""

import os

import json

from typing import List, Optional, Dict,Union

from functools import reduce, partial

from icecream import ic

import seaborn as sns

import pandas as pd

from config import df_config


def transform_to_df(data: dict, tag: str) -> pd.DataFrame:
    """Transform financial data to pandas

    Args:
        data (dict): Data in Edgar dict form
        tag (str): Tag of the data to be transformed

    Returns:
        pd.DataFrame: Result Dataframe
    """
    return pd.DataFrame(data["facts"]["us-gaap"][tag]["units"]["USD"])


def extract_yearly_data(
    data: dict, tag: str, col_nm: str, norm_mill:Optional[bool]=None
) -> pd.DataFrame:
    """Extract yearly data for a given tag

    Args:
        data (dict): Data in Edgar dict form
        tag (str): Tag of the values of interests
        col_nm (str): Name of the column of the resulting data
        norm_mill (bool): Flag for normalization values in millions

    Returns:
        pd.DataFrame: Resulting dataframe
    """
    # pylint: disable=E1101
    df_res = transform_to_df(data, tag)
    df_res = df_res[df_res["fp"] == "FY"]
    df_res["end"] = pd.to_datetime(df_res["end"], format="%Y-%m-%d")
    df_res["year"] = pd.DatetimeIndex(df_res["end"]).year
    df_res = (
        df_res.sort_values(["year", "fy"], ascending=False)
        .groupby("year")
        .head(1)[["year", "val"]]
        .reset_index(drop=True)
    )

    if norm_mill:
        df_res["val"] = df_res["val"] / 1000000

    if col_nm:
        df_res = df_res.rename(columns={"val": col_nm})

    return df_res


def extract_yearly_data_norm(tag: str, col_nm: str, data: dict) -> pd.DataFrame:
    """Extract yearly data from tag normalized to millions

    Args:
        data (dict): Company data in Edgar form
        tag (str): Tag of financial
        col_nm (str): _description_

    Returns:
        pd.DataFrame: _description_
    """

    return extract_yearly_data(data, tag, col_nm, norm_mill=True)


class FinancialAnalyze:
    """Class for analysing financial data from Edgar dict"""

    def __init__(
        self, comp_nm: str, data: dict, dict_df: Optional[Dict[str,pd.DataFrame]] = None
    ) -> None:
        """Initialization

        Args:
            comp_nm (str): Name of the Company
            data (dict): Data in Edgar Form
        """
        self.comp_nm = comp_nm
        self.data = data
        self.dict_df = dict_df if dict_df else {}

    def transform_to_df(self, tag: str) -> pd.DataFrame:
        """ Given a tag in Edgar data,
        show the corresponding data in DF
        """
        return transform_to_df(data=self.data,tag=tag)

    def extract_yearly_data(
        self, tag: str, col_nm: str, norm_mill: Optional[bool]=None
    ) -> pd.DataFrame:
        """Extract yearly data for a given tag

        Args:
            tag (str): Tag of the values of interests
            col_nm (str): Name of the column of the resulting data
            norm_mill (bool): Flag for normalization values in millions

        Returns:
            pd.DataFrame: Resulting dataframe
        """
        return extract_yearly_data(
            data=self.data, tag=tag, col_nm=col_nm, norm_mill=norm_mill
        )

    def extract_yearly_data_norm(self, tag: str, col_nm: str) -> pd.DataFrame:
        """Extract yearly data from tag normalized to millions

        Args:
            tag (str): Tag of financial
            col_nm (str): _description_

        Returns:
            pd.DataFrame: _description_
        """

        return self.extract_yearly_data(tag=tag, col_nm=col_nm, norm_mill=True)
    
    @staticmethod
    def _concat_yearly_data(li_df_concat:List[pd.DataFrame]):
        return pd.concat(li_df_concat, ignore_index=True).drop_duplicates(ignore_index=True).sort_values("year", ascending=False, ignore_index=True)

    @staticmethod
    def _merge_yearly_data(li_df_merge:List[pd.DataFrame]):
        return reduce(
            lambda df_old, df_to_add: pd.merge(df_old, df_to_add,how="outer", on=["year"]),
            li_df_merge
        ).sort_values("year", ascending=False, ignore_index=True)

    def add_yearly_data_norm_to_col(
        self, tags: List[str], col_nm: str, df_nm: str, init_col: Optional[bool]=False,init_df: Optional[bool]=False,overwrite_col:Optional[bool]=False, overwrite_df:Optional[bool]=False
    ) -> None:
        """Add yearly data - normalized to millions
        insert data to the existing column as default.
        If init_column == True then one can also add data to non-existing column
        If init_df==True then one can add data to non-existing dataframe by first
            initializing df

        Args:
            tag (str): GAAP taxonomy tag of interest
            col_nm (str): Column name of the column needs 
                to be added by data
            df_nm (str): Name of the DF in the class dict
        """
        
        ## Exceptions
        
        # Raise error if the init_df option is chosen, dataframe already
        # exists but overwrite_df option is not chosen 
        if init_df and df_nm in self.dict_df.keys() and not overwrite_df:
            raise ValueError(
                f"""
                DF {df_nm} already exists.
                Initialization DF not possible!
                If overwrite is desired, please set the option:
                overwrite_df=True
                """
            )

        if df_nm in self.dict_df.keys():
            # Raise error if the init_col option is chosen, column already
            # exists but overwrite_col option is not chosen
            if init_col and col_nm in self.dict_df[df_nm].columns and not overwrite_col:
                raise ValueError(
                    f"""
                    DF {df_nm} does have a column with name {col_nm}.
                    Adding data not possible!
                    If overwrite is desired, please set the option:
                    overwrite_col=True
                    """
                )
            
            if not init_col and not col_nm in self.dict_df[df_nm].columns:
                raise ValueError(
                    f"""
                    DF {df_nm} does not have a column with name {col_nm}.
                    Adding data not possible!
                    Consider to first initialize the column.
                    """
                )
        
        # Fix function to extract data
        def func_fixed(tag)->List[pd.DataFrame]:
            return self.extract_yearly_data_norm(col_nm=col_nm, tag=tag)
        
        # Apply to the list of tags to obtain a list of data

        _dfs_temp = map(func_fixed, tags)

        # Add the data to the specified column

        _dfs_to_concat = _dfs_temp 

        if not init_df and not init_col:
            _dfs_to_concat=[self.dict_df[df_nm][col_nm],*_dfs_to_concat]
            self.dict_df[df_nm]=self.dict_df[df_nm].drop(columns=[col_nm])

        concat_df=self._concat_yearly_data(_dfs_to_concat)

        if init_df:
            # Initialize DF by years
            self.dict_df[df_nm]=pd.DataFrame(concat_df["year"])
            
        self.dict_df[df_nm] = self._merge_yearly_data([self.dict_df[df_nm],concat_df])

    def add_data_from_other_col(
        self,
        source_df: str,
        source_col: str,
        target_df: str,
        col_nm: Optional[str] = None,
    ):
        """Function to add data to a given df from other df

        Args:
            source_df (str): Source dataframe
            source_col (str): Source column in source_df
            target_df (str): df where data in source_col should be inserted
            col_nm (_type_, optional): Name of the resulting column. Defaults to None.
        """
        col_nm = source_col if not col_nm else col_nm

        _df_temp = self.dict_df[source_df][["year", source_col]].rename(
            columns={source_col: col_nm}
        )
        self.dict_df[target_df] = pd.merge(
            self.dict_df[target_df], _df_temp, on=["year"], how="outer"
        )

    def add_cols(self, li_col_nm: List[str], res_col_nm: str, df_nm: str, drop=True):
        """Sum multiple columns. Needs perhaps to be refactored

        Args:
            li_col_nm (List[str]): List of column names to be summed
            res_col_nm (str): Column name of the summation result
            df_nm (str): Name of the dataframe to be edited
            drop (bool, optional): Drop the old columns. Defaults to True.
        """
        self.dict_df[df_nm][res_col_nm] = self.dict_df[df_nm][li_col_nm].sum(axis=1)

        if drop:
            self.dict_df[df_nm] = self.dict_df[df_nm].drop(columns=li_col_nm)

    @staticmethod
    def _compute_row_change(col_input:pd.Series)->pd.Series:
        """Compute change of rows for a series
        """
        return (col_input - col_input.shift(-1))*100/col_input
    def compute_row_change_columns(
        self, col_nms: List[str], df_nm: str, res_col_nms:Optional[Dict[str,str]]=None
    ) -> pd.DataFrame:
        """Compute yearly change in percent of several columns
        Args:
            col_nm (str): Name of column to compute
            df_nm (str): Name of the corresponding dataframe
            res_col_nm (str, optional): Name of the corresponding result column. Defaults to "".

        Returns:
            pd.DataFrame: _description_
        """
        if not res_col_nms:
            res_col_nms = {col_nm:f"{col_nm}_change_in_perc" for col_nm in col_nms}
        
        for col_nm in col_nms:
            self.dict_df[df_nm][res_col_nms[col_nm]] = self._compute_row_change(self.dict_df[df_nm][col_nm])
    
    @staticmethod
    def _compute_ratio(numer_col:pd.Series,denom_col:pd.Series)->pd.Series:
        return (numer_col/denom_col)*100


    def compute_ratio_multiple(
        self, list_to_compute:List[Dict[str,str]], df_nm: str
    ) -> None:
        """
        Compute yearly

        Input: list_to_compute--> list of dicts with each contains res_col_nm,numer_col_nm, and denom_col_nm
        """

        for to_compute in list_to_compute:
            self.dict_df[df_nm][to_compute["res_col_nm"]]=self._compute_ratio(self.dict_df[df_nm][to_compute["numer_col_nm"]],self.dict_df[df_nm][to_compute["denom_col_nm"]])
    
    def generate_df(self, df_nm: str, how_gen: List[Dict[str,Union[List[str],List[int]]]],property:Optional[Dict]=None)->None:
        """
        Args:
            df_nm (str): Name of dataframe to be generated
            how_gen (dict): List of Dict with config how to specify how to generate the DF.
                posible key and values:
                    'init': col_nm , list tags
                    'add': col_nm , list tags
                    'drop': col_nm , list rows
        """
        for gen in how_gen:
            init_df=True if how_gen.index(gen)==0 else False
            init_col=True if how_gen.index(gen)>0 else False

            self.add_yearly_data_norm_to_col(
                    tags=gen["list_tags"], col_nm=gen["col_nm"], df_nm=df_nm,
                    init_df=init_df,init_col=init_col) 

            if "drop" in gen:
                self.dict_df[df_nm] = (
                    self.dict_df[df_nm].drop(gen["drop"]).reset_index(drop=True)
                )
        self.dict_df[df_nm] = self.dict_df[df_nm].dropna()

        if df_nm in df_config:
            property=df_config[df_nm]

        if not property:
            return
        
        # Create yearly change data
        if "yearly_change" in property:
            self.compute_row_change_columns(col_nms=property["yearly_change"],df_nm=df_nm)

        if "ratio" in property:
            self.compute_ratio_multiple(
                list_to_compute=property["ratio"], df_nm= df_nm)




    # Services for labels and descriptions

    def output_entry(self, position: str, key: str) -> str:
        """
        Output the JSON entry
        Usage e.g. output_entry('FeesAndCommissionsCreditCards','description')
        """
        return self.data["facts"]["us-gaap"][position][key]

    def find_in_dict_entry(self, key: str, query: str) -> list:
        """
        Usage e.g. find_in_dict_entry(key='description',query='loan')
        -> Find position for which the word loans occurs in the description
        """
        return [
            position
            for position in self.data["facts"]["us-gaap"].keys()
            if self.data["facts"]["us-gaap"][position][key]
            and self.data["facts"]["us-gaap"][position][key].lower().find(query) != -1
        ]


def generate_fin_analyze_class(cik: str, comp_nm: str, path: str) -> FinancialAnalyze:
    """
    Function to generate FinancialAnalyze Class of a company
    given the CIK
    """
    file_nm = "CIK" + cik
    with open(os.path.join(path, f"{file_nm}.json"), encoding="utf-8") as file:
        # Define service
        return FinancialAnalyze(comp_nm=comp_nm, data=json.load(file))


class InterCompaniesAnalyze:
    """
    Class for comparing financials of multiple companies
    """

    def __init__(
        self,
        li_comp_obj: List[FinancialAnalyze],
        dict_df: Optional[pd.DataFrame] = None,
    ) -> None:
        """
        Initialization
        :Inputs: List of FinancialAnalyze Objects
        """
        self.li_comp_obj = li_comp_obj
        self.dict_df = dict_df if dict_df else {}

    def merge_by_quant(self, df_name: str, quantity: str) -> None:
        """
        Create join data of certain quantity for
        different companies
        """

        def _extract_quantity(comp_obj: FinancialAnalyze):
            return comp_obj.dict_df[df_name][["year", quantity]].rename(
                columns={quantity: comp_obj.comp_nm}
            )

        self.dict_df[quantity] = reduce(
            lambda left, right: pd.merge(left, right, on=["year"], how="outer"),
            map(_extract_quantity, self.li_comp_obj),
        )

    def plot_df(
        self, key: str, xlabel: str, ylabel: str, width: float, height: float
    ) -> None:
        """
        Plot the comparison table of a certain quantity
        """
        df_analyze = self.dict_df[key].melt(
            id_vars=["year"], var_name="company", value_name=key
        )

        sns.set_theme(style="whitegrid")

        sns.set(rc={"figure.figsize": (width, height)})
        ax_plot = sns.barplot(data=df_analyze, x="year", y=key, hue="company")

        ax_plot.set(xlabel=xlabel, ylabel=ylabel)
