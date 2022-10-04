"""
Module for analyzing financial data
"""

import os

import json

from typing import List

from functools import reduce

import seaborn as sns

import pandas as pd


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
    data: dict, tag: str, col_nm: str, norm_mill: bool
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


def extract_yearly_data_norm(data: dict, tag: str, col_nm: str) -> pd.DataFrame:
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
    """Class for analysing financial data from Edgar dict
    """

    def __init__(self, comp_nm: str, data: dict) -> None:
        """Initialization

        Args:
            comp_nm (str): Name of the Company
            data (dict): Data in Edgar Form
        """
        self.comp_nm = comp_nm
        self.data = data
        self.dict_df = {}

    def init_yearly_data_norm(self, tag: str, col_nm: str, df_nm: str) -> None:
        """Initialize Table from dict with a given tag

        Args:
            tag (str): GAAP taxonomy tag of interest
            col_nm (str): Column name of the added data
            df_nm (str): Name of the DF in the class dict
        """
        self.dict_df[df_nm] = extract_yearly_data(
            self.data, tag, col_nm, norm_mill=True
        )

    def add_yearly_data_norm(
        self, tag: str, col_nm: str, df_nm: str, add_row=True
    ) -> None:
        """ Add yearly data - normalized to millions
        By default insert data to existing row.
        Set add_row to False in order to add new column

        Args:
            tag (str): GAAP taxonomy tag of interest
            col_nm (str): Column name of the added data
            df_nm (str): Name of the DF in the class dict
            add_row (bool, optional): _description_. Defaults to True.
        """
        if add_row:
            _df_temp = extract_yearly_data_norm(self.data, tag=tag, col_nm=col_nm)

            self.dict_df[df_nm] = (
                pd.concat([self.dict_df[df_nm], _df_temp], ignore_index=True)
                .drop_duplicates(ignore_index=True)
                .sort_values("year", ascending=False, ignore_index=True)
            )

        else:
            _df_temp = extract_yearly_data_norm(self.data, tag=tag, col_nm=col_nm)

            self.dict_df[df_nm] = pd.merge(
                self.dict_df[df_nm], _df_temp, on=["year"], how="outer"
            )

    def add_data_from_other_col(
        self, source_df: str, source_col: str, target_df: str, col_nm=None
    ):
        """Function to add data to a given df from other df

        Args:
            source_df (str): Source dataframe
            source_col (str): Source column in source_df
            target_df (str): df where data in source_col should be inserted
            col_nm (_type_, optional): Name of the resulting column. Defaults to None.
        """
        if col_nm is None:
            col_nm = source_col

        _df_temp = self.dict_df[source_df][["year", source_col]]
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

    def compute_row_change(
        self, col_nm: str, df_nm: str, res_col_nm=""
    ) -> pd.DataFrame:
        """Compute yearly change in percent

        Args:
            col_nm (str): Name of column to compute
            df_nm (str): Name of the corresponding dataframe
            res_col_nm (str, optional): Name of the corresponding result column. Defaults to "".

        Returns:
            pd.DataFrame: _description_
        """
        if not res_col_nm:
            res_col_nm = f"{col_nm}_change_in_perc"

        self.dict_df[df_nm][res_col_nm] = (
            (self.dict_df[df_nm][col_nm] - self.dict_df[df_nm][col_nm].shift(-1))
            / self.dict_df[df_nm][col_nm]
        ) * 100

    def compute_ratio(
        self, numer_col_nm: str, denom_col_nm: str, res_col_nm: str, df_nm: str
    ) -> None:
        """
        Compute yearly
        """
        self.dict_df[df_nm][res_col_nm] = (
            self.dict_df[df_nm][numer_col_nm] / self.dict_df[df_nm][denom_col_nm]
        ) * 100

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
        li_positions = []

        for position in self.data["facts"]["us-gaap"].keys():
            if self.data["facts"]["us-gaap"][position][key]:
                if (
                    self.data["facts"]["us-gaap"][position][key].lower().find(query)
                    != -1
                ):
                    li_positions.append(position)

        return li_positions


def generate_fin_analyze_class(cik: str, comp_nm: str, path: str) -> FinancialAnalyze:
    """
    Function to generate FinancialAnalyze Class of a company
    given the CIK
    """
    file_nm = "CIK" + cik
    with open(os.path.join(path,f"{file_nm}.json")) as file:
        # Define service
        return FinancialAnalyze(comp_nm=comp_nm, data=json.load(file))


class InterCompaniesAnalyze:
    """
    Class for comparing financials of multiple companies
    """

    def __init__(self, li_comp_obj: List[FinancialAnalyze]) -> None:
        """
        Initialization
        :Inputs: List of FinancialAnalyze Objects
        """
        self.li_comp_obj = li_comp_obj
        self.dict_df = {}

    def merge_by_quant(self, df_name: str, quantity: str) -> None:
        """
        Create join data of certain quantity for
        different companies
        """

        li_df = []

        for comp_serv in self.li_comp_obj:
            comp_nm = comp_serv.comp_nm

            if df_name in comp_serv.dict_df.keys():
                df_comp = comp_serv.dict_df[df_name]
                li_col_to_drop = [
                    col
                    for col in list(df_comp.columns)
                    if col not in ["year", quantity]
                ]
                df_comp = df_comp.drop(columns=li_col_to_drop).rename(
                    columns={quantity: comp_nm}
                )
                li_df.append(df_comp)

        self.dict_df[quantity] = reduce(
            lambda left, right: pd.merge(left, right, on=["year"], how="outer"), li_df
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
        ax = sns.barplot(data=df_analyze, x="year", y=key, hue="company")

        ax.set(xlabel=xlabel, ylabel=ylabel)
