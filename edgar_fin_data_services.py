"""
Module for analyzing financial data
"""


from typing import List

from functools import reduce

import seaborn as sns

import pandas as pd


def transform_to_df(data: dict, tag: str) -> pd.DataFrame:
    """
    Transform financial data to pandas
    """
    return pd.DataFrame(data["facts"]["us-gaap"][tag]["units"]["USD"])


def extract_yearly_data(
    data: dict, tag: str, col_nm: str, norm_mill: bool
) -> pd.DataFrame:
    """
    Extract yearly data for a given tag
    """

    df = transform_to_df(data, tag)
    df = df[df["fp"] == "FY"]
    df["end"] = pd.to_datetime(df["end"], format="%Y-%m-%d")
    df["year"] = pd.DatetimeIndex(df["end"]).year
    df = (
        df.sort_values(["year", "fy"], ascending=False)
        .groupby("year")
        .head(1)[["year", "val"]]
        .reset_index(drop=True)
    )

    if norm_mill:
        df["val"] = df["val"] / 1000000

    if col_nm:
        df = df.rename(columns={"val": col_nm})

    return df


def extract_yearly_data_norm(data: dict, tag: str, col_nm: str) -> pd.DataFrame:
    """
    Extract yearly data from tag normalized to millions
    """

    return extract_yearly_data(data, tag, col_nm, norm_mill=True)


class FinancialAnalyze:
    """
    Class for analysing financial data from
    Edgar dict
    """

    def __init__(self, comp_nm: str, data: dict) -> None:
        self.comp_nm = comp_nm
        self.data = data
        self.dict_df = {}
        self.df = None

    def init_yearly_data_norm(self, tag: str, col_nm: str, df_nm: str) -> None:
        """
        Initialize Table from dict with a given tag
        """
        self.dict_df[df_nm] = extract_yearly_data(
            self.data, tag, col_nm, norm_mill=True
        )

    def add_yearly_data_norm(
        self, tag: str, col_nm: str, df_nm: str, add_row=True
    ) -> None:
        """
        Add yearly data - normalized to millions
        By default insert data to existing row
        Set add_row to False in order to add new column
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
        if col_nm is None:
            col_nm = source_col

        _df_temp = self.dict_df[source_df][["year", source_col]]
        self.dict_df[target_df] = pd.merge(
            self.dict_df[target_df], _df_temp, on=["year"], how="outer"
        )

    def add_cols(self, li_col_nm: List[str], res_col_nm: str, df_nm: str, drop=True):
        """
        Add multiple columns. Needs perhaps to be refactored
        """
        self.dict_df[df_nm][res_col_nm] = self.dict_df[df_nm][li_col_nm].sum(axis=1)

        if drop:
            self.dict_df[df_nm] = self.dict_df[df_nm].drop(columns=li_col_nm)

    def compute_row_change(
        self, col_nm: str, df_nm: str, res_col_nm=""
    ) -> pd.DataFrame:
        """
        Compute yearly change in percent
        """
        if not res_col_nm:
            res_col_nm = col_nm + "_change_in_perc"

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

        for CompService in self.li_comp_obj:
            comp_nm = CompService.comp_nm

            if df_name in CompService.dict_df.keys():
                df = CompService.dict_df[df_name]
                li_col_to_drop = [
                    col for col in list(df.columns) if col not in ["year", quantity]
                ]
                df = df.drop(columns=li_col_to_drop).rename(columns={quantity: comp_nm})
                li_df.append(df)

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
