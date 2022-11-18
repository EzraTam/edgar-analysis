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


def transform_to_df(data: dict, tag: str, unit:Optional[str]="USD") -> pd.DataFrame:
    """Transform financial data to pandas

    Args:
        data (dict): Data in Edgar dict form
        tag (str): Tag of the data to be transformed

    Returns:
        pd.DataFrame: Result Dataframe
    """
    return pd.DataFrame(data["facts"]["us-gaap"][tag]["units"][unit])


def extract_yearly_data(
    data: dict, tag: str, col_nm: str, norm_mill: Optional[bool] = None, unit:Optional[str]="USD"
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
    df_res = transform_to_df(data, tag,unit)
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
        self,
        comp_nm: str,
        data: dict,
        dict_df: Optional[Dict[str, pd.DataFrame]] = None,
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
        """Given a tag in Edgar data,
        show the corresponding data in DF
        """
        return transform_to_df(data=self.data, tag=tag)

    def extract_yearly_data(
        self, tag: str, col_nm: str, norm_mill: Optional[bool] = None
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
    def _concat_yearly_data(li_df_concat: List[pd.DataFrame]):
        return (
            pd.concat(li_df_concat, ignore_index=True)
            .drop_duplicates(ignore_index=True)
            .sort_values("year", ascending=False, ignore_index=True)
        )

    @staticmethod
    def _merge_yearly_data(li_df_merge: List[pd.DataFrame]):
        return reduce(
            lambda df_old, df_to_add: pd.merge(
                df_old, df_to_add, how="outer", on=["year"]
            ),
            li_df_merge,
        ).sort_values("year", ascending=False, ignore_index=True)

    def add_yearly_data_norm_to_col(
        self,
        tags: List[str],
        col_nm: str,
        df_nm: str,
        init_col: Optional[bool] = False,
        init_df: Optional[bool] = False,
        overwrite_col: Optional[bool] = False,
        overwrite_df: Optional[bool] = False,
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
        if init_df and df_nm in self.dict_df and not overwrite_df:
            raise ValueError(
                f"""
                DF {df_nm} already exists.
                Initialization DF not possible!
                If overwrite is desired, please set the option:
                overwrite_df=True
                """
            )

        if df_nm in self.dict_df:
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
        def func_fixed(tag) -> List[pd.DataFrame]:
            return self.extract_yearly_data_norm(col_nm=col_nm, tag=tag)

        # Apply to the list of tags to obtain a list of data

        _dfs_temp = map(func_fixed, tags)

        # Add the data to the specified column

        _dfs_to_concat = _dfs_temp

        if not init_df and not init_col:
            _dfs_to_concat = [self.dict_df[df_nm][col_nm], *_dfs_to_concat]
            self.dict_df[df_nm] = self.dict_df[df_nm].drop(columns=[col_nm])

        concat_df = self._concat_yearly_data(_dfs_to_concat)

        if init_df:
            # Initialize DF by years
            self.dict_df[df_nm] = pd.DataFrame(concat_df["year"])

        self.dict_df[df_nm] = self._merge_yearly_data([self.dict_df[df_nm], concat_df])

    def add_data_from_other_col(
        self,
        source_df: str,
        source_col: str,
        target_df: str,
        col_nm: Optional[str] = None,
        init : Optional[str]= False
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
        self.dict_df[target_df] = _df_temp if init else pd.merge(
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
    def _compute_row_change(col_input: pd.Series) -> pd.Series:
        """Compute change of rows for a series"""
        return (col_input - col_input.shift(-1)) * 100 / col_input

    def compute_row_change_columns(
        self,
        col_nms: List[str],
        df_nm: str,
        res_col_nms: Optional[Dict[str, str]] = None,
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
            res_col_nms = {col_nm: f"{col_nm}_change_in_perc" for col_nm in col_nms}

        for col_nm in col_nms:
            self.dict_df[df_nm][res_col_nms[col_nm]] = self._compute_row_change(
                self.dict_df[df_nm][col_nm]
            )

    @staticmethod
    def _compute_ratio(numer_col: pd.Series, denom_col: pd.Series) -> pd.Series:
        return (numer_col / denom_col) * 100

    def compute_ratio_multiple(
        self, list_to_compute: List[Dict[str, str]], df_nm: str
    ) -> None:
        """
        Compute yearly

        Input: list_to_compute--> list of dicts with each contains res_col_nm,numer_col_nm, and denom_col_nm
        """

        for to_compute in list_to_compute:
            self.dict_df[df_nm][to_compute["res_col_nm"]] = self._compute_ratio(
                self.dict_df[df_nm][to_compute["numer_col_nm"]],
                self.dict_df[df_nm][to_compute["denom_col_nm"]]
            )

    def generate_df(
        self,
        df_nm: str,
        how_gen: List[Dict[str, Union[List[str], List[int]]]],
        add_gen: Optional[Dict] = None,
    ) -> None:
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

            if "list_tags" in gen: 
                # Initialize DF in the beginning of the workflow
                init_df = True if how_gen.index(gen) == 0 else False

                # Initialize DF columns
                init_col = True if how_gen.index(gen) > 0 else False

                self.add_yearly_data_norm_to_col(
                    tags=gen["list_tags"],
                    col_nm=gen["col_nm"],
                    df_nm=df_nm,
                    init_df=init_df,
                    init_col=init_col,
                )

            if "add_tags" in gen:
                _li_df=[self.extract_yearly_data_norm(tag=tag, col_nm=tag) for tag in gen["add_tags"]]
                _df_temp=self._merge_yearly_data(li_df_merge=_li_df)
                _df_temp[gen["col_nm"]]=_df_temp.sum(axis=1)
                _df_temp=_df_temp.drop(columns=gen["add_tags"])
                self.dict_df[df_nm]=self._merge_yearly_data(li_df_merge=[self.dict_df[df_nm],_df_temp])

            
            if "drop" in gen:
                self.dict_df[df_nm] = (
                    self.dict_df[df_nm].drop(gen["drop"]).reset_index(drop=True)
                )

            if "add_col_from_df" in gen:
                init= True if how_gen.index(gen)==0 else False
                self.add_data_from_other_col(source_df=gen["add_col_from_df"],source_col=gen["col_nm"],target_df=df_nm,init=init)


        self.dict_df[df_nm] = self.dict_df[df_nm].dropna()

        # Additional generation of columns

        if df_nm in df_config:
            add_gen = df_config[df_nm]

        if not add_gen:
            return

        # Create yearly change data
        if "yearly_change" in add_gen:
            self.compute_row_change_columns(
                col_nms=add_gen["yearly_change"], df_nm=df_nm
            )

        if "ratio" in add_gen:
            self.compute_ratio_multiple(list_to_compute=add_gen["ratio"], df_nm=df_nm)

    @staticmethod
    def _pick_year_data(df: pd.DataFrame, col_nm: str, year: int):
        return df[df["year"] == year].reset_index(drop=True)[col_nm][0]

    # CAGR
    @staticmethod
    def _compute_cagr(begin_value: float, end_value: float, num_years: int) -> float:
        return (pow(end_value / begin_value, 1 / num_years) - 1) * 100

    def compute_yearly_cagr(
        self,
        df_nm: str,
        col_nm: str,
        interval: Dict[str, int],
        round_num: Optional[int] = None,
    ):
        _df_temp = self.dict_df[df_nm]
        begin_value = self._pick_year_data(
            _df_temp, col_nm=col_nm, year=interval["begin"]
        )
        end_value = self._pick_year_data(_df_temp, col_nm=col_nm, year=interval["end"])
        num_years = interval["end"] - interval["begin"]
        result = self._compute_cagr(
            begin_value=begin_value, end_value=end_value, num_years=num_years
        )

        return result if not round_num else round(result, 2)

    def compute_stats(
        self,
        df_nm: str,
        col_nm: str,
        interval: Dict[str, int],
        round_num: Optional[int] = None,
        show_pd: Optional[bool] = False,
    ):
        stat_to_extract = ["mean", "median", "max", "min"]

        _df_temp = self.dict_df[df_nm]
        _ser_temp = _df_temp[
            _df_temp["year"].isin(range(interval["begin"], interval["end"] + 1))
        ][col_nm]

        stat = {
            stat_method: getattr(pd.Series, stat_method)(_ser_temp)
            if not round_num
            else round(getattr(pd.Series, stat_method)(_ser_temp), round_num)
            for stat_method in stat_to_extract
        }

        return stat if not show_pd else pd.DataFrame(data=stat, index=[0])

    # Other services
    def delete_df(self,df_nm:str)->None:
        del self.dict_df[df_nm]

    # Services for labels and descriptions

    def output_entry(self, position: str, key: str) -> str:
        """
        Output the JSON entry
        Usage e.g. output_entry('FeesAndCommissionsCreditCards','description')
        """
        return self.data["facts"]["us-gaap"][position][key]

    def show_description(self,tag:str)->str:
        return self.output_entry(position=tag, key= "description")


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

        li_comp_obj_compare=[comp_obj for comp_obj in self.li_comp_obj if df_name in comp_obj.dict_df]

        def _extract_quantity(comp_obj: FinancialAnalyze):
            return comp_obj.dict_df[df_name][["year", quantity]].rename(
                columns={quantity: comp_obj.comp_nm}
            )

        self.dict_df[quantity] = reduce(
            lambda left, right: pd.merge(left, right, on=["year"], how="outer"),
            map(_extract_quantity, li_comp_obj_compare),
        )

    def compute_cagr(
        self,
        df_nm: str,
        col_nm: str,
        interval: Dict[str, int],
        round_num: Optional[int] = None,
    ):

        result_cagr = [
            (
                comp_obj.comp_nm,
                comp_obj.compute_yearly_cagr(
                    df_nm=df_nm, col_nm=col_nm, interval=interval, round_num=round_num
                ),
            )
            for comp_obj in self.li_comp_obj
        ]

        return pd.DataFrame(result_cagr, columns=["company", "cagr"])

    def compute_stats(
        self,
        df_nm: str,
        col_nm: str,
        interval: Dict[str, int],
        round_num: Optional[int] = None,
    ) -> pd.DataFrame:
        """Compute statistics of certain column

        Args:
            df_nm (str): Name of the DF
            col_nm (str): Column for which the stats should be computed
            interval (Dict[str,int]): Year interval
            round_num (Optional[int], optional): Round to hom many digit?. Defaults to None.

        Returns:
            pd.DataFrame: _description_
        """
        dfs = [
            comp_obj.compute_stats(
                df_nm=df_nm,
                col_nm=col_nm,
                interval=interval,
                round_num=round_num,
                show_pd=True,
            )
            for comp_obj in self.li_comp_obj
        ]
        columns = dfs[0].columns
        df = pd.concat(dfs)
        df["company"] = [comp_obj.comp_nm for comp_obj in self.li_comp_obj]
        return df.reset_index(drop=True)[["company", *columns]]

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
