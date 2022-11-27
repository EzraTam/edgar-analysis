from __future__ import annotations
from typing import List, Dict, Union,Optional
import os
import json
import numpy as np

import pandas as pd

def norm_and_rename(method):
    """ Decorator for EdgarData method
    This decorator normalize the values to million,
    and change the corresponding column name
    """
    def inner(self):
        df_output=method(self)
        if self.norm_mill:
            df_output["val"] = df_output["val"] / 1000000
        if self.res_col_nm:
            df_output = df_output.rename(columns={"val": df_output.res_col_nm})  
        return df_output
    return inner


class EdgarData:
    """Class for processing raw financial data from a company
    """
    
    def __init__(self,data:List[Dict[str,Union[str,int,float]]],unit:Optional[str]=None,company_name:Optional[str]=None,cik:Optional[str]=None,res_col_nm:Optional[str]=None, norm_mill: Optional[bool] = None)->None:
        
        self.data=data

        # Input used the by the decorator norm_and_rename
        self.res_col_nm=res_col_nm
        self.norm_mill=norm_mill
        self.unit=unit
        self.company_name=company_name
        self.cik=cik

    @classmethod
    def from_tag(cls,data_comp:Dict,tag:str,which:str,res_col_nm:Optional[str]=None, norm_mill: Optional[bool] = None)->EdgarData:
        """Create Financial DF of a tag

        Args:
            data_comp (Dict): _description_
            tag (str): _description_
            res_col_nm (Optional[str], optional): _description_. Defaults to None.
            norm_mill (Optional[bool], optional): _description_. Defaults to None.

        Returns:
            EdgarData: _description_
        """
        list_which=data_comp["facts"].keys()

        assert which in list_which, f"Please choose which only from the following list: {list(list_which)}"

        data_to_extract=data_comp["facts"][which]
        
        assert len(data_to_extract[tag]["units"])<= 1, f"There are more than 1 data with different units! Units are: {list(data_to_extract[tag]['units'].keys())}"

        assert len(data_to_extract[tag]["units"])> 0, "No data is given or data is directly given in unit. Please check" 

        unit = list(data_to_extract[tag]["units"].keys())[0]
        company_name=data_comp["entityName"]
        cik=data_comp["cik"]

        return EdgarData(data=data_to_extract[tag]["units"][unit],unit=unit,res_col_nm=res_col_nm,norm_mill=norm_mill,company_name=company_name,cik=cik)

    @property
    def raw(self)->pd.DataFrame:
        """Raw DF created from the data input into the class
        """
        _df=pd.DataFrame(self.data)
        _df["end"] = pd.to_datetime(_df["end"], format="%Y-%m-%d")
        if "start" in _df.columns: 
            _df["start"] = pd.to_datetime(_df["start"], format="%Y-%m-%d")
        _df["year"] = pd.DatetimeIndex(_df["end"]).year
        _df['diff_months'] = (_df['end'] - _df['start']) / np.timedelta64(1, 'M')
        _df['diff_months'] = _df['diff_months'].round(0).astype(int)
        return _df
    
    @property
    @norm_and_rename
    def yearly(self)->pd.DataFrame:
        """Yearly Data in DF-Format
        """
        df_res = self.raw[self.raw["fp"] == "FY"]

        return (
            df_res.sort_values(["year", "fy"], ascending=False)
            .groupby("year")
            .head(1)[["year", "val"]]
            .reset_index(drop=True)
        )
    
    @property
    @norm_and_rename
    def quarterly(self)->pd.DataFrame:
        """Quarterly Data in DF-Fromat
        """
        
        df_quarter=self.raw[self.raw["diff_months"]==3]
        df_quarter=df_quarter.sort_values(by=["end","start","filed"])
        df_quarter=df_quarter.groupby(["end","start"]).head(1)
        df_quarter=df_quarter.sort_values(by="end",ascending=False)
        df_quarter=df_quarter[["year","fp","val"]]

        df_temp=df_quarter.groupby("year")["val"].agg(['sum', 'count']).reset_index()
        year_not_comp=df_temp[df_temp["count"]==3]
        df_temp=pd.merge(self.yearly,year_not_comp,on="year",how="inner")
        df_temp["Q4"]=df_temp["val"]-df_temp["sum"]
        df_temp=df_temp.drop(columns=["sum","count","val"]).rename(columns={"Q4":"val"})
        df_temp["fp"]="Q4"

        df_quarter=pd.concat([df_quarter,df_temp])
        df_quarter.loc[df_quarter["fp"]=="FY","fp"]="Q4"
        df_quarter=df_quarter.sort_values(by=["year","fp"],ascending=[False,False]).reset_index(drop=True)

        return df_quarter.sort_values(by=["year","fp"],ascending=[False,False]).reset_index(drop=True).rename(columns={"fp":"quarter"})
    
    # TODO: Plot services

class EdgarDataComp:
    """Class for Company financial
    """
    def __init__(self,interval:str,data_comp:Optional[Dict]=None,path_data:Optional[str]=None,path_folder:Optional[str]=None,cik:Optional[str]=None,norm_mill: Optional[bool] = True)->None:
        
        
        self.norm_mill=norm_mill
        self.interval=interval
        
        if data_comp:
            self.data_comp=data_comp
            return

        if not path_data and (path_folder and cik):
            path_data= os.path.join(path_folder,"CIK" + cik+".json")
        
        with open(os.path.join(path_data), encoding="utf-8") as file:
            self.data_comp=json.load(file)



    def by_tag(self,tag:str,which:str,res_col_nm:Optional[str]=None)->pd.DataFrame:
        """Extract financial position given by a tag

        Args:
            tag (str): Tag to extract
            interval (str): which data interval (yearly/quarterly?)
            res_col_nm (str): name of the resulting column

        Returns:
            pd.DataFrame: _description_
        """
        tag_data=EdgarData.from_tag(
            data_comp=self.data_comp,
            which=which,
            tag=tag,
            res_col_nm=res_col_nm,
            norm_mill=self.norm_mill
            )
        
        return getattr(tag_data,self.interval)

    # Document and Entity Information (DEI)

    # TODO: Check Whether DEI has to be treated separately as there is no start!

    @property
    def shares_outstanding(self)->pd.DataFrame:
        """Number of shares outstanding
        """
        _df_shares=pd.DataFrame(self.data_comp["facts"]["dei"]["EntityCommonStockSharesOutstanding"]["units"]["shares"])[["fy","fp","val"]]
        _df_shares=_df_shares.rename(columns={"fy":"year","fp":"quarter"})
        if self.interval == "yearly":
            _df_shares=_df_shares[_df_shares["quarter"]=="FY"]
            return _df_shares[["year","val"]]

        _df_shares.loc[_df_shares["quarter"]=="FY", ['quarter']] = "Q4"
        if self.norm_mill:
            _df_shares["val"]=_df_shares["val"]/1000000
        return _df_shares.sort_values(["year", "quarter"], ascending=[False,False],ignore_index=True)   

    @property
    def market_capitalization(self)->pd.DataFrame:
        """ Market Capitalization
        """   
        _df_mc=pd.DataFrame(self.data_comp["facts"]["dei"]['EntityPublicFloat']["units"]["USD"])[["fy","val"]].rename(columns={"fy":"year"}) 
        if self.norm_mill:
            _df_mc["val"]=_df_mc["val"]/1_000_000
        return _df_mc.sort_values("year",ascending=False,ignore_index=True)

    @property
    def avg_share_price(self)->pd.DataFrame:
        """Average share price
        """
        interval_old=self.interval
        self.interval="yearly"
        _df_res=pd.merge(self.shares_outstanding.rename(columns={"val":"so"}),self.market_capitalization.rename(columns={"val":"mc"}),on="year")
        _df_res["val"]=_df_res["mc"]/_df_res["so"]

        self.interval=interval_old
        return _df_res.drop(columns=["mc","so"])
