from __future__ import annotations
from typing import List, Dict, Union,Optional
import numpy as np

import pandas as pd

def norm_and_rename(method):
    def inner(self):
        df_output=method(self)
        if self.norm_mill:
            df_output["val"] = df_output["val"] / 1000000
        if self.col_nm:
            df_output = df_output.rename(columns={"val": df_output.col_nm})  
        return df_output
    return inner


class EdgarData:
    
    def __init__(self,data:List[Dict[str,Union[str,int,float]]],unit:Optional[str]=None,company_name:Optional[str]=None,cik:Optional[str]=None,col_nm:Optional[str]=None, norm_mill: Optional[bool] = None)->None:
        
        self.data=data
        self.col_nm=col_nm
        self.norm_mill=norm_mill
        self.unit=unit
        self.company_name=company_name
        self.cik=cik


    @classmethod
    def from_tag(cls,data_comp: Dict,tag:str,col_nm:Optional[str]=None, norm_mill: Optional[bool] = None)->EdgarData:
        
        assert len(data_comp["facts"]["us-gaap"][tag]["units"])<= 1, f"There are more than 1 data with different units! Units are: {list(data_comp['facts']['us-gaap'][tag]['units'].keys())}"

        assert len(data_comp["facts"]["us-gaap"][tag]["units"])> 0, "No data is given or data is directly given in unit. Please check" 

        unit = list(data_comp["facts"]["us-gaap"][tag]["units"].keys())[0]
        company_name=data_comp["entityName"]
        cik=data_comp["cik"]

        return EdgarData(data=data_comp["facts"]["us-gaap"][tag]["units"][unit],unit=unit,col_nm=col_nm,norm_mill=norm_mill,company_name=company_name,cik=cik)

    @property
    def raw(self)->pd.DataFrame:
        """Raw DF created from the data input into the class
        """
        _df=pd.DataFrame(self.data)
        _df["end"] = pd.to_datetime(_df["end"], format="%Y-%m-%d")
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

        return df_quarter.sort_values(by=["year","fp"],ascending=[False,False]).reset_index(drop=True)
