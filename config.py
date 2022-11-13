"""Modul for Configurations
"""

df_config={
    "ib_deposits":{
        "yearly_change": ["interest_bearing_deposit_liabilities","interest_expense_deposits"],
        "ratio":[{"res_col_nm":"interest_rate_deposits","numer_col_nm":"interest_expense_deposits",
                        "denom_col_nm":"interest_bearing_deposit_liabilities"}]
    }
}