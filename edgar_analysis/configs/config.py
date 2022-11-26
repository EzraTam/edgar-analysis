"""Modul for Configurations
"""

df_config={
    "ib_deposits":{
        "yearly_change": ["interest_bearing_deposit_liabilities","interest_expense_deposits"],
        "ratio":[{"res_col_nm":"interest_rate_deposits","numer_col_nm":"interest_expense_deposits",
                        "denom_col_nm":"interest_bearing_deposit_liabilities"}]
    }
    ,
    "loan_receivables":{
        "yearly_change": ["loan_receivables","interest_loans"],
        "ratio":[{"res_col_nm":"interest_rate_loan_receivables","numer_col_nm":"interest_loans",
        "denom_col_nm":"loan_receivables"}]
    },
    "loan_deposit":{
        "ratio":[{"res_col_nm":"cover_rate_loan_deposit","numer_col_nm":"interest_bearing_deposit_liabilities",
        "denom_col_nm":"loan_receivables"}]
    }
}
