"""Modul for Configurations
"""

df_config={
    "ib_deposits":{
        "init":{
            "col_nm": "interest_bearing_deposit_liabilities"
        },
        "add":
            [
                {
                    "col_nm":"interest_bearing_deposit_liabilities",
                    "method":"yearly_change"
                },
                {
                    "col_nm":"interest_expense_deposits",
                    "method":"add_data"
                },
                {
                    "col_nm":"interest_rate_deposits",
                    "method":"compute_ratio",
                    "how":
                        {
                        "numer_col_nm":"interest_expense_deposits",
                        "denom_col_nm":"interest_bearing_deposit_liabilities"
                        }
                }

            ]
    },
    "loan_receivables":{
        "init":{
            "col_nm":"loan_receivables"
        },
        "add":
            [
                {
                    "col_nm":"loan_receivables",
                    "method":"yearly_change"
                },
                {
                    "col_nm":"interest_loans",
                    "method":"add_data"
                },
                {
                    "col_nm":"interest_rate_loan_receivables",
                    "method":"compute_ratio",
                    "how":
                        {
                        "numer_col_nm":"interest_loans",
                        "denom_col_nm":"loan_receivables"
                        }
                }

            ]
    }
}
