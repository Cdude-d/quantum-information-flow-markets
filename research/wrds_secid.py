"""Find OptionMetrics secid for key tickers (and the secnmd column layout)."""
from wrds_lib import q

cols = q("""SELECT column_name FROM information_schema.columns
            WHERE table_schema='optionm' AND table_name='secnmd'
            ORDER BY ordinal_position""")
print("secnmd columns:", list(cols["column_name"]))

s = q("""SELECT * FROM optionm.secnmd
         WHERE ticker IN ('SPX','SPY','QQQ','NDX','RUT','AAPL','TSLA')
         ORDER BY ticker""")
print(s.to_string(index=False))
