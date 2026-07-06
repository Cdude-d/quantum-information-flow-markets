"""Discover what WRDS products this account can actually access. Run after adding
WRDS_USERNAME / WRDS_PASSWORD to .env.

  python research/wrds_explore.py
"""
from wrds_lib import q

# Libraries that matter for the strategies we've been testing.
WANT = {
    "crsp": "stock returns, survivorship-free, with delisting returns",
    "optionm": "OptionMetrics — historical option prices/IV/greeks (1996+)",
    "comp": "Compustat — point-in-time fundamentals",
    "ibes": "analyst estimates — for earnings drift (PEAD)",
    "ff": "Fama-French factors",
    "taqmsec": "intraday trades & quotes",
}


def main():
    print("Connecting to WRDS...", flush=True)
    who = q("SELECT current_user AS u, current_date AS d")
    print(f"Connected as {who['u'][0]} on {who['d'][0]}\n", flush=True)

    libs = q("""SELECT DISTINCT table_schema FROM information_schema.tables
                ORDER BY table_schema""")
    have = set(libs["table_schema"])
    print(f"{len(have)} schemas visible. Key products for our work:")
    for lib, desc in WANT.items():
        mark = "YES" if lib in have else " no"
        print(f"  [{mark}] {lib:10} — {desc}")

    # peek at CRSP daily stock file dimensions if present
    if "crsp" in have:
        print("\nCRSP sample (daily stock file dsf):")
        try:
            rng = q("SELECT MIN(date) lo, MAX(date) hi, COUNT(*) n FROM crsp.dsf "
                    "WHERE date > '2020-01-01'")
            print(f"   dsf since 2020: {rng['lo'][0]}..{rng['hi'][0]}, {int(rng['n'][0]):,} rows")
        except Exception as e:  # noqa: BLE001
            print(f"   (couldn't sample crsp.dsf: {e})")
    if "optionm" in have:
        print("\nOptionMetrics present — option-level history available.")


if __name__ == "__main__":
    main()
