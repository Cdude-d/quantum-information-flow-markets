"""Determine what we can ACTUALLY read (visibility != access). Probes the real
tables each candidate experiment needs, and the OptionMetrics-sample coverage."""
from wrds_lib import q


def readable(schema, table, cols="*"):
    try:
        r = q(f"SELECT {cols} FROM {schema}.{table} LIMIT 1")
        return True, list(r.columns)
    except Exception as e:  # noqa: BLE001
        return False, str(e).splitlines()[0][:60]


def main():
    print("=== READ-ACCESS PROBE (small LIMIT queries) ===\n")
    targets = [
        ("optionmsamp_us", "secnmd"), ("optionmsamp_us", "vsurfd2014"),
        ("crsp_a_stock", "dsf"), ("crsp_q_stock", "dsf"), ("crsp", "dsf"),
        ("crsp_a_stock", "msf"), ("crsp_a_indexes", "msix"),
        ("comp", "funda"), ("comp", "secd"), ("comp_na_daily_all", "secd"),
        ("ibes", "statsum_epsus"), ("ibes", "det_epsus"),
        ("ff", "factors_daily"), ("ff_all", "factors_daily"),
    ]
    ok = []
    for s, t in targets:
        good, info = readable(s, t)
        print(f"  [{'OK ' if good else 'no '}] {s}.{t}"
              + (f"  -> {info[:7]}" if good else f"  ({info})"))
        if good:
            ok.append((s, t))

    # OptionMetrics SAMPLE coverage — how usable is it?
    print("\n=== OptionMetrics SAMPLE (optionmsamp_us) coverage ===")
    try:
        vtabs = q("""SELECT table_name FROM information_schema.tables
                     WHERE table_schema='optionmsamp_us' AND table_name LIKE 'vsurfd%'
                     ORDER BY table_name""")
        print("  vol-surface years:", [t[-4:] for t in vtabs["table_name"]])
    except Exception as e:  # noqa: BLE001
        print("  vsurf list err:", e)
    try:
        nm = q("""SELECT secid, ticker, issuer FROM optionmsamp_us.secnmd
                  ORDER BY ticker LIMIT 40""")
        print(f"  securities in sample: {len(nm)} shown")
        print(nm.to_string(index=False))
    except Exception as e:  # noqa: BLE001
        print("  secnmd err:", str(e).splitlines()[0])


if __name__ == "__main__":
    main()
