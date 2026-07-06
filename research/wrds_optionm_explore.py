"""Explore the OptionMetrics schema: table inventory, coverage, key columns."""
from wrds_lib import q


def main():
    t = q("""SELECT table_name FROM information_schema.tables
             WHERE table_schema='optionm' ORDER BY table_name""")
    names = list(t["table_name"])
    opprcd = [n for n in names if "opprcd" in n]
    vsurf = [n for n in names if "vsurf" in n]
    other = [n for n in names if not any(k in n for k in ("opprcd", "vsurf", "secprd"))]
    print(f"optionm has {len(names)} tables.")
    print(f"  option price (opprcd): {len(opprcd)} -> {opprcd[:3]} ... {opprcd[-3:]}")
    print(f"  vol surface (vsurf):   {len(vsurf)} -> {vsurf[:3]} ... {vsurf[-3:]}")
    print(f"  other: {other[:25]}")

    for tbl in ("vsurfd2020", "opprcd2020", "secprd2020"):
        try:
            cols = q("""SELECT column_name FROM information_schema.columns
                        WHERE table_schema='optionm' AND table_name=:t
                        ORDER BY ordinal_position""", t=tbl)
            print(f"\n{tbl} columns: {list(cols['column_name'])}")
        except Exception as e:  # noqa: BLE001
            print(f"\n{tbl}: {e}")

    # find SPX and SPY secids in the security name table
    for nm in ("secnmd", "securd", "secnm"):
        try:
            s = q("""SELECT secid, ticker, issuer, index_flag FROM optionm.""" + nm + """
                     WHERE ticker IN ('SPX','SPY','QQQ') ORDER BY ticker""")
            print(f"\nsec lookup via optionm.{nm}:")
            print(s.to_string(index=False))
            break
        except Exception as e:  # noqa: BLE001
            print(f"  (optionm.{nm} not it: {str(e)[:80]})")


if __name__ == "__main__":
    main()
