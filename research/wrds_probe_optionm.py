"""Find which OptionMetrics schema this account can actually SELECT rows from
(the optionm views route to optionm_all, which may be gated)."""
from wrds_lib import q

# all schemas that look option-related
sch = q("""SELECT schema_name FROM information_schema.schemata
           WHERE schema_name ILIKE '%option%' ORDER BY schema_name""")
cands = list(sch["schema_name"])
print("option-related schemas:", cands)

# for each, find a vsurfd/secnmd table and try to read 1 row
for s in cands:
    tabs = q("""SELECT table_name FROM information_schema.tables
                WHERE table_schema=:s AND (table_name LIKE 'vsurfd%' OR table_name='secnmd'
                      OR table_name LIKE 'opprcd%') ORDER BY table_name LIMIT 3""", s=s)
    tnames = list(tabs["table_name"])
    if not tnames:
        print(f"  {s}: (no opt tables)")
        continue
    probe = tnames[0]
    try:
        r = q(f"SELECT * FROM {s}.{probe} LIMIT 1")
        print(f"  {s}.{probe}: READABLE  cols={list(r.columns)[:6]}...")
    except Exception as e:  # noqa: BLE001
        print(f"  {s}.{probe}: denied ({str(e).splitlines()[0][:70]})")


if __name__ == "__main__":
    pass
