"""Check service databases for narrator data."""
import sys, sqlite3
sys.stdout.reconfigure(encoding="utf-8")

BASE = r"C:\Users\titan\AppData\Local\Temp\shamela_test\database\service"

for dbname in ["trajim", "hadeeth", "S1", "S2", "tafseer"]:
    path = f"{BASE}\\{dbname}.db"
    try:
        conn = sqlite3.connect(path)
        cur = conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [t[0] for t in cur.fetchall()]
        print(f"\n{'='*50}")
        print(f"{dbname}.db: {len(tables)} tables")
        for tname in tables[:5]:
            cur.execute(f"SELECT COUNT(*) FROM [{tname}]")
            nr = cur.fetchone()[0]
            print(f"  [{tname}]: {nr:,} rows")
            if nr > 0:
                cur.execute(f"SELECT * FROM [{tname}] LIMIT 1")
                cols = [d[0] for d in cur.description]
                print(f"    cols: {cols[:15]}")
                row = cur.fetchone()
                if row:
                    for c, v in zip(cols[:8], row[:8]):
                        if v:
                            vstr = str(v)[:120]
                            print(f"    {c}: {vstr}")
        conn.close()
    except Exception as e:
        print(f"\n{dbname}.db: ERROR: {e}")
