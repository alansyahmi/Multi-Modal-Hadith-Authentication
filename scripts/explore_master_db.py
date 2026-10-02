"""Explore Shamela master.db for book/page metadata."""
import sys, sqlite3
sys.stdout.reconfigure(encoding="utf-8")

conn = sqlite3.connect(r"C:\Users\titan\AppData\Local\Temp\shamela_test\database\master.db")
cur = conn.cursor()

cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = [t[0] for t in cur.fetchall()]
print(f"Tables ({len(tables)}):")
for name in tables:
    try:
        cur.execute(f"SELECT COUNT(*) FROM [{name}]")
        nr = cur.fetchone()[0]
        cur.execute(f"SELECT * FROM [{name}] LIMIT 1")
        cols = [d[0] for d in cur.description]
        print(f"  [{name}]: {nr} rows, cols={cols[:15]}")
    except Exception as e:
        print(f"  [{name}]: ERROR: {e}")

# Search all tables for our book IDs
for name in tables:
    try:
        cols = [d[0] for d in cur.execute(f"SELECT * FROM [{name}] LIMIT 0").description]
        for col in cols:
            if "id" in col.lower():
                for bid in [3722, 1299]:
                    try:
                        cur.execute(f"SELECT * FROM [{name}] WHERE [{col}]=? LIMIT 1", (bid,))
                        row = cur.fetchone()
                        if row:
                            print(f"\nBook {bid} FOUND in [{name}].{col}:")
                            for c, v in zip(cols, row):
                                if v and c != "schemas":
                                    print(f"  {c}: {str(v)[:300]}")
                    except:
                        pass
    except:
        pass

conn.close()
