import csv
from pathlib import Path

p = Path("data/raw/valve1_0.csv")
print("файл існує:", p.exists())

with p.open() as f:
    reader = csv.DictReader(f, delimiter=";")
    rows = list(reader)

print("рядків:", len(rows))
print("перший рядок:", rows[0])
print("тип значення Current:", type(rows[0]["Current"]))

values = [float(r["Current"]) for r in rows]

print("значень:", len(values))
print("min:", min(values))
print("max:", max(values))
print("mean:", sum(values) / len(values))

numeric_cols = [c for c in rows[0] if c != "datetime"]
print("числові колонки:", numeric_cols)

for col in numeric_cols:
    vals = [float(r[col]) for r in rows]
    mean = sum(vals) / len(vals)
    print(f"{col:<22} min={min(vals):>10.4f}  max={max(vals):>10.4f}  mean={mean:>10.4f}")
