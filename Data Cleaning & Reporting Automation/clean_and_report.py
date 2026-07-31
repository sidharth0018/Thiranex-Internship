"""
Data Cleaning & Reporting Automation
Cleans a raw customer dataset (missing values, duplicates, inconsistent
formatting, invalid entries) and auto-generates a summary report with
visuals — end to end, no manual steps.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

RAW_PATH = "data/raw_customer_data.csv"
CLEAN_PATH = "outputs/cleaned_customer_data.csv"
LOG_PATH = "outputs/cleaning_log.txt"
REPORT_PATH = "outputs/summary_report.html"

log_lines = []


def log(msg):
    print(msg)
    log_lines.append(msg)


# ---------- 1. Load ----------
df = pd.read_csv(RAW_PATH)
raw_rows = len(df)
log(f"Loaded raw data: {raw_rows} rows, {df.shape[1]} columns")

# ---------- 2. Remove exact duplicates ----------
dupes = df.duplicated().sum()
df = df.drop_duplicates()
log(f"Removed {dupes} duplicate rows")

# ---------- 3. Standardize text fields (City casing/whitespace) ----------
df["City"] = df["City"].astype(str).str.strip().str.title()
df["City"] = df["City"].replace("None", np.nan)
log("Standardized City field casing and whitespace")

# ---------- 4. Fix invalid Age values ----------
invalid_age = ((df["Age"] < 0) | (df["Age"] > 100) | df["Age"].isna()).sum()
median_age = df.loc[(df["Age"] >= 0) & (df["Age"] <= 100), "Age"].median()
df.loc[(df["Age"] < 0) | (df["Age"] > 100), "Age"] = np.nan
df["Age"] = df["Age"].fillna(median_age).round().astype(int)
log(f"Fixed {invalid_age} invalid/missing Age values (imputed with median={median_age:.0f})")

# ---------- 5. Handle missing Income (impute with median) ----------
missing_income = df["Income"].isna().sum()
median_income = df["Income"].median()
df["Income"] = df["Income"].fillna(median_income).round(2)
log(f"Imputed {missing_income} missing Income values with median={median_income:.0f}")

# ---------- 6. Fix invalid PurchaseAmount (negatives → NaN → impute) ----------
invalid_purchase = ((df["PurchaseAmount"] < 0) | df["PurchaseAmount"].isna()).sum()
df.loc[df["PurchaseAmount"] < 0, "PurchaseAmount"] = np.nan
median_purchase = df["PurchaseAmount"].median()
df["PurchaseAmount"] = df["PurchaseAmount"].fillna(median_purchase).round(2)
log(f"Fixed {invalid_purchase} invalid/missing PurchaseAmount values (imputed with median={median_purchase:.0f})")

# ---------- 7. Missing City → 'Unknown' ----------
missing_city = df["City"].isna().sum()
df["City"] = df["City"].fillna("Unknown")
log(f"Filled {missing_city} missing City values with 'Unknown'")

# ---------- 8. Parse dates ----------
df["SignupDate"] = pd.to_datetime(df["SignupDate"], errors="coerce")
log("Parsed SignupDate to datetime")

clean_rows = len(df)
df.to_csv(CLEAN_PATH, index=False)
log(f"\nFinal cleaned dataset: {clean_rows} rows (from {raw_rows} raw rows)")

with open(LOG_PATH, "w") as f:
    f.write("\n".join(log_lines))

# ---------- 9. Visual summaries ----------
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

fig, axes = plt.subplots(2, 2, figsize=(11, 8))

df["City"].value_counts().plot(kind="bar", ax=axes[0, 0], color="#4C72B0")
axes[0, 0].set_title("Customers by City")

axes[0, 1].hist(df["Age"], bins=20, color="#55A868")
axes[0, 1].set_title("Age Distribution")

axes[1, 0].hist(df["PurchaseAmount"], bins=25, color="#C44E52")
axes[1, 0].set_title("Purchase Amount Distribution")

monthly = df.set_index("SignupDate").resample("ME").size()
axes[1, 1].plot(monthly.index, monthly.values, marker="o", color="#8172B2")
axes[1, 1].set_title("Signups Over Time")
axes[1, 1].tick_params(axis="x", rotation=45)

plt.tight_layout()
plt.savefig("outputs/summary_visuals.png", dpi=120)
plt.close()

# ---------- 10. Auto-generate HTML report ----------
top_cities = df["City"].value_counts().head(5)
html = f"""
<html>
<head>
<title>Data Cleaning & Reporting Summary</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 40px; color: #222; }}
h1 {{ color: #2c3e50; }}
h2 {{ color: #34495e; border-bottom: 2px solid #eee; padding-bottom: 4px; }}
table {{ border-collapse: collapse; width: 60%; margin-bottom: 20px; }}
th, td {{ border: 1px solid #ddd; padding: 8px 12px; text-align: left; }}
th {{ background-color: #f5f6fa; }}
.metric {{ display: inline-block; margin: 10px 25px 10px 0; }}
.metric .value {{ font-size: 28px; font-weight: bold; color: #2980b9; }}
.metric .label {{ font-size: 13px; color: #777; }}
img {{ max-width: 100%; border: 1px solid #eee; margin-top: 10px; }}
</style>
</head>
<body>
<h1>Data Cleaning & Reporting Summary</h1>
<p>Generated automatically on {datetime.now().strftime('%d %b %Y, %H:%M')}</p>

<h2>Cleaning Overview</h2>
<div class="metric"><div class="value">{raw_rows}</div><div class="label">Raw Rows</div></div>
<div class="metric"><div class="value">{clean_rows}</div><div class="label">Clean Rows</div></div>
<div class="metric"><div class="value">{dupes}</div><div class="label">Duplicates Removed</div></div>
<div class="metric"><div class="value">{invalid_age + missing_income + invalid_purchase + missing_city}</div><div class="label">Values Fixed/Imputed</div></div>

<h2>Cleaning Log</h2>
<pre>{chr(10).join(log_lines)}</pre>

<h2>Top 5 Cities by Customer Count</h2>
<table>
<tr><th>City</th><th>Customers</th></tr>
{''.join(f"<tr><td>{c}</td><td>{v}</td></tr>" for c, v in top_cities.items())}
</table>

<h2>Visual Summary</h2>
<img src="summary_visuals.png" alt="Summary visuals">

</body>
</html>
"""

with open(REPORT_PATH, "w") as f:
    f.write(html)

log(f"\nReport generated: {REPORT_PATH}")
