"""
Generates tableau/Fraud_Detection.twbx: a dark green dashboard with a row of
eight KPI tiles and six charts (3 x 2 grid), built from the pipeline outputs.

Same layout and styling approach as the Retail and Sales portfolio dashboards
(dark background, centred big-number tiles, borderless zones, no gridlines,
automatic sizing), in the green theme shared with the Sales dashboard.

Tableau Public only opens workbooks whose data sources are extracts, so every
input table is written to a .hyper file (Tableau Hyper API) and packaged with
the workbook XML into a .twbx.

Runs on the Windows host or inside the Airflow container. The original CSV path
recorded in the workbook (used only for "Refresh Extract") should be the host
path: set TABLEAU_DATA_DIR when running in Docker.
"""
import csv
import json
import os
import re
import shutil
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path
from xml.sax.saxutils import escape, quoteattr

import pandas as pd
from tableauhyperapi import (Connection, CreateMode, HyperProcess, SqlType, TableDefinition,
                             TableName, Telemetry, escape_string_literal)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = Path(os.environ.get("PROCESSED_DIR", PROJECT_ROOT / "data" / "processed"))
OUTPUTS_DIR = Path(os.environ.get("OUTPUTS_DIR", PROJECT_ROOT / "outputs"))
TABLEAU_DIR = Path(os.environ.get("TABLEAU_DIR", PROJECT_ROOT / "tableau"))
WORKBOOK_NAME = "Fraud_Detection"
TWBX_PATH = TABLEAU_DIR / f"{WORKBOOK_NAME}.twbx"
DATA_DIR_IN_WORKBOOK = os.environ.get("TABLEAU_DATA_DIR", str(PROCESSED_DIR))

THEME = {  # same values as the Sales dashboard
    "background": "#0e1f16",
    "text": "#9ccfae",
    "title": "#ffffff",
    "kpi_label": "#7fd49a",
    "kpi_value": "#ffffff",
    "mark": "#3fa968",
    # Sequential gradient for measures on Color: light green (low) -> dark green (high).
    "gradient": ("#c7ebd2", "#14532d"),
}

HYPER_TYPES = {
    "integer": SqlType.big_int(),
    "real": SqlType.double(),
    "date": SqlType.date(),
    "datetime": SqlType.timestamp(),
    "string": SqlType.text(),
}
EXTRACT_TABLE = TableName("Extract", "Extract")

# prefix -> (derivation, column-instance type)
AGGREGATIONS = {
    "none": ("None", "nominal"),
    "sum": ("Sum", "quantitative"),
    "avg": ("Avg", "quantitative"),
    "cnt": ("Count", "quantitative"),
    "twk": ("Week-Trunc", "quantitative"),
}

CAPTIONS = {
    "hour_band": "Hour of Day",
    "weekday": "Day of Week",
    "fraud_rate": "Fraud Rate",
    "merchant_category": "Merchant Category",
    "txn_date": "Transaction Date",
    "fraud_amount": "Fraud Losses ($)",
    "transaction_country": "Transaction Country",
    "probability_band": "Fraud Probability",
    "transactions": "Flagged Transactions",
    "outcome": "Outcome",
    "precision": "Precision",
    "recall": "Recall",
    "threshold": "Threshold",
}

HOUR_BANDS = ["0-2", "3-5", "6-8", "9-11", "12-14", "15-17", "18-20", "21-23"]
WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
PG_DOW = {1: "Mon", 2: "Tue", 3: "Wed", 4: "Thu", 5: "Fri", 6: "Sat", 0: "Sun"}  # Postgres dow
PROB_BANDS = ["0.5-0.6", "0.6-0.7", "0.7-0.8", "0.8-0.9", "0.9-1.0"]

KPIS = [  # (column in dash_kpis.csv, tile label)
    ("transactions", "TRANSACTIONS"),
    ("fraud_cases", "FRAUD CASES"),
    ("fraud_rate", "FRAUD RATE"),
    ("fraud_losses", "FRAUD LOSSES"),
    ("avg_fraud", "AVG FRAUD"),
    ("auc_pr", "MODEL AUC-PR"),
    ("precision", "PRECISION"),
    ("recall", "RECALL"),
]

DATASOURCES = {
    "kpi": {"caption": "KPIs", "csv": "dash_kpis.csv"},
    "heat": {"caption": "Fraud Rate by Hour and Weekday", "csv": "dash_fraud_heatmap.csv"},
    "category": {"caption": "Fraud by Merchant Category", "csv": "dash_fraud_by_category.csv"},
    "weekly": {"caption": "Weekly Fraud Losses", "csv": "dash_fraud_weekly.csv"},
    "country": {"caption": "Fraud by Country", "csv": "dash_fraud_by_country.csv"},
    "bands": {"caption": "Flagged by Probability", "csv": "dash_flagged_by_band.csv"},
    "pr": {"caption": "Precision and Recall by Threshold", "csv": "dash_threshold_curve.csv"},
}

CHARTS = [
    # Heatmap: fraud rate for each (weekday, 3-hour band) cell.
    {"name": "Fraud Rate: Hour of Day x Weekday", "ds": "heat", "mark": "Square",
     "rows": [("none", "hour_band")], "cols": [("none", "weekday")],
     "color": ("sum", "fraud_rate"), "text": [("sum", "fraud_rate")], "labels": True,
     "manual_order": [("weekday", WEEKDAYS), ("hour_band", HOUR_BANDS)]},
    {"name": "Top 10 Merchant Categories by Fraud Rate", "ds": "category", "mark": "Bar",
     "rows": [("none", "merchant_category")], "cols": [("sum", "fraud_rate")],
     "color": ("sum", "fraud_rate"), "labels": True,
     "sort_desc_by": ("merchant_category", ("sum", "fraud_rate"))},
    {"name": "Weekly Fraud Losses ($)", "ds": "weekly", "mark": "Area",
     "rows": [("sum", "fraud_amount")], "cols": [("twk", "txn_date")]},
    # Treemap: share of fraud losses by the country the transaction came from.
    {"name": "Fraud Losses by Transaction Country", "ds": "country", "mark": "Square",
     "rows": [], "cols": [], "size": ("sum", "fraud_amount"), "color": ("sum", "fraud_amount"),
     "text": [("none", "transaction_country"), ("sum", "fraud_amount")], "labels": True},
    # Stacked bar: every test-period transaction the model flagged, split by outcome.
    # Colour is a 0/1 flag on the green gradient (categorical colour maps written into
    # the XML were ignored by Tableau in every placement tried), so the key is in the title.
    {"name": "Flagged in Test (Jul-Aug): Fraud (dark) vs False Alarm (light)", "ds": "bands", "mark": "Bar",
     "rows": [("sum", "transactions")], "cols": [("none", "probability_band")],
     "color": ("avg", "fraud_flag"), "detail": ("none", "outcome"), "labels": True,
     "manual_order": [("probability_band", PROB_BANDS)]},
    # Scatter: one dot per decision threshold (a precision-recall curve).
    {"name": "Precision vs Recall by Threshold (Test)", "ds": "pr", "mark": "Circle",
     "rows": [("sum", "precision")], "cols": [("sum", "recall")],
     "detail": ("none", "threshold_label"),
     "text": [("none", "threshold_label")], "labels": True},
]

DASHBOARD_NAME = "Fraud Detection Overview"
KPI_HEIGHT, GAP = 12000, 600
CHART_W = (100000 - 4 * GAP) // 3
CHART_H = (100000 - KPI_HEIGHT - 4 * GAP) // 2
KPI_W = (100000 - (len(KPIS) + 1) * GAP) // len(KPIS)

CLEAN_RULES = """
          <style-rule element='gridline'>
            <format attr='line-visibility' scope='rows' value='off' />
            <format attr='line-visibility' scope='cols' value='off' />
          </style-rule>
          <style-rule element='zeroline'>
            <format attr='line-visibility' value='off' />
          </style-rule>
          <style-rule element='axis'>
            <format attr='line-visibility' value='off' />
          </style-rule>
          <style-rule element='table-div'>
            <format attr='line-visibility' scope='rows' value='off' />
            <format attr='line-visibility' scope='cols' value='off' />
          </style-rule>"""

FORMATS = {"fraud_rate": "p0.0%", "fraud_amount": "c\"$\"#,##0", "precision": "p0%", "recall": "p0%"}

ID_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


# ---------------------------------------------------------------- dashboard data

def fmt_money(value):
    if abs(value) >= 1e6:
        return f"${value / 1e6:.2f}M"
    return f"${value / 1e3:,.0f}K" if abs(value) >= 1e4 else f"${value:,.0f}"


def prepare_dashboard_data():
    """Aggregates the pipeline outputs into the small tables the dashboard plots."""
    txn = pd.read_csv(PROCESSED_DIR / "fact_transactions.csv", parse_dates=["transaction_date"])
    flagged = pd.read_csv(OUTPUTS_DIR / "flagged_transactions.csv")
    thresholds = pd.read_csv(OUTPUTS_DIR / "threshold_metrics.csv")
    metrics = json.loads((OUTPUTS_DIR / "model_metrics.json").read_text())
    fraud = txn[txn["is_fraud"] == 1]

    txn["hour_band"] = pd.cut(txn["transaction_hour"], range(-1, 24, 3), labels=HOUR_BANDS)
    txn["weekday"] = txn["transaction_dow"].map(PG_DOW)
    heat = txn.groupby(["weekday", "hour_band"], observed=True)["is_fraud"].agg(
        transactions="size", fraud="sum").reset_index()
    heat["fraud_rate"] = (heat["fraud"] / heat["transactions"]).round(4)
    heat.to_csv(PROCESSED_DIR / "dash_fraud_heatmap.csv", index=False)

    cat = txn.groupby("merchant_category")["is_fraud"].agg(transactions="size", fraud="sum").reset_index()
    cat["fraud_rate"] = (cat["fraud"] / cat["transactions"]).round(4)
    cat = cat.nlargest(10, "fraud_rate")
    cat.to_csv(PROCESSED_DIR / "dash_fraud_by_category.csv", index=False)

    # Daily rows; the chart truncates to weeks. The last week (Aug 25-31) is complete.
    daily = fraud.groupby("transaction_date")["amount"].agg(fraud_amount="sum", fraud="size").reset_index()
    daily = daily.rename(columns={"transaction_date": "txn_date"})
    daily["fraud_amount"] = daily["fraud_amount"].round(2)
    daily.to_csv(PROCESSED_DIR / "dash_fraud_weekly.csv", index=False)

    country = fraud.groupby("transaction_country")["amount"].agg(fraud_amount="sum", fraud="size").reset_index()
    country["fraud_amount"] = country["fraud_amount"].round(2)
    country.to_csv(PROCESSED_DIR / "dash_fraud_by_country.csv", index=False)

    # flagged_transactions.csv holds every test-period transaction with probability >= 0.5
    # (true positives + false positives), so the bars add up to the confusion matrix.
    flagged = flagged.merge(txn[["transaction_id", "is_fraud"]], on="transaction_id", how="left")
    flagged["probability_band"] = pd.cut(flagged["fraud_probability"], [0.5, 0.6, 0.7, 0.8, 0.9, 1.0001],
                                         labels=PROB_BANDS, right=False)
    flagged["outcome"] = flagged["is_fraud"].map({1: "Confirmed fraud", 0: "False alarm"})
    bands = flagged.groupby(["probability_band", "outcome"], observed=True).size().rename(
        "transactions").reset_index()
    bands["fraud_flag"] = (bands["outcome"] == "Confirmed fraud").astype(int)
    bands.to_csv(PROCESSED_DIR / "dash_flagged_by_band.csv", index=False)

    pr = thresholds[thresholds["threshold"].round(2).isin([0.1, 0.3, 0.5, 0.7, 0.9])].copy()
    pr["threshold_label"] = "t=" + pr["threshold"].map(lambda t: f"{t:.1f}")
    pr.to_csv(PROCESSED_DIR / "dash_threshold_curve.csv", index=False)

    m = metrics["winner_metrics"]
    tn, fp, fn, tp = (v for row in m["confusion_matrix"] for v in row)
    assert len(flagged) == tp + fp, f"flagged rows {len(flagged)} != TP+FP {tp + fp}"
    kpis = {
        "transactions": f"{len(txn):,}",
        "fraud_cases": f"{len(fraud):,}",
        "fraud_rate": f"{len(fraud) / len(txn):.2%}",
        "fraud_losses": fmt_money(fraud["amount"].sum()),
        "avg_fraud": f"${fraud['amount'].mean():,.0f}",
        "auc_pr": f"{m['auc_pr']:.3f}",
        "precision": f"{m['precision']:.1%}",
        "recall": f"{m['recall']:.1%}",
    }
    pd.DataFrame([kpis]).to_csv(PROCESSED_DIR / "dash_kpis.csv", index=False)
    print("KPIs:", kpis)


# ---------------------------------------------------------------- workbook XML

def infer_type(values):
    """Returns (datatype, role, type) for a CSV column from sample values."""
    vals = [v for v in values if v not in ("", None)]
    if vals and all(re.fullmatch(r"-?\d+", v) for v in vals):
        return "integer", "measure", "quantitative"
    if vals and all(re.fullmatch(r"-?\d+(\.\d+)?([eE][-+]?\d+)?", v) for v in vals):
        return "real", "measure", "quantitative"
    if vals and all(re.fullmatch(r"\d{4}-\d{2}-\d{2}", v) for v in vals):
        return "date", "dimension", "ordinal"
    if vals and all(re.fullmatch(r"\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}(:\d{2})?", v) for v in vals):
        return "datetime", "dimension", "ordinal"
    return "string", "dimension", "nominal"


def read_schema(csv_path, sample_rows=1000):
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        samples = [row for _, row in zip(range(sample_rows), reader)]
    schema = []
    for i, name in enumerate(header):
        if not ID_RE.match(name):
            raise ValueError(f"{csv_path.name}: column {name!r} needs quoting support")
        schema.append((name, i, *infer_type([r[i] for r in samples if i < len(r)])))
    return schema


def caption(col):
    return CAPTIONS.get(col, col.replace("_", " ").title())


def ds_name(key):
    return f"federated.{key}"


def instance_name(prefix, col):
    kind = "nk" if AGGREGATIONS[prefix][1] == "nominal" else "qk"
    return f"[{prefix}:{col}:{kind}]"


def field_ref(ds_key, prefix, col):
    return f"[{ds_name(ds_key)}].{instance_name(prefix, col)}"


def write_hyper(csv_path, schema, hyper_path, hyper):
    """Loads csv_path into Extract.Extract in a new .hyper file; returns row count."""
    table = TableDefinition(EXTRACT_TABLE, [
        TableDefinition.Column(name, HYPER_TYPES[dt]) for name, _, dt, _, _ in schema
    ])
    with Connection(hyper.endpoint, hyper_path, CreateMode.CREATE_AND_REPLACE) as conn:
        conn.catalog.create_schema(EXTRACT_TABLE.schema_name)
        conn.catalog.create_table(table)
        return conn.execute_command(
            f"COPY {EXTRACT_TABLE} FROM {escape_string_literal(str(csv_path))} "
            "WITH (format csv, header, NULL '')"
        )


def extract_xml(key, row_count):
    now = datetime.now()
    return f"""      <extract count='-1' enabled='true' units='records'>
        <connection access_mode='readonly' authentication='auth-none' author-locale='en_US' class='hyper' dbname='Data/Extracts/{key}.hyper' default-settings='hyper' schema='Extract' sslmode='' tablename='Extract' update-time={quoteattr(now.strftime('%m/%d/%Y %I:%M:%S %p'))} username='tableau'>
          <relation name='Extract' table='[Extract].[Extract]' type='table' />
          <refresh>
            <refresh-event add-from-file-path={quoteattr(key)} increment-value='%null%' refresh-type='create' rows-inserted='{row_count}' timestamp-start={quoteattr(now.strftime('%Y-%m-%d %H:%M:%S.000'))} />
          </refresh>
        </connection>
      </extract>"""



def gradient_xml(field):
    """Green sequential colour ramp for a continuous measure on Color."""
    low, high = THEME["gradient"]
    return f"""
          <style-rule element='mark'>
            <encoding attr='color' field='{field}' type='custom-interpolated'>
              <color-palette custom='true' name='' type='ordered-sequential'>
                <color>{low}</color>
                <color>{high}</color>
              </color-palette>
            </encoding>
          </style-rule>"""


def datasource_xml(key, spec, schema, row_count):
    csv_file = spec["csv"]
    stem = Path(csv_file).stem
    relation_cols = "\n".join(
        f"            <column datatype='{dt}' name={quoteattr(n)} ordinal='{i}' />"
        for n, i, dt, _, _ in schema
    )
    ds_cols = "\n".join(
        f"      <column caption={quoteattr(caption(n))} datatype='{dt}'"
        + (f" default-format={quoteattr(FORMATS[n])}" if n in FORMATS else "")
        + f" name='[{n}]' role='{role}' type='{typ}' />"
        for n, _, dt, role, typ in schema
    )
    return f"""    <datasource caption={quoteattr(spec['caption'])} inline='true' name='{ds_name(key)}' version='18.1'>
      <connection class='federated'>
        <named-connections>
          <named-connection caption={quoteattr(stem)} name='textscan.{key}'>
            <connection class='textscan' directory={quoteattr(DATA_DIR_IN_WORKBOOK.replace(chr(92), '/'))} filename={quoteattr(csv_file)} password='' server='' />
          </named-connection>
        </named-connections>
        <relation connection='textscan.{key}' name={quoteattr(csv_file)} table='[{stem}#csv]' type='table'>
          <columns character-set='UTF-8' header='yes' locale='en_US' separator=','>
{relation_cols}
          </columns>
        </relation>
      </connection>
      <aliases enabled='yes' />
{ds_cols}
{extract_xml(key, row_count)}
    </datasource>"""


def title_xml(text, color, size, align=None):
    align_attr = " fontalignment='1'" if align == "center" else ""
    return f"""      <layout-options>
        <title>
          <formatted-text>
            <run bold='true' fontcolor='{color}' fontsize='{size}'{align_attr}>{escape(text)}</run>
          </formatted-text>
        </title>
      </layout-options>"""


def sheet_style_xml(extra_rules="", labels_in_marks=False):
    """labels_in_marks: colour only axes and headers, leaving mark labels on Tableau's
    automatic colour, which contrasts with each mark (dark text on light cells)."""
    if labels_in_marks:
        text = "".join(f"""
          <style-rule element='{el}'>
            <format attr='color' value='{THEME["text"]}' />
          </style-rule>""" for el in ("axis", "header", "label"))
    else:
        text = f"""
          <style-rule element='worksheet'>
            <format attr='color' value='{THEME["text"]}' />
          </style-rule>"""
    return f"""        <style>
          <style-rule element='table'>
            <format attr='background-color' value='{THEME["background"]}' />
          </style-rule>{text}{CLEAN_RULES}{extra_rules}
        </style>"""


def pane_style_xml(labels, fixed_color):
    formats = []
    if labels:
        formats.append("<format attr='mark-labels-show' value='true' />")
    if fixed_color:
        formats.append(f"<format attr='mark-color' value='{THEME['mark']}' />")
    if not formats:
        return ""
    body = "\n".join(f"                {f}" for f in formats)
    return f"""
            <style>
              <style-rule element='mark'>
{body}
              </style-rule>
            </style>"""


def shelf(key, fields):
    """Multiple fields on one shelf are nested (Tableau's '/' operator); empty -> ''."""
    refs = [field_ref(key, p, c) for p, c in fields]
    if not refs:
        return ""
    return refs[0] if len(refs) == 1 else "(" + " / ".join(refs) + ")"


def shelf_xml(tag, key, fields):
    value = shelf(key, fields)
    return f"<{tag}>{value}</{tag}>" if value else f"<{tag} />"


def dependencies_xml(key, types, used):
    deps = []
    for col in dict.fromkeys(c for _, c in used):
        dt, role, typ = types[col]
        deps.append(f"            <column caption={quoteattr(caption(col))} datatype='{dt}' "
                    + (f"default-format={quoteattr(FORMATS[col])} " if col in FORMATS else "")
                    + f"name='[{col}]' role='{role}' type='{typ}' />")
        for prefix, c in used:
            if c == col:
                derivation, itype = AGGREGATIONS[prefix]
                deps.append(
                    f"            <column-instance column='[{col}]' derivation='{derivation}' "
                    f"name='{instance_name(prefix, col)}' pivot='key' type='{itype}' />"
                )
    return "\n".join(deps)


def chart_xml(ws, schemas):
    key = ws["ds"]
    types = {n: (dt, role, typ) for n, _, dt, role, typ in schemas[key]}
    extra = [ws[k] for k in ("color", "detail", "size") if k in ws] + ws.get("text", [])
    used = list(dict.fromkeys(ws["rows"] + ws["cols"] + extra))
    filter_xml = ""
    if "sort_desc_by" in ws:
        dim, measure = ws["sort_desc_by"]
        filter_xml += f"""
          <sort class='computed' column='{field_ref(key, "none", dim)}' direction='DESC' using='{field_ref(key, *measure)}' />"""
    for dim, values in ws.get("manual_order", []):
        buckets = "\n".join(f"              <bucket>{escape(chr(34) + v + chr(34))}</bucket>" for v in values)
        filter_xml += f"""
          <sort class='manual' column='{field_ref(key, "none", dim)}' direction='ASC'>
            <dictionary>
{buckets}
            </dictionary>
          </sort>"""
    enc = []
    color = ws.get("color")
    if color:
        enc.append(f"<color column='{field_ref(key, *color)}' />")
    if "size" in ws:
        enc.append(f"<size column='{field_ref(key, *ws['size'])}' />")
    enc += [f"<text column='{field_ref(key, *t)}' />" for t in ws.get("text", [])]
    if "detail" in ws:
        enc.append(f"<lod column='{field_ref(key, *ws['detail'])}' />")
    encodings = ""
    if enc:
        body = "\n".join(f"              {e}" for e in enc)
        encodings = f"""
            <encodings>
{body}
            </encodings>"""
    encodings += pane_style_xml(ws.get("labels", False), fixed_color=color is None)
    color_rules = gradient_xml(field_ref(key, *ws["color"])) if "color" in ws else ""
    return f"""    <worksheet name={quoteattr(ws['name'])}>
{title_xml(ws['name'], THEME['title'], 12)}
      <table>
        <view>
          <datasources>
            <datasource caption={quoteattr(DATASOURCES[key]['caption'])} name='{ds_name(key)}' />
          </datasources>
          <datasource-dependencies datasource='{ds_name(key)}'>
{dependencies_xml(key, types, used)}
          </datasource-dependencies>{filter_xml}
          <aggregation value='true' />
        </view>
{sheet_style_xml(color_rules, labels_in_marks=color is not None and ws.get("labels", False))}
        <panes>
          <pane selection-relaxation-option='selection-relaxation-allow'>
            <view>
              <breakdown value='auto' />
            </view>
            <mark class='{ws['mark']}' />{encodings}
          </pane>
        </panes>
        {shelf_xml('rows', key, ws['rows'])}
        {shelf_xml('cols', key, ws['cols'])}
      </table>
    </worksheet>"""


def kpi_style_xml():
    return f"""        <style>
          <style-rule element='table'>
            <format attr='background-color' value='{THEME["background"]}' />
          </style-rule>
          <style-rule element='cell'>
            <format attr='font-size' value='20' />
            <format attr='font-weight' value='bold' />
            <format attr='color' value='{THEME["kpi_value"]}' />
            <format attr='text-align' value='center' />
            <format attr='vertical-align' value='center' />
          </style-rule>{CLEAN_RULES}
        </style>"""


def kpi_xml(column, label, schemas):
    """A big-number tile: one text mark showing a preformatted value, titled with the label."""
    types = {n: (dt, role, typ) for n, _, dt, role, typ in schemas["kpi"]}
    ref = field_ref("kpi", "none", column)
    return f"""    <worksheet name={quoteattr(label)}>
{title_xml(label, THEME['kpi_label'], 9, align='center')}
      <table>
        <view>
          <datasources>
            <datasource caption='KPIs' name='{ds_name("kpi")}' />
          </datasources>
          <datasource-dependencies datasource='{ds_name("kpi")}'>
{dependencies_xml("kpi", types, [("none", column)])}
          </datasource-dependencies>
          <aggregation value='true' />
        </view>
{kpi_style_xml()}
        <panes>
          <pane selection-relaxation-option='selection-relaxation-allow'>
            <view>
              <breakdown value='auto' />
            </view>
            <mark class='Text' />
            <encodings>
              <text column='{ref}' />
            </encodings>
            <customized-label>
              <formatted-text>
                <run bold='true' fontalignment='1' fontcolor='{THEME["kpi_value"]}' fontsize='22'>&lt;{ref}&gt;</run>
              </formatted-text>
            </customized-label>
          </pane>
        </panes>
        <rows />
        <cols />
      </table>
    </worksheet>"""


ZONE_STYLE = ("            <zone-style>\n"
              "              <format attr='border-style' value='none' />\n"
              "              <format attr='border-width' value='0' />\n"
              "              <format attr='margin' value='4' />\n"
              "            </zone-style>\n")


def zones():
    """KPI tiles across the top, charts in a 3 x 2 grid below ."""
    out, zone_id = [], 2
    for i, (_, label) in enumerate(KPIS):
        x = GAP + i * (KPI_W + GAP)
        out.append(f"          <zone h='{KPI_HEIGHT}' id='{zone_id}' name={quoteattr(label)} "
                   f"w='{KPI_W}' x='{x}' y='{GAP}'>\n{ZONE_STYLE}          </zone>")
        zone_id += 1
    for i, chart in enumerate(CHARTS):
        col, row = i % 3, i // 3
        x = GAP + col * (CHART_W + GAP)
        y = KPI_HEIGHT + 2 * GAP + row * (CHART_H + GAP)
        w = CHART_W
        out.append(f"          <zone h='{CHART_H}' id='{zone_id}' name={quoteattr(chart['name'])} "
                   f"w='{w}' x='{x}' y='{y}'>\n{ZONE_STYLE}          </zone>")
        zone_id += 1
    return "\n".join(out)


def dashboard_xml():
    return f"""    <dashboard name={quoteattr(DASHBOARD_NAME)}>
      <style>
        <style-rule element='table'>
          <format attr='background-color' value='{THEME["background"]}' />
        </style-rule>
      </style>
      <size maxheight='900' maxwidth='1600' minheight='900' minwidth='1600' />
      <zones>
        <zone h='100000' id='1' type-v2='layout-basic' w='100000' x='0' y='0'>
{zones()}
        </zone>
      </zones>
    </dashboard>"""


def windows_xml():
    names = [label for _, label in KPIS] + [c["name"] for c in CHARTS]
    viewpoints = "\n".join(
        f"        <viewpoint name={quoteattr(n)}>\n          <zoom type='entire-view' />\n        </viewpoint>"
        for n in names
    )
    return f"""  <windows source-height='30'>
    <window class='dashboard' maximized='true' name={quoteattr(DASHBOARD_NAME)}>
      <viewpoints>
{viewpoints}
      </viewpoints>
      <active id='-1' />
    </window>
  </windows>"""


def build():
    prepare_dashboard_data()
    TABLEAU_DIR.mkdir(parents=True, exist_ok=True)
    work_dir = Path(tempfile.mkdtemp(prefix="twbx_"))
    schemas, row_counts, hyper_files = {}, {}, {}
    # log_dir keeps hyperd.log out of the working directory; work_dir is deleted below.
    with HyperProcess(Telemetry.DO_NOT_SEND_USAGE_DATA_TO_TABLEAU,
                      parameters={"log_dir": str(work_dir)}) as hyper:
        for key, spec in DATASOURCES.items():
            path = PROCESSED_DIR / spec["csv"]
            if not path.exists():
                raise FileNotFoundError(f"{path} not found; run export_marts.py first")
            schemas[key] = read_schema(path)
            hyper_files[key] = work_dir / f"{key}.hyper"
            row_counts[key] = write_hyper(path, schemas[key], hyper_files[key], hyper)
            print(f"  {spec['csv']}: {row_counts[key]} rows -> {key}.hyper")
    datasources = "\n".join(
        datasource_xml(k, s, schemas[k], row_counts[k]) for k, s in DATASOURCES.items()
    )
    worksheets = "\n".join(
        [kpi_xml(col, label, schemas) for col, label in KPIS] + [chart_xml(c, schemas) for c in CHARTS]
    )
    xml = f"""<?xml version='1.0' encoding='utf-8' ?>
<workbook original-version='18.1' source-build='2026.2.0 (20262.26.0819.2015)' source-platform='win' version='18.1' xmlns:user='http://www.tableausoftware.com/xml/user'>
  <preferences>
    <preference name='ui.encoding.shelf.height' value='24' />
    <preference name='ui.shelf.height' value='26' />
  </preferences>
  <datasources>
{datasources}
  </datasources>
  <worksheets>
{worksheets}
  </worksheets>
  <dashboards>
{dashboard_xml()}
  </dashboards>
{windows_xml()}
</workbook>
"""
    with zipfile.ZipFile(TWBX_PATH, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr(f"{WORKBOOK_NAME}.twb", xml)
        for key, hyper_path in hyper_files.items():
            z.write(hyper_path, f"Data/Extracts/{key}.hyper")
    shutil.rmtree(work_dir, ignore_errors=True)
    print(f"Wrote {TWBX_PATH}")


if __name__ == "__main__":
    build()
