"""Proof that we read the data exactly as it was supplied, before trusting anything built on it.

Each check reports a row: what was checked, on which table, what it found, and whether it passed.
Nothing here repairs data. If a check fails, stop and find out why.
Run it with `quote-followup audit --zip ../data.zip`.

Every file is first read as plain text and only then converted, so a value that can't be read
properly gets counted as a problem instead of slipping through as a blank.
"""

from __future__ import annotations

import csv
import hashlib
import zipfile
from pathlib import Path

import pandas as pd

TABLES = ("service_providers", "customers", "properties", "quotes", "jobs", "visits", "invoices",
          "payments", "product_events", "daily_sp_metrics")

# For each table, the records it points to. A quote must point to a real customer of the same
# business, a job to a real property of the same customer and business, and so on down the chain.
LINKS = {
    "customers": [(["sp_id"], "service_providers", ["sp_id"])],
    "properties": [(["customer_id", "sp_id"], "customers", ["customer_id", "sp_id"])],
    "quotes": [(["customer_id", "sp_id"], "customers", ["customer_id", "sp_id"]),
               (["property_id", "customer_id", "sp_id"], "properties", ["property_id", "customer_id", "sp_id"])],
    "jobs": [(["property_id", "customer_id", "sp_id"], "properties", ["property_id", "customer_id", "sp_id"]),
             (["quote_id", "customer_id", "sp_id"], "quotes", ["quote_id", "customer_id", "sp_id"])],
    "visits": [(["job_id", "customer_id", "property_id", "sp_id"], "jobs", ["job_id", "customer_id", "property_id", "sp_id"])],
    "invoices": [(["job_id", "customer_id", "property_id", "sp_id"], "jobs", ["job_id", "customer_id", "property_id", "sp_id"])],
    "payments": [(["invoice_id", "customer_id", "sp_id"], "invoices", ["invoice_id", "customer_id", "sp_id"])],
    "product_events": [(["sp_id"], "service_providers", ["sp_id"])],
    "daily_sp_metrics": [(["sp_id"], "service_providers", ["sp_id"])],
}

Row = tuple[str, str, str, bool]


def read_raw(data_dir: Path) -> dict[str, pd.DataFrame]:
    """Every table exactly as written in the file, with empty cells marked as missing."""
    return {t: pd.read_csv(data_dir / f"{t}.csv", dtype=str, keep_default_na=False, na_values=[""])
            for t in TABLES}


def zip_fidelity(data_dir: Path, zip_path: Path) -> list[Row]:
    """Each file on disk is identical, byte for byte, to the copy inside the supplied zip."""
    rows = []
    with zipfile.ZipFile(zip_path) as archive:
        members = {Path(n).name: n for n in archive.namelist() if n.endswith(".csv") and "__MACOSX" not in n}
        for table in TABLES:
            packed = hashlib.sha256(archive.read(members[f"{table}.csv"])).hexdigest()
            on_disk = hashlib.sha256((data_dir / f"{table}.csv").read_bytes()).hexdigest()
            rows.append(("matches zip", table, f"sha256 {on_disk[:12]}", packed == on_disk))
    return rows


def row_counts(data_dir: Path, raw: dict[str, pd.DataFrame]) -> list[Row]:
    """We load every row: the count matches a simple line-by-line read of the file."""
    rows = []
    for table in TABLES:
        with open(data_dir / f"{table}.csv", newline="") as handle:
            records = sum(1 for _ in csv.reader(handle)) - 1
        rows.append(("row count", table, f"{records:,} rows", records == len(raw[table])))
    return rows


def _parses(text: pd.Series, data_type: str) -> pd.Series:
    """True where a value can be read as the type the data dictionary says it is."""
    match data_type:
        case "timestamp" | "date":
            return pd.to_datetime(text, format="%Y-%m-%dT%H:%M:%S", errors="coerce").notna()
        case "integer":
            return text.str.fullmatch(r"-?\d+(\.0)?")
        case "decimal":
            return pd.to_numeric(text, errors="coerce").notna()
        case "boolean":
            return text.isin(["True", "False"])
        case _:
            return text.str.len() > 0


def dictionary(raw: dict[str, pd.DataFrame], dictionary_path: Path) -> list[Row]:
    """Every table has the columns the data dictionary lists, of the right type, with no gaps where
    the dictionary says there shouldn't be any, and no duplicate ids."""
    spec = pd.read_csv(dictionary_path)
    rows = []
    for table, columns in spec.groupby("table", sort=False):
        frame = raw[table]
        rows.append(("columns", table, f"{len(frame.columns)} columns", list(frame.columns) == list(columns["column"])))
        bad_parse, bad_null = [], []
        for _, col in columns.iterrows():
            values = frame[col["column"]]
            present = values.dropna()
            failed = (~_parses(present, col["data_type"])).sum()
            if failed:
                bad_parse.append(f"{col['column']} ({failed})")
            if not col["nullable"] and values.isna().any():
                bad_null.append(col["column"])
        rows.append(("types parse", table, ", ".join(bad_parse) or "every value", not bad_parse))
        rows.append(("required present", table, ", ".join(bad_null) or "no gaps", not bad_null))
        key = columns.loc[columns["key"].fillna("").str.startswith("PK"), "column"].tolist()
        rows.append(("key unique", table, "+".join(key), not frame.duplicated(key).any()))
    return rows


def links(raw: dict[str, pd.DataFrame]) -> list[Row]:
    """Every record points to something that exists, and never to another business's records."""
    rows = []
    for child, references in LINKS.items():
        for child_cols, parent, parent_cols in references:
            refs = raw[child][child_cols].dropna(subset=[child_cols[0]])
            keys = raw[parent][parent_cols].drop_duplicates()
            matched = refs.merge(keys, left_on=child_cols, right_on=parent_cols, how="left", indicator=True)
            orphans = int(matched["_merge"].eq("left_only").sum())
            rows.append(("links", child, f"{'+'.join(child_cols)} -> {parent}: {orphans} orphans", orphans == 0))
    events = raw["product_events"].dropna(subset=["entity_id"])
    for entity, (table, column) in {"quote": ("quotes", "quote_id"), "job": ("jobs", "job_id"),
                                    "visit": ("visits", "visit_id"), "invoice": ("invoices", "invoice_id"),
                                    "payment": ("payments", "payment_id")}.items():
        refs = events[events["entity_type"] == entity]
        target = raw[table][[column, "sp_id"]].rename(columns={column: "entity_id"})
        orphans = int((~refs.set_index(["entity_id", "sp_id"]).index.isin(target.set_index(["entity_id", "sp_id"]).index)).sum())
        rows.append(("links", "product_events", f"{entity} events -> {table}: {orphans} orphans", orphans == 0))
    return rows


def quote_rules(quotes: pd.DataFrame) -> list[Row]:
    """Common-sense rules about a quote's life that the model depends on."""
    def rule(name: str, broken: pd.Series) -> Row:
        return ("quote rule", "quotes", f"{name}: {int(broken.sum())} break it", not broken.any())

    status = quotes["status"]
    return [
        rule("created <= sent < expires", ~((quotes["created_at"] <= quotes["sent_at"]) & (quotes["sent_at"] < quotes["expires_at"]))),
        rule("accepted or declined has decided_at", status.isin(["accepted", "declined"]) & quotes["decided_at"].isna()),
        rule("expired or open has no decided_at", status.isin(["expired", "open"]) & quotes["decided_at"].notna()),
        rule("decided after sent", quotes["decided_at"].notna() & (quotes["decided_at"] < quotes["sent_at"])),
        rule("decline reason only on declines", quotes["decline_reason"].notna() & status.ne("declined")),
        rule("amount positive", quotes["quoted_amount_usd"] <= 0),
    ]


def reconciliation(raw: dict[str, pd.DataFrame], quotes: pd.DataFrame) -> list[Row]:
    """Our quote counts match two separate records kept by the platform itself.

    `daily_sp_metrics` counts quotes created, sent and accepted per business per day, and the
    product event log records every quote's creation. If our reading of the quotes table were off,
    these would disagree.
    """
    daily = raw["daily_sp_metrics"].assign(day=lambda d: pd.to_datetime(d["reporting_date"]).dt.normalize())
    rows = []
    for metric, stamp, mask in [
        ("quote_created_count", "created_at", slice(None)),
        ("quote_sent_count", "sent_at", slice(None)),
        ("quote_accepted_count", "decided_at", quotes["status"].eq("accepted")),
    ]:
        ours = quotes.loc[mask].groupby(["sp_id", quotes.loc[mask, stamp].dt.normalize().rename("day")]).size()
        theirs = daily.set_index(["sp_id", "day"])[metric].astype(int)
        theirs = theirs[theirs > 0]
        window = ours[ours.index.get_level_values("day") <= daily["day"].max()]
        joined = pd.concat([window.rename("ours"), theirs.rename("theirs")], axis=1).fillna(0)
        mismatched = int((joined["ours"] != joined["theirs"]).sum())
        rows.append(("reconciles", "daily_sp_metrics", f"{metric}: {mismatched} SP-days differ of {len(joined):,}", mismatched == 0))

    events = raw["product_events"]
    created = events[events["event_name"] == "quote_created"]
    same = set(created["entity_id"]) == set(quotes["quote_id"])
    at = created.set_index("entity_id")["event_at"].pipe(pd.to_datetime)
    timing = (at.reindex(quotes["quote_id"]).to_numpy() == quotes["created_at"].to_numpy()).mean()
    rows.append(("reconciles", "product_events", f"quote_created events cover every quote: {same}; same timestamp: {timing:.1%}",
                 same and timing == 1.0))
    return rows


def run(data_dir: Path, dictionary_path: Path, quotes: pd.DataFrame, zip_path: Path | None = None) -> pd.DataFrame:
    raw = read_raw(data_dir)
    rows = zip_fidelity(data_dir, zip_path) if zip_path else []
    rows += row_counts(data_dir, raw) + dictionary(raw, dictionary_path) + links(raw)
    rows += quote_rules(quotes) + reconciliation(raw, quotes)
    return pd.DataFrame(rows, columns=["check", "table", "detail", "ok"])
