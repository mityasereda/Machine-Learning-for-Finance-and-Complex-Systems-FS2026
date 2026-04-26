#!/usr/bin/env python3
"""Extract a WRDS subset for this project.

The script pulls:
- TAQ daily trades tables into raw per-symbol daily files.
- CRSP daily prices/dividends into processed daily files.
- 1-minute intraday bars with the engineered columns expected by the repo.

It keeps the file layout aligned with the optional WRDS loader introduced in
`momentum/data.py` and `portfolio-rebalancing/data.py`.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import wrds
except ImportError as exc:  # pragma: no cover - optional dependency
    wrds = None
    WRDS_IMPORT_ERROR = exc
else:
    WRDS_IMPORT_ERROR = None


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ASSET_MAP = REPO_ROOT / "data" / "wrds" / "asset_map.example.csv"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "data" / "wrds"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-map", type=Path, default=DEFAULT_ASSET_MAP)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--start-date", default="2021-05-09")
    parser.add_argument("--end-date", default="2022-12-09")
    parser.add_argument("--file-format", choices=["parquet", "csv"], default="parquet")
    parser.add_argument("--wrds-username", default=None)
    parser.add_argument(
        "--list-taq-access",
        action="store_true",
        help="List TAQ-like WRDS libraries and sample table names, then exit.",
    )
    parser.add_argument("--taq-library", default="taq")
    parser.add_argument("--taq-prefix", default="ct_")
    parser.add_argument("--crsp-daily-table", default="crsp.dsf")
    parser.add_argument("--crsp-dist-table", default="crsp.dsedist")
    return parser.parse_args()


def ensure_wrds_available():
    if wrds is None:
        raise ImportError(
            "The 'wrds' package is required for extraction. Install it in the "
            "environment you use for WRDS access."
        ) from WRDS_IMPORT_ERROR


def load_asset_map(path: Path) -> pd.DataFrame:
    asset_map = pd.read_csv(path, dtype={"permno": "Int64"})
    required = {"project_symbol", "taq_symbol", "permno", "valid_from", "valid_to"}
    missing = required - set(asset_map.columns)
    if missing:
        raise ValueError(f"Asset map is missing required columns: {sorted(missing)}")

    asset_map["valid_from"] = pd.to_datetime(asset_map["valid_from"]).dt.date
    asset_map["valid_to"] = pd.to_datetime(asset_map["valid_to"]).dt.date

    if asset_map["permno"].isna().any():
        missing_permnos = asset_map.loc[asset_map["permno"].isna(), "project_symbol"].unique()
        raise ValueError(
            "Fill the PERMNO values in the asset map before extraction. Missing for: "
            + ", ".join(sorted(missing_permnos))
        )

    return asset_map


def connect_wrds(username: str | None):
    kwargs = {"wrds_username": username} if username else {}
    return wrds.Connection(**kwargs)


def sql_quote(values):
    return ", ".join(f"'{value}'" for value in values)


def split_table_name(table_name: str) -> tuple[str, str]:
    parts = table_name.split(".", 1)
    if len(parts) != 2:
        raise ValueError(f"Expected fully-qualified table name like schema.table, got: {table_name}")
    return parts[0], parts[1]


def get_table_columns(db, schema: str, table: str) -> list[str]:
    query = f"""
        select column_name
        from information_schema.columns
        where table_schema = '{schema}'
          and table_name = '{table}'
        order by ordinal_position
    """
    df = db.raw_sql(query)
    return df["column_name"].tolist()


def find_existing_table(db, candidates: list[tuple[str, str]]) -> tuple[str, str] | None:
    for schema, table in candidates:
        query = f"""
            select 1
            from pg_tables
            where schemaname = '{schema}'
              and tablename = '{table}'
            limit 1
        """
        df = db.raw_sql(query)
        if not df.empty:
            return schema, table
    return None


def unique_preserving_order(items: list[tuple[str, str]]) -> list[tuple[str, str]]:
    seen = set()
    result = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


def write_frame(df: pd.DataFrame, path: Path, file_format: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    if file_format == "parquet":
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)


def prepare_intraday_features(minute_bars: pd.DataFrame, daily_dividends: pd.DataFrame) -> pd.DataFrame:
    df = minute_bars.copy()
    df["caldt"] = pd.to_datetime(df["caldt"])
    df.sort_values("caldt", inplace=True)
    df["day"] = df["caldt"].dt.date
    df.set_index("caldt", inplace=True, drop=False)

    daily_groups = df.groupby("day")
    all_days = df["day"].unique()

    df["move_open"] = np.nan
    df["vwap"] = np.nan
    df["spy_dvol"] = np.nan

    spy_ret = pd.Series(index=all_days, dtype=float)

    for d in range(1, len(all_days)):
        current_day = all_days[d]
        prev_day = all_days[d - 1]

        current_day_data = daily_groups.get_group(current_day)
        prev_day_data = daily_groups.get_group(prev_day)

        hlc = (current_day_data["high"] + current_day_data["low"] + current_day_data["close"]) / 3
        vol_x_hlc = current_day_data["volume"] * hlc
        cum_vol_x_hlc = vol_x_hlc.cumsum()
        cum_volume = current_day_data["volume"].cumsum()

        df.loc[current_day_data.index, "vwap"] = cum_vol_x_hlc / cum_volume

        open_price = current_day_data["open"].iloc[0]
        df.loc[current_day_data.index, "move_open"] = (current_day_data["close"] / open_price - 1).abs()

        spy_ret.loc[current_day] = (
            current_day_data["close"].iloc[-1] / prev_day_data["close"].iloc[-1] - 1
        )

        if d > 14:
            df.loc[current_day_data.index, "spy_dvol"] = spy_ret.iloc[d - 15:d - 1].std(skipna=False)

    df["min_from_open"] = ((df.index - df.index.normalize()) / pd.Timedelta(minutes=1)) - (9 * 60 + 30) + 1
    df["minute_of_day"] = df["min_from_open"].round().astype(int)

    minute_groups = df.groupby("minute_of_day")
    df["move_open_rolling_mean"] = minute_groups["move_open"].transform(
        lambda x: x.rolling(window=14, min_periods=13).mean()
    )
    df["sigma_open"] = minute_groups["move_open_rolling_mean"].transform(lambda x: x.shift(1))

    df = df.reset_index(drop=True)
    dividends = daily_dividends[["caldt", "dividend"]].copy()
    dividends["day"] = pd.to_datetime(dividends["caldt"]).dt.date
    df = df.merge(dividends[["day", "dividend"]], on="day", how="left")
    df["dividend"] = df["dividend"].fillna(0)

    df["caldt"] = pd.to_datetime(df["caldt"])
    df["day"] = pd.to_datetime(df["day"]).dt.date
    return df


def fetch_taq_day(db, taq_library: str, taq_prefix: str, day: pd.Timestamp, symbols: list[str]) -> pd.DataFrame:
    yyyymmdd = day.strftime("%Y%m%d")
    yearly_library = f"taqm_{day.year}"
    libraries = [taq_library, yearly_library, "taqmsec"]
    table_names = [
        f"{taq_prefix}{yyyymmdd}",
        f"ctm_{yyyymmdd}",
        f"ct_{yyyymmdd}",
    ]
    candidates = unique_preserving_order([
        (library, table_name)
        for library in libraries
        for table_name in table_names
    ] + [
        (taq_library, f"{taq_prefix}{yyyymmdd}"),
        (taq_library, f"ctm_{yyyymmdd}"),
        (taq_library, f"ct_{yyyymmdd}"),
        ("taqmsec", f"{taq_prefix}{yyyymmdd}"),
        ("taqmsec", f"ctm_{yyyymmdd}"),
        ("taqmsec", f"ct_{yyyymmdd}"),
    ])
    table_ref = find_existing_table(db, candidates)
    if table_ref is None:
        tried = ", ".join(f"{schema}.{table}" for schema, table in candidates)
        raise RuntimeError(f"No TAQ trade table found for {yyyymmdd}. Tried: {tried}")

    schema, table = table_ref
    columns = get_table_columns(db, schema, table)

    symbol_col = "symbol" if "symbol" in columns else "sym_root"
    price_col = "price"
    size_col = "size"
    time_col = "time" if "time" in columns else "time_m"

    select_date = "date as trade_date" if "date" in columns else f"date '{day.date().isoformat()}' as trade_date"
    query = f"""
        select
            {symbol_col} as symbol,
            {select_date},
            {time_col} as trade_time,
            {price_col} as price,
            {size_col} as size
        from {schema}.{table}
        where {symbol_col} in ({sql_quote(symbols)})
          and {time_col} between '09:30:00' and '15:59:59'
          and {price_col} > 0
          and {size_col} > 0
        order by {symbol_col}, {time_col}
    """
    return db.raw_sql(query)


def fetch_crsp_daily(db, table_name: str, permnos: list[int], start_date: str, end_date: str) -> pd.DataFrame:
    schema, table = split_table_name(table_name)
    columns = get_table_columns(db, schema, table)

    if "date" not in columns:
        raise ValueError(f"{table_name} does not have a 'date' column")

    query = f"""
        select permno, date as caldt, openprc, askhi, bidlo, prc, vol, ret
        from {table_name}
        where permno in ({", ".join(str(int(permno)) for permno in permnos)})
          and date between '{start_date}' and '{end_date}'
        order by permno, caldt
    """
    df = db.raw_sql(query)
    df["caldt"] = pd.to_datetime(df["caldt"])
    return df


def fetch_crsp_dividends(db, table_name: str, permnos: list[int], start_date: str, end_date: str) -> pd.DataFrame:
    schema, table = split_table_name(table_name)
    columns = get_table_columns(db, schema, table)

    if "permno" not in columns:
        raise ValueError(f"{table_name} does not have a 'permno' column")

    amount_col = "divamt" if "divamt" in columns else None
    if amount_col is None:
        raise ValueError(f"{table_name} does not expose a dividend amount column like 'divamt'")

    date_col = None
    for candidate in ["exdt", "date", "distdt"]:
        if candidate in columns:
            date_col = candidate
            break
    if date_col is None:
        raise ValueError(f"{table_name} does not expose a distribution date column like exdt/date/distdt")

    query = f"""
        select permno, {date_col} as caldt, sum(coalesce({amount_col}, 0)) as dividend
        from {table_name}
        where permno in ({", ".join(str(int(permno)) for permno in permnos)})
          and {date_col} between '{start_date}' and '{end_date}'
        group by permno, {date_col}
        order by permno, caldt
    """
    df = db.raw_sql(query)
    if df.empty:
        return pd.DataFrame(
            {
                "permno": pd.Series(dtype="Int64"),
                "caldt": pd.Series(dtype="datetime64[ns]"),
                "dividend": pd.Series(dtype="float64"),
            }
        )
    df["caldt"] = pd.to_datetime(df["caldt"])
    return df


def list_taq_access(db):
    libraries = sorted(lib for lib in db.list_libraries() if "taq" in lib.lower())
    if not libraries:
        print("No TAQ-like libraries are visible in this WRDS account.")
        return

    print("TAQ-like libraries:")
    for library in libraries:
        print(f"- {library}")
        try:
            tables = db.list_tables(library=library)
        except Exception as exc:  # pragma: no cover - depends on WRDS access
            print(f"  FAILED to list tables: {exc}")
            continue

        sample = [table for table in tables if table.startswith(("ct", "cq"))][:20]
        if sample:
            print(f"  sample tables: {', '.join(sample)}")
        else:
            print("  no ct*/cq* tables found")


def main():
    args = parse_args()
    ensure_wrds_available()

    db = connect_wrds(args.wrds_username)
    if args.list_taq_access:
        list_taq_access(db)
        db.close()
        return

    asset_map = load_asset_map(args.asset_map)
    output_root = args.output_root.resolve()
    date_range = pd.date_range(args.start_date, args.end_date, freq="B")

    minute_bars_by_symbol: dict[str, list[pd.DataFrame]] = defaultdict(list)

    for day in date_range:
        active_rows = asset_map[
            (asset_map["valid_from"] <= day.date()) & (asset_map["valid_to"] >= day.date())
        ]
        if active_rows.empty:
            continue

        active_symbols = sorted(active_rows["taq_symbol"].unique())
        symbol_map = dict(zip(active_rows["taq_symbol"], active_rows["project_symbol"]))

        try:
            trades = fetch_taq_day(db, args.taq_library, args.taq_prefix, day, active_symbols)
        except Exception as exc:  # pragma: no cover - depends on WRDS access
            print(f"Skipping {day.date()} because TAQ extraction failed: {exc}")
            continue

        if trades.empty:
            continue

        trades["timestamp"] = pd.to_datetime(trades["trade_date"]) + pd.to_timedelta(
            trades["trade_time"].astype(str)
        )
        trades["project_symbol"] = trades["symbol"].map(symbol_map)
        trades["value"] = trades["price"] * trades["size"]
        trades = trades[["project_symbol", "symbol", "timestamp", "price", "size", "value"]]

        for project_symbol, symbol_trades in trades.groupby("project_symbol"):
            raw_path = output_root / "raw" / "taq_trades" / project_symbol / f"{day.date().isoformat()}.{args.file_format}"
            write_frame(symbol_trades, raw_path, args.file_format)

            minute_bars = (
                symbol_trades.set_index("timestamp")
                .sort_index()
                .resample("1min")
                .agg(
                    open=("price", "first"),
                    high=("price", "max"),
                    low=("price", "min"),
                    close=("price", "last"),
                    volume=("size", "sum"),
                )
                .dropna(subset=["open", "close"])
                .reset_index()
                .rename(columns={"timestamp": "caldt"})
            )
            minute_bars_by_symbol[project_symbol].append(minute_bars)

    crsp_daily = fetch_crsp_daily(
        db,
        args.crsp_daily_table,
        sorted(asset_map["permno"].dropna().astype(int).unique()),
        args.start_date,
        args.end_date,
    )
    crsp_dividends = fetch_crsp_dividends(
        db,
        args.crsp_dist_table,
        sorted(asset_map["permno"].dropna().astype(int).unique()),
        args.start_date,
        args.end_date,
    )

    for row in asset_map.itertuples(index=False):
        daily = crsp_daily[
            (crsp_daily["permno"] == int(row.permno))
            & (crsp_daily["caldt"].dt.date >= row.valid_from)
            & (crsp_daily["caldt"].dt.date <= row.valid_to)
        ].copy()

        if daily.empty:
            continue

        daily["project_symbol"] = row.project_symbol
        daily["open"] = daily["openprc"].abs()
        daily["high"] = daily["askhi"].abs()
        daily["low"] = daily["bidlo"].abs()
        daily["close"] = daily["prc"].abs()
        daily["volume"] = daily["vol"]
        dividends = crsp_dividends[
            (crsp_dividends["permno"] == int(row.permno))
            & (crsp_dividends["caldt"].dt.date >= row.valid_from)
            & (crsp_dividends["caldt"].dt.date <= row.valid_to)
        ][["permno", "caldt", "dividend"]].copy()
        daily = daily.merge(dividends, on=["permno", "caldt"], how="left")
        daily["dividend"] = daily["dividend"].fillna(0)

        daily = daily[
            ["project_symbol", "permno", "caldt", "open", "high", "low", "close", "volume", "ret", "dividend"]
        ]

        daily_path = output_root / "processed" / "daily" / f"{row.project_symbol}_daily.{args.file_format}"
        existing = None
        if daily_path.exists():
            if args.file_format == "parquet":
                existing = pd.read_parquet(daily_path)
            else:
                existing = pd.read_csv(daily_path, parse_dates=["caldt"])
        merged_daily = pd.concat([existing, daily], ignore_index=True) if existing is not None else daily
        merged_daily.drop_duplicates(subset=["project_symbol", "caldt"], inplace=True)
        merged_daily.sort_values("caldt", inplace=True)
        write_frame(merged_daily, daily_path, args.file_format)

    for project_symbol, minute_frames in minute_bars_by_symbol.items():
        if not minute_frames:
            continue

        minute_bars = pd.concat(minute_frames, ignore_index=True)
        minute_bars.drop_duplicates(subset=["caldt"], inplace=True)
        minute_bars.sort_values("caldt", inplace=True)

        daily_path = output_root / "processed" / "daily" / f"{project_symbol}_daily.{args.file_format}"
        if args.file_format == "parquet":
            daily_frame = pd.read_parquet(daily_path)
        else:
            daily_frame = pd.read_csv(daily_path, parse_dates=["caldt"])

        intraday = prepare_intraday_features(minute_bars, daily_frame)
        intraday_path = output_root / "processed" / "intraday" / f"{project_symbol}_1min.{args.file_format}"
        write_frame(intraday, intraday_path, args.file_format)

    db.close()
    print(f"WRDS extraction complete. Files written under {output_root}")


if __name__ == "__main__":
    main()
