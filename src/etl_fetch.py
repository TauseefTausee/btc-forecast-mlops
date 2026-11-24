"""Extract step: pull daily BTC-USD OHLCV data and write it to raw storage.

Sources are tried in order (yfinance -> Binance -> CoinGecko) so a single
provider outage or rate limit does not break the daily pipeline. Every source
is normalised to the same schema:

    date (YYYY-MM-DD, UTC) | open | high | low | close | volume

Usage:
    python -m src.etl_fetch --output shared/models/data/btc_raw.csv
"""

import argparse
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests

from src import config

log = logging.getLogger(__name__)

COLUMNS = ["date", "open", "high", "low", "close", "volume"]
MIN_ROWS = 120
HTTP_TIMEOUT = 30


def _utc_today() -> pd.Timestamp:
    return pd.Timestamp(datetime.now(timezone.utc).date())


def fetch_yfinance(days: int, ticker: str = config.TICKER) -> pd.DataFrame:
    import yfinance as yf  # imported lazily: heavy and only needed here

    start = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d")
    hist = yf.Ticker(ticker).history(start=start, interval="1d", auto_adjust=False)
    if hist is None or hist.empty:
        raise RuntimeError("yfinance returned no rows")
    if isinstance(hist.columns, pd.MultiIndex):
        hist.columns = hist.columns.get_level_values(0)
    df = hist.reset_index().rename(columns=str.lower)
    df = df.rename(columns={"datetime": "date"})
    df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_localize(None).dt.normalize()
    return df[COLUMNS]


def fetch_binance(days: int, symbol: str = "BTCUSDT") -> pd.DataFrame:
    url = "https://api.binance.com/api/v3/klines"
    start_ms = int((datetime.now(timezone.utc) - timedelta(days=days)).timestamp() * 1000)
    rows = []
    while True:
        resp = requests.get(
            url,
            params={"symbol": symbol, "interval": "1d", "startTime": start_ms, "limit": 1000},
            timeout=HTTP_TIMEOUT,
        )
        resp.raise_for_status()
        batch = resp.json()
        if not batch:
            break
        rows.extend(batch)
        if len(batch) < 1000:
            break
        start_ms = batch[-1][0] + 86_400_000
    if not rows:
        raise RuntimeError("Binance returned no rows")
    df = pd.DataFrame(rows).iloc[:, :6]
    df.columns = COLUMNS
    df["date"] = pd.to_datetime(df["date"], unit="ms").dt.normalize()
    return df


def fetch_coingecko(days: int) -> pd.DataFrame:
    # The free CoinGecko tier only serves 365 days and has no daily OHLC, so
    # open/high/low are approximated by the daily close. Last-resort source.
    days = min(days, 365)
    headers = {"x-cg-demo-api-key": config.COINGECKO_API_KEY} if config.COINGECKO_API_KEY else {}
    resp = requests.get(
        "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart",
        params={"vs_currency": "usd", "days": days, "interval": "daily"},
        headers=headers,
        timeout=HTTP_TIMEOUT,
    )
    resp.raise_for_status()
    payload = resp.json()
    prices = pd.DataFrame(payload["prices"], columns=["ts", "close"])
    volumes = pd.DataFrame(payload["total_volumes"], columns=["ts", "volume"])
    df = prices.merge(volumes, on="ts", how="left")
    df["date"] = pd.to_datetime(df["ts"], unit="ms").dt.normalize()
    df["open"] = df["high"] = df["low"] = df["close"]
    return df[COLUMNS]


SOURCES = {
    "yfinance": fetch_yfinance,
    "binance": fetch_binance,
    "coingecko": fetch_coingecko,
}


def clean_ohlcv(df: pd.DataFrame, drop_partial_day: bool = True) -> pd.DataFrame:
    """Validate and normalise raw OHLCV rows."""
    missing = set(COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    out = df[COLUMNS].copy()
    out["date"] = pd.to_datetime(out["date"]).dt.normalize()
    for col in COLUMNS[1:]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=["close"])
    out = out[out["close"] > 0]
    if drop_partial_day:
        # Today's candle is still forming; training on it leaks noise.
        out = out[out["date"] < _utc_today()]
    out = out.sort_values("date").drop_duplicates("date", keep="last")
    # Fill the odd missing calendar day so features see a regular daily grid.
    out = out.set_index("date").asfreq("D")
    out["close"] = out["close"].ffill()
    for col in ("open", "high", "low"):
        out[col] = out[col].fillna(out["close"])
    out["volume"] = out["volume"].fillna(0.0)
    out = out.reset_index()
    return out[COLUMNS]


def fetch_btc_data(days: int = config.HISTORY_DAYS, source: str = "auto") -> pd.DataFrame:
    order = list(SOURCES) if source == "auto" else [source]
    errors = []
    for name in order:
        try:
            log.info("Fetching %s days of BTC data from %s", days, name)
            df = clean_ohlcv(SOURCES[name](days))
            if len(df) < MIN_ROWS:
                raise RuntimeError(f"only {len(df)} rows")
            df.attrs["source"] = name
            log.info(
                "Got %d rows from %s (%s -> %s)",
                len(df),
                name,
                df["date"].min().date(),
                df["date"].max().date(),
            )
            return df
        except Exception as exc:  # noqa: BLE001 - try the next provider
            log.warning("Source %s failed: %s", name, exc)
            errors.append(f"{name}: {exc}")
    raise RuntimeError("All data sources failed -> " + " | ".join(errors))


def save_raw(df: pd.DataFrame, output: Path) -> Path:
    """Write the latest raw file plus a dated snapshot for auditability."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(".tmp")
    df.to_csv(tmp, index=False, date_format="%Y-%m-%d")
    tmp.replace(output)
    snap_dir = output.parent / "raw_snapshots"
    snap_dir.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
    df.to_csv(snap_dir / f"btc_raw_{stamp}.csv", index=False, date_format="%Y-%m-%d")
    return output


def load_raw(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    return clean_ohlcv(df, drop_partial_day=False)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Fetch daily BTC-USD data")
    parser.add_argument("--output", default=str(config.DATA_DIR / "btc_raw.csv"))
    parser.add_argument("--days", type=int, default=config.HISTORY_DAYS)
    parser.add_argument("--source", default="auto", choices=["auto", *SOURCES])
    args = parser.parse_args(argv)
    logging.basicConfig(level=config.LOG_LEVEL, format="%(asctime)s %(levelname)s %(message)s")
    df = fetch_btc_data(args.days, args.source)
    path = save_raw(df, Path(args.output))
    print(f"Saved {len(df)} rows from {df.attrs.get('source')} to {path}")


if __name__ == "__main__":
    main()
