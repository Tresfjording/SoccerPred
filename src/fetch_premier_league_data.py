import argparse
from pathlib import Path

import pandas as pd
import requests


LEAGUE_CODE = "E0"  # Premier League hos football-data.co.uk
BASE_URL = "https://www.football-data.co.uk/mmz4281/{season}/{league}.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Hent gratis Premier League kampstatistikk fra football-data.co.uk"
    )
    parser.add_argument(
        "--seasons",
        type=str,
        default="2526",
        help="Kommaseparerte sesongkoder, f.eks. 2526,2425,2324",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/premier_league",
        help="Mappe for lagring av nedlastede og prosesserte filer.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=30,
        help="Timeout i sekunder for HTTP-kall.",
    )
    return parser.parse_args()


def parse_season_codes(raw: str) -> list[str]:
    seasons = [s.strip() for s in raw.split(",") if s.strip()]
    if not seasons:
        raise ValueError("Ingen sesonger oppgitt.")

    invalid = [s for s in seasons if not (len(s) == 4 and s.isdigit())]
    if invalid:
        raise ValueError(
            "Ugyldig sesongkode. Bruk fire siffer, for eksempel 2526,2425. "
            f"Fant: {invalid}"
        )

    return seasons


def download_season_csv(season: str, timeout: int) -> pd.DataFrame:
    url = BASE_URL.format(season=season, league=LEAGUE_CODE)
    headers = {"User-Agent": "SoccerPred/1.0"}

    response = requests.get(url, headers=headers, timeout=timeout)
    if response.status_code == 404:
        raise FileNotFoundError(f"Fant ikke data for sesong {season}: {url}")
    response.raise_for_status()

    df = pd.read_csv(url)
    if df.empty:
        raise ValueError(f"Tom CSV for sesong {season}: {url}")

    df["Season"] = season
    return df


def normalize_matches(df: pd.DataFrame) -> pd.DataFrame:
    required = {
        "Date": "Date",
        "HomeTeam": "HomeTeam",
        "AwayTeam": "AwayTeam",
        "FTHG": "HomeGoals",
        "FTAG": "AwayGoals",
        "FTR": "Result",
    }

    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Mangler nødvendige kolonner i kildefil: {missing}")

    selected_cols = [
        "Season",
        "Date",
        "HomeTeam",
        "AwayTeam",
        "FTHG",
        "FTAG",
        "FTR",
        "HS",
        "AS",
        "HST",
        "AST",
        "HC",
        "AC",
        "HY",
        "AY",
        "HR",
        "AR",
    ]
    available_cols = [c for c in selected_cols if c in df.columns]

    normalized = df[available_cols].rename(columns=required).copy()
    normalized["Date"] = pd.to_datetime(
        normalized["Date"], dayfirst=True, errors="coerce"
    )

    for goal_col in ("HomeGoals", "AwayGoals"):
        normalized[goal_col] = pd.to_numeric(normalized[goal_col], errors="coerce")

    normalized = normalized.dropna(subset=["HomeTeam", "AwayTeam", "HomeGoals", "AwayGoals"])
    normalized = normalized.sort_values(["Season", "Date", "HomeTeam", "AwayTeam"])
    normalized.reset_index(drop=True, inplace=True)
    return normalized


def main() -> int:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = (project_root / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        seasons = parse_season_codes(args.seasons)
    except ValueError as exc:
        print(f"Feil: {exc}")
        return 1

    season_frames: list[pd.DataFrame] = []
    for season in seasons:
        try:
            df = download_season_csv(season, timeout=args.timeout)
            season_frames.append(df)
            print(f"Lastet ned sesong {season}: {len(df)} kamper")
        except Exception as exc:
            print(f"Advarsel: hoppet over sesong {season} ({exc})")

    if not season_frames:
        print("Ingen sesongdata ble lastet ned.")
        return 1

    raw_combined = pd.concat(season_frames, ignore_index=True)
    normalized = normalize_matches(raw_combined)

    raw_path = output_dir / "epl_raw_combined.csv"
    matches_path = output_dir / "epl_matches.csv"

    raw_combined.to_csv(raw_path, index=False)
    normalized.to_csv(matches_path, index=False)

    print(f"Lagret rådata: {raw_path}")
    print(f"Lagret normalisert kampfil: {matches_path}")
    print("Bruk denne videre i Excel-skriptet via --matches-dataset.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())