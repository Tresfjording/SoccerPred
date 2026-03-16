import argparse
from pathlib import Path

import pandas as pd


def finn_kolonne(df: pd.DataFrame, kandidater: list[str]) -> str | None:
    normalisert = {c.strip().lower(): c for c in df.columns}
    for kandidat in kandidater:
        if kandidat in normalisert:
            return normalisert[kandidat]
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyser toppscorer-datasett.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="",
        help="Valgfri sti til CSV-fil. Hvis utelatt brukes standardfil i data-mappen.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    prosjektrot = Path(__file__).resolve().parents[1]
    data_dir = prosjektrot / "data"
    standard_dataset = data_dir / "fifa_world_cup_top_scorers.csv"
    dataset_fil = Path(args.dataset) if args.dataset else standard_dataset
    if not dataset_fil.is_absolute():
        dataset_fil = (prosjektrot / dataset_fil).resolve()

    output_fil = data_dir / "scorers_analyse.xlsx"

    if not dataset_fil.exists():
        print("Fant ikke datasettfilen.")
        print(f"Forventet fil: {dataset_fil}")

        filer = sorted([p.name for p in data_dir.glob("*")]) if data_dir.exists() else []
        if filer:
            print("Tilgjengelige filer i data-mappen:")
            for navn in filer:
                print(f" - {navn}")
        else:
            print("Data-mappen er tom.")

        print(
            "Bruk standardfilen data/fifa_world_cup_top_scorers.csv "
            "eller angi fil med --dataset."
        )
        return 1

    print(f"Bruker dataset: {dataset_fil.name}")
    df = pd.read_csv(dataset_fil)

    print("\nDatasett lastet:")
    print(df.head())

    player_col = finn_kolonne(df, ["player", "player_name", "name"])
    goals_col = finn_kolonne(df, ["goals", "goal", "gols"])

    if not player_col or not goals_col:
        print("Kunne ikke finne nødvendige kolonner for analyse.")
        print(f"Forventet spillerkolonne, fant: {list(df.columns)}")
        print(f"Forventet målkolonne, fant: {list(df.columns)}")
        return 1

    # Grunnleggende analyse: topp 10 scorere basert på tilgjengelig målkolonne.
    top_scorers = (
        df.groupby(player_col)[goals_col]
        .sum()
        .sort_values(ascending=False)
        .head(10)
    )
    print("\nTopp 10 scorere:")
    print(top_scorers)

    df.to_excel(output_fil, index=False)
    print(f"\nData eksportert til: {output_fil}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())