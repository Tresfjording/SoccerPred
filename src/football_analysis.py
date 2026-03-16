import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def finn_kolonne(df: pd.DataFrame, kandidater: list[str]) -> str | None:
    normalisert = {c.strip().lower(): c for c in df.columns}
    for kandidat in kandidater:
        if kandidat in normalisert:
            return normalisert[kandidat]
    return None


@dataclass(frozen=True)
class MatchKolonner:
    dato: str
    hjemmelag: str
    bortelag: str
    hjemmemaal: str
    bortemaal: str
    turnering: str | None


def finn_match_kolonner(df: pd.DataFrame) -> MatchKolonner | None:
    dato = finn_kolonne(df, ["date", "match_date", "dato"])
    hjemmelag = finn_kolonne(df, ["home_team", "team_home", "hjemmelag"])
    bortelag = finn_kolonne(df, ["away_team", "team_away", "bortelag"])
    hjemmemaal = finn_kolonne(df, ["home_score", "home_goals", "hjemmemaal"])
    bortemaal = finn_kolonne(df, ["away_score", "away_goals", "bortemaal"])
    turnering = finn_kolonne(df, ["tournament", "competition", "turnering"])

    if not all([dato, hjemmelag, bortelag, hjemmemaal, bortemaal]):
        return None

    return MatchKolonner(
        dato=dato,
        hjemmelag=hjemmelag,
        bortelag=bortelag,
        hjemmemaal=hjemmemaal,
        bortemaal=bortemaal,
        turnering=turnering,
    )


def parse_date_column(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df[df[date_col].notna()].copy()
    return df


def bygg_lagkamper(df: pd.DataFrame, kolonner: MatchKolonner) -> pd.DataFrame:
    hjemme = pd.DataFrame(
        {
            "date": df[kolonner.dato],
            "team": df[kolonner.hjemmelag],
            "opponent": df[kolonner.bortelag],
            "goals_for": pd.to_numeric(df[kolonner.hjemmemaal], errors="coerce"),
            "goals_against": pd.to_numeric(df[kolonner.bortemaal], errors="coerce"),
            "is_home": 1,
            "tournament": (
                df[kolonner.turnering]
                if kolonner.turnering
                else pd.Series(["unknown"] * len(df), index=df.index)
            ),
        }
    )

    borte = pd.DataFrame(
        {
            "date": df[kolonner.dato],
            "team": df[kolonner.bortelag],
            "opponent": df[kolonner.hjemmelag],
            "goals_for": pd.to_numeric(df[kolonner.bortemaal], errors="coerce"),
            "goals_against": pd.to_numeric(df[kolonner.hjemmemaal], errors="coerce"),
            "is_home": 0,
            "tournament": (
                df[kolonner.turnering]
                if kolonner.turnering
                else pd.Series(["unknown"] * len(df), index=df.index)
            ),
        }
    )

    lagkamper = pd.concat([hjemme, borte], ignore_index=True)
    lagkamper["goal_diff"] = lagkamper["goals_for"] - lagkamper["goals_against"]
    lagkamper["points"] = 0
    lagkamper.loc[lagkamper["goal_diff"] > 0, "points"] = 3
    lagkamper.loc[lagkamper["goal_diff"] == 0, "points"] = 1
    lagkamper = lagkamper.sort_values(["team", "date"]).reset_index(drop=True)
    return lagkamper


def bygg_form_features(lagkamper: pd.DataFrame) -> pd.DataFrame:
    df = lagkamper.copy()

    grouped = df.groupby("team", group_keys=False)
    df["form_points_5"] = grouped["points"].transform(
        lambda s: s.shift(1).rolling(5, min_periods=1).mean()
    )
    df["form_goal_diff_5"] = grouped["goal_diff"].transform(
        lambda s: s.shift(1).rolling(5, min_periods=1).mean()
    )
    df["goals_for_5"] = grouped["goals_for"].transform(
        lambda s: s.shift(1).rolling(5, min_periods=1).mean()
    )
    df["goals_against_5"] = grouped["goals_against"].transform(
        lambda s: s.shift(1).rolling(5, min_periods=1).mean()
    )
    df["matches_played"] = grouped.cumcount()
    return df


def bygg_hjemmekamp_features(
    matcher: pd.DataFrame,
    lag_features: pd.DataFrame,
    kolonner: MatchKolonner,
) -> pd.DataFrame:
    base = pd.DataFrame(
        {
            "date": matcher[kolonner.dato],
            "home_team": matcher[kolonner.hjemmelag],
            "away_team": matcher[kolonner.bortelag],
            "home_score": pd.to_numeric(matcher[kolonner.hjemmemaal], errors="coerce"),
            "away_score": pd.to_numeric(matcher[kolonner.bortemaal], errors="coerce"),
            "tournament": (
                matcher[kolonner.turnering]
                if kolonner.turnering
                else pd.Series(["unknown"] * len(matcher), index=matcher.index)
            ),
        }
    )

    home_features = lag_features[lag_features["is_home"] == 1].copy()
    away_features = lag_features[lag_features["is_home"] == 0].copy()

    home_features = home_features.rename(
        columns={
            "team": "home_team",
            "date": "date",
            "form_points_5": "home_form_points_5",
            "form_goal_diff_5": "home_form_goal_diff_5",
            "goals_for_5": "home_goals_for_5",
            "goals_against_5": "home_goals_against_5",
            "matches_played": "home_matches_played",
        }
    )
    away_features = away_features.rename(
        columns={
            "team": "away_team",
            "date": "date",
            "form_points_5": "away_form_points_5",
            "form_goal_diff_5": "away_form_goal_diff_5",
            "goals_for_5": "away_goals_for_5",
            "goals_against_5": "away_goals_against_5",
            "matches_played": "away_matches_played",
        }
    )

    home_cols = [
        "date",
        "home_team",
        "home_form_points_5",
        "home_form_goal_diff_5",
        "home_goals_for_5",
        "home_goals_against_5",
        "home_matches_played",
    ]
    away_cols = [
        "date",
        "away_team",
        "away_form_points_5",
        "away_form_goal_diff_5",
        "away_goals_for_5",
        "away_goals_against_5",
        "away_matches_played",
    ]

    features = base.merge(home_features[home_cols], on=["date", "home_team"], how="left")
    features = features.merge(away_features[away_cols], on=["date", "away_team"], how="left")

    features["result"] = "D"
    features.loc[features["home_score"] > features["away_score"], "result"] = "H"
    features.loc[features["home_score"] < features["away_score"], "result"] = "A"
    return features


def tren_resultatmodell(features: pd.DataFrame) -> tuple[Pipeline, pd.DataFrame, str]:
    modell_df = features.dropna(subset=["home_score", "away_score", "result"]).copy()
    modell_df = modell_df.sort_values("date").reset_index(drop=True)

    if len(modell_df) < 40:
        raise ValueError(
            "For få kamper til modellering. Trenger minst 40 rader for en meningsfull baseline."
        )

    numeriske = [
        "home_form_points_5",
        "home_form_goal_diff_5",
        "home_goals_for_5",
        "home_goals_against_5",
        "home_matches_played",
        "away_form_points_5",
        "away_form_goal_diff_5",
        "away_goals_for_5",
        "away_goals_against_5",
        "away_matches_played",
    ]
    kategoriske = ["home_team", "away_team", "tournament"]

    split_idx = max(int(len(modell_df) * 0.8), 1)
    train_df = modell_df.iloc[:split_idx].copy()
    test_df = modell_df.iloc[split_idx:].copy()
    if test_df.empty:
        raise ValueError("Testsettet ble tomt. Legg til flere kamper i datasettet.")

    x_train = train_df[numeriske + kategoriske]
    x_test = test_df[numeriske + kategoriske]
    y_train = train_df["result"]
    y_test = test_df["result"]

    pre = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeriske,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "onehot",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ),
                    ]
                ),
                kategoriske,
            ),
        ]
    )

    modell = Pipeline(
        steps=[
            ("preprocess", pre),
            ("model", LogisticRegression(max_iter=2000, multi_class="auto")),
        ]
    )
    modell.fit(x_train, y_train)

    test_pred = modell.predict(x_test)
    score = accuracy_score(y_test, test_pred)
    rapport = classification_report(y_test, test_pred, zero_division=0)

    pred_df = test_df[["date", "home_team", "away_team", "home_score", "away_score", "result"]].copy()
    pred_df["pred_result"] = test_pred
    pred_df["is_correct"] = (pred_df["result"] == pred_df["pred_result"]).astype(int)

    probas = modell.predict_proba(x_test)
    klasse_rekkefolge = list(modell.named_steps["model"].classes_)
    for idx, label in enumerate(klasse_rekkefolge):
        pred_df[f"proba_{label}"] = probas[:, idx]

    metrics_text = (
        f"Accuracy: {score:.3f}\n"
        f"Train rows: {len(train_df)}\n"
        f"Test rows: {len(test_df)}\n\n"
        f"Classification report:\n{rapport}"
    )
    return modell, pred_df, metrics_text


def lag_lagsammendrag(lagkamper: pd.DataFrame) -> pd.DataFrame:
    group = lagkamper.groupby("team")
    summary = group.agg(
        kamper=("team", "size"),
        maal_for=("goals_for", "sum"),
        maal_mot=("goals_against", "sum"),
        poeng=("points", "sum"),
    ).reset_index()
    summary["maal_diff"] = summary["maal_for"] - summary["maal_mot"]
    summary["poeng_per_kamp"] = (summary["poeng"] / summary["kamper"]).round(3)
    summary = summary.sort_values(["poeng_per_kamp", "maal_diff"], ascending=False)
    return summary


def lag_form_plot(lagkamper: pd.DataFrame, output_dir: Path, topp_n: int = 8) -> Path:
    latest = (
        lagkamper.sort_values("date")
        .groupby("team", as_index=False)
        .tail(1)
        .sort_values("form_points_5", ascending=False)
        .head(topp_n)
    )
    top_teams = latest["team"].tolist()

    plot_df = lagkamper[lagkamper["team"].isin(top_teams)].copy()
    if plot_df.empty:
        raise ValueError("Ingen data tilgjengelig for plotting.")

    plt.figure(figsize=(12, 6))
    for lag in top_teams:
        team_df = plot_df[plot_df["team"] == lag].sort_values("date")
        plt.plot(team_df["date"], team_df["form_points_5"], label=lag)

    plt.title("Form (snitt poeng siste 5 kamper)")
    plt.xlabel("Dato")
    plt.ylabel("Poengsnitt")
    plt.legend(loc="best", fontsize=8, ncol=2)
    plt.tight_layout()

    output_file = output_dir / "landslag_form.png"
    plt.savefig(output_file, dpi=150)
    plt.close()
    return output_file


def analyser_landslag(dataset_fil: Path, output_fil: Path, prosjektrot: Path) -> int:
    if not dataset_fil.exists():
        print("Fant ikke landslagsdatasettfilen.")
        print(f"Forventet fil: {dataset_fil}")
        print(
            "Bruk --dataset data/min_fil.csv og sørg for kolonner som date, "
            "home_team, away_team, home_score, away_score."
        )
        return 1

    matcher = pd.read_csv(dataset_fil)
    kolonner = finn_match_kolonner(matcher)
    if kolonner is None:
        print("Fant ikke nødvendige kolonner i landslagsdatasettet.")
        print("Må minst inneholde: date, home_team, away_team, home_score, away_score")
        print(f"Tilgjengelige kolonner: {list(matcher.columns)}")
        return 1

    matcher = parse_date_column(matcher, kolonner.dato)
    matcher = matcher.sort_values(kolonner.dato).reset_index(drop=True)

    lagkamper = bygg_lagkamper(matcher, kolonner)
    lagkamper = bygg_form_features(lagkamper)
    features = bygg_hjemmekamp_features(matcher, lagkamper, kolonner)
    lagsammendrag = lag_lagsammendrag(lagkamper)

    try:
        _, pred_df, metrics_text = tren_resultatmodell(features)
    except ValueError as exc:
        pred_df = pd.DataFrame()
        metrics_text = f"Modell ikke trent: {exc}"

    form_plot = None
    try:
        form_plot = lag_form_plot(lagkamper, prosjektrot / "data")
    except ValueError:
        pass

    with pd.ExcelWriter(output_fil, engine="openpyxl") as writer:
        matcher.to_excel(writer, sheet_name="RawMatches", index=False)
        lagkamper.to_excel(writer, sheet_name="TeamFeatures", index=False)
        lagsammendrag.to_excel(writer, sheet_name="TeamSummary", index=False)
        if not pred_df.empty:
            pred_df.to_excel(writer, sheet_name="Predictions", index=False)
        pd.DataFrame({"metrics": [metrics_text]}).to_excel(
            writer, sheet_name="ModelMetrics", index=False
        )

    print(f"Landslagsanalyse eksportert til: {output_fil}")
    print("\nModelMetrics:")
    print(metrics_text)
    if form_plot:
        print(f"Formplot lagret til: {form_plot}")
    return 0


def analyser_toppscorere(dataset_fil: Path, output_fil: Path, data_dir: Path) -> int:
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyser toppscorere eller landslagskamper."
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["topscorers", "national-team"],
        default="topscorers",
        help="Velg analysemodus.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="",
        help="Valgfri sti til CSV-fil. Hvis utelatt brukes standardfil for valgt modus.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Valgfri sti til output Excel-fil.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    prosjektrot = Path(__file__).resolve().parents[1]
    data_dir = prosjektrot / "data"

    standard_dataset = (
        data_dir / "fifa_world_cup_top_scorers.csv"
        if args.mode == "topscorers"
        else data_dir / "national_team_matches.csv"
    )
    dataset_fil = Path(args.dataset) if args.dataset else standard_dataset
    if not dataset_fil.is_absolute():
        dataset_fil = (prosjektrot / dataset_fil).resolve()

    default_output = (
        data_dir / "scorers_analyse.xlsx"
        if args.mode == "topscorers"
        else data_dir / "landslag_analyse.xlsx"
    )
    output_fil = Path(args.output) if args.output else default_output
    if not output_fil.is_absolute():
        output_fil = (prosjektrot / output_fil).resolve()

    if args.mode == "topscorers":
        return analyser_toppscorere(dataset_fil, output_fil, data_dir)
    return analyser_landslag(dataset_fil, output_fil, prosjektrot)


if __name__ == "__main__":
    raise SystemExit(main())