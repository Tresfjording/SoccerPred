import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score, log_loss, make_scorer
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from openpyxl.worksheet.datavalidation import DataValidation

try:
    from xgboost import XGBClassifier

    HAS_XGBOOST = True
except ImportError:
    XGBClassifier = None
    HAS_XGBOOST = False


def hybrid_metric(y_true: pd.Series, y_pred: pd.Series) -> float:
    return 0.7 * accuracy_score(y_true, y_pred) + 0.3 * f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )


def beregn_optimaliseringsscore(y_true: pd.Series, y_pred: pd.Series, optimize_metric: str) -> float:
    if optimize_metric == "macro-f1":
        return float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    if optimize_metric == "hybrid":
        return float(hybrid_metric(y_true, y_pred))
    return float(accuracy_score(y_true, y_pred))


def bygg_scorer(optimize_metric: str):
    if optimize_metric == "macro-f1":
        return make_scorer(f1_score, average="macro", zero_division=0)
    if optimize_metric == "hybrid":
        return make_scorer(hybrid_metric)
    return "accuracy"


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
    dato = finn_kolonne(df, ["date", "match_date", "matchdate", "dato"])
    hjemmelag = finn_kolonne(
        df,
        ["home_team", "team_home", "hjemmelag", "hometeam"],
    )
    bortelag = finn_kolonne(
        df,
        ["away_team", "team_away", "bortelag", "awayteam"],
    )
    hjemmemaal = finn_kolonne(
        df,
        ["home_score", "home_goals", "hjemmemaal", "fthg", "homegoals"],
    )
    bortemaal = finn_kolonne(
        df,
        ["away_score", "away_goals", "bortemaal", "ftag", "awaygoals"],
    )
    turnering = finn_kolonne(df, ["tournament", "competition", "turnering", "div"])

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
    dato_str = df[date_col].astype(str)
    slash_ratio = dato_str.str.contains(r"^\d{1,2}/\d{1,2}/\d{2,4}$", regex=True).mean()
    dayfirst = slash_ratio > 0.5

    parsed = pd.to_datetime(df[date_col], errors="coerce", dayfirst=dayfirst)
    if parsed.notna().sum() == 0 and dayfirst:
        parsed = pd.to_datetime(df[date_col], errors="coerce", dayfirst=False)

    df[date_col] = parsed
    df = df[df[date_col].notna()].copy()
    return df


def last_ned_og_kombiner_sesonger(
    sesonger: list[str],
    data_dir: Path,
    timeout: int = 30,
) -> pd.DataFrame | None:
    try:
        import requests as _requests
    except ImportError:
        print("Advarsel: 'requests' ikke installert. Kan ikke laste ned ekstra sesonger.")
        return None

    BASE_URL = "https://www.football-data.co.uk/mmz4281/{season}/E0.csv"
    frames: list[pd.DataFrame] = []

    for sesong in sesonger:
        lokal = data_dir / f"E0_{sesong}.csv"
        if lokal.exists():
            df = pd.read_csv(lokal)
            df["_season"] = sesong
            frames.append(df)
            print(f"Bruker lokal fil for sesong {sesong}: {len(df)} kamper")
            continue
        try:
            url = BASE_URL.format(season=sesong)
            resp = _requests.get(url, headers={"User-Agent": "SoccerPred/1.0"}, timeout=timeout)
            resp.raise_for_status()
            lokal.write_bytes(resp.content)
            df = pd.read_csv(lokal)
            df["_season"] = sesong
            frames.append(df)
            print(f"Lastet ned sesong {sesong}: {len(df)} kamper")
        except Exception as exc:
            print(f"Advarsel: Hoppet over sesong {sesong} ({exc})")

    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


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
    df["days_since_last_match"] = grouped["date"].transform(lambda s: s.diff().dt.days)
    venue_grouped = df.groupby(["team", "is_home"], group_keys=False)
    df["venue_form_points_5"] = venue_grouped["points"].transform(
        lambda s: s.shift(1).rolling(5, min_periods=1).mean()
    )
    df["matches_played"] = grouped.cumcount()
    for _col in ["shots_for", "shots_against", "shots_on_tgt_for", "shots_on_tgt_against",
                 "corners_for", "corners_against"]:
        if _col in df.columns:
            df[f"{_col}_5"] = df.groupby("team", group_keys=False)[_col].transform(
                lambda s: s.shift(1).rolling(5, min_periods=1).mean()
            )
    return df


def legg_til_ekstra_stats(
    lagkamper: pd.DataFrame,
    matcher: pd.DataFrame,
    kolonner: MatchKolonner,
) -> pd.DataFrame:
    stat_pars = [
        ("shots_for", "shots_against", ["hs"], ["as"]),
        ("shots_on_tgt_for", "shots_on_tgt_against", ["hst"], ["ast"]),
        ("corners_for", "corners_against", ["hc"], ["ac"]),
    ]
    hjemme_ekstra: dict[str, object] = {}
    borte_ekstra: dict[str, object] = {}
    for h_name, a_name, h_cands, a_cands in stat_pars:
        h_col = finn_kolonne(matcher, h_cands)
        a_col = finn_kolonne(matcher, a_cands)
        if not h_col or not a_col:
            continue
        h_vals = pd.to_numeric(matcher[h_col], errors="coerce").values
        a_vals = pd.to_numeric(matcher[a_col], errors="coerce").values
        hjemme_ekstra[h_name] = h_vals
        hjemme_ekstra[a_name] = a_vals
        borte_ekstra[h_name] = a_vals
        borte_ekstra[a_name] = h_vals
    if not hjemme_ekstra:
        return lagkamper
    hjemme_df = pd.DataFrame(
        {"date": matcher[kolonner.dato].values,
         "team": matcher[kolonner.hjemmelag].values,
         "is_home": 1} | hjemme_ekstra
    )
    borte_df = pd.DataFrame(
        {"date": matcher[kolonner.dato].values,
         "team": matcher[kolonner.bortelag].values,
         "is_home": 0} | borte_ekstra
    )
    ekstra = pd.concat([hjemme_df, borte_df], ignore_index=True)
    return lagkamper.merge(ekstra, on=["date", "team", "is_home"], how="left")


def legg_til_sesong_tabell(lagkamper: pd.DataFrame) -> pd.DataFrame:
    df = lagkamper.copy()
    df["_sy"] = df["date"].dt.year.where(df["date"].dt.month >= 8, df["date"].dt.year - 1)
    df["season_cum_points"] = (
        df.groupby(["team", "_sy"], group_keys=False)["points"]
        .transform(lambda s: s.shift(1).expanding().sum())
        .fillna(0)
    )
    return df.drop(columns=["_sy"])


def beregn_elo_features(
    matcher: pd.DataFrame,
    kolonner: MatchKolonner,
    base_elo: float = 1500.0,
    k_factor: float = 24.0,
    home_advantage: float = 60.0,
) -> pd.DataFrame:
    sorterte = matcher.sort_values(kolonner.dato).reset_index(drop=True)
    elo: dict[str, float] = {}
    rows: list[dict[str, object]] = []

    for _, rad in sorterte.iterrows():
        hjemmelag = rad[kolonner.hjemmelag]
        bortelag = rad[kolonner.bortelag]
        hjem_elo = float(elo.get(hjemmelag, base_elo))
        borte_elo = float(elo.get(bortelag, base_elo))

        expected_home = 1.0 / (1.0 + 10.0 ** ((borte_elo - (hjem_elo + home_advantage)) / 400.0))

        home_score = pd.to_numeric(pd.Series([rad[kolonner.hjemmemaal]]), errors="coerce").iloc[0]
        away_score = pd.to_numeric(pd.Series([rad[kolonner.bortemaal]]), errors="coerce").iloc[0]
        if pd.isna(home_score) or pd.isna(away_score):
            actual_home = 0.5
        elif home_score > away_score:
            actual_home = 1.0
        elif home_score < away_score:
            actual_home = 0.0
        else:
            actual_home = 0.5

        delta = k_factor * (actual_home - expected_home)
        elo[hjemmelag] = hjem_elo + delta
        elo[bortelag] = borte_elo - delta

        rows.append(
            {
                "date": rad[kolonner.dato],
                "home_team": hjemmelag,
                "away_team": bortelag,
                "home_elo": hjem_elo,
                "away_elo": borte_elo,
                "elo_diff": hjem_elo - borte_elo,
            }
        )

    return pd.DataFrame(rows)


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
            "form_points_5": "home_form_points_5",
            "form_goal_diff_5": "home_form_goal_diff_5",
            "goals_for_5": "home_goals_for_5",
            "goals_against_5": "home_goals_against_5",
            "days_since_last_match": "home_rest_days",
            "venue_form_points_5": "home_venue_form_points_5",
            "matches_played": "home_matches_played",
            **{c: f"home_{c}" for c in ["shots_for_5", "shots_against_5",
                                         "shots_on_tgt_for_5", "shots_on_tgt_against_5",
                                         "corners_for_5", "corners_against_5",
                                         "season_cum_points"]
               if c in home_features.columns},
        }
    )
    away_features = away_features.rename(
        columns={
            "team": "away_team",
            "form_points_5": "away_form_points_5",
            "form_goal_diff_5": "away_form_goal_diff_5",
            "goals_for_5": "away_goals_for_5",
            "goals_against_5": "away_goals_against_5",
            "days_since_last_match": "away_rest_days",
            "venue_form_points_5": "away_venue_form_points_5",
            "matches_played": "away_matches_played",
            **{c: f"away_{c}" for c in ["shots_for_5", "shots_against_5",
                                         "shots_on_tgt_for_5", "shots_on_tgt_against_5",
                                         "corners_for_5", "corners_against_5",
                                         "season_cum_points"]
               if c in away_features.columns},
        }
    )

    _opt_base = ["shots_for_5", "shots_against_5", "shots_on_tgt_for_5",
                 "shots_on_tgt_against_5", "corners_for_5", "corners_against_5",
                 "season_cum_points"]
    home_cols = (
        ["date", "home_team", "home_form_points_5", "home_form_goal_diff_5",
         "home_goals_for_5", "home_goals_against_5", "home_rest_days",
         "home_venue_form_points_5", "home_matches_played"]
        + [f"home_{c}" for c in _opt_base if f"home_{c}" in home_features.columns]
    )
    away_cols = (
        ["date", "away_team", "away_form_points_5", "away_form_goal_diff_5",
         "away_goals_for_5", "away_goals_against_5", "away_rest_days",
         "away_venue_form_points_5", "away_matches_played"]
        + [f"away_{c}" for c in _opt_base if f"away_{c}" in away_features.columns]
    )

    elo_features = beregn_elo_features(matcher, kolonner)

    features = base.merge(home_features[home_cols], on=["date", "home_team"], how="left")
    features = features.merge(away_features[away_cols], on=["date", "away_team"], how="left")
    features = features.merge(elo_features, on=["date", "home_team", "away_team"], how="left")

    # Bookmaker-odds → normaliserte implisitte sannsynligheter
    h_odds_col = finn_kolonne(matcher, ["b365h", "psh", "bsh"])
    d_odds_col = finn_kolonne(matcher, ["b365d", "psd", "bsd"])
    a_odds_col = finn_kolonne(matcher, ["b365a", "psa", "bsa"])
    if h_odds_col and d_odds_col and a_odds_col:
        odds_src = matcher[
            [kolonner.dato, kolonner.hjemmelag, kolonner.bortelag,
             h_odds_col, d_odds_col, a_odds_col]
        ].copy()
        odds_src.columns = ["date", "home_team", "away_team", "_oh", "_od", "_oa"]
        raw_h = 1.0 / pd.to_numeric(odds_src["_oh"], errors="coerce")
        raw_d = 1.0 / pd.to_numeric(odds_src["_od"], errors="coerce")
        raw_a = 1.0 / pd.to_numeric(odds_src["_oa"], errors="coerce")
        total = raw_h + raw_d + raw_a
        odds_src["imp_prob_H"] = raw_h / total
        odds_src["imp_prob_D"] = raw_d / total
        features = features.merge(
            odds_src[["date", "home_team", "away_team", "imp_prob_H", "imp_prob_D"]],
            on=["date", "home_team", "away_team"],
            how="left",
        )

    features["form_points_diff"] = (
        features["home_form_points_5"] - features["away_form_points_5"]
    )
    features["form_goal_diff_diff"] = (
        features["home_form_goal_diff_5"] - features["away_form_goal_diff_5"]
    )
    features["goals_for_diff"] = features["home_goals_for_5"] - features["away_goals_for_5"]
    features["goals_against_diff"] = (
        features["home_goals_against_5"] - features["away_goals_against_5"]
    )
    features["rest_days_diff"] = features["home_rest_days"] - features["away_rest_days"]
    features["matches_played_diff"] = (
        features["home_matches_played"] - features["away_matches_played"]
    )

    features["result"] = "D"
    features.loc[features["home_score"] > features["away_score"], "result"] = "H"
    features.loc[features["home_score"] < features["away_score"], "result"] = "A"
    return features


def velg_confidence_terskler(
    pred_df: pd.DataFrame,
    klasse_rekkefolge: list[str],
    min_confidence: float,
    confidence_mode: str,
    min_coverage: float,
    confidence_scope: str,
) -> tuple[pd.Series, float, str, dict[str, float]]:
    """Velg global/per-class confidence-terskel og returner prediksjonsmasken."""
    selected_confidence = min_confidence
    auto_note = "fixed"
    per_class_thresholds: dict[str, float] = {}

    if confidence_mode != "auto":
        is_predicted = (pred_df["pred_confidence"] >= selected_confidence).astype(int)
        return is_predicted, selected_confidence, auto_note, per_class_thresholds

    thresholds = [round(v, 2) for v in np.arange(0.35, 0.91, 0.05)]
    if confidence_scope == "per-class":
        combined_mask = pd.Series(False, index=pred_df.index)
        class_notes: list[str] = []
        for label in klasse_rekkefolge:
            class_base_mask = pred_df["pred_label_raw"] == label
            class_count = int(class_base_mask.sum())
            if class_count == 0:
                continue

            class_min_count = max(1, int(np.ceil(class_count * min_coverage)))
            class_candidates: list[tuple[float, int, float]] = []
            for thr in thresholds:
                class_mask = class_base_mask & (pred_df["pred_confidence"] >= thr)
                selected_count = int(class_mask.sum())
                if selected_count < class_min_count:
                    continue
                class_acc = float(
                    (pred_df.loc[class_mask, "result"] == pred_df.loc[class_mask, "pred_label_raw"]).mean()
                )
                class_candidates.append((thr, selected_count, class_acc))

            if class_candidates:
                best_thr, best_count, best_acc = max(
                    class_candidates,
                    key=lambda x: (x[2], x[1], x[0]),
                )
            else:
                best_thr = min_confidence
                fallback_mask = class_base_mask & (pred_df["pred_confidence"] >= best_thr)
                best_count = int(fallback_mask.sum())
                best_acc = float(
                    (pred_df.loc[fallback_mask, "result"] == pred_df.loc[fallback_mask, "pred_label_raw"]).mean()
                ) if best_count else 0.0

            per_class_thresholds[label] = float(best_thr)
            combined_mask = combined_mask | (
                class_base_mask & (pred_df["pred_confidence"] >= float(best_thr))
            )
            class_notes.append(
                f"{label}:{best_thr:.2f} ({best_count}/{class_count}, acc={best_acc:.3f})"
            )

        if combined_mask.any():
            selected_confidence = 0.0
            auto_note = (
                f"auto-selected per-class (class min coverage={min_coverage:.2f}): "
                + ", ".join(class_notes)
            )
        else:
            auto_note = (
                f"auto-fallback per-class (ingen klasse oppfylte krav, bruker min-confidence={min_confidence:.2f})"
            )

        is_predicted = pd.Series(0, index=pred_df.index)
        for label, threshold in per_class_thresholds.items():
            class_mask = (pred_df["pred_label_raw"] == label) & (pred_df["pred_confidence"] >= threshold)
            is_predicted.loc[class_mask] = 1
        return is_predicted.astype(int), selected_confidence, auto_note, per_class_thresholds

    candidates: list[tuple[float, float, float]] = []
    for thr in thresholds:
        mask = pred_df["pred_confidence"] >= thr
        coverage_thr = float(mask.mean()) if len(pred_df) else 0.0
        if coverage_thr < min_coverage or mask.sum() == 0:
            continue
        acc_thr = float((pred_df.loc[mask, "result"] == pred_df.loc[mask, "pred_result"]).mean())
        candidates.append((thr, coverage_thr, acc_thr))

    if candidates:
        best_thr, best_cov, best_acc = max(candidates, key=lambda x: (x[2], x[1], x[0]))
        selected_confidence = float(best_thr)
        auto_note = (
            f"auto-selected (target min coverage={min_coverage:.2f}, "
            f"best coverage={best_cov:.3f}, best acc={best_acc:.3f})"
        )
    else:
        selected_confidence = min_confidence
        auto_note = (
            f"auto-fallback (ingen terskel oppfylte min coverage={min_coverage:.2f}, "
            f"bruker min-confidence={min_confidence:.2f})"
        )

    is_predicted = (pred_df["pred_confidence"] >= selected_confidence).astype(int)
    return is_predicted, selected_confidence, auto_note, per_class_thresholds


def bygg_kandidatmodeller() -> dict[str, object]:
    """Bygg standard kandidater for modellvalg."""
    kandidatmodeller: dict[str, object] = {
        "LogisticRegression": LogisticRegression(max_iter=2000, class_weight="balanced", C=0.8),
        "RandomForest": RandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
        ),
        "ExtraTrees": ExtraTreesClassifier(
            n_estimators=300,
            max_depth=10,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            max_iter=300,
            max_leaf_nodes=31,
            learning_rate=0.05,
            class_weight="balanced",
            random_state=42,
        ),
    }
    if HAS_XGBOOST and XGBClassifier is not None:
        kandidatmodeller["XGBoost"] = XGBClassifier(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=1.0,
            random_state=42,
            objective="multi:softprob",
            tree_method="hist",
            eval_metric="mlogloss",
            n_jobs=-1,
        )
    return kandidatmodeller


def bygg_param_grids() -> dict[str, dict]:
    """Hyperparameter-ruter for lett tuning av valgt modell."""
    param_grids: dict[str, dict] = {
        "LogisticRegression": {
            "model__C": [0.05, 0.1, 0.3, 0.5, 1.0, 3.0, 5.0],
            "model__solver": ["lbfgs", "saga"],
        },
        "RandomForest": {
            "model__n_estimators": [200, 400, 600],
            "model__max_depth": [6, 10, 14, None],
            "model__min_samples_leaf": [1, 2, 4],
            "model__max_features": ["sqrt", "log2"],
        },
        "ExtraTrees": {
            "model__n_estimators": [300, 500, 700],
            "model__max_depth": [8, 14, None],
            "model__min_samples_leaf": [1, 2, 4],
            "model__max_features": ["sqrt", "log2"],
        },
        "HistGradientBoosting": {
            "model__max_iter": [200, 400, 600],
            "model__learning_rate": [0.02, 0.05, 0.1, 0.15],
            "model__max_leaf_nodes": [15, 31, 63],
            "model__min_samples_leaf": [10, 20, 40],
            "model__l2_regularization": [0.0, 0.1, 1.0],
        },
    }
    if HAS_XGBOOST and XGBClassifier is not None:
        param_grids["XGBoost"] = {
            "model__n_estimators": [200, 400, 700],
            "model__max_depth": [4, 6, 8],
            "model__learning_rate": [0.02, 0.05, 0.1],
            "model__subsample": [0.7, 0.85, 1.0],
            "model__colsample_bytree": [0.7, 0.85, 1.0],
            "model__reg_lambda": [0.5, 1.0, 2.0],
            "model__min_child_weight": [1, 3, 5],
        }
    return param_grids


def velg_kalibrerte_prediksjoner(
    modell: Pipeline,
    x_train: pd.DataFrame,
    y_train_enc: pd.Series,
    x_test: pd.DataFrame,
    y_test_enc: pd.Series,
    test_pred_raw: np.ndarray,
    raw_probas: np.ndarray,
    optimize_metric: str,
    calibration_mode: str,
) -> tuple[np.ndarray, np.ndarray, bool, str, object | None]:
    """Velg rå eller kalibrerte prediksjoner basert på ønsket modus og score."""
    calibrated_model = None
    calibrated_used = False
    calibration_note = "not attempted"
    test_pred = test_pred_raw
    probas = raw_probas

    if calibration_mode == "off":
        return test_pred, probas, calibrated_used, "off", calibrated_model

    try:
        calibrated_model = CalibratedClassifierCV(modell, method="sigmoid", cv=3)
        calibrated_model.fit(x_train, y_train_enc)
        calibrated_probas = calibrated_model.predict_proba(x_test)
        calibrated_pred = np.argmax(calibrated_probas, axis=1)

        raw_selected = beregn_optimaliseringsscore(y_test_enc, test_pred_raw, optimize_metric)
        cal_selected = beregn_optimaliseringsscore(y_test_enc, calibrated_pred, optimize_metric)

        if calibration_mode == "force":
            calibrated_used = True
            test_pred = calibrated_pred
            probas = calibrated_probas
            calibration_note = "forced"
        elif cal_selected >= raw_selected:
            calibrated_used = True
            test_pred = calibrated_pred
            probas = calibrated_probas
            calibration_note = "auto-selected"
        else:
            calibration_note = "auto-rejected"
    except Exception:
        calibrated_model = None
        calibration_note = "failed"

    return test_pred, probas, calibrated_used, calibration_note, calibrated_model


def evaluer_kandidater_walkforward(
    wf_base: pd.DataFrame,
    kandidatmodeller: dict[str, object],
    lag_preprocessor,
    numeriske: list[str],
    kategoriske: list[str],
    label_til_id: dict[str, int],
    draw_id: int | None,
    draw_weight: float,
    optimize_metric: str,
) -> tuple[dict[str, float], dict[str, float]]:
    """Kjør walk-forward CV for hver kandidatmodell og returner scoretabeller."""
    tscv = TimeSeriesSplit(n_splits=4)
    cv_scores: dict[str, float] = {}
    cv_acc_scores: dict[str, float] = {}

    for cv_navn, estimator in kandidatmodeller.items():
        fold_scores: list[float] = []
        fold_acc_scores: list[float] = []
        for fold_train_idx, fold_val_idx in tscv.split(wf_base):
            fold_train = wf_base.iloc[fold_train_idx]
            fold_val = wf_base.iloc[fold_val_idx]
            if len(fold_train) < 15 or len(fold_val) < 5:
                continue

            pipe = Pipeline(
                steps=[
                    ("preprocess", lag_preprocessor()),
                    ("model", clone(estimator)),
                ]
            )
            fold_y_train = fold_train["result"].map(label_til_id)
            fold_y_val = fold_val["result"].map(label_til_id)
            fold_weights = pd.Series(1.0, index=fold_y_train.index)
            if draw_id is not None:
                fold_weights[fold_y_train == draw_id] = draw_weight

            pipe.fit(
                fold_train[numeriske + kategoriske],
                fold_y_train,
                model__sample_weight=fold_weights.values,
            )
            preds = pipe.predict(fold_val[numeriske + kategoriske])

            fold_scores.append(
                beregn_optimaliseringsscore(fold_y_val, preds, optimize_metric)
            )

            fold_acc_scores.append(accuracy_score(fold_y_val, preds))

        cv_scores[cv_navn] = (sum(fold_scores) / len(fold_scores)) if fold_scores else 0.0
        cv_acc_scores[cv_navn] = (
            (sum(fold_acc_scores) / len(fold_acc_scores)) if fold_acc_scores else 0.0
        )

    return cv_scores, cv_acc_scores


def bygg_metrics_text(
    beste_cv_navn: str,
    metric_label: str,
    confidence_mode: str,
    confidence_scope: str,
    selected_confidence: float,
    auto_note: str,
    search: RandomizedSearchCV,
    cv_scores: dict[str, float],
    cv_acc_scores: dict[str, float],
    score: float,
    macro_f1: float,
    predicted_count: int,
    total_count: int,
    coverage: float,
    confident_accuracy: float,
    baseline_hjemme_score: float,
    baseline_hjemme_f1: float,
    majoritetsklasse: str,
    baseline_majoritet_score: float,
    baseline_majoritet_f1: float,
    score_raw: float,
    macro_f1_raw: float,
    raw_log_loss: float,
    cal_log_loss: float,
    calibration_mode: str,
    calibration_note: str,
    calibrated_used: bool,
    draw_weight: float,
    train_rows: int,
    test_rows: int,
    rapport: str,
) -> str:
    """Bygg samlet tekstblokk med modellresultater og diagnostikk."""
    best_params_str = "\n".join(f"  {k}: {v}" for k, v in sorted(search.best_params_.items()))
    cv_str = "\n".join(
        f"  {n}: selected={cv_scores[n]:.3f}, accuracy={cv_acc_scores[n]:.3f}"
        for n in cv_scores
    )

    return (
        f"Best model (tuned): {beste_cv_navn}\n"
        f"Optimize metric: {metric_label}\n"
        f"Confidence mode: {confidence_mode}\n"
        f"Confidence scope: {confidence_scope}\n"
        f"Min confidence for prediction: {selected_confidence:.2f}\n"
        f"Confidence selection note: {auto_note}\n"
        f"Best hyperparams:\n{best_params_str}\n\n"
        f"Walk-forward CV (4 folds, treningssett, selected={metric_label}):\n{cv_str}\n\n"
        f"Test accuracy (tuned): {score:.3f}\n"
        f"Test macro-F1 (tuned): {macro_f1:.3f}\n"
        f"Confident tips: {predicted_count}/{total_count} (coverage={coverage:.3f})\n"
        f"Accuracy on confident tips: {confident_accuracy:.3f}\n"
        f"Baseline (alltid H): accuracy={baseline_hjemme_score:.3f}, macro-F1={baseline_hjemme_f1:.3f}\n"
        f"Baseline (majoritet={majoritetsklasse}): accuracy={baseline_majoritet_score:.3f}, macro-F1={baseline_majoritet_f1:.3f}\n"
        f"Raw test accuracy: {score_raw:.3f}, Raw test macro-F1: {macro_f1_raw:.3f}\n"
        f"Log-loss raw: {raw_log_loss:.4f}, Log-loss calibrated/final: {cal_log_loss:.4f}\n"
        f"Calibration mode: {calibration_mode}, status: {calibration_note}\n"
        f"Calibration used for final predictions: {'ja' if calibrated_used else 'nei'}\n"
        f"Draw class weight: {draw_weight:.2f}\n"
        f"XGBoost tilgjengelig: {'ja' if HAS_XGBOOST else 'nei'}\n"
        f"Train rows: {train_rows}, Test rows: {test_rows}\n\n"
        f"Classification report ({beste_cv_navn}):\n{rapport}"
    )


def tren_resultatmodell(
    features: pd.DataFrame,
    optimize_metric: str = "macro-f1",
    calibration_mode: str = "auto",
    min_confidence: float = 0.0,
    confidence_mode: str = "fixed",
    min_coverage: float = 0.4,
    confidence_scope: str = "global",
) -> tuple[Pipeline, pd.DataFrame, str]:
    modell_df = features.dropna(subset=["home_score", "away_score", "result"]).copy()
    modell_df = modell_df.sort_values("date").reset_index(drop=True)

    if len(modell_df) < 40:
        raise ValueError(
            "For få kamper til modellering. Trenger minst 40 rader for en meningsfull baseline."
        )

    numeriske = [
        "home_form_points_5", "home_form_goal_diff_5",
        "home_goals_for_5", "home_goals_against_5",
        "home_rest_days", "home_venue_form_points_5", "home_matches_played",
        "away_form_points_5", "away_form_goal_diff_5",
        "away_goals_for_5", "away_goals_against_5",
        "away_rest_days", "away_venue_form_points_5", "away_matches_played",
        "home_elo", "away_elo", "elo_diff",
        "form_points_diff", "form_goal_diff_diff",
        "goals_for_diff", "goals_against_diff",
        "rest_days_diff", "matches_played_diff",
    ]
    _opt_numeriske = [
        "home_shots_for_5", "home_shots_against_5",
        "home_shots_on_tgt_for_5", "home_shots_on_tgt_against_5",
        "home_corners_for_5", "home_corners_against_5", "home_season_cum_points",
        "away_shots_for_5", "away_shots_against_5",
        "away_shots_on_tgt_for_5", "away_shots_on_tgt_against_5",
        "away_corners_for_5", "away_corners_against_5", "away_season_cum_points",
        "imp_prob_H", "imp_prob_D",
    ]
    numeriske = numeriske + [c for c in _opt_numeriske if c in modell_df.columns]
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

    klasse_rekkefolge = sorted(y_train.astype(str).unique().tolist())
    label_til_id = {label: idx for idx, label in enumerate(klasse_rekkefolge)}
    id_til_label = {idx: label for label, idx in label_til_id.items()}
    y_train_enc = y_train.map(label_til_id)
    y_test_enc = y_test.map(label_til_id)

    # Gi uavgjort høyere vekt i balanserte moduser.
    draw_weight = 2.0 if optimize_metric == "macro-f1" else (1.4 if optimize_metric == "hybrid" else 1.0)
    draw_id = label_til_id.get("D")
    train_weights = pd.Series(1.0, index=y_train_enc.index)
    if draw_id is not None:
        train_weights[y_train_enc == draw_id] = draw_weight

    metric_label = (
        "macro-F1"
        if optimize_metric == "macro-f1"
        else ("hybrid(0.7*accuracy + 0.3*macro-F1)" if optimize_metric == "hybrid" else "accuracy")
    )

    def lag_preprocessor() -> ColumnTransformer:
        return ColumnTransformer(
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

    kandidatmodeller = bygg_kandidatmodeller()

    wf_base = modell_df.iloc[:split_idx].reset_index(drop=True)
    cv_scores, cv_acc_scores = evaluer_kandidater_walkforward(
        wf_base,
        kandidatmodeller,
        lag_preprocessor,
        numeriske,
        kategoriske,
        label_til_id,
        draw_id,
        draw_weight,
        optimize_metric,
    )

    beste_cv_navn = max(cv_scores, key=lambda n: cv_scores[n])

    # --- Light hyperparameter tuning on best candidate ---
    param_grids = bygg_param_grids()

    tuning_pipeline = Pipeline(
        steps=[
            ("preprocess", lag_preprocessor()),
            ("model", kandidatmodeller[beste_cv_navn]),
        ]
    )
    search = RandomizedSearchCV(
        tuning_pipeline,
        param_distributions=param_grids[beste_cv_navn],
        n_iter=12,
        cv=TimeSeriesSplit(n_splits=4),
        scoring=bygg_scorer(optimize_metric),
        n_jobs=-1,
        random_state=42,
        refit=True,
        error_score=0.0,
    )
    search.fit(x_train, y_train_enc, model__sample_weight=train_weights.values)

    modell = search.best_estimator_
    test_pred_raw = modell.predict(x_test)
    raw_probas = modell.predict_proba(x_test)
    score_raw = accuracy_score(y_test_enc, test_pred_raw)
    macro_f1_raw = f1_score(y_test_enc, test_pred_raw, average="macro", zero_division=0)

    test_pred, probas, calibrated_used, calibration_note, _ = velg_kalibrerte_prediksjoner(
        modell,
        x_train,
        y_train_enc,
        x_test,
        y_test_enc,
        test_pred_raw,
        raw_probas,
        optimize_metric,
        calibration_mode,
    )

    score = accuracy_score(y_test_enc, test_pred)
    macro_f1 = f1_score(y_test_enc, test_pred, average="macro", zero_division=0)
    rapport = classification_report(
        y_test_enc,
        test_pred,
        labels=list(range(len(klasse_rekkefolge))),
        target_names=klasse_rekkefolge,
        zero_division=0,
    )

    raw_log_loss = log_loss(y_test_enc, raw_probas, labels=list(range(len(klasse_rekkefolge))))
    cal_log_loss = log_loss(y_test_enc, probas, labels=list(range(len(klasse_rekkefolge))))

    baseline_hjemme = pd.Series([label_til_id["H"]] * len(y_test_enc), index=y_test_enc.index)
    baseline_hjemme_score = accuracy_score(y_test_enc, baseline_hjemme)
    baseline_hjemme_f1 = f1_score(y_test_enc, baseline_hjemme, average="macro", zero_division=0)
    majoritetsklasse = y_train.mode().iloc[0]
    majoritetsklasse_id = label_til_id[majoritetsklasse]
    baseline_majoritet = pd.Series([majoritetsklasse_id] * len(y_test_enc), index=y_test_enc.index)
    baseline_majoritet_score = accuracy_score(y_test_enc, baseline_majoritet)
    baseline_majoritet_f1 = f1_score(
        y_test_enc,
        baseline_majoritet,
        average="macro",
        zero_division=0,
    )

    pred_df = test_df[["date", "home_team", "away_team", "home_score", "away_score", "result"]].copy()
    pred_df["model_used"] = beste_cv_navn
    pred_df["calibration_used"] = int(calibrated_used)
    pred_df["pred_result"] = pd.Series(test_pred, index=pred_df.index).map(id_til_label)
    pred_df["pred_label_raw"] = pred_df["pred_result"]
    pred_df["pred_confidence"] = np.max(probas, axis=1)
    pred_df["is_predicted"], selected_confidence, auto_note, per_class_thresholds = velg_confidence_terskler(
        pred_df,
        klasse_rekkefolge,
        min_confidence=min_confidence,
        confidence_mode=confidence_mode,
        min_coverage=min_coverage,
        confidence_scope=confidence_scope,
    )
    pred_df.loc[pred_df["is_predicted"] == 0, "pred_result"] = "SKIP"
    pred_df["is_correct"] = (pred_df["result"] == pred_df["pred_result"]).astype(int)

    for idx, label in enumerate(klasse_rekkefolge):
        pred_df[f"proba_{label}"] = probas[:, idx]

    predicted_mask = pred_df["is_predicted"] == 1
    predicted_count = int(predicted_mask.sum())
    total_count = len(pred_df)
    coverage = (predicted_count / total_count) if total_count else 0.0
    if predicted_count > 0:
        confident_accuracy = float(pred_df.loc[predicted_mask, "is_correct"].mean())
    else:
        confident_accuracy = 0.0

    metrics_text = bygg_metrics_text(
        beste_cv_navn=beste_cv_navn,
        metric_label=metric_label,
        confidence_mode=confidence_mode,
        confidence_scope=confidence_scope,
        selected_confidence=selected_confidence,
        auto_note=auto_note,
        search=search,
        cv_scores=cv_scores,
        cv_acc_scores=cv_acc_scores,
        score=score,
        macro_f1=macro_f1,
        predicted_count=predicted_count,
        total_count=total_count,
        coverage=coverage,
        confident_accuracy=confident_accuracy,
        baseline_hjemme_score=baseline_hjemme_score,
        baseline_hjemme_f1=baseline_hjemme_f1,
        majoritetsklasse=majoritetsklasse,
        baseline_majoritet_score=baseline_majoritet_score,
        baseline_majoritet_f1=baseline_majoritet_f1,
        score_raw=score_raw,
        macro_f1_raw=macro_f1_raw,
        raw_log_loss=raw_log_loss,
        cal_log_loss=cal_log_loss,
        calibration_mode=calibration_mode,
        calibration_note=calibration_note,
        calibrated_used=calibrated_used,
        draw_weight=draw_weight,
        train_rows=len(train_df),
        test_rows=len(test_df),
        rapport=rapport,
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


def lag_bane_fp_sammendrag(lagkamper: pd.DataFrame) -> pd.DataFrame:
    siste = (
        lagkamper.sort_values("date")
        .groupby(["team", "is_home"], as_index=False)
        .tail(1)
        .copy()
    )
    siste["venue_label"] = siste["is_home"].map({1: "home", 0: "away"})

    kolonner = [
        "venue_form_points_5",
        "goals_for_5",
        "goals_against_5",
        "form_goal_diff_5",
    ]
    deler: list[pd.DataFrame] = []
    for venue_label, prefix in [("home", "home"), ("away", "away")]:
        venue_df = siste[siste["venue_label"] == venue_label][["team"] + kolonner].copy()
        venue_df = venue_df.rename(
            columns={
                "venue_form_points_5": f"{prefix}_fp_5",
                "goals_for_5": f"{prefix}_goals_for_5",
                "goals_against_5": f"{prefix}_goals_against_5",
                "form_goal_diff_5": f"{prefix}_goal_diff_5",
            }
        )
        deler.append(venue_df)

    if not deler:
        return pd.DataFrame(
            columns=[
                "team",
                "home_fp_5",
                "away_fp_5",
                "home_goals_for_5",
                "home_goals_against_5",
                "home_goal_diff_5",
                "away_goals_for_5",
                "away_goals_against_5",
                "away_goal_diff_5",
            ]
        )

    pivot = deler[0]
    for del_df in deler[1:]:
        pivot = pivot.merge(del_df, on="team", how="outer")

    forventede = [
        "home_fp_5",
        "away_fp_5",
        "home_goals_for_5",
        "home_goals_against_5",
        "home_goal_diff_5",
        "away_goals_for_5",
        "away_goals_against_5",
        "away_goal_diff_5",
    ]
    for kolonne in forventede:
        if kolonne not in pivot.columns:
            pivot[kolonne] = np.nan
    return pivot.sort_values("team").reset_index(drop=True)


def bygg_lagvalg_ark(workbook, lag_fp_df: pd.DataFrame) -> None:
    if "Lagvalg" in workbook.sheetnames:
        del workbook["Lagvalg"]

    ark = workbook.create_sheet("Lagvalg")
    ark["A1"] = "Velg hjemmelag"
    ark["B1"] = "Velg bortelag"
    ark["A2"] = lag_fp_df["team"].iloc[0] if not lag_fp_df.empty else ""
    ark["B2"] = lag_fp_df["team"].iloc[1] if len(lag_fp_df) > 1 else ark["A2"].value

    ark["A4"] = "Hjemmelag FP (basert pa hjemmekamper)"
    ark["B4"] = "Bortelag FP (basert pa bortekamper)"
    ark["A5"] = '=IFERROR(INDEX(TeamVenueFP!$B:$B, MATCH($A$2, TeamVenueFP!$A:$A, 0)), "")'
    ark["B5"] = '=IFERROR(INDEX(TeamVenueFP!$C:$C, MATCH($B$2, TeamVenueFP!$A:$A, 0)), "")'

    ark["A7"] = "Hjemmelag mal for siste 5 hjemme"
    ark["B7"] = "Bortelag mal for siste 5 borte"
    ark["A8"] = '=IFERROR(INDEX(TeamVenueFP!$D:$D, MATCH($A$2, TeamVenueFP!$A:$A, 0)), "")'
    ark["B8"] = '=IFERROR(INDEX(TeamVenueFP!$G:$G, MATCH($B$2, TeamVenueFP!$A:$A, 0)), "")'

    ark["A10"] = "Hjemmelag mal mot siste 5 hjemme"
    ark["B10"] = "Bortelag mal mot siste 5 borte"
    ark["A11"] = '=IFERROR(INDEX(TeamVenueFP!$E:$E, MATCH($A$2, TeamVenueFP!$A:$A, 0)), "")'
    ark["B11"] = '=IFERROR(INDEX(TeamVenueFP!$H:$H, MATCH($B$2, TeamVenueFP!$A:$A, 0)), "")'

    ark["A13"] = "Hjemmelag måldiff siste 5 hjemme"
    ark["B13"] = "Bortelag måldiff siste 5 borte"
    ark["A14"] = '=IFERROR(INDEX(TeamVenueFP!$F:$F, MATCH($A$2, TeamVenueFP!$A:$A, 0)), "")'
    ark["B14"] = '=IFERROR(INDEX(TeamVenueFP!$I:$I, MATCH($B$2, TeamVenueFP!$A:$A, 0)), "")'

    ark["D1"] = "Sammenligning"
    ark["D4"] = "FP-diff hjem-borte"
    ark["E4"] = '=IF(OR($A$5="", $B$5=""), "", $A$5-$B$5)'
    ark["D7"] = "Mal for diff"
    ark["E7"] = '=IF(OR($A$8="", $B$8=""), "", $A$8-$B$8)'
    ark["D10"] = "Mal mot diff"
    ark["E10"] = '=IF(OR($A$11="", $B$11=""), "", $A$11-$B$11)'
    ark["D13"] = "Måldiff diff"
    ark["E13"] = '=IF(OR($A$14="", $B$14=""), "", $A$14-$B$14)'

    ark.column_dimensions["A"].width = 34
    ark.column_dimensions["B"].width = 34
    ark.column_dimensions["D"].width = 24
    ark.column_dimensions["E"].width = 18

    if not lag_fp_df.empty:
        team_end_row = len(lag_fp_df) + 1
        validation = DataValidation(
            type="list",
            formula1=f"=TeamVenueFP!$A$2:$A${team_end_row}",
            allow_blank=False,
        )
        ark.add_data_validation(validation)
        validation.add(ark["A2"])
        validation.add(ark["B2"])


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


def analyser_landslag(
    dataset_fil: Path,
    output_fil: Path,
    prosjektrot: Path,
    sesonger: list[str] | None = None,
    optimize_metric: str = "macro-f1",
    calibration_mode: str = "auto",
    min_confidence: float = 0.0,
    confidence_mode: str = "fixed",
    min_coverage: float = 0.4,
    confidence_scope: str = "global",
) -> int:
    data_dir = prosjektrot / "data"
    frames: list[pd.DataFrame] = []

    if sesonger:
        nedlastet = last_ned_og_kombiner_sesonger(sesonger, data_dir)
        if nedlastet is not None:
            frames.append(nedlastet)

    if dataset_fil.exists():
        frames.append(pd.read_csv(dataset_fil))
    elif not frames:
        print("Fant ikke landslagsdatasettfilen.")
        print(f"Forventet fil: {dataset_fil}")
        print(
            "Bruk --dataset data/min_fil.csv, eller angi --seasons 2526,2425 "
            "for automatisk nedlasting fra football-data.co.uk."
        )
        return 1

    matcher = pd.concat(frames, ignore_index=True).drop_duplicates()
    kolonner = finn_match_kolonner(matcher)
    if kolonner is None:
        print("Fant ikke nødvendige kolonner i landslagsdatasettet.")
        print("Må minst inneholde: date, home_team, away_team, home_score, away_score")
        print(f"Tilgjengelige kolonner: {list(matcher.columns)}")
        return 1

    matcher = parse_date_column(matcher, kolonner.dato)
    matcher = matcher.sort_values(kolonner.dato).reset_index(drop=True)

    lagkamper = bygg_lagkamper(matcher, kolonner)
    lagkamper = legg_til_ekstra_stats(lagkamper, matcher, kolonner)
    lagkamper = bygg_form_features(lagkamper)
    lagkamper = legg_til_sesong_tabell(lagkamper)
    features = bygg_hjemmekamp_features(matcher, lagkamper, kolonner)
    lagsammendrag = lag_lagsammendrag(lagkamper)
    bane_fp_sammendrag = lag_bane_fp_sammendrag(lagkamper)

    try:
        _, pred_df, metrics_text = tren_resultatmodell(
            features,
            optimize_metric=optimize_metric,
            calibration_mode=calibration_mode,
            min_confidence=min_confidence,
            confidence_mode=confidence_mode,
            min_coverage=min_coverage,
            confidence_scope=confidence_scope,
        )
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
        bane_fp_sammendrag.to_excel(writer, sheet_name="TeamVenueFP", index=False)
        if not pred_df.empty:
            pred_df.to_excel(writer, sheet_name="Predictions", index=False)
        pd.DataFrame({"metrics": [metrics_text]}).to_excel(
            writer, sheet_name="ModelMetrics", index=False
        )
        bygg_lagvalg_ark(writer.book, bane_fp_sammendrag)

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
    parser.add_argument(
        "--seasons",
        type=str,
        default="",
        help=(
            "Kommaseparerte sesongkoder for automatisk nedlasting fra football-data.co.uk, "
            "f.eks. 2526,2425,2324. Kombineres med --dataset hvis begge er angitt."
        ),
    )
    parser.add_argument(
        "--optimize",
        type=str,
        choices=["accuracy", "macro-f1", "hybrid"],
        default="macro-f1",
        help="Velg optimaliseringsmål for modellvalg og tuning.",
    )
    parser.add_argument(
        "--calibration",
        type=str,
        choices=["off", "auto", "force"],
        default="auto",
        help="Styrer sannsynlighetskalibrering: av, automatisk, eller tvungen.",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.0,
        help=(
            "Minste sannsynlighet for å gi et tips. Kamper under terskelen markeres som SKIP. "
            "Eksempel: 0.45 gir ofte høyere treffprosent, men færre tips."
        ),
    )
    parser.add_argument(
        "--confidence-mode",
        type=str,
        choices=["fixed", "auto"],
        default="fixed",
        help="Fast terskel (fixed) eller automatisk valg av terskel (auto).",
    )
    parser.add_argument(
        "--min-coverage",
        type=float,
        default=0.4,
        help="Minste dekning brukt ved --confidence-mode auto (mellom 0 og 1).",
    )
    parser.add_argument(
        "--confidence-scope",
        type=str,
        choices=["global", "per-class"],
        default="global",
        help="Bruk én terskel for alle tips eller egne terskler per utfallsklasse.",
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

    sesonger = [s.strip() for s in args.seasons.split(",") if s.strip()] if args.seasons else []

    if args.mode == "topscorers":
        return analyser_toppscorere(dataset_fil, output_fil, data_dir)
    return analyser_landslag(
        dataset_fil,
        output_fil,
        prosjektrot,
        sesonger=sesonger or None,
        optimize_metric=args.optimize,
        calibration_mode=args.calibration,
        min_confidence=max(0.0, min(1.0, args.min_confidence)),
        confidence_mode=args.confidence_mode,
        min_coverage=max(0.0, min(1.0, args.min_coverage)),
        confidence_scope=args.confidence_scope,
    )


if __name__ == "__main__":
    raise SystemExit(main())