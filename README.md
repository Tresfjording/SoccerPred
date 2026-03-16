# Football Statistics Analysis

Dette prosjektet analyserer fotballstatistikk, spesielt toppscorere fra europeiske ligaer, for å forutsi kampresultater.

## Oppsett
1. Last ned datasettet fra Kaggle: [Top Football Leagues Scorers Dataset](https://www.kaggle.com/datasets/mohammadtalib786/top-football-leagues-scorers-dataset)
2. Plasser CSV-filen i `data/`-mappen.
3. Installer avhengigheter: `pip install -r requirements.txt`
4. Kjør skriptet: `python src/football_analysis.py`

## Kjoring med valgfritt datasett
- Standard (bruker `data/fifa_world_cup_top_scorers.csv`): `python src/football_analysis.py`
- Eget datasett: `python src/football_analysis.py --dataset data/min_fil.csv`

## Excel for hjemme- og bortelag
- Generer regnearket: `python src/build_match_excel.py`
- Filen lages som `data/kampanalyse.xlsx`
- Fanen `Kampanalyse` har dropdown for hjemme- og bortelag i `B2` og `B3`
- Statistikk vises automatisk under valgene (Entries, Goals, snitt, osv.)
- Fanen `Kamper` brukes for kamp-for-kamp-data (template opprettes automatisk)
- Du kan laste inn kamp-CSV: `python src/build_match_excel.py --matches-dataset data/matches.csv`
- Fanen `Tabell` beregnes automatisk fra `Kamper` (MP, W, D, L, GF, GA, Pts og Pos)
- Fanen `LiveTabell` viser tabellen i faktisk sortert rekkefølge etter `Pos`

## Struktur
- `data/`: Datasett og eksporterte filer
- `src/`: Python-skript
- `notebooks/`: Jupyter notebooks for utforskning
- `requirements.txt`: Avhengigheter# SoccerPred
