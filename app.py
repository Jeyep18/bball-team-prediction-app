"""
Streamlit dashboard for the NBA Game Outcome & Live Performance Predictor.

The app lets users scrape Basketball-Reference data, choose model variables,
train Logistic Regression, make predictions, and interpret model performance
without running separate terminal commands.

Run with:
    python -m streamlit run app.py
"""

from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

from scrape_nba_data import scrape_multiple_teams
from train_model import train_logistic_model


DATA_FILE = Path("clean_nba_data.csv")
MODEL_FILE = Path("nba_regression_model.pkl")

FEATURE_COLUMNS = [
    "FG%",
    "3P%",
    "FT%",
    "FTA",
    "TRB",
    "AST",
    "TOV",
    "STL",
    "BLK",
    "Point_Differential",
    "Opponent_Win_Rate",
    "Home",
]
TARGET_COLUMN = "Win"
DEFAULT_TEAMS = ["LAL", "BOS", "GSW"]

NBA_TEAM_CODES = [
    "ATL",
    "BOS",
    "BRK",
    "CHO",
    "CHI",
    "CLE",
    "DAL",
    "DEN",
    "DET",
    "GSW",
    "HOU",
    "IND",
    "LAC",
    "LAL",
    "MEM",
    "MIA",
    "MIL",
    "MIN",
    "NOP",
    "NYK",
    "OKC",
    "ORL",
    "PHI",
    "PHO",
    "POR",
    "SAC",
    "SAS",
    "TOR",
    "UTA",
    "WAS",
]

VARIABLE_GUIDE = {
    "FG%": "Field Goal Percentage: how efficiently the team made all field goal attempts.",
    "3P%": "Three-Point Percentage: how efficiently the team made shots from beyond the arc.",
    "FT%": "Free Throw Percentage: how efficiently the team made free throws.",
    "FTA": "Free Throw Attempts: how often the team got to the free throw line.",
    "TRB": "Total Rebounds: how many missed shots the team recovered.",
    "AST": "Assists: how often made baskets were created by passes.",
    "TOV": "Turnovers: how often the team lost possession before attempting a shot.",
    "STL": "Steals: how often the team forced a live-ball turnover.",
    "BLK": "Blocks: how often the team blocked an opponent shot attempt.",
    "Point_Differential": "Score margin from the team's view. Positive means leading/won by that many points.",
    "Opponent_Win_Rate": "Opponent strength proxy based on how often that opponent won in the scraped games.",
    "Home": "Venue indicator: 1 means home game, 0 means away game.",
    "Win": "Target variable: 1 means the team won, 0 means the team lost.",
    "Team": "NBA team code for the row.",
    "Opponent": "Opponent team code for the row.",
}


st.set_page_config(
    page_title="NBA Courtside Analytics Dashboard",
    page_icon=":basketball:",
    layout="wide",
)


@st.cache_data
def load_dataset() -> pd.DataFrame:
    """Load the cleaned dataset from disk."""
    if not DATA_FILE.exists():
        return pd.DataFrame()

    data = pd.read_csv(DATA_FILE)
    required_columns = FEATURE_COLUMNS + [TARGET_COLUMN]
    missing_columns = [column for column in required_columns if column not in data.columns]
    if missing_columns:
        raise ValueError(f"The dataset is missing required columns: {missing_columns}")

    return data.dropna(subset=required_columns).copy()


@st.cache_resource
def load_model_bundle() -> dict | None:
    """Load the trained model bundle from disk, if one exists."""
    if not MODEL_FILE.exists():
        return None

    saved_object = joblib.load(MODEL_FILE)

    # Older project versions saved only the model. This compatibility wrapper
    # keeps those files usable, while all new training saves the full bundle.
    if isinstance(saved_object, dict) and "model" in saved_object:
        return saved_object

    return {
        "model": saved_object,
        "features": FEATURE_COLUMNS,
        "accuracy": None,
        "confusion_matrix": None,
        "train_rows": None,
        "test_rows": None,
    }


def save_dataset(data: pd.DataFrame) -> None:
    """Persist freshly scraped data and refresh Streamlit's cached dataset."""
    data.to_csv(DATA_FILE, index=False)
    load_dataset.clear()


def save_model_bundle(model_bundle: dict) -> None:
    """Persist the trained model bundle and refresh Streamlit's cached model."""
    joblib.dump(model_bundle, MODEL_FILE)
    load_model_bundle.clear()


def build_prediction_inputs(model_features: list[str]) -> pd.DataFrame:
    """
    Render input widgets only for variables used by the trained model.

    The returned DataFrame uses the exact feature order learned during training,
    which prevents accidental mismatches between sliders and model columns.
    """
    values: dict[str, float | int] = {}
    columns = st.columns(3)

    if "FG%" in model_features:
        with columns[0]:
            values["FG%"] = st.slider("Field Goal %", 30.0, 65.0, 45.0, 0.1) / 100
    if "3P%" in model_features:
        with columns[1]:
            values["3P%"] = st.slider("Three-Point %", 15.0, 55.0, 35.0, 0.1) / 100
    if "TRB" in model_features:
        with columns[2]:
            values["TRB"] = st.slider("Total Rebounds", 20, 65, 40, 1)
    if "FT%" in model_features:
        with columns[0]:
            values["FT%"] = st.slider("Free Throw %", 40.0, 100.0, 78.0, 0.1) / 100
    if "FTA" in model_features:
        with columns[1]:
            values["FTA"] = st.slider("Free Throw Attempts", 0, 45, 22, 1)
    if "AST" in model_features:
        with columns[0]:
            values["AST"] = st.slider("Assists", 10, 40, 22, 1)
    if "TOV" in model_features:
        with columns[1]:
            values["TOV"] = st.slider("Turnovers", 0, 30, 12, 1)
    if "STL" in model_features:
        with columns[2]:
            values["STL"] = st.slider("Steals", 0, 20, 7, 1)
    if "BLK" in model_features:
        with columns[0]:
            values["BLK"] = st.slider("Blocks", 0, 18, 5, 1)
    if "Point_Differential" in model_features:
        with columns[1]:
            values["Point_Differential"] = st.slider("Projected Score Margin", -40, 40, 0, 1)
    if "Opponent_Win_Rate" in model_features:
        with columns[2]:
            values["Opponent_Win_Rate"] = st.slider("Opponent Win Rate %", 0.0, 100.0, 50.0, 0.1) / 100
    if "Home" in model_features:
        with columns[2]:
            venue = st.radio("Venue", ["Home Game", "Away Game"], horizontal=True)
            values["Home"] = 1 if venue == "Home Game" else 0

    return pd.DataFrame([{feature: values[feature] for feature in model_features}])


def show_confusion_matrix(matrix) -> None:
    """Display a labeled confusion matrix chart."""
    figure, axis = plt.subplots(figsize=(6, 4))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Predicted Loss", "Predicted Win"],
        yticklabels=["Actual Loss", "Actual Win"],
        ax=axis,
    )
    axis.set_title("Confusion Matrix")
    axis.set_xlabel("Model Prediction")
    axis.set_ylabel("Actual Result")
    st.pyplot(figure, use_container_width=True)
    plt.close(figure)


def explain_selected_features(selected_features: list[str]) -> None:
    """Show a compact explanation of the variables used by the model."""
    explanation_rows = [
        {"Variable": feature, "Meaning": VARIABLE_GUIDE[feature]}
        for feature in selected_features
    ]
    st.dataframe(pd.DataFrame(explanation_rows), use_container_width=True, hide_index=True)


st.title("NBA Game Outcome & Live Performance Predictor")
st.caption("Courtside Analytics Dashboard for scraping, training, predicting, and interpreting NBA game outcomes.")

st.write(
    "This data app predicts whether one team performance profile is more likely to result in a WIN or LOSS. "
    "The model learns from historical team box-score rows scraped from Basketball-Reference. "
    "It does not use injuries, betting odds, player availability, or live play-by-play context. "
    "For live games, counting stats should be entered as projected full-game values."
)

with st.sidebar:
    st.header("Data Setup")
    selected_teams = st.multiselect(
        "Choose NBA Teams to Scrape",
        options=NBA_TEAM_CODES,
        default=DEFAULT_TEAMS,
        help="More teams usually means more rows and a stronger classroom dataset.",
    )

    scrape_button = st.button("Scrape Latest Data", use_container_width=True)

    st.divider()
    st.header("Model Setup")
    selected_features = st.multiselect(
        "Choose Model Variables",
        options=FEATURE_COLUMNS,
        default=FEATURE_COLUMNS,
        help="Win is always the target variable. Select the input variables the model should learn from.",
    )

    train_button = st.button("Train Logistic Regression Model", use_container_width=True)

    st.caption("Target variable: Win, where 1 = WIN and 0 = LOSS.")


if scrape_button:
    if not selected_teams:
        st.sidebar.error("Select at least one team before scraping.")
    else:
        st.info("Scraping can take several minutes because the app waits 3 seconds between requests.")
        progress_log = st.empty()
        progress_messages: list[str] = []

        def dashboard_progress(message: str) -> None:
            progress_messages.append(message)
            progress_log.info(progress_messages[-1])

        scraped_data = scrape_multiple_teams(selected_teams, progress_callback=dashboard_progress)
        if scraped_data.empty:
            st.error("No data was scraped. Basketball-Reference may be unavailable, or no completed games were found.")
        else:
            save_dataset(scraped_data)
            st.success(f"Scraped and saved {len(scraped_data):,} rows to {DATA_FILE}.")
            st.rerun()


try:
    nba_data = load_dataset()
except ValueError as error:
    st.error(str(error))
    nba_data = pd.DataFrame()

model_bundle = load_model_bundle()

if train_button:
    if nba_data.empty:
        st.sidebar.error("Scrape or load a dataset before training.")
    elif len(selected_features) < 2:
        st.sidebar.error("Select at least 2 model variables before training.")
    else:
        try:
            trained_bundle = train_logistic_model(nba_data, selected_features)
            save_model_bundle(trained_bundle)
            st.sidebar.success("Model trained and saved successfully.")
            st.rerun()
        except ValueError as error:
            st.sidebar.error(str(error))


st.header("1. Live Dataset Overview")

if nba_data.empty:
    st.warning(
        "No dataset is loaded yet. Use the sidebar to select teams and click "
        "Scrape Latest Data. After scraping, train the model from the dashboard."
    )
    st.caption(
        "Data source: Basketball-Reference.com team game logs and box scores, "
        "2025-2026 NBA season. https://www.basketball-reference.com/"
    )
    st.stop()

st.write(
    "Each row in this table represents one completed game for one selected NBA team. "
    "The input variables describe that team's box-score performance, opponent context, and score margin, "
    "while Win is the outcome the model tries to predict."
)

metric_1, metric_2, metric_3, metric_4, metric_5 = st.columns(5)
metric_1.metric("Rows", f"{len(nba_data):,}")
metric_2.metric("Teams", nba_data["Team"].nunique() if "Team" in nba_data.columns else "N/A")
metric_3.metric("Variables", len(FEATURE_COLUMNS))
metric_4.metric("Win Rate", f"{nba_data[TARGET_COLUMN].mean() * 100:.1f}%")
metric_5.metric("Home Games", f"{int(nba_data['Home'].sum()):,}")

if len(nba_data) < 100:
    st.warning(
        "This dataset currently has fewer than 100 rows. For the project requirement, scrape more teams "
        "until the row count reaches at least 100."
    )

st.dataframe(nba_data, use_container_width=True)

with st.expander("Column Guide", expanded=True):
    guide_rows = [
        {"Column": column, "Interpretation": explanation}
        for column, explanation in VARIABLE_GUIDE.items()
        if column in nba_data.columns
    ]
    st.dataframe(pd.DataFrame(guide_rows), use_container_width=True, hide_index=True)


st.header("2. Data Interpretations")
st.write(
    "These tables and charts help describe the dataset before modeling. This matters because a model is only as useful "
    "as the data patterns it learns from."
)

summary_tab, outcome_tab, team_tab = st.tabs(["Summary Statistics", "Win/Loss Patterns", "Team Comparison"])

with summary_tab:
    st.write(
        "The summary statistics table shows the center and spread of each numeric variable. "
        "For example, the mean gives the average team performance, while the minimum and maximum show the observed range."
    )
    st.dataframe(nba_data[FEATURE_COLUMNS + [TARGET_COLUMN]].describe(), use_container_width=True)

with outcome_tab:
    st.write(
        "The win/loss chart shows whether the dataset is balanced. A very uneven dataset can make accuracy look better "
        "than it really is because the model may learn to favor the majority class."
    )
    counts = nba_data[TARGET_COLUMN].map({0: "Loss", 1: "Win"}).value_counts().reindex(["Loss", "Win"], fill_value=0)
    figure, axis = plt.subplots(figsize=(6, 3))
    sns.barplot(x=counts.index, y=counts.values, ax=axis, palette=["#C62828", "#2E7D32"], hue=counts.index, legend=False)
    axis.set_xlabel("Game Result")
    axis.set_ylabel("Number of Rows")
    st.pyplot(figure, use_container_width=True)
    plt.close(figure)

    st.write(
        "The average stat comparison table shows how team performance differs between losses and wins. "
        "Large gaps can suggest variables that may be helpful for prediction."
    )
    averages_by_result = nba_data.groupby(TARGET_COLUMN)[FEATURE_COLUMNS].mean().rename(index={0: "Loss", 1: "Win"})
    st.dataframe(averages_by_result, use_container_width=True)

with team_tab:
    st.write(
        "The team comparison table summarizes each scraped team. This helps check whether one team dominates the dataset "
        "or whether the model is learning from a broader mix of performances."
    )
    if "Team" in nba_data.columns:
        team_summary = nba_data.groupby("Team").agg(
            Games=("Win", "count"),
            Win_Rate=("Win", "mean"),
            Avg_FG_Pct=("FG%", "mean"),
            Avg_3P_Pct=("3P%", "mean"),
            Avg_TRB=("TRB", "mean"),
            Avg_AST=("AST", "mean"),
            Avg_TOV=("TOV", "mean"),
        )
        st.dataframe(team_summary, use_container_width=True)
    else:
        st.info("Team comparison is unavailable because the Team column is missing.")


st.header("3. Model Selection and Training")
st.write(
    "The model is Logistic Regression, a classification method used when the target has two possible outcomes. "
    "Here, the two outcomes are WIN and LOSS. The model estimates probabilities by learning how the selected variables "
    "relate to past game results."
)

st.write("Selected dashboard variables:")
explain_selected_features(selected_features)

if model_bundle is None:
    st.warning("No trained model is loaded yet. Choose variables in the sidebar and click Train Logistic Regression Model.")
    st.caption(
        "Data source: Basketball-Reference.com team game logs and box scores, "
        "2025-2026 NBA season. https://www.basketball-reference.com/"
    )
    st.stop()

model = model_bundle["model"]
model_features = model_bundle["features"]

st.success(f"Loaded trained Logistic Regression model using: {', '.join(model_features)}")

if set(model_features) != set(selected_features):
    st.info(
        "The loaded model was trained with a different variable set than the current sidebar selection. "
        "Retrain the model if you want predictions to use the newly selected variables."
    )


st.header("4. Input Fields for Prediction")
st.write(
    "Enter a projected team stat line for one game. The dashboard will compare this profile to patterns learned from "
    "the scraped dataset and estimate whether it looks more like a win or a loss."
)

prediction_input = build_prediction_inputs(model_features)


st.header("5. Prediction Result")
if st.button("Predict Game Outcome", type="primary", use_container_width=True):
    predicted_class = int(model.predict(prediction_input)[0])
    class_probabilities = model.predict_proba(prediction_input)[0]
    loss_probability = class_probabilities[0]
    win_probability = class_probabilities[1]

    result_col_1, result_col_2 = st.columns(2)
    result_col_1.metric("Win Probability", f"{win_probability * 100:.1f}%")
    result_col_2.metric("Loss Probability", f"{loss_probability * 100:.1f}%")

    if predicted_class == 1:
        st.success("Predicted Outcome: WIN")
        st.write(
            "Interpretation: the model estimates this stat line is more similar to past wins than past losses "
            "in the scraped dataset."
        )
    else:
        st.error("Predicted Outcome: LOSS")
        st.write(
            "Interpretation: the model estimates this stat line is more similar to past losses than past wins "
            "in the scraped dataset."
        )

    st.dataframe(prediction_input, use_container_width=True, hide_index=True)


st.divider()
st.header("6. Model Evaluation")
st.write(
    "Model evaluation checks how well the trained model performed on the 20% test portion of the dataset that was "
    "not used for fitting the model."
)

accuracy = model_bundle.get("accuracy")
matrix = model_bundle.get("confusion_matrix")
train_rows = model_bundle.get("train_rows")
test_rows = model_bundle.get("test_rows")

eval_col_1, eval_col_2 = st.columns([1, 2])

with eval_col_1:
    if accuracy is None:
        st.info("This model was loaded from an older file. Retrain from the dashboard to display evaluation metrics.")
    else:
        st.metric("Accuracy Score", f"{accuracy * 100:.1f}%")
        st.write(f"Training rows: {train_rows}")
        st.write(f"Testing rows: {test_rows}")
        st.warning(
            "Accuracy is useful, but it is only one metric. It can be affected by dataset size, team selection, "
            "and whether wins and losses are balanced."
        )

with eval_col_2:
    if matrix is not None:
        show_confusion_matrix(matrix)

st.write(
    "Confusion matrix interpretation: top-left means losses correctly predicted as losses; top-right means losses "
    "incorrectly predicted as wins; bottom-left means wins incorrectly predicted as losses; bottom-right means wins "
    "correctly predicted as wins."
)

st.caption(
    "Data source: Basketball-Reference.com team game logs and box scores, "
    "2025-2026 NBA season. https://www.basketball-reference.com/"
)
