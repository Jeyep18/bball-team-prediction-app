"""
Phase 1: Automated live web scraper for NBA team game logs.

This script collects 2025-2026 regular-season game logs from
Basketball-Reference and saves a clean machine-learning dataset to:

    clean_nba_data.csv

Important note for students:
Basketball-Reference uses team schedule pages where every row is one game.
The schedule table includes team and opponent box-score metrics. We use the
team metrics as model features and the win/loss result as the prediction target.
"""

from __future__ import annotations

import time
from pathlib import Path
from urllib.parse import urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup


# Keep all generated files relative to the project folder.
OUTPUT_FILE = Path("clean_nba_data.csv")

# Add or remove team codes here without changing the scraping logic.
TEAM_CODES = ["LAL", "BOS", "GSW"]

# Basketball-Reference asks users to keep request volume low. Sleeping for
# 3 seconds keeps us at 20 requests per minute or less.
REQUEST_DELAY_SECONDS = 3

SEASON_YEAR = 2026
BASE_URL = "https://www.basketball-reference.com/teams/{team_code}/{season}_games.html"
BASKETBALL_REFERENCE_HOME = "https://www.basketball-reference.com"

# A browser-like User-Agent reduces the chance of being rejected by basic
# anti-bot filters while still behaving respectfully with a slow request rate.
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/125.0 Safari/537.36"
    )
}

# Final feature names used throughout scraping, training, and Streamlit.
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


_last_request_time = 0.0


def fetch_page(url: str) -> str | None:
    """
    Download a Basketball-Reference page with polite rate limiting.

    This function centralizes request handling so every schedule page and every
    box-score page follows the same defensive behavior.
    """
    global _last_request_time

    elapsed = time.time() - _last_request_time
    if _last_request_time > 0 and elapsed < REQUEST_DELAY_SECONDS:
        wait_time = REQUEST_DELAY_SECONDS - elapsed
        print(f"Waiting {wait_time:.1f} seconds before the next request...")
        time.sleep(wait_time)

    try:
        response = requests.get(url, headers=HEADERS, timeout=20)
        _last_request_time = time.time()
        response.raise_for_status()
        return response.text
    except requests.exceptions.RequestException as error:
        print(f"ERROR: Could not download {url}. Details: {error}")
        return None


def extract_team_totals_from_box_score(box_score_url: str, team_code: str) -> dict[str, float] | None:
    """Extract one team's basic box-score totals from a completed game page."""
    html = fetch_page(box_score_url)
    if html is None:
        return None

    try:
        soup = BeautifulSoup(html, "html.parser")
        basic_table = soup.find("table", id=f"box-{team_code}-game-basic")

        if basic_table is None:
            print(f"ERROR: Could not find {team_code} basic box-score table at {box_score_url}")
            return None

        totals_row = basic_table.find("tfoot")
        if totals_row is None:
            print(f"ERROR: Could not find team totals row at {box_score_url}")
            return None

        def read_stat(data_stat: str) -> float:
            cell = totals_row.find(attrs={"data-stat": data_stat})
            if cell is None:
                raise KeyError(data_stat)
            return float(cell.get_text(strip=True))

        return {
            "FG%": read_stat("fg_pct"),
            "3P%": read_stat("fg3_pct"),
            "FT%": read_stat("ft_pct"),
            "FTA": read_stat("fta"),
            "TRB": read_stat("trb"),
            "AST": read_stat("ast"),
            "TOV": read_stat("tov"),
            "STL": read_stat("stl"),
            "BLK": read_stat("blk"),
        }
    except (KeyError, TypeError, ValueError) as error:
        print(f"ERROR: Could not parse team totals from {box_score_url}. Details: {error}")
        return None


def scrape_team_games(team_code: str, progress_callback=None) -> pd.DataFrame:
    """
    Scrape one team's schedule page and return cleaned game-level rows.

    The returned DataFrame contains only the model-ready columns:
    FG%, 3P%, TRB, AST, TOV, Home, Win, Team.
    """
    url = BASE_URL.format(team_code=team_code, season=SEASON_YEAR)
    message = f"Scraping {team_code}: {url}"
    print(message)
    if progress_callback:
        progress_callback(message)

    html = fetch_page(url)
    if html is None:
        return pd.DataFrame()

    try:
        soup = BeautifulSoup(html, "html.parser")
        table = soup.find("table", id="games")

        if table is None:
            print(f"ERROR: Could not find the regular-season games table for {team_code}.")
            return pd.DataFrame()

        rows: list[dict[str, float | int | str]] = []
        opponent_records: dict[str, list[int]] = {}

        for schedule_row in table.select("tbody tr"):
            # Header rows inside the tbody have class="thead"; skip them.
            if "thead" in schedule_row.get("class", []):
                continue

            result_cell = schedule_row.find(attrs={"data-stat": "game_result"})
            box_score_cell = schedule_row.find(attrs={"data-stat": "box_score_text"})
            location_cell = schedule_row.find(attrs={"data-stat": "game_location"})
            opponent_cell = schedule_row.find(attrs={"data-stat": "opp_name"})
            team_points_cell = schedule_row.find(attrs={"data-stat": "pts"})
            opponent_points_cell = schedule_row.find(attrs={"data-stat": "opp_pts"})

            if result_cell is None or box_score_cell is None:
                continue

            result = result_cell.get_text(strip=True)
            box_score_link = box_score_cell.find("a")

            # Future games do not have results or box-score links yet.
            if result not in {"W", "L"} or box_score_link is None:
                continue

            opponent_link = opponent_cell.find("a") if opponent_cell else None
            opponent_code = ""
            if opponent_link and "/teams/" in opponent_link.get("href", ""):
                opponent_code = opponent_link["href"].split("/teams/")[1].split("/")[0]

            team_points = pd.to_numeric(team_points_cell.get_text(strip=True), errors="coerce") if team_points_cell else None
            opponent_points = (
                pd.to_numeric(opponent_points_cell.get_text(strip=True), errors="coerce")
                if opponent_points_cell
                else None
            )
            point_differential = team_points - opponent_points if pd.notna(team_points) and pd.notna(opponent_points) else None

            box_score_url = urljoin(BASKETBALL_REFERENCE_HOME, box_score_link["href"])
            if progress_callback:
                progress_callback(f"Reading {team_code} box score: {box_score_url}")
            box_score_stats = extract_team_totals_from_box_score(box_score_url, team_code)
            if box_score_stats is None:
                continue

            location_text = location_cell.get_text(strip=True) if location_cell else ""
            rows.append(
                {
                    **box_score_stats,
                    "Opponent": opponent_code,
                    "Point_Differential": point_differential,
                    # "@" means away. A blank location means home.
                    "Home": 0 if location_text == "@" else 1,
                    "Win": 1 if result == "W" else 0,
                    "Team": team_code,
                }
            )

            if opponent_code:
                opponent_records.setdefault(opponent_code, []).append(0 if result == "W" else 1)

        clean_team_data = pd.DataFrame(rows)

        if clean_team_data.empty:
            print(f"ERROR: No completed games with box-score stats were collected for {team_code}.")
            return pd.DataFrame()

        if "Opponent" in clean_team_data.columns:
            opponent_win_rates = {
                opponent: sum(results) / len(results)
                for opponent, results in opponent_records.items()
                if results
            }
            clean_team_data["Opponent_Win_Rate"] = clean_team_data["Opponent"].map(opponent_win_rates)

        numeric_columns = FEATURE_COLUMNS + [TARGET_COLUMN]
        for column in numeric_columns:
            clean_team_data[column] = pd.to_numeric(clean_team_data[column], errors="coerce")

        before_drop = len(clean_team_data)
        clean_team_data = clean_team_data.dropna(subset=numeric_columns).reset_index(drop=True)
        after_drop = len(clean_team_data)

        message = f"Collected {after_drop} clean rows for {team_code} ({before_drop - after_drop} rows removed)."
        print(message)
        if progress_callback:
            progress_callback(message)
        return clean_team_data

    except (KeyError, ValueError, IndexError) as error:
        print(f"ERROR: The page structure for {team_code} was not in the expected format. Details: {error}")
        return pd.DataFrame()
    except Exception as error:  # Defensive catch for a classroom-friendly script.
        print(f"ERROR: Unexpected scraping problem for {team_code}. Details: {error}")
        return pd.DataFrame()


def main() -> None:
    """Scrape all configured teams, combine them, and save the dataset."""
    combined_data = scrape_multiple_teams(TEAM_CODES)

    if combined_data.empty:
        print("ERROR: No team data was scraped. The CSV file was not created.")
        return
    combined_data.to_csv(OUTPUT_FILE, index=False)

    print(f"Saved {len(combined_data)} total rows to {OUTPUT_FILE}")
    print("Columns:", ", ".join(combined_data.columns))


def scrape_multiple_teams(team_codes: list[str], progress_callback=None) -> pd.DataFrame:
    """
    Scrape several teams and return one clean combined dataset.

    Streamlit calls this function directly so users can gather data from the
    dashboard instead of running this script in a terminal.
    """
    all_team_frames: list[pd.DataFrame] = []

    for team_code in team_codes:
        team_frame = scrape_team_games(team_code, progress_callback=progress_callback)
        if not team_frame.empty:
            all_team_frames.append(team_frame)

    if not all_team_frames:
        return pd.DataFrame()

    combined_data = pd.concat(all_team_frames, ignore_index=True).drop_duplicates().reset_index(drop=True)

    # Once several teams are combined, this becomes a better opponent-strength
    # proxy: each opponent receives the win rate observed for that team in the
    # scraped dataset. Opponents outside the selected team list keep the schedule
    # estimate or fall back to a neutral 0.500.
    if {"Team", "Opponent", "Win"}.issubset(combined_data.columns):
        team_win_rates = combined_data.groupby("Team")["Win"].mean()
        combined_data["Opponent_Win_Rate"] = (
            combined_data["Opponent"].map(team_win_rates).fillna(combined_data["Opponent_Win_Rate"]).fillna(0.500)
        )

    return combined_data


if __name__ == "__main__":
    main()
