# NBA Game Outcome & Live Performance Predictor

This Streamlit data app scrapes NBA team game logs and box scores from
Basketball-Reference, trains a Logistic Regression model, and predicts whether a
team performance profile is more likely to result in a win or loss.

## What the model predicts

The model predicts whether a team with the entered statistical profile is more
similar to past wins or past losses in the scraped dataset. It is not a guarantee
of a real game result.

For live use, enter projected full-game statistics. For example, halftime
counting stats such as rebounds, assists, turnovers, steals, blocks, and free
throw attempts should be scaled toward a 48-minute estimate. Percentages should
not be doubled.

## Features

- Scrape latest team data directly inside the Streamlit dashboard
- Select NBA teams to include in the dataset
- Select model variables before training
- Train Logistic Regression inside the dashboard
- Predict win/loss outcome with probability estimates
- View dataset summaries, win/loss patterns, team comparisons, accuracy, and a
  confusion matrix

## Improved model variables

The app can train with:

- Field Goal Percentage
- Three-Point Percentage
- Free Throw Percentage
- Free Throw Attempts
- Total Rebounds
- Assists
- Turnovers
- Steals
- Blocks
- Projected Score Margin / Point Differential
- Opponent Win Rate
- Home/Away indicator

## How to run

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Then use the sidebar to scrape data and train the model.

## Data source

Basketball-Reference.com team game logs and box scores, 2025-2026 NBA season:
https://www.basketball-reference.com/
