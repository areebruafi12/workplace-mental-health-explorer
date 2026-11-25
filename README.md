# mental-health-treatment-app
# Workplace Mental Health Survey Explorer

An interactive Streamlit app that explores workplace mental-health support patterns from the 2014 OSMI Mental Health in Tech Survey.

The app includes a compact machine-learning demo that estimates the historical survey-response pattern for whether a respondent reported seeking mental-health treatment. It is built for learning and portfolio demonstration, not for clinical or employment decisions.

## Live app

Add your deployed Streamlit link here after deployment:

```text
https://workplace-mental-health-survey-explorer.streamlit.app
```

## What the app does

- Provides an interactive 10-question workplace-support form.
- Predicts the survey target `treatment_sought`:
  - `1` - respondent reported seeking treatment.
  - `0` - respondent did not report seeking treatment.
- Shows the estimated probability for each target response.
- Includes dataset insights and a downloadable cleaned CSV.
- Uses a high-contrast dark UI designed for easy reading.

## Model inputs

The model uses only workplace-related inputs, arranged into two sections.

### Workplace setup

1. Self-employed status
2. Company size
3. Remote-work status
4. Whether the employer is primarily a tech company

### Support and privacy

5. Mental-health benefits
6. Awareness of available care options
7. Mental-health wellness programme
8. Resources to seek help
9. Anonymity protection
10. Ease of taking mental-health medical leave

Age, gender, country, family history, and self-reported work interference are excluded from the app model.

## Dataset

- **Source:** [OSMI Mental Health in Tech Survey on Kaggle](https://www.kaggle.com/datasets/osmi/mental-health-in-tech-survey)
- **Original data:** 1,259 survey responses and 27 columns.
- **Cleaned project data:** 1,259 rows and 24 columns.
- **Target:** `treatment_sought`
- **Class distribution:** 622 responses with `0`, 637 responses with `1`.

Cleaning removed the timestamp, granular US state field, and free-text comments. No survey rows were removed. Invalid or outlier ages were grouped as unknown in the cleaned project data.

## Model

- **Algorithm:** Logistic Regression
- **Preprocessing:** One-hot encoding for categorical features
- **Split:** Stratified 80/20 train-validation split
- **Validation F1 score:** approximately 64.8%

This score is only a learning-project result from one validation split. It is not a measure of clinical reliability.

## Project structure

```text
workplace-mental-health-explorer/
├── .streamlit/
│   └── config.toml
├── app.py
├── mental_health_tech_clean.csv
├── mental_health_treatment_model.pkl
├── requirements.txt
└── README.md
```

## Run locally

1. Clone or download this repository.

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Start the app:

```bash
streamlit run app.py
```

4. Open the local URL shown in the terminal, usually:

```text
http://localhost:8501
```

## Deploy on Streamlit Community Cloud

1. Upload all project files to a GitHub repository.
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Select **Create app**.
4. Choose the repository and branch `main`.
5. Set the main file path to `app.py`.
6. Choose a valid subdomain, for example:

```text
workplace-mental-health-survey-explorer
```

7. Click **Deploy**.

## Important note

This project is for educational exploration of anonymized historical survey data.

It does not diagnose mental-health conditions, recommend a type of treatment, predict an individual's needs, or support hiring, insurance, screening, or workplace decisions.

