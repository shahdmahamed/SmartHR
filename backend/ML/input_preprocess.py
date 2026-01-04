# input_preprocess.py
import pandas as pd

MAX_SKILLS = 7  # num of actual dataset

def input_preprocess(data, model_feature_names):
    """
    Preprocess input JSON for prediction.
    data: CandidateInput (Pydantic model)
    model_feature_names: feature names list from trained model
    """
    input_dict = {}

    # ---- Validate skill_count ----
    skill_count = min(data.skill_count, MAX_SKILLS)
    input_dict['skill_count'] = skill_count

    # ---- Years experience ----
    input_dict['years_experience'] = data.years_experience

    # ---- Validate job_state ----
    job_state_cols = [c for c in model_feature_names if c.startswith('job_state_')]
    job_state_names = [c.replace('job_state_', '') for c in job_state_cols]

    if data.job_state not in job_state_names:
        for col in job_state_cols:
            input_dict[col] = 0
    else:
        for col in job_state_cols:
            input_dict[col] = 1 if col == f"job_state_{data.job_state}" else 0

    # ---- Fill any other columns with 0 (for safety) ----
    for col in model_feature_names:
        if col not in input_dict:
            input_dict[col] = 0

    df_input = pd.DataFrame([input_dict])
    df_input = df_input[model_feature_names]
    return df_input



