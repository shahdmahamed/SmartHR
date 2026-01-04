# Smart HR - Recruitment Assistant

## Overview

**Smart HR** is a project designed to help HR teams:

- **Predict a candidate's expected salary** based on experience, skill count, and job location (state).  
- Provide a **simple API** for quick salary estimation.


## Features
- Predict salary using Machine Learning
- FastAPI backend
- Simple HTML frontend
- Docker support

---

##  Project Structure
SMART_HR/
├── backend/
│   ├── ML/
│   │   ├── __pycache__/
│   │   ├── app.py
│   │   ├── input_preprocess.py
│   │   ├── train.py
│   │   └── models/
│   │       └── model.pkl
│   ├── Notebook/
│   │   └── preprocess.ipynb
│   └── data/
│       ├── processed/
│       │   └── processed_jobs.csv
│       └── raw/
│           └── glassdoor_jobs.csv
├── FrontEnd/
│   └── index.html
├── test/
│   └── test_inputs.json
├── .dockerignore
├── Dockerfile
├── README.md
└── requirements.txt
