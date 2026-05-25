Deployment instructions

Streamlit Community Cloud (recommended)

1. Go to https://share.streamlit.io and sign in with GitHub.
2. Click "New app" → choose repository `ayushsinha0804/heart` → branch `main` → file `app.py`.
3. Click "Deploy". The app will build from `requirements.txt` and go live.

Render (example)

1. Create a new Web Service on https://render.com and connect your GitHub repo.
2. Set the build command to empty and the start command to:

   streamlit run app.py --server.port $PORT --server.address 0.0.0.0

Heroku (example)

1. Create an app on Heroku.
2. Ensure `Procfile` exists (this repo includes one).
3. Push to Heroku remote: `git push heroku main`.

Notes

- `models/pipeline.pkl` and `models/feature_columns.pkl` are present in the repo — no extra download required.
- If model files get large, consider using Git LFS or hosting them externally and downloading at startup.
