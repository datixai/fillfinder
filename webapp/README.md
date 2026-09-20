# FillFinder web app

Browser version of FillFinder. Upload a design, see each step explained, download the results.

## Run locally

    cd webapp
    pip install -r requirements.txt
    python local_server.py

Open http://localhost:8000

## Deploy on Vercel

1. Push the repository to GitHub.
2. In Vercel, import the repo and set **Root Directory** to `webapp`.
3. Deploy. The static page is served from `public/`, and `api/process.py` runs as a Python function.

Limits on Vercel: 12 MB upload, 60 second run time, about 4.5 MB response. Very large designs are better handled by the desktop version.

## Files

    webapp/
    ├── api/process.py           HTTP endpoint (also used by the local server)
    ├── api/fillfinder_core.py   the engine: reading, outline detection, area finding, fills
    ├── public/index.html        the whole front end
    ├── local_server.py          local test server
    ├── requirements.txt
    └── vercel.json
