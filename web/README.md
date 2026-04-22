# g4f Web Chat (Frontend)

Quick static frontend to chat with a running g4f server.

Usage

1. Start the g4f API server (example):

```bash
python -m g4f --port 8080
```

2. Open `web/index.html` in your browser (or serve the `web/` folder).

3. If your server uses a different URL or endpoint, update `API_URL` inside `web/app.js`.

Notes

- The frontend posts JSON `{message: string}` to `POST ${API_URL}/chat` and expects JSON like `{reply: string}`. Adjust `app.js` to match your server's actual contract if needed.
- If the server runs on a different host/port, ensure CORS is enabled or serve the frontend via a small static server.

- A small Python proxy is included at [web/server.py](web/server.py#L1). It forwards `POST /chat` to an upstream g4f server (env `G4F_UPSTREAM`, default `http://localhost:8080`) and enables CORS.

Run the proxy (recommended for local testing):

```bash
python -m venv .venv
# Unix/macOS
source .venv/bin/activate
# Windows PowerShell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python server.py
```

If you run the proxy on port 5000, set `API_URL` in `web/app.js` to `http://localhost:5000`.

Deploy to Vercel
-----------------

This project includes a Vercel serverless proxy at `api/chat.js` so you can deploy the frontend and proxy together on Vercel.

1. Install the Vercel CLI (optional) and log in:

```bash
npm i -g vercel
vercel login
```

2. From the `web/` folder, run:

```bash
vercel --prod
```

3. Set the environment variable `G4F_UPSTREAM` in your Vercel project (Project Settings → Environment Variables) to point to your running g4f backend (for example `http://your-server:8080`). The serverless function will forward requests to that URL.

Notes:
- The frontend calls the relative path `/api/chat` so it will use the serverless proxy when deployed on Vercel.
- If you prefer to host only static files, you can remove the `api/` folder and set `API_URL` in `app.js` to a publicly accessible g4f server URL.
