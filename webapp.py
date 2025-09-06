import os
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import sqlite3
from typing import Dict, List, Tuple

ROOT = os.path.abspath(os.path.dirname(__file__))
DB_PATH = os.path.join(ROOT, "tracking.sqlite3")

app = FastAPI(title="Tracking Dashboard")

templates = Jinja2Templates(directory=os.path.join(ROOT, "templates"))
static_dir = os.path.join(ROOT, "static")
if not os.path.exists(static_dir):
    os.makedirs(static_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")


def _q(conn, sql: str, params: Tuple = ()):
    cur = conn.cursor()
    cur.execute(sql, params)
    return cur.fetchall()


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/api/summary")
def api_summary():
    conn = sqlite3.connect(DB_PATH)
    total_events = _q(conn, "SELECT COUNT(*) FROM events")[0][0]
    by_object = _q(
        conn,
        "SELECT class_id, COUNT(*) cnt FROM events WHERE class_id IS NOT NULL GROUP BY class_id ORDER BY cnt DESC LIMIT 10",
    )
    by_tracker = _q(
        conn, "SELECT tracker, COUNT(*) cnt FROM events GROUP BY tracker ORDER BY cnt DESC"
    )
    conn.close()
    return {
        "total_events": total_events,
        "by_object": [{"class_id": c, "count": n} for c, n in by_object],
        "by_tracker": [{"tracker": t, "count": n} for t, n in by_tracker],
    }


@app.get("/api/directions")
def api_directions(video: str = ""):
    # Direction computed from first and last centroid per track
    conn = sqlite3.connect(DB_PATH)
    rows = _q(
        conn,
        """
        WITH track_points AS (
            SELECT video, track_id,
                   MIN(frame_idx) AS fmin,
                   MAX(frame_idx) AS fmax
            FROM events
            WHERE track_id IS NOT NULL AND video LIKE ?
            GROUP BY video, track_id
        )
        SELECT e1.video, e1.track_id,
               e1.cx AS cx_start, e1.cy AS cy_start,
               e2.cx AS cx_end,   e2.cy AS cy_end
        FROM track_points tp
        JOIN events e1 ON e1.video = tp.video AND e1.track_id = tp.track_id AND e1.frame_idx = tp.fmin
        JOIN events e2 ON e2.video = tp.video AND e2.track_id = tp.track_id AND e2.frame_idx = tp.fmax
        """,
        (f"%{video}%",),
    )
    conn.close()
    def direction(cx0, cy0, cx1, cy1):
        dx, dy = cx1 - cx0, cy1 - cy0
        if abs(dx) > abs(dy):
            return "right" if dx > 0 else "left"
        else:
            return "down" if dy > 0 else "up"
    data = [
        {
            "video": v,
            "track_id": tid,
            "direction": direction(cx0, cy0, cx1, cy1),
            "dx": cx1 - cx0,
            "dy": cy1 - cy0,
        }
        for v, tid, cx0, cy0, cx1, cy1 in rows
    ]
    return {"tracks": data}


@app.get("/api/top-objects")
def api_top_objects(limit: int = 10):
    conn = sqlite3.connect(DB_PATH)
    rows = _q(
        conn,
        "SELECT class_id, COUNT(*) AS cnt FROM events WHERE class_id IS NOT NULL GROUP BY class_id ORDER BY cnt DESC LIMIT ?",
        (limit,),
    )
    conn.close()
    return {"top": [{"class_id": c, "count": n} for c, n in rows]}


