from typing import Optional
from pathlib import Path
import time, threading
import asyncio
from collections import deque

from fastapi import FastAPI, Body, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi import UploadFile, File, Form, HTTPException

from pydantic import BaseModel
from google.oauth2 import service_account
from googleapiclient.discovery import build

from datetime import datetime, timezone, timedelta
import os
import shutil
import uuid
import json
import traceback
import io
import zipfile

from .simulate_logic import predict_mmi_ai
from . import logic
from . import simulate_logic
import joblib

from fastapi import Cookie
import hashlib
import hmac
import pandas as pd

GOOGLE_SHEETS_SPREADSHEET_ID = os.getenv("GOOGLE_SHEETS_SPREADSHEET_ID", "1n6Q5kY7AG3d1bf4C6RIhh4JNao2JubObIeZAKaOoiJc")

GOOGLE_SHEETS_REPORT_RANGE = os.getenv("GOOGLE_SHEETS_REPORT_RANGE", "MMIReports!A:Q")
GOOGLE_SHEETS_TRAIN_RANGE  = os.getenv("GOOGLE_SHEETS_TRAIN_RANGE", "TrainingData!A:H")

GOOGLE_SERVICE_ACCOUNT_FILE = os.getenv(
    "GOOGLE_SERVICE_ACCOUNT_FILE",
    r"c:\project\secret\focal-sight-490506-e2-e0cd6f286f88.json"
)

AUTO_TRAIN_EVERY = int(os.getenv("AUTO_TRAIN_EVERY", "5"))
AUTO_TRAIN_MIN_ROWS = int(os.getenv("AUTO_TRAIN_MIN_ROWS", "30"))
AUTO_TRAIN_N_ITER = int(os.getenv("AUTO_TRAIN_N_ITER", "25"))
AUTO_TRAIN_MODEL_TYPE = os.getenv("AUTO_TRAIN_MODEL_TYPE", "classifier")

AUTO_TRAIN_STATE_FILE = Path("app/state/auto_train_state.json")
MODEL_META_STATE_FILE = Path("app/state/model_meta.json")

_auto_train_lock = threading.Lock()
_auto_train_running = False

OPS_SECRET_CODE = os.getenv("OPS_SECRET_CODE", "123456")
OPS_SESSION_SECRET = os.getenv("OPS_SESSION_SECRET", "dev-secret-change-this")
OPS_COOKIE_NAME = os.getenv("OPS_COOKIE_NAME", "ops_session")
OPS_SESSION_TTL_SEC = int(os.getenv("OPS_SESSION_TTL_SEC", "300"))

class MMIReportRequest(BaseModel):
    user_uid: str | None = None
    user_lat: float
    user_lon: float

    mmi_value: int
    mmi_code: str
    feeling_text: str | None = None

    event_time_th: str | None = None
    event_time_utc: str | None = None
    event_lat: float
    event_lon: float
    event_mag: float | None = None
    event_depth_km: float | None = None
    event_changwat: str | None = None

    distance_km: float | None = None
    estimated_pga_percent_g: float | None = None
    source: str | None = "manual_user_report"

class OpsLoginRequest(BaseModel):
    code: str

from .logic import (
    fetch_latest_event_in_thailand,
    compute_overlay_from_event,
    simulate_event,
    get_soil_info,
)
# ==== ดึงฟังก์ชันจาก logic.py ของโปรเจกคุณ ====
# - fetch_latest_event_in_thailand() : ดึงเมตาเหตุการณ์ล่าสุด
# - compute_overlay_from_event(ev)   : คำนวณ/สร้างผลลัพธ์แผนที่จาก ev
# - simulate_event(lat, lon, depth_km, mag) : จำลองเหตุการณ์


app = FastAPI(title="SHAKEMAP API", version="1.3.0")

from .logic import debug_vs30_paths

@app.get("/api/soil_debug")
def soil_debug():
    return debug_vs30_paths()


# CORS (เปิดกว้างสำหรับ dev)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ================== In-memory Cache (คำนวณครั้งแรก) ==================
_CACHE_LOCK = threading.Lock()
_CACHE = {
    "data": None,        # JSON ผลลัพธ์เต็ม (รวม data URL/HTML/เมตา)
    "event_key": None,   # คีย์อ้างอิงเหตุการณ์ล่าสุดที่คำนวณแล้ว
    "ts": 0.0,           # เวลาที่คำนวณ (epoch)
}

# ตั้ง TTL ถ้าอยากให้รีเฟรชอัตโนมัติเมื่อพ้นเวลา; None = ไม่หมดอายุเอง
CACHE_TTL_SEC: Optional[int] = None  # เช่น 600 = 10 นาที

def _safe_name(name: str) -> str:
    return "".join(c for c in name if c.isalnum() or c in ("-", "_", ".")).strip() or "upload.csv"

def _make_event_key(meta: dict) -> str:
    return f"{meta.get('time_utc') or meta.get('time_th')}|{meta.get('lat')}|{meta.get('lon')}|{meta.get('mag')}|{meta.get('depth_km')}"

def _get_cached_ok() -> bool:
    if _CACHE["data"] is None:
        return False
    if CACHE_TTL_SEC is None:
        return True
    return (time.time() - (_CACHE["ts"] or 0)) < CACHE_TTL_SEC

def _compute_and_store() -> dict:
    """คำนวณผลจากเหตุการณ์ล่าสุด แล้วเก็บลงแคช (ต้องเรียกภายใต้ LOCK)"""
    ev = fetch_latest_event_in_thailand()
    data = compute_overlay_from_event(ev)
    meta = data.get("meta", {})
    _CACHE["data"] = data
    _CACHE["event_key"] = _make_event_key(meta)
    _CACHE["ts"] = time.time()
    return data

def _get_or_compute(force: bool = False) -> dict:
    # 1) บังคับคำนวณใหม่
    if force:
        with _CACHE_LOCK:
            return _compute_and_store()

    # 2) ถ้ายังไม่มีแคช → คำนวณใหม่
    if _CACHE["data"] is None:
        with _CACHE_LOCK:
            if _CACHE["data"] is None:
                return _compute_and_store()
            return _CACHE["data"]

    # 3) มีแคชแล้ว → เช็กว่ามี "เหตุการณ์ใหม่" ไหม (เปรียบเทียบ event_key)
    try:
        ev = fetch_latest_event_in_thailand()  # ดึง meta ล่าสุด (ไม่เรนเดอร์ภาพ)
        if ev:
            # สร้างคีย์จาก meta ล่าสุด
            new_key = _make_event_key({
                "time_utc":  ev.get("time_utc"),
                "time_th":   ev.get("time_th"),
                "lat":       ev.get("lat"),
                "lon":       ev.get("lon"),
                "mag":       ev.get("mag"),
                "depth_km":  ev.get("depth"),
            })
            # ถ้าไม่ใช่เหตุการณ์เดิม → คำนวณใหม่ด้วย ev นี้
            if new_key and new_key != _CACHE["event_key"]:
                with _CACHE_LOCK:
                    data = compute_overlay_from_event(ev)
                    _CACHE["data"] = data
                    _CACHE["event_key"] = new_key
                    _CACHE["ts"] = time.time()
                    return data
    except Exception:
        # ถ้าเช็ก meta ล้มเหลว ให้ใช้ของเดิมไปก่อน
        pass

    # 4) เหตุการณ์เดิม → ใช้แคช
    return _CACHE["data"]



# ================== Queue: จำกัดผู้ใช้พร้อมกัน 10 คน ==================
MAX_ACTIVE = 10           # จำกัดผู้ใช้งานพร้อมกัน
HEARTBEAT_TIMEOUT = 45.0     # วินาที: ไม่ส่ง heartbeat เกินนี้ = หลุด
PROMOTE_BATCH = 5            # โปรโมตทีละกี่คนจากคิว (กันกรณีพุ่งพร้อมกัน)

_q_lock = asyncio.Lock()
_active = {}         # key=(client_id, tab_id) -> last_seen_ts
_queue = deque()     # item=(client_id, tab_id, enq_ts)

def _now() -> float:
    return time.time()

def _queue_position(client_id: str, tab_id: str):
    pos = 1
    for (c, t, _) in _queue:
        if c == client_id and t == tab_id:
            return pos
        pos += 1
    return None

async def _maintain_and_promote():
    """ลบ active ที่หมดอายุ และโปรโมตจากคิวตามช่องว่าง"""
    now = _now()
    # ตัด active หมดอายุ
    expired = [k for k, ts in list(_active.items()) if (now - ts) > HEARTBEAT_TIMEOUT]
    for k in expired:
        _active.pop(k, None)

    # โปรโมตจากคิว
    slots = max(0, MAX_ACTIVE - len(_active))
    moved = 0
    while slots > 0 and _queue and moved < PROMOTE_BATCH:
        c_id, t_id, _ = _queue[0]
        key = (c_id, t_id)
        if key in _active:
            _queue.popleft()
            continue
        _active[key] = _now()
        _queue.popleft()
        slots -= 1
        moved += 1

def append_mmi_report_to_google_sheet(payload: MMIReportRequest):
    if not GOOGLE_SHEETS_SPREADSHEET_ID:
        raise RuntimeError("Missing GOOGLE_SHEETS_SPREADSHEET_ID")
    if not GOOGLE_SERVICE_ACCOUNT_FILE:
        raise RuntimeError("Missing GOOGLE_SERVICE_ACCOUNT_FILE")

    scopes = ["https://www.googleapis.com/auth/spreadsheets"]
    creds = service_account.Credentials.from_service_account_file(
        GOOGLE_SERVICE_ACCOUNT_FILE,
        scopes=scopes,
    )

    service = build("sheets", "v4", credentials=creds)
    thai_now = datetime.now(timezone(timedelta(hours=7))).strftime("%Y-%m-%d %H:%M:%S")
    values = [[
        thai_now,
        payload.user_uid,
        payload.user_lat,
        payload.user_lon,
        payload.mmi_value,
        payload.mmi_code,
        payload.feeling_text,
        payload.event_time_th,
        payload.event_time_utc,
        payload.event_lat,
        payload.event_lon,
        payload.event_mag,
        payload.event_depth_km,
        payload.event_changwat,
        payload.distance_km,
        payload.estimated_pga_percent_g,
        payload.source,
    ]]

    body = {"values": values}

    service.spreadsheets().values().append(
        spreadsheetId=GOOGLE_SHEETS_SPREADSHEET_ID,
        range=GOOGLE_SHEETS_REPORT_RANGE,
        valueInputOption="RAW",
        insertDataOption="INSERT_ROWS",
        body=body,
    ).execute()

def append_training_row_to_google_sheet(payload: MMIReportRequest):
    if not GOOGLE_SHEETS_SPREADSHEET_ID:
        raise RuntimeError("Missing GOOGLE_SHEETS_SPREADSHEET_ID")
    if not GOOGLE_SERVICE_ACCOUNT_FILE:
        raise RuntimeError("Missing GOOGLE_SERVICE_ACCOUNT_FILE")

    # ถ้าข้อมูลสำคัญไม่ครบ ก็ข้ามไม่เอาเข้า training
    if payload.distance_km is None or payload.distance_km <= 0:
        return {"ok": False, "skipped": True, "reason": "invalid_distance"}
    if payload.estimated_pga_percent_g is None or payload.estimated_pga_percent_g <= 0:
        return {"ok": False, "skipped": True, "reason": "invalid_pga"}
    if payload.mmi_value is None:
        return {"ok": False, "skipped": True, "reason": "missing_mmi"}

    scopes = ["https://www.googleapis.com/auth/spreadsheets"]
    creds = service_account.Credentials.from_service_account_file(
        GOOGLE_SERVICE_ACCOUNT_FILE,
        scopes=scopes,
    )
    service = build("sheets", "v4", credentials=creds)

    event_id = (
        f"{payload.event_time_utc or payload.event_time_th or 'unknown'}|"
        f"{round(payload.event_lat, 4)}|"
        f"{round(payload.event_lon, 4)}|"
        f"{payload.event_mag if payload.event_mag is not None else 'na'}"
    )

    values = [[
        event_id,                                # event_id
        payload.event_mag,                       # mag
        payload.user_lat,                        # lat
        payload.user_lon,                        # lon
        payload.distance_km,                     # dist
        payload.estimated_pga_percent_g / 100.0, # pga -> แปลง %g เป็น g
        payload.mmi_value,                       # mmi
        payload.mmi_code,                        # mmi_roman
    ]]

    body = {"values": values}

    service.spreadsheets().values().append(
        spreadsheetId=GOOGLE_SHEETS_SPREADSHEET_ID,
        range=GOOGLE_SHEETS_TRAIN_RANGE,
        valueInputOption="RAW",
        insertDataOption="INSERT_ROWS",
        body=body,
    ).execute()

    return {"ok": True, "skipped": False}

def _get_google_sheets_service():
    if not GOOGLE_SHEETS_SPREADSHEET_ID:
        raise RuntimeError("Missing GOOGLE_SHEETS_SPREADSHEET_ID")
    if not GOOGLE_SERVICE_ACCOUNT_FILE:
        raise RuntimeError("Missing GOOGLE_SERVICE_ACCOUNT_FILE")

    scopes = ["https://www.googleapis.com/auth/spreadsheets"]
    creds = service_account.Credentials.from_service_account_file(
        GOOGLE_SERVICE_ACCOUNT_FILE,
        scopes=scopes,
    )
    return build("sheets", "v4", credentials=creds)

def read_training_sheet_as_dataframe() -> pd.DataFrame:
    service = _get_google_sheets_service()
    result = service.spreadsheets().values().get(
        spreadsheetId=GOOGLE_SHEETS_SPREADSHEET_ID,
        range=GOOGLE_SHEETS_TRAIN_RANGE,
    ).execute()

    values = result.get("values", [])
    if not values:
        return pd.DataFrame(columns=["event_id", "mag", "lat", "lon", "dist", "pga", "mmi", "mmi_roman"])

    header = values[0]
    rows = values[1:]

    if not rows:
        return pd.DataFrame(columns=header)

    max_len = max(len(header), max(len(r) for r in rows))
    header = header + [f"extra_{i}" for i in range(len(header), max_len)]
    rows = [r + [""] * (max_len - len(r)) for r in rows]

    df = pd.DataFrame(rows, columns=header)

    for col in ["mag", "lat", "lon", "dist", "pga", "mmi"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df

def get_training_row_count() -> int:
    df = read_training_sheet_as_dataframe()
    if df.empty:
        return 0
    if "event_id" in df.columns:
        return int(df["event_id"].astype(str).str.strip().ne("").sum())
    return int(len(df))



def load_auto_train_state():
    default_state = {
        "enabled": True,
        "last_trained_row_count": 0,
        "last_trained_at": None,
        "last_status": None,
        "last_error": None,
        "last_report": None,
    }

    if AUTO_TRAIN_STATE_FILE.exists():
        try:
            loaded = json.loads(AUTO_TRAIN_STATE_FILE.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                state = dict(default_state)
                state.update(loaded)
                state.setdefault("enabled", True)
                return state
        except Exception:
            pass

    return dict(default_state)

def save_auto_train_state(state: dict):
    state = dict(state or {})
    state.setdefault("enabled", True)
    AUTO_TRAIN_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    AUTO_TRAIN_STATE_FILE.write_text(
        json.dumps(state, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

def _sync_auto_train_state_on_startup():
    """
    หลัง restart ให้ auto train ทำงานต่อได้เลยโดยไม่ต้องกดปิด/เปิดใหม่
    """
    global _auto_train_running

    try:
        state = load_auto_train_state()
        state.setdefault("enabled", True)

        current_count = get_training_row_count()
        last_count_raw = state.get("last_trained_row_count", 0)

        try:
            last_count = int(last_count_raw or 0)
        except Exception:
            last_count = 0

        changed = False

        # หลังบูตใหม่ ไม่ควรถือว่ากำลังเทรนค้าง
        if _auto_train_running:
            _auto_train_running = False

        # ถ้า state เพี้ยน เช่น นับเกินจำนวนแถวจริง ให้ reset baseline
        if last_count < 0 or last_count > current_count:
            state["last_trained_row_count"] = current_count
            state["last_status"] = "armed_after_restart"
            state["last_error"] = None
            save_auto_train_state(state)
            last_count = current_count
            changed = True

        pending_new_rows = max(0, current_count - last_count)

        # ถ้ามี pending ครบเงื่อนไขอยู่แล้ว ให้เริ่มต่อเองหลัง restart
        if state.get("enabled", True) and current_count >= AUTO_TRAIN_MIN_ROWS and pending_new_rows >= AUTO_TRAIN_EVERY:
            threading.Timer(1.0, maybe_trigger_auto_retrain).start()

        # ถ้าก่อน restart เปิด auto train ค้างไว้ แต่ยังไม่มี baseline ที่ใช้ได้
        # ให้ถือจำนวนปัจจุบันเป็น baseline เลย จะได้ไม่ต้องกด toggle ใหม่
        elif state.get("enabled", True) and (
            state.get("last_trained_at") is None
            and int(state.get("last_trained_row_count", 0) or 0) == 0
            and current_count > 0
            and not changed
        ):
            state["last_trained_row_count"] = current_count
            state["last_status"] = "armed_after_restart"
            state["last_error"] = None
            save_auto_train_state(state)

    except Exception as e:
        print("[AUTO_TRAIN] startup sync failed:", repr(e))
        print(traceback.format_exc())

@app.on_event("startup")
def _startup_auto_train_hook():
    _sync_auto_train_state_on_startup()

def _thai_now_str() -> str:
    return datetime.now(timezone(timedelta(hours=7))).strftime("%Y-%m-%d %H:%M:%S")


def _thai_str_from_epoch(ts: float | None):
    if ts is None:
        return None
    try:
        return datetime.fromtimestamp(float(ts), tz=timezone(timedelta(hours=7))).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return None


def _normalize_model_target(target: str | None):
    key = str(target or "").strip().lower()
    mapping = {
        "default": "default",
        "north": "default",
        "main": "default",
        "west": "west",
        "south": "south",
    }
    return mapping.get(key)


def _model_filename_for_target(target: str) -> str:
    mapping = {
        "default": "mmi_model.joblib",
        "west": "mmi_west.joblib",
        "south": "mmi_south.joblib",
    }
    return mapping[target]


def _model_region_for_target(target: str) -> str:
    mapping = {
        "default": "north",
        "west": "west",
        "south": "south",
    }
    return mapping[target]


def _model_label_for_target(target: str) -> str:
    mapping = {
        "default": "Model North",
        "west": "Model West",
        "south": "Model South",
    }
    return mapping[target]


def load_model_meta_registry():
    if MODEL_META_STATE_FILE.exists():
        try:
            return json.loads(MODEL_META_STATE_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def save_model_meta_registry(registry: dict):
    MODEL_META_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    MODEL_META_STATE_FILE.write_text(
        json.dumps(registry or {}, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )


def _safe_float(v):
    try:
        if v is None or v == "":
            return None
        return float(v)
    except Exception:
        return None


def _best_result_from_report(report: dict | None):
    if not isinstance(report, dict):
        return {}
    best_name = report.get("best_model_name")
    all_results = report.get("all_results") or {}
    if isinstance(all_results, dict) and isinstance(best_name, str):
        best_result = all_results.get(best_name)
        if isinstance(best_result, dict):
            return best_result
    return {}


def _extract_report_accuracy_exact(report: dict | None):
    best_result = _best_result_from_report(report)
    value = _safe_float(best_result.get("cv_accuracy_exact"))
    if value is not None:
        return value

    if isinstance(report, dict):
        value = _safe_float(report.get("best_cv_accuracy_exact"))
        if value is not None:
            return value

        if str(report.get("selection_metric") or "").strip().lower() == "cv_accuracy_exact":
            value = _safe_float(report.get("best_value"))
            if value is not None:
                return value

    return None


def _extract_report_accuracy_pm1(report: dict | None):
    best_result = _best_result_from_report(report)
    value = _safe_float(best_result.get("cv_accuracy_pm1"))
    if value is not None:
        return value
    if isinstance(report, dict):
        return _safe_float(report.get("best_cv_accuracy_pm1"))
    return None


def _summarize_train_report(report: dict | None):
    if not isinstance(report, dict):
        return None

    best_result = _best_result_from_report(report)
    out = {
        "model_type": report.get("model_type"),
        "best_model_name": report.get("best_model_name"),
        "selection_metric": report.get("selection_metric"),
        "best_value": report.get("best_value"),
        "n_rows": report.get("n_rows"),
        "cv_strategy": report.get("cv_strategy"),
        "cv_accuracy_exact": _extract_report_accuracy_exact(report),
        "cv_accuracy_pm1": _extract_report_accuracy_pm1(report),
        "cv_mae": _safe_float(best_result.get("cv_mae")),
        "cv_rmse": _safe_float(best_result.get("cv_rmse")),
    }
    return {k: v for k, v in out.items() if v is not None}


def _upsert_model_meta(
    target: str,
    *,
    trained_at: str | None = None,
    source: str | None = None,
    report: dict | None = None,
    model_type: str | None = None,
    n_iter: int | None = None,
    original_filename: str | None = None,
    note: str | None = None,
):
    norm = _normalize_model_target(target)
    if not norm:
        raise ValueError("invalid target")

    registry = load_model_meta_registry()
    prev = dict(registry.get(norm) or {})

    model_dir = Path("app/models")
    path = model_dir / _model_filename_for_target(norm)
    exists = path.exists()
    stat = path.stat() if exists else None

    entry = {
        "target": norm,
        "region": _model_region_for_target(norm),
        "label": _model_label_for_target(norm),
        "filename": path.name,
        "exists": exists,
        "size_bytes": stat.st_size if stat else None,
        "file_updated_at": _thai_str_from_epoch(stat.st_mtime) if stat else None,
        "trained_at": trained_at or prev.get("trained_at") or (_thai_str_from_epoch(stat.st_mtime) if stat else None),
        "source": source or prev.get("source") or ("file_mtime" if stat else None),
        "report_summary": _summarize_train_report(report) if report is not None else prev.get("report_summary"),
        "model_type": model_type or prev.get("model_type"),
        "n_iter": n_iter if n_iter is not None else prev.get("n_iter"),
        "original_filename": original_filename if original_filename is not None else prev.get("original_filename"),
        "note": note if note is not None else prev.get("note"),
    }

    registry[norm] = entry
    save_model_meta_registry(registry)
    return entry


def _get_model_info(target: str | None):
    norm = _normalize_model_target(target)
    if not norm:
        return None

    registry = load_model_meta_registry()
    if norm not in registry:
        return _upsert_model_meta(norm)

    return _upsert_model_meta(
        norm,
        trained_at=registry[norm].get("trained_at"),
        source=registry[norm].get("source"),
        report=registry[norm].get("report_summary"),
        model_type=registry[norm].get("model_type"),
        n_iter=registry[norm].get("n_iter"),
        original_filename=registry[norm].get("original_filename"),
        note=registry[norm].get("note"),
    )


def _current_exact_from_deployed_model(target: str):
    info = _get_model_info(target) or {}
    current_exact = _safe_float((info.get("report_summary") or {}).get("cv_accuracy_exact"))
    current_source = "model_meta.report_summary" if current_exact is not None else None

    if current_exact is None:
        state = load_auto_train_state()
        current_exact = _extract_report_accuracy_exact(state.get("last_report"))
        if current_exact is not None:
            current_source = "auto_train_state.last_report"

    return {
        "exists": bool(info.get("exists")),
        "current_exact": current_exact,
        "current_exact_source": current_source,
    }


def _should_replace_model(target: str, report: dict | None):
    baseline = _current_exact_from_deployed_model(target)
    new_exact = _extract_report_accuracy_exact(report)

    if not baseline["exists"]:
        should_replace = True
        reason = "no_current_model"
    elif new_exact is None:
        should_replace = False
        reason = "new_exact_missing"
    elif baseline["current_exact"] is None:
        should_replace = True
        reason = "no_current_exact"
    elif float(new_exact) > float(baseline["current_exact"]):
        should_replace = True
        reason = "better_exact"
    else:
        should_replace = False
        reason = "not_better_exact"

    return {
        "target": _normalize_model_target(target),
        "should_replace": should_replace,
        "reason": reason,
        "new_exact": new_exact,
        "current_exact": baseline["current_exact"],
        "current_exact_source": baseline["current_exact_source"],
    }


def _ops_sign(raw: str) -> str:
    return hmac.new(
        OPS_SESSION_SECRET.encode("utf-8"),
        raw.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()

def _make_ops_token() -> str:
    exp = int(time.time()) + OPS_SESSION_TTL_SEC
    raw = f"ops:{exp}"
    sig = _ops_sign(raw)
    return f"{raw}:{sig}"

def _verify_ops_token(token: str | None) -> bool:
    if not token:
        return False

    try:
        parts = token.split(":")
        if len(parts) != 3:
            return False

        role, exp_str, sig = parts
        if role != "ops":
            return False

        exp = int(exp_str)
        if exp < int(time.time()):
            return False

        raw = f"{role}:{exp}"
        expected = _ops_sign(raw)
        return hmac.compare_digest(sig, expected)
    except Exception:
        return False

# ================== Routes ==================
@app.get("/")
def index():
    # เสิร์ฟหน้าเว็บ
    return FileResponse(str(STATIC_DIR / "index.html"))

@app.post("/api/ops/login")
def api_ops_login(payload: OpsLoginRequest):
    if not OPS_SECRET_CODE:
        return JSONResponse(
            status_code=500,
            content={"ok": False, "error": "OPS_SECRET_CODE is not configured"}
        )

    if payload.code != OPS_SECRET_CODE:
        return JSONResponse(
            status_code=401,
            content={"ok": False, "error": "รหัสไม่ถูกต้อง"}
        )

    token = _make_ops_token()

    resp = JSONResponse({"ok": True})
    resp.set_cookie(
        key=OPS_COOKIE_NAME,
        value=token,
        max_age=OPS_SESSION_TTL_SEC,
        httponly=True,
        samesite="lax",
        secure=False,   # ถ้า deploy https จริง ค่อยเปลี่ยนเป็น True
        path="/",
    )
    return resp


@app.get("/api/ops/me")
def api_ops_me(ops_session: str | None = Cookie(default=None, alias=OPS_COOKIE_NAME)):
    ok = _verify_ops_token(ops_session)
    return {"ok": True, "is_ops": ok}


@app.post("/api/ops/logout")
def api_ops_logout():
    resp = JSONResponse({"ok": True})
    resp.delete_cookie(key=OPS_COOKIE_NAME, path="/")
    return resp

def require_ops(ops_session: str | None = Cookie(default=None, alias=OPS_COOKIE_NAME)):
    if not _verify_ops_token(ops_session):
        raise HTTPException(status_code=403, detail="Ops permission required")

@app.get("/simulate")
def simulate_page():
    return FileResponse(str(STATIC_DIR / "simulate.html"))

@app.post("/api/report_mmi")
def api_report_mmi(payload: MMIReportRequest):
    try:
        append_mmi_report_to_google_sheet(payload)
        train_result = append_training_row_to_google_sheet(payload)

        if train_result.get("ok") and not train_result.get("skipped"):
            auto_retrain = maybe_trigger_auto_retrain()
        else:
            auto_retrain = {
                "triggered": False,
                "reason": "training_row_skipped",
                "train_row_reason": train_result.get("reason"),
            }

        return {
            "ok": True,
            "train_row": train_result,
            "auto_retrain": auto_retrain,
        }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"ok": False, "detail": str(e)}
        )


@app.post("/api/retrain_now")
def api_retrain_now(ops_session: str | None = Cookie(default=None, alias=OPS_COOKIE_NAME)):
    global _auto_train_running
    require_ops(ops_session)

    with _auto_train_lock:
        if _auto_train_running:
            return {"ok": False, "error": "auto retrain already running"}
        _auto_train_running = True

    current_count = get_training_row_count()
    threading.Thread(
        target=_auto_retrain_worker,
        args=(current_count,),
        daemon=True,
    ).start()

    return {"ok": True, "message": "started retrain", "current_count": current_count}


@app.get("/api/auto_train_state")
def api_auto_train_state(ops_session: str | None = Cookie(default=None, alias=OPS_COOKIE_NAME)):
    require_ops(ops_session)

    state = load_auto_train_state()
    current_count = get_training_row_count()
    last_count = int(state.get("last_trained_row_count", 0) or 0)

    state.setdefault("enabled", True)
    state["running"] = _auto_train_running
    state["current_training_rows"] = current_count
    state["last_trained_row_count"] = last_count
    state["pending_new_rows"] = max(0, current_count - last_count)
    state["auto_train_every"] = AUTO_TRAIN_EVERY
    state["auto_train_min_rows"] = AUTO_TRAIN_MIN_ROWS
    state["auto_train_model_type"] = AUTO_TRAIN_MODEL_TYPE
    return {"ok": True, "state": state}


@app.post("/api/auto_train_toggle")
def api_auto_train_toggle(
    payload: dict = Body(default_factory=dict),
    ops_session: str | None = Cookie(default=None, alias=OPS_COOKIE_NAME),
):
    require_ops(ops_session)

    enabled = bool(payload.get("enabled", True))
    state = load_auto_train_state()
    prev_enabled = bool(state.get("enabled", True))
    current_count = get_training_row_count()

    state["enabled"] = enabled

    # ตอนเปิดจาก False -> True ให้เริ่มนับจากจำนวนแถวปัจจุบัน
    if enabled and not prev_enabled:
        state["last_trained_row_count"] = current_count
        state["last_status"] = "armed"
        state["last_error"] = None

    save_auto_train_state(state)

    state["running"] = _auto_train_running
    state["current_training_rows"] = current_count
    state["pending_new_rows"] = max(0, current_count - int(state.get("last_trained_row_count", 0) or 0))
    state["auto_train_every"] = AUTO_TRAIN_EVERY
    state["auto_train_min_rows"] = AUTO_TRAIN_MIN_ROWS
    state["auto_train_model_type"] = AUTO_TRAIN_MODEL_TYPE

    return {"ok": True, "state": state}


@app.post("/api/train_from_sheet")
def api_train_from_sheet(
    payload: dict = Body(default_factory=dict),
    ops_session: str | None = Cookie(default=None, alias=OPS_COOKIE_NAME),
):
    require_ops(ops_session)

    model_type = str(payload.get("model_type") or AUTO_TRAIN_MODEL_TYPE).strip().lower()
    n_iter = int(payload.get("n_iter") or AUTO_TRAIN_N_ITER)

    if model_type not in {"classifier", "regression"}:
        return JSONResponse(
            status_code=400,
            content={"ok": False, "error": "model_type ต้องเป็น classifier หรือ regression"}
        )

    df = read_training_sheet_as_dataframe()
    if len(df) < AUTO_TRAIN_MIN_ROWS:
        return JSONResponse(
            status_code=400,
            content={
                "ok": False,
                "error": f"training rows too low: {len(df)} < {AUTO_TRAIN_MIN_ROWS}",
                "current_count": int(len(df)),
                "min_rows": AUTO_TRAIN_MIN_ROWS,
            }
        )

    with _auto_train_lock:
        if _auto_train_running:
            return JSONResponse(
                status_code=409,
                content={"ok": False, "error": "training is already running"}
            )
        globals()["_auto_train_running"] = True

    try:
        best_model, report = train_best_model_from_dataframe(
            df=df,
            model_type=model_type,
            n_iter=n_iter,
        )

        trained_at = _thai_now_str()
        deployment = []
        replaced_any = False

        for target in ("default", "west", "south"):
            decision = _should_replace_model(target, report)
            if decision["should_replace"]:
                save_runtime_model(
                    target,
                    best_model,
                    meta={
                        "trained_at": trained_at,
                        "source": "train_from_sheet",
                        "report": report,
                        "model_type": model_type,
                        "n_iter": n_iter,
                        "note": "trained from Google Sheets TrainingData",
                    },
                )
                replaced_any = True
            deployment.append(decision)

        report["deployment"] = deployment

        state = load_auto_train_state()
        state.setdefault("enabled", True)
        state["last_trained_row_count"] = int(len(df))
        state["last_trained_at"] = trained_at
        state["last_status"] = "ok_replaced" if replaced_any else "ok_no_replace"
        state["last_error"] = None
        state["last_report"] = report
        save_auto_train_state(state)

        return {
            "ok": True,
            "message": "trained from Google Sheets TrainingData",
            "report": report,
            "deployment": deployment,
            "replaced_any": replaced_any,
            "current_count": int(len(df)),
            "model_type": model_type,
            "n_iter": n_iter,
        }
    except Exception as e:
        state = load_auto_train_state()
        state.setdefault("enabled", True)
        state["last_status"] = "error"
        state["last_error"] = str(e)
        save_auto_train_state(state)
        return JSONResponse(
            status_code=500,
            content={"ok": False, "error": str(e), "traceback": traceback.format_exc()}
        )
    finally:
        with _auto_train_lock:
            globals()["_auto_train_running"] = False


# --------- Queue APIs ----------
@app.post("/api/queue/enter")
async def queue_enter(req: Request):
    body = await req.json()
    client_id = body.get("client_id")
    tab_id    = body.get("tab_id")
    if not client_id or not tab_id:
        return JSONResponse({"error": "missing client_id/tab_id"}, status_code=400)

    async with _q_lock:
        await _maintain_and_promote()
        key = (client_id, tab_id)

        # อยู่ active แล้ว
        if key in _active:
            _active[key] = _now()
            return {"state": "active", "active": len(_active), "limit": MAX_ACTIVE}

        # ยังมีช่องว่าง
        if len(_active) < MAX_ACTIVE:
            _active[key] = _now()
            return {"state": "active", "active": len(_active), "limit": MAX_ACTIVE}

        # เต็ม -> เข้าคิวถ้ายังไม่อยู่
        if _queue_position(client_id, tab_id) is None:
            _queue.append((client_id, tab_id, _now()))
        pos = _queue_position(client_id, tab_id)
        return {"state": "queued", "position": pos, "active": len(_active), "limit": MAX_ACTIVE}

@app.get("/api/queue/status")
async def queue_status(client_id: str, tab_id: str):
    async with _q_lock:
        await _maintain_and_promote()
        key = (client_id, tab_id)
        if key in _active:
            return {"state": "active", "active": len(_active), "limit": MAX_ACTIVE}
        pos = _queue_position(client_id, tab_id)
        if pos is not None:
            return {"state": "queued", "position": pos, "active": len(_active), "limit": MAX_ACTIVE}
        return {"state": "none", "active": len(_active), "limit": MAX_ACTIVE}

@app.post("/api/queue/heartbeat")
async def queue_heartbeat(req: Request):
    body = await req.json()
    client_id = body.get("client_id")
    tab_id    = body.get("tab_id")
    if not client_id or not tab_id:
        return JSONResponse({"error": "missing client_id/tab_id"}, status_code=400)

    async with _q_lock:
        await _maintain_and_promote()
        key = (client_id, tab_id)
        if key in _active:
            _active[key] = _now()
            return {"state": "active", "active": len(_active), "limit": MAX_ACTIVE}
        pos = _queue_position(client_id, tab_id)
        if pos is not None:
            return {"state": "queued", "position": pos, "active": len(_active), "limit": MAX_ACTIVE}
        return {"state": "none", "active": len(_active), "limit": MAX_ACTIVE}

@app.post("/api/queue/leave")
async def queue_leave(req: Request):
    body = await req.json()
    client_id = body.get("client_id")
    tab_id    = body.get("tab_id")
    if not client_id or not tab_id:
        return JSONResponse({"error": "missing client_id/tab_id"}, status_code=400)

    async with _q_lock:
        key = (client_id, tab_id)
        _active.pop(key, None)
        # ลบจากคิวหากมี
        for i, (c, t, ts) in enumerate(list(_queue)):
            if c == client_id and t == tab_id:
                try:
                    _queue.remove((c, t, ts))
                except Exception:
                    pass
                break
        await _maintain_and_promote()
        return {"ok": True, "active": len(_active), "limit": MAX_ACTIVE}


# --------- Data APIs (เดิม) ----------
# GET สำหรับเปิดในเบราว์เซอร์/เทส
@app.get("/api/run")
def api_run_get():
    try:
        data = _get_or_compute(force=False)
        return JSONResponse(data)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

# POST ใช้งานจริงจากหน้าเว็บ
@app.post("/api/run")
def api_run(body: dict = Body(default_factory=dict)):
    """
    โหมดปกติ: POST /api/run  (body ว่างก็ได้)
    โหมดจำลอง: POST /api/run { "mode":"simulate", "lat":..., "lon":..., "depth":..., "mag":... }
    บังคับรีเฟรช: POST /api/run { "force": true }
    """
    try:
        # โหมดจำลอง
        if body.get("mode") == "simulate":
            lat   = float(body["lat"])
            lon   = float(body["lon"])
            depth = float(body["depth"])
            mag   = float(body["mag"])
            data = simulate_event(lat=lat, lon=lon, depth_km=depth, mag=mag)
            return JSONResponse(data)

        force = bool(body.get("force"))
        data = _get_or_compute(force=force)
        return JSONResponse(data)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

# รีเฟรชเหตุการณ์ล่าสุดแบบบังคับ (สำหรับแอดมิน/DevTools)
@app.post("/api/refresh")
def api_refresh(ops_session: str | None = Cookie(default=None, alias=OPS_COOKIE_NAME)):
    require_ops(ops_session)
    try:
        data = _get_or_compute(force=True)
        return JSONResponse({"ok": True, "meta": data.get("meta", {}), "event_key": _CACHE["event_key"]})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

# ดูสถานะแคช (debug)
@app.get("/api/cache_state")
def api_cache_state():
    return {
        "has_cache": _CACHE["data"] is not None,
        "event_key": _CACHE["event_key"],
        "ts": _CACHE["ts"],
        "ttl_sec": CACHE_TTL_SEC,
    }



# ชั้นดิน/ประเภทชั้นดิน (Site Class จาก Vs30)
@app.get("/api/soil")
def api_soil(lat: float, lon: float):
    try:
        return JSONResponse(get_soil_info(lat=lat, lon=lon))
    except Exception as e:
        return JSONResponse({"status": "error", "error": str(e), "lat": lat, "lon": lon}, status_code=500)

@app.post("/api/simulate")
def api_simulate(body: dict = Body(...)):
    try:
        lat   = float(body["lat"])
        lon   = float(body["lon"])
        depth = float(body["depth"])
        mag   = float(body["mag"])
        data = simulate_event(lat=lat, lon=lon, depth_km=depth, mag=mag)
        return JSONResponse(data)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
    
from pydantic import BaseModel

class MMIPredictRequest(BaseModel):
    dist: float
    pga: float
    mag: float
    lat: float | None = None
    lon: float | None = None


@app.post("/api/predict_mmi")
def api_predict_mmi(payload: MMIPredictRequest):
    try:
        result = predict_mmi_ai(
        dist=payload.dist,
        pga=payload.pga / 100.0,   # payload มาจากเว็บเป็น %g
        mag=payload.mag,
        lat=payload.lat,
        lon=payload.lon,
    )
        return result
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
    
    
@app.post("/api/train_model")
async def train_model_api(
    ops_session: str | None = Cookie(default=None, alias=OPS_COOKIE_NAME),
    file: UploadFile = File(...),
    model_type: str = Form(...),
    n_iter: int = Form(25),
):
    require_ops(ops_session)
    try:
        upload_dir = Path("app/uploads")
        upload_dir.mkdir(parents=True, exist_ok=True)

        ext = Path(file.filename).suffix.lower()
        if ext not in [".xlsx", ".xls", ".csv"]:
            return JSONResponse(
                status_code=400,
                content={"ok": False, "error": "รองรับเฉพาะ .xlsx .xls .csv"}
            )

        file_id = uuid.uuid4().hex[:10]
        safe_filename = _safe_name(file.filename)
        saved_path = upload_dir / f"{file_id}_{safe_filename}"

        with saved_path.open("wb") as f:
            shutil.copyfileobj(file.file, f)

        # import ฟังก์ชัน train ตามประเภท
        if model_type == "classifier":
            from train_mmi_classifier_tuned import (
                load_input_folder_or_file,
                prepare_dataset,
                choose_cv,
                build_search_spaces,
                tune_one_model,
            )
            selection_metric = "cv_accuracy_exact"
            better = "higher"

        elif model_type == "regression":
            from train_mmi_regression_tuned import (
                load_input_folder_or_file,
                prepare_dataset,
                choose_cv,
                build_search_spaces,
                tune_one_model,
            )
            selection_metric = "cv_mae"
            better = "lower"

        else:
            return JSONResponse(
                status_code=400,
                content={"ok": False, "error": "model_type ต้องเป็น classifier หรือ regression"}
            )

        df = load_input_folder_or_file(str(saved_path))
        clean_df, X_raw, X, y, groups = prepare_dataset(df)
        cv, cv_name = choose_cv(y, groups) if model_type == "classifier" else choose_cv(groups)

        spaces = build_search_spaces()
        all_results = {}

        best_name = None
        best_estimator = None
        best_value = None

        from sklearn.base import clone
        import joblib

        trained_models = {}
        for name, (model, param_dist) in spaces.items():
            
            est, result = tune_one_model(
                name=name,
                model=model,
                param_dist=param_dist,
                X=X,
                y=y,
                cv=cv,
                groups=groups,
                n_iter=n_iter,
            )
            result["cv_strategy"] = cv_name
            all_results[name] = result

            fitted_est = clone(est)
            fitted_est.fit(X, y)

            trained_models[name] = fitted_est

            current_value = result[selection_metric]
            if best_value is None:
                best_value = current_value
                best_name = name
                best_estimator = clone(est)
            else:
                if better == "higher" and current_value > best_value:
                    best_value = current_value
                    best_name = name
                    best_estimator = clone(est)
                elif better == "lower" and current_value < best_value:
                    best_value = current_value
                    best_name = name
                    best_estimator = clone(est)

        best_estimator.fit(X, y)

        report = {
            "ok": True,
            "model_type": model_type,
            "n_rows": int(len(clean_df)),
            "cv_strategy": cv_name,
            "feature_columns_raw": list(X_raw.columns),
            "feature_columns_engineered": list(X.columns),
            "best_model_name": best_name,
            "selection_metric": selection_metric,
            "best_value": float(best_value),
            "all_results": all_results,
            "uploaded_file": file.filename,
        }

        zip_buffer = io.BytesIO()

        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            best_bytes = io.BytesIO()
            joblib.dump(best_estimator, best_bytes)
            best_bytes.seek(0)
            zf.writestr(f"{file_id}_best_{model_type}.joblib", best_bytes.read())

            for name, model_obj in trained_models.items():
                model_bytes = io.BytesIO()
                joblib.dump(model_obj, model_bytes)
                model_bytes.seek(0)
                zf.writestr(f"{file_id}_{name}.joblib", model_bytes.read())

            zf.writestr(
                f"{file_id}_train_report.json",
                json.dumps(report, indent=2, ensure_ascii=False)
            )

        zip_buffer.seek(0)

        try:
            saved_path.unlink(missing_ok=True)
        except Exception:
            pass

        return StreamingResponse(
            zip_buffer,
            media_type="application/zip",
            headers={
                "Content-Disposition": f'attachment; filename="{file_id}_trained_models.zip"'
            }
        )

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "ok": False,
                "error": str(e),
                "traceback": traceback.format_exc(),
            }
        )

@app.post("/api/upload_model")
async def api_upload_model(
    ops_session: str | None = Cookie(default=None, alias=OPS_COOKIE_NAME),
    file: UploadFile = File(...),
    target: str = Form("default"),
):
    require_ops(ops_session)
    try:
        ext = Path(file.filename).suffix.lower()
        if ext != ".joblib":
            return JSONResponse(
                status_code=400,
                content={"ok": False, "error": "รองรับเฉพาะไฟล์ .joblib"}
            )

        model_dir = Path("app/models")
        model_dir.mkdir(parents=True, exist_ok=True)

        if target == "default":
            out_name = "mmi_model.joblib"
            label = "north/default"
        elif target == "west":
            out_name = "mmi_west.joblib"
            label = "west"
        elif target == "south":
            out_name = "mmi_south.joblib"
            label = "south"
        else:
            return JSONResponse(
                status_code=400,
                content={"ok": False, "error": "target ไม่ถูกต้อง"}
            )

        out_path = model_dir / out_name

        with out_path.open("wb") as f:
            shutil.copyfileobj(file.file, f)

        import joblib
        from . import logic

        loaded_model = joblib.load(out_path)

        if target == "default":
            logic.MMI_MODEL = loaded_model
            logic.MMI_MODELS["north"] = loaded_model

            simulate_logic.MMI_MODEL = loaded_model
            simulate_logic.MMI_MODELS["north"] = loaded_model

        elif target == "west":
            logic.MMI_MODELS["west"] = loaded_model
            if logic.MMI_MODEL is None:
                logic.MMI_MODEL = loaded_model

            simulate_logic.MMI_MODELS["west"] = loaded_model
            if simulate_logic.MMI_MODEL is None:
                simulate_logic.MMI_MODEL = loaded_model

        elif target == "south":
            logic.MMI_MODELS["south"] = loaded_model
            if logic.MMI_MODEL is None:
                logic.MMI_MODEL = loaded_model

            simulate_logic.MMI_MODELS["south"] = loaded_model
            if simulate_logic.MMI_MODEL is None:
                simulate_logic.MMI_MODEL = loaded_model

        _upsert_model_meta(
            target,
            trained_at=_thai_now_str(),
            source="upload_model",
            model_type=None,
            n_iter=None,
            original_filename=file.filename,
            note="manual uploaded model",
        )

        return {
            "ok": True,
            "message": "อัปโหลดโมเดลสำเร็จ",
            "target": target,
            "saved_to": str(out_path).replace("\\", "/"),
            "filename": file.filename,
            "label": label,
            "synced_modules": ["logic", "simulate_logic"],
        }

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"ok": False, "error": str(e), "traceback": traceback.format_exc()}
        )
    
@app.get("/api/model_info/{target}")
def api_model_info(target: str):
    info = _get_model_info(target)
    if info is None:
        return JSONResponse(
            status_code=400,
            content={"ok": False, "error": "target ไม่ถูกต้อง"}
        )
    return {"ok": True, "model": info}


@app.get("/api/current_models")
def api_current_models(ops_session: str | None = Cookie(default=None, alias=OPS_COOKIE_NAME)):
    require_ops(ops_session)

    result = []
    for target in ("default", "west", "south"):
        info = _get_model_info(target)
        exists = bool(info.get("exists")) if info else False
        result.append({
            "target": target,
            "label": info.get("label") if info else _model_label_for_target(target),
            "filename": info.get("filename") if info else _model_filename_for_target(target),
            "exists": exists,
            "size_bytes": info.get("size_bytes") if info else None,
            "download_url": f"/api/download_model/{target}" if exists else None,
            "trained_at": info.get("trained_at") if info else None,
            "file_updated_at": info.get("file_updated_at") if info else None,
            "source": info.get("source") if info else None,
            "model_type": info.get("model_type") if info else None,
            "report_summary": info.get("report_summary") if info else None,
            "note": info.get("note") if info else None,
        })

    return {"ok": True, "models": result}


@app.get("/api/download_model/{target}")
def api_download_model(
    target: str,
    ops_session: str | None = Cookie(default=None, alias=OPS_COOKIE_NAME),
):
    require_ops(ops_session)
    model_dir = Path("app/models")

    mapping = {
        "default": ("mmi_model.joblib", "mmi_model.joblib"),
        "west": ("mmi_west.joblib", "mmi_west.joblib"),
        "south": ("mmi_south.joblib", "mmi_south.joblib"),
    }

    if target not in mapping:
        return JSONResponse(
            status_code=400,
            content={"ok": False, "error": "target ไม่ถูกต้อง"}
        )

    real_name, download_name = mapping[target]
    path = model_dir / real_name

    if not path.exists():
        return JSONResponse(
            status_code=404,
            content={"ok": False, "error": "ไม่พบไฟล์โมเดล"}
        )

    return FileResponse(
        path=str(path),
        filename=download_name,
        media_type="application/octet-stream",
    )


def save_runtime_model(target: str, model_obj, meta: dict | None = None):
    model_dir = Path("app/models")
    model_dir.mkdir(parents=True, exist_ok=True)

    if target == "default":
        out_name = "mmi_model.joblib"
    elif target == "west":
        out_name = "mmi_west.joblib"
    elif target == "south":
        out_name = "mmi_south.joblib"
    else:
        raise ValueError("invalid target")

    out_path = model_dir / out_name
    joblib.dump(model_obj, out_path)

    loaded_model = joblib.load(out_path)

    if target == "default":
        logic.MMI_MODEL = loaded_model
        logic.MMI_MODELS["north"] = loaded_model
        simulate_logic.MMI_MODEL = loaded_model
        simulate_logic.MMI_MODELS["north"] = loaded_model

    elif target == "west":
        logic.MMI_MODELS["west"] = loaded_model
        if logic.MMI_MODEL is None:
            logic.MMI_MODEL = loaded_model

        simulate_logic.MMI_MODELS["west"] = loaded_model
        if simulate_logic.MMI_MODEL is None:
            simulate_logic.MMI_MODEL = loaded_model

    elif target == "south":
        logic.MMI_MODELS["south"] = loaded_model
        if logic.MMI_MODEL is None:
            logic.MMI_MODEL = loaded_model

        simulate_logic.MMI_MODELS["south"] = loaded_model
        if simulate_logic.MMI_MODEL is None:
            simulate_logic.MMI_MODEL = loaded_model

    if isinstance(meta, dict):
        _upsert_model_meta(
            target,
            trained_at=meta.get("trained_at"),
            source=meta.get("source"),
            report=meta.get("report"),
            model_type=meta.get("model_type"),
            n_iter=meta.get("n_iter"),
            original_filename=meta.get("original_filename"),
            note=meta.get("note"),
        )
    else:
        _upsert_model_meta(target)

    return str(out_path).replace("\\", "/")


def train_best_model_from_dataframe(df: pd.DataFrame, model_type: str = "classifier", n_iter: int = 25):
    from sklearn.base import clone

    if model_type == "classifier":
        from train_mmi_classifier_tuned import (
            prepare_dataset,
            choose_cv,
            build_search_spaces,
            tune_one_model,
        )
        selection_metric = "cv_accuracy_exact"
        better = "higher"

    elif model_type == "regression":
        from train_mmi_regression_tuned import (
            prepare_dataset,
            choose_cv,
            build_search_spaces,
            tune_one_model,
        )
        selection_metric = "cv_mae"
        better = "lower"

    else:
        raise ValueError("model_type must be classifier or regression")

    clean_df, X_raw, X, y, groups = prepare_dataset(df)
    cv, cv_name = choose_cv(y, groups) if model_type == "classifier" else choose_cv(groups)

    spaces = build_search_spaces()

    best_name = None
    best_estimator = None
    best_value = None
    all_results = {}

    for name, (model, param_dist) in spaces.items():
        est, result = tune_one_model(
            name=name,
            model=model,
            param_dist=param_dist,
            X=X,
            y=y,
            cv=cv,
            groups=groups,
            n_iter=n_iter,
        )
        result["cv_strategy"] = cv_name
        all_results[name] = result

        current_value = result[selection_metric]

        if best_value is None:
            best_value = current_value
            best_name = name
            best_estimator = clone(est)
        else:
            if better == "higher" and current_value > best_value:
                best_value = current_value
                best_name = name
                best_estimator = clone(est)
            elif better == "lower" and current_value < best_value:
                best_value = current_value
                best_name = name
                best_estimator = clone(est)

    best_estimator.fit(X, y)

    report = {
        "ok": True,
        "model_type": model_type,
        "n_rows": int(len(clean_df)),
        "cv_strategy": cv_name,
        "feature_columns_raw": list(X_raw.columns),
        "feature_columns_engineered": list(X.columns),
        "best_model_name": best_name,
        "selection_metric": selection_metric,
        "best_value": float(best_value),
        "all_results": all_results,
    }

    return best_estimator, report


def _auto_retrain_worker(current_count_at_start: int):
    global _auto_train_running

    try:
        df = read_training_sheet_as_dataframe()

        if len(df) < AUTO_TRAIN_MIN_ROWS:
            state = load_auto_train_state()
            state["last_status"] = "skipped_min_rows"
            state["last_error"] = f"training rows too low: {len(df)} < {AUTO_TRAIN_MIN_ROWS}"
            save_auto_train_state(state)
            return

        best_model, report = train_best_model_from_dataframe(
            df=df,
            model_type=AUTO_TRAIN_MODEL_TYPE,
            n_iter=AUTO_TRAIN_N_ITER,
        )

        trained_at = _thai_now_str()
        deployment = []
        replaced_any = False

        for target in ("default", "west", "south"):
            decision = _should_replace_model(target, report)
            if decision["should_replace"]:
                save_runtime_model(
                    target,
                    best_model,
                    meta={
                        "trained_at": trained_at,
                        "source": "auto_train",
                        "report": report,
                        "model_type": AUTO_TRAIN_MODEL_TYPE,
                        "n_iter": AUTO_TRAIN_N_ITER,
                        "note": "auto trained from Google Sheets TrainingData",
                    },
                )
                replaced_any = True
            deployment.append(decision)

        report["deployment"] = deployment

        state = load_auto_train_state()
        state["last_trained_row_count"] = current_count_at_start
        state["last_trained_at"] = trained_at
        state["last_status"] = "ok_replaced" if replaced_any else "ok_no_replace"
        state["last_error"] = None
        state["last_report"] = report
        save_auto_train_state(state)

        print(
            f"[AUTO_TRAIN] done rows={current_count_at_start} best={report['best_model_name']} "
            f"value={report['best_value']} replaced_any={replaced_any}"
        )

    except Exception as e:
        state = load_auto_train_state()
        state["last_status"] = "error"
        state["last_error"] = str(e)
        save_auto_train_state(state)
        print("[AUTO_TRAIN] failed:", repr(e))
        print(traceback.format_exc())

    finally:
        with _auto_train_lock:
            _auto_train_running = False


def maybe_trigger_auto_retrain():
    global _auto_train_running

    current_count = get_training_row_count()
    state = load_auto_train_state()
    enabled = bool(state.get("enabled", True))
    last_count = int(state.get("last_trained_row_count", 0) or 0)
    new_rows = max(0, current_count - last_count)

    if not enabled:
        return {
            "triggered": False,
            "reason": "disabled",
            "current_count": current_count,
            "last_trained_row_count": last_count,
        }

    if current_count < AUTO_TRAIN_MIN_ROWS:
        return {
            "triggered": False,
            "reason": "below_min_rows",
            "current_count": current_count,
            "min_rows": AUTO_TRAIN_MIN_ROWS,
        }

    if new_rows < AUTO_TRAIN_EVERY:
        return {
            "triggered": False,
            "reason": "not_enough_new_rows",
            "new_rows": new_rows,
            "needed": AUTO_TRAIN_EVERY,
            "current_count": current_count,
            "last_trained_row_count": last_count,
        }

    with _auto_train_lock:
        if _auto_train_running:
            return {
                "triggered": False,
                "reason": "already_running",
                "new_rows": new_rows,
                "current_count": current_count,
            }

        _auto_train_running = True

    threading.Thread(
        target=_auto_retrain_worker,
        args=(current_count,),
        daemon=True,
    ).start()

    return {
        "triggered": True,
        "reason": "started",
        "new_rows": new_rows,
        "current_count": current_count,
    }
