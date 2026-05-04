import numpy as np
import pandas as pd
from pathlib import Path

from .features import make_features


# --- AI MMI models (regional joblib) ---
MMI_MODEL = None
MMI_MODELS = {}

_MODEL_DIR = Path(__file__).resolve().parent / "models"
_MODEL_PATH = _MODEL_DIR / "mmi_model.joblib"      # north/default
_MODEL_WEST_PATH = _MODEL_DIR / "mmi_west.joblib"
_MODEL_SOUTH_PATH = _MODEL_DIR / "mmi_south.joblib"


def _ensure_model_shims():
    import sys
    try:
        from . import features as _app_features
        sys.modules.setdefault("features", _app_features)
    except Exception:
        pass

    try:
        import __main__
        if not hasattr(__main__, "make_features"):
            from .features import make_features as _mf
            setattr(__main__, "make_features", _mf)
    except Exception:
        pass


def _load_one_model(path: Path, label: str):
    try:
        import joblib
        if path.exists():
            m = joblib.load(path)
            print(f"[MMI_MODEL] loaded {label}: {path}")
            return m
        print(f"[MMI_MODEL] not found {label}: {path}")
        return None
    except Exception as e:
        print(f"[MMI_MODEL] load failed {label}: {repr(e)}")
        return None


try:
    _ensure_model_shims()

    north_model = _load_one_model(_MODEL_PATH, "north/default")
    west_model = _load_one_model(_MODEL_WEST_PATH, "west")
    south_model = _load_one_model(_MODEL_SOUTH_PATH, "south")

    MMI_MODELS = {
        "north": north_model,
        "west": west_model,
        "south": south_model,
    }

    # fallback หลัก
    MMI_MODEL = north_model or west_model or south_model

except Exception as e:
    print("[MMI_MODEL] init failed:", repr(e))
    MMI_MODEL = None
    MMI_MODELS = {}


def _point_in_box(lat: float, lon: float, box: dict) -> bool:
    return (
        box["lat_min"] <= lat <= box["lat_max"]
        and box["lon_min"] <= lon <= box["lon_max"]
    )

REGION_BLOCKS = {
    "south": [
        {"lat_min": 5.0, "lat_max": 10.80, "lon_min": 97.0, "lon_max": 106.5},
        {"lat_min": 10.80, "lat_max": 11.80, "lon_min": 98.00, "lon_max": 101.80},
        {"lat_min": 11.80, "lat_max": 12.35, "lon_min": 98.45, "lon_max": 100.80},
    ],
    "west": [
        {"lat_min": 15.00, "lat_max": 19.90, "lon_min": 97.35, "lon_max": 98.25},
        {"lat_min": 11.70, "lat_max": 15.00, "lon_min": 98.00, "lon_max": 99.25},
    ],
}

def _thai_region_from_epicenter(lat: float, lon: float) -> str:
    lat = float(lat)
    lon = float(lon)

    if not (5.0 <= lat <= 21.5 and 97.0 <= lon <= 106.5):
        return "north"

    for region_name in ("south", "west"):
        for box in REGION_BLOCKS[region_name]:
            if _point_in_box(lat, lon, box):
                return region_name

    return "north"


# PGA (%g) → MMI (Worden+2012)
PGA_THRESH = [0.05, 0.3, 2.8, 6.2, 12, 22, 40, 75, 139]
MMI_CODES = ["I", "II–III", "IV", "V", "VI", "VII", "VIII", "IX", "X", "X+"]
TH_SHAKING = [
    "ไม่รู้สึก",
    "อ่อนมาก–อ่อน",
    "อ่อน",
    "ปานกลาง",
    "ค่อนข้างแรง",
    "แรงมาก",
    "รุนแรง",
    "รุนแรงมาก",
    "รุนแรงมาก (Violent)",
    "รุนแรงที่สุด (Extreme)",
]
TH_DAMAGE = [
    "ไม่มี",
    "ไม่มี",
    "ไม่มี",
    "มีรอยแตกของผนังและกระจกหน้าต่างบ้างเล็กน้อย",
    "ผนัง/เพดานร่วงหล่นบางจุด หน้าต่างบานเกร็ดแตกหรือร้าว",
    "บ้าน/ตึกแถวที่ก่อสร้างไม่ได้มาตรฐานเริ่มเสียหายหนัก (ผนังแตกร้าว, เสาปูนบางต้นร้าว)",
    "อาคารที่สร้างไม่แข็งแรง (บ้านก่ออิฐไม่เสริมเหล็ก, ตึกแถวเก่า) เสียหายหนัก บางส่วนอาจพังทลาย",
    "อาคาร/บ้านเรือนที่ไม่ได้ออกแบบให้ทนแผ่นดินไหวเสียหายหนัก",
    "อาคาร/บ้านเรือนพังถล่มเกือบทั้งหมด",
    "อาคาร/บ้านเรือนพังถล่ม",
]


def mmi_from_pga(pga_percent_g: float):
    """รับค่า PGA เป็นหน่วย %g แล้วคืน dict ข้อมูล MMI"""
    idx = 0
    while idx < len(PGA_THRESH) and pga_percent_g >= PGA_THRESH[idx]:
        idx += 1

    code = MMI_CODES[idx]
    shake = TH_SHAKING[idx]
    damage = TH_DAMAGE[idx]
    lower = 0 if idx == 0 else PGA_THRESH[idx - 1]
    upper = float("inf") if idx == len(PGA_THRESH) else PGA_THRESH[idx]
    range_text = (
        f"< {PGA_THRESH[0]}%g"
        if idx == 0
        else (f"≥ {lower}%g" if upper == float("inf") else f"{lower}–{upper}%g")
    )

    return {
        "code": code,
        "shake_th": shake,
        "damage_th": damage,
        "range_text": range_text,
        "bin_index": idx,
    }


def cap_mmi_from_pga_g(pred: float, pga_g: float) -> float:
    """
    คุมเพดาน MMI จากค่า PGA โดยตรง
    pga_g เป็นหน่วย g
    """
    if pga_g <= 0.0001:      # <= 0.01 %g
        return min(pred, 1.0)
    elif pga_g <= 0.0003:    # <= 0.03 %g
        return min(pred, 2.5)
    elif pga_g <= 0.0030:    # <= 0.30 %g
        return min(pred, 3.0)
    elif pga_g <= 0.0280:    # <= 2.80 %g
        return min(pred, 4.0)

    return pred


def apply_physical_guardrail(pred: float, pga_g: float, dist_km: float, mag: float) -> float:
    # ถ้าแมกนิจูดต่ำมาก และ PGA ก็ต่ำจริง ให้เป็น MMI 1 ไปเลย
    if mag < 2.0 and pga_g <= 0.001:
        return 1.0

    pred = cap_mmi_from_pga_g(pred, pga_g)

    if dist_km >= 150 and pga_g <= 0.002:
        pred = min(pred, 2.5)
    if dist_km >= 100 and pga_g <= 0.005:
        pred = min(pred, 3.0)

    return pred


def _hybrid_mmi_from_model_and_table(pred_mmi: float, pga_percent_g: float):
    """
    pred_mmi      : ค่าทศนิยมจากโมเดล
    pga_percent_g : PGA หน่วย %g

    กติกา:
    - ถ้า PGA >= 40 %g -> ยึดค่าจากตารางทันที
    - ถ้า model กับ table ต่างกัน >= 2 ระดับ -> ใช้ table
    - ถ้าต่างกัน 0 หรือ 1 ระดับ -> ใช้ model
    """
    pred_mmi = float(np.clip(pred_mmi, 1.0, 12.0))
    model_level = int(np.clip(np.rint(pred_mmi), 1, 12))

    table_info = mmi_from_pga(float(pga_percent_g))
    table_level = int(table_info["bin_index"] + 1)

    diff = abs(model_level - table_level)

    if float(pga_percent_g) >= 40.0:
        final_level = table_level
        final_pred = float(table_level)
        source = "table_forced_high_pga"
    elif diff >= 2:
        final_level = table_level
        final_pred = float(table_level)
        source = "table"
    else:
        final_level = model_level
        final_pred = pred_mmi
        source = "model"

    return {
        "mmi_pred": float(final_pred),
        "mmi_level": int(final_level),
        "model_level": int(model_level),
        "table_level": int(table_level),
        "diff": int(diff),
        "source": source,
        "table": table_info,
    }


def predict_mmi_ai(
    dist: float,
    pga: float,
    mag: float = None,
    lat: float = None,
    lon: float = None,
):
    """
    predict MMI จาก dist/pga/(mag)/lat/lon
    - pga ที่รับเข้ามาเป็นหน่วย g
    - รองรับโมเดลที่ train มาไม่เท่ากันเรื่อง feature columns
    """
    model, model_region = _get_mmi_model_for_region(lat, lon)
    print(f"[MMI_POINT] selected region={model_region} lat={lat} lon={lon}")

    if model is None:
        return None

    row = {
        "dist": float(dist),
        "pga": float(pga),
    }

    if mag is not None:
        row["mag"] = float(mag)
    if lat is not None:
        row["lat"] = float(lat)
    if lon is not None:
        row["lon"] = float(lon)

    try:
        X_raw = pd.DataFrame([row])
        X_pred_full = make_features(X_raw)

        if hasattr(model, "feature_names_in_"):
            expected_cols = list(model.feature_names_in_)
            for c in expected_cols:
                if c not in X_pred_full.columns:
                    X_pred_full[c] = 0.0
            X_pred = X_pred_full[expected_cols]
        else:
            X_pred = X_pred_full

        pred = float(model.predict(X_pred)[0])
        print("[DEBUG] before guardrail =", pred)

        pred = apply_physical_guardrail(
            pred=pred,
            pga_g=float(pga),
            dist_km=float(dist),
            mag=float(mag) if mag is not None else 5.0,
        )
        print("[DEBUG] after guardrail =", pred)

        pred = float(np.clip(pred, 1.0, 12.0))

        hybrid = _hybrid_mmi_from_model_and_table(
            pred_mmi=pred,
            pga_percent_g=float(pga) * 100.0,   # pga ในฟังก์ชันนี้เป็น g
        )

        return hybrid

    except Exception as e:
        print("[MMI_MODEL] predict point failed:", repr(e))
        return None