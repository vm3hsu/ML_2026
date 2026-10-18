"""房價預測 v8：v7 + 2 個新的「舊版特徵 + LightGBM」模型

會沿用 v7_parts 裡已有的模型預測，只訓練缺少的模型（約 4 分鐘）。
輸出：
  submission-v8-raw.csv     ensemble（未套規則）
  submission-v8-rule.csv    ensemble + 安置住宅規則   ← 上傳這個
用法：python train_v8.py
"""
import re
import time
import unicodedata
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

ROOT = Path(__file__).resolve().parent
TRAIN_FILE = ROOT / "train-v2.csv"
VALID_FILE = ROOT / "valid-v2.xlsx"
TEST_FILE = ROOT / "test-reindex-test-v2.2.xlsx"
OUT_RAW = ROOT / "submission-v8-raw.csv"
OUT_RULE = ROOT / "submission-v8-rule.csv"
PARTS_DIR = ROOT / "v7_parts"     # 與 v7 共用，已有的模型直接讀檔

# ---------------- 設定 ----------------
REUSE_PARTS = True       # 已有的模型直接讀檔；改了特徵想全部重訓時設 False
PER_PING = 180_000       # 安置住宅每坪單價（不含車位）
PARK = 1_000_000
OLD_HGB_MAX_ITER = 3000
MEMBER_WEIGHTS = {       # 幾何平均權重（自動正規化；設 0 = 不使用）
    "old_hgb": 0.20,
    "old_lgb": 0.12,
    "old_lgb_s7": 0.12,  # 新：舊版特徵 + LightGBM，換一組隨機往前推月數
    "old_l2": 0.06,      # 新：舊版特徵 + LightGBM L2
    "v5_l1_w": 1 / 6,
    "v5_l1_nok": 1 / 6,
    "v5_l2": 1 / 6,
}

V5_PARAMS = dict(
    learning_rate=0.1, num_leaves=127, min_child_samples=40, subsample=0.8,
    subsample_freq=1, colsample_bytree=0.7, reg_lambda=1.0, max_bin=255,
    metric="None", n_jobs=-1, verbosity=-1,
)
V5_VARIANTS = [
    dict(name="v5_l1_w", objective="regression_l1", weight=True, kdist_cat=True, n_iter=2400),
    dict(name="v5_l1_nok", objective="regression_l1", weight=False, kdist_cat=False, n_iter=2400),
    dict(name="v5_l2", objective="regression", weight=False, kdist_cat=True, n_iter=1500),
]
OLD_LGB_PARAMS = dict(
    objective="regression_l1", learning_rate=0.1, num_leaves=127, min_child_samples=40,
    reg_lambda=1.0, max_bin=255, metric="None", n_jobs=-1, verbosity=-1,
)
OLD_LGB_ITER = 2000


# =========================================================
# 共用
# =========================================================
def read_any(p):
    if p.suffix == ".csv":
        df = pd.read_csv(p, encoding="utf-8-sig", low_memory=False)
    else:
        df = pd.read_excel(p)
    df.columns = [str(c).replace(" ", "").replace("\u3000", "") for c in df.columns]
    return df


def load():
    t0 = time.time()
    tr, va, te = read_any(TRAIN_FILE), read_any(VALID_FILE), read_any(TEST_FILE)
    full = pd.concat([tr, va], ignore_index=True)
    full = full[full["編號"].astype(str) != "serial number"]
    full = full.drop_duplicates("編號")
    full["總價元"] = pd.to_numeric(full["總價元"], errors="coerce")
    full = full[full["總價元"] > 0].reset_index(drop=True)
    print(f"讀檔: train+valid={len(full):,} test={len(te):,} ({time.time() - t0:.0f}s)")
    return full, te.reset_index(drop=True)


CN = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
FLOOR_RE = re.compile(r"(地下)?([一二三四五六七八九十]+)層")


def cn2int(s):
    if "十" in s:
        a, b = s.split("十", 1)
        return (CN.get(a, 1) if a else 1) * 10 + (CN.get(b, 0) if b else 0)
    return CN.get(s, np.nan)


def parse_floor_cn(s):
    if not isinstance(s, str):
        return np.nan, np.nan, np.nan
    fl = [(-1 if m.group(1) else 1) * cn2int(m.group(2)) for m in FLOOR_RE.finditer(s)]
    return (fl[0], max(fl), len(fl)) if fl else (np.nan, np.nan, 0)


def num(df, c):
    return pd.to_numeric(df[c], errors="coerce") if c in df else pd.Series(np.nan, index=df.index)


def _cum_stats(k, mi, y):
    g = pd.DataFrame({"k": np.asarray(k, dtype=object), "mi": np.asarray(mi, float),
                      "y": np.asarray(y, float)}).dropna()
    st = g.groupby(["k", "mi"])["y"].agg(["sum", "count"]).reset_index().sort_values(["k", "mi"])
    st["csum"] = st.groupby("k")["sum"].cumsum()
    st["ccnt"] = st.groupby("k")["count"].cumsum()
    return st[["k", "mi", "csum", "ccnt"]].sort_values("mi").reset_index(drop=True)


def _lookup(st, k, c):
    q = pd.DataFrame({"k": np.asarray(k, dtype=object), "c": np.asarray(c, float),
                      "i": np.arange(len(k))})
    q = q[q["k"].notna() & q["c"].notna()].sort_values("c")
    s, n, last = np.zeros(len(k)), np.zeros(len(k)), np.full(len(k), np.nan)
    if len(q) and len(st):
        r = st.assign(lastmi=st["mi"]).rename(columns={"mi": "c"})
        mg = pd.merge_asof(q, r, on="c", by="k", direction="backward")
        idx = mg["i"].values
        s[idx] = mg["csum"].fillna(0).values
        n[idx] = mg["ccnt"].fillna(0).values
        last[idx] = mg["lastmi"].values
    return s, n, last


def area_eff(df):
    a_b = num(df, "建物移轉總面積平方公尺").fillna(0)
    a_l = num(df, "土地移轉總面積平方公尺").fillna(0)
    a_p = num(df, "車位移轉總面積平方公尺").fillna(0)
    return np.where(a_b > 0, a_b, np.where(a_l > 0, a_l, np.where(a_p > 0, a_p, 1.0)))


# =========================================================
# A. 舊版（prep.py + train.py）特徵
# =========================================================
O_CAT = ["city", "交易標的", "建物型態", "主要用途", "主要建材", "車位類別",
         "都市土地使用分區", "非都市土地使用分區", "非都市土地使用編定", "建物現況格局-隔間"]
O_KEYS = ["k_dist", "k_road", "k_seg", "k_proj", "k_bld"]
O_ROAD_RE = re.compile(r"^(.+?(?:路|街|大道))([一二三四五六七八九十]+段)?")
O_REMARK = {
    "rm_deco": "裝潢", "rm_furn": "傢俱|家具|家電", "rm_rel": "親友|特殊關係|員工|關係人",
    "rm_urgent": "急", "rm_rooftop": "頂樓|露台|平台", "rm_park": "車位", "rm_shop": "店面|商業",
}


def old_base(df):
    f = pd.DataFrame(index=df.index)
    addr = df["土地位置建物門牌"].astype(str).str.replace("台", "臺")
    city = addr.str[:3]
    f["city"] = city
    dist = df["鄉鎮市區"].fillna(addr.str[3:6]).astype(str)
    f["k_dist"] = city + dist
    rest = pd.Series([a.split(d, 1)[-1] for a, d in zip(addr, dist)], index=df.index)
    rs = rest.str.extract(O_ROAD_RE)
    road, seg = rs[0], rs[1].fillna("")
    f["k_road"] = np.where(road.notna(), f["k_dist"] + "|" + road.fillna(""), None)
    f["k_seg"] = np.where(road.notna(), f["k_road"].astype(str) + seg, None)
    f["is_cross"] = rest.str.contains("交叉|與").astype(int)
    f["is_lot"] = rest.str.contains("地號").astype(int)
    proj = df["建案名稱"].fillna("").astype(str).str.strip()
    f["k_proj"] = np.where(proj != "", f["k_dist"] + "|" + proj, None)
    bld = df["棟及號"].fillna("").astype(str).str.split("-").str[0].str.strip()
    f["k_bld"] = np.where((proj != "") & (bld != ""), f["k_proj"].astype(str) + "|" + bld, None)

    d = num(df, "交易年月日")
    d = d.where((d > 1000000) & (d < 1200000))
    y, m = d // 10000, (d // 100) % 100
    f["t"] = y + (m - 1) / 12
    f["month"] = m

    nums = df["交易筆棟數"].astype(str).str.extract(r"土地(\d+)建物(\d+)車位(\d+)").astype(float)
    f["n_land"], f["n_bld"], f["n_park"] = nums[0], nums[1], nums[2]
    fl = df["移轉層次"].map(parse_floor_cn)
    f["floor"] = [a for a, _, _ in fl]
    f["floor_max"] = [b for _, b, _ in fl]
    f["n_floor"] = [c for _, _, c in fl]
    f["floor_all"] = df["移轉層次"].astype(str).str.contains("全").astype(int)
    f["total_floor"] = num(df, "總樓層數")
    f["floor_ratio"] = f["floor"] / f["total_floor"].replace(0, np.nan)
    f["is_top"] = (f["floor_max"] == f["total_floor"]).astype(int)

    a_b = num(df, "建物移轉總面積平方公尺").fillna(0)
    a_l = num(df, "土地移轉總面積平方公尺").fillna(0)
    a_p = num(df, "車位移轉總面積平方公尺").fillna(0)
    f["a_bld"], f["a_land"], f["a_park"] = a_b, a_l, a_p
    f["a_main"] = (a_b - a_p).clip(lower=0)
    f["log_a_bld"] = np.log1p(a_b)
    f["land_ratio"] = a_l / a_b.replace(0, np.nan)
    f["park_ratio"] = a_p / a_b.replace(0, np.nan)
    for c in ["房", "廳", "衛"]:
        f[c] = num(df, f"建物現況格局-{c}")
    f["rooms"] = f[["房", "廳", "衛"]].sum(axis=1)
    f["a_per_room"] = f["a_main"] / f["房"].replace(0, np.nan)

    rm = df["備註"].fillna("").astype(str)
    for k, pat in O_REMARK.items():
        f[k] = rm.str.contains(pat).astype(int)
    f["has_remark"] = (rm != "").astype(int)
    f["cancel"] = df["解約情形"].notna().astype(int) if "解約情形" in df else 0
    for c in O_CAT[1:]:
        f[c] = df[c].astype(str) if c in df else "nan"
    f["area_eff"] = area_eff(df)
    return f


def old_project(f_all):
    g = f_all.dropna(subset=["k_proj"]).groupby("k_proj")
    agg = pd.DataFrame({
        "proj_units": g.size(),
        "proj_max_floor": g["total_floor"].max(),
        "proj_mean_area": g["a_bld"].mean(),
        "proj_med_rooms": g["房"].median(),
        "proj_first_t": g["t"].min(),
        "proj_park_rate": g["n_park"].apply(lambda s: (s > 0).mean()),
    })
    out = f_all[["k_proj"]].join(agg, on="k_proj").drop(columns="k_proj")
    out["t_since_launch"] = f_all["t"] - out["proj_first_t"]
    out["area_vs_proj"] = f_all["a_bld"] / out["proj_mean_area"]
    out["dist_units"] = f_all.groupby("k_dist")["k_dist"].transform("size")
    return out


def old_prepare(full, test):
    t0 = time.time()
    ff, ft = old_base(full), old_base(test)
    n = len(ff)
    fa = pd.concat([ff, ft], ignore_index=True)
    fa = pd.concat([fa, old_project(fa)], axis=1)
    ff, ft = fa.iloc[:n].reset_index(drop=True), fa.iloc[n:].reset_index(drop=True)
    ff["總價元"] = full["總價元"].astype(float).values
    print(f"[舊版特徵] 完成 ({time.time() - t0:.0f}s)")
    return ff, ft


O_RECENT = 18
O_TE_SPEC = [
    ("k_dist", "k_dist", 20, None, None, True),
    ("k_road", "k_road", 20, "k_dist", None, True),
    ("k_seg", "k_seg", 20, "k_road", None, True),
    ("k_proj", "k_proj", 5, "k_seg", None, True),
    ("k_bld", "k_bld", 5, "k_proj", None, True),
    ("dist_recent", "k_dist", 20, "k_dist", O_RECENT, True),
    ("road_recent", "k_road", 20, "dist_recent", O_RECENT, False),
    ("seg_recent", "k_seg", 20, "road_recent", O_RECENT, False),
    ("dist_type", "dist_type", 20, "k_dist", None, False),
    ("road_type", "road_type", 20, "dist_type", None, False),
    ("dist_fl", "dist_fl", 20, "k_dist", None, False),
    ("dist_fl_recent", "dist_fl", 20, "dist_fl", O_RECENT, False),
]


def old_keys(f):
    fb = pd.cut(f["total_floor"], [-1, 5, 10, 15, 20, 30, 999], labels=False).astype(str)
    return {
        "k_dist": f["k_dist"], "k_road": f["k_road"], "k_seg": f["k_seg"],
        "k_proj": f["k_proj"], "k_bld": f["k_bld"],
        "dist_type": f["k_dist"] + "|" + f["建物型態"],
        "road_type": f["k_road"] + "|" + f["建物型態"],
        "dist_fl": f["k_dist"] + "|" + fb,
    }


def old_te_block(f_fit, y, f_q, cutoff):
    mi_fit = np.round(f_fit["t"] * 12)
    kf, kq = old_keys(f_fit), old_keys(f_q)
    stats = {k: _cum_stats(v, mi_fit, y) for k, v in kf.items()}
    gmean = float(y.mean())
    out = pd.DataFrame(index=f_q.index)
    for name, key, m, parent, win, with_cnt in O_TE_SPEC:
        s, n, _ = _lookup(stats[key], kq[key], cutoff)
        if win:
            s0, n0, _ = _lookup(stats[key], kq[key], cutoff - win)
            s, n = s - s0, n - n0
        prior = out[f"te_{parent}"].values if parent else gmean
        with np.errstate(invalid="ignore", divide="ignore"):
            out[f"te_{name}"] = np.where(n > 0, (s + m * prior) / (n + m), prior)
        if with_cnt:
            out[f"cnt_{name}"] = n
    out["dist_trend"] = out["te_dist_recent"] - out["te_k_dist"]
    return out


def old_build(of, ot, seed=42):
    t0 = time.time()
    y = np.log(of["總價元"] / of["area_eff"])
    lo, hi = y.quantile([0.002, 0.998])
    keep = y.between(lo, hi).to_numpy()
    of, y = of[keep].reset_index(drop=True), y[keep].reset_index(drop=True)
    rng = np.random.default_rng(seed)
    mi = np.round(of["t"] * 12).to_numpy()
    lag = rng.integers(1, 13, size=len(of))
    enc_fit = old_te_block(of, y, of, mi - lag)
    enc_app = old_te_block(of, y, ot, np.full(len(ot), np.nanmax(mi)))
    cats = {c: list(of[c].astype(str).value_counts().index[:250]) for c in O_CAT}
    drop = set(O_KEYS + ["area_eff", "總價元"])

    def mat(f, enc):
        X = pd.concat([f.drop(columns=[c for c in drop if c in f.columns]).reset_index(drop=True),
                       enc.reset_index(drop=True)], axis=1)
        for c in O_CAT:
            X[c] = pd.Categorical(X[c].astype(str), categories=cats[c])
        return X

    X_fit, X_app = mat(of, enc_fit), mat(ot, enc_app)
    print(f"  舊版歷史特徵(seed={seed}) {time.time() - t0:.0f}s, 特徵數 {X_fit.shape[1]}")
    return X_fit, y, X_app


# =========================================================
# B. v5 特徵
# =========================================================
CAT_COLS = ["k_city", "k_dist", "交易標的", "建物型態", "主要用途", "主要建材", "車位類別",
            "都市土地使用分區", "非都市土地使用分區", "非都市土地使用編定", "建物現況格局-隔間"]
CITIES = ["台北市", "新北市", "桃園市", "台中市", "台南市", "高雄市", "新竹市", "嘉義市",
          "基隆市", "新竹縣", "苗栗縣", "彰化縣", "南投縣", "雲林縣", "嘉義縣", "屏東縣",
          "宜蘭縣", "花蓮縣", "台東縣", "澎湖縣", "金門縣", "連江縣"]
OLD_CITY = {"桃園縣": "桃園市", "台北縣": "新北市", "台中縣": "台中市",
            "台南縣": "台南市", "高雄縣": "高雄市"}
CITY_RE = re.compile("^(" + "|".join(CITIES + list(OLD_CITY)) + ")")
ADMIN_RE = re.compile(r"^[\u4e00-\u9fff]{1,3}?[區鄉鎮市]")
VILLAGE_RE = re.compile(
    r"^(?:[\u4e00-\u9fff]{1,3}[里村]\d+鄰|[\u4e00-\u9fff]{2}里(?=[\u4e00-\u9fff]{1,6}(?:路|街|大道)))")
ROAD_RE = re.compile(r"^([\u4e00-\u9fff]{1,8}?(?:大道|路|街))([一二三四五六七八九十]+段)?")
ROAD2_RE = re.compile(r"(?:與|及|和|跟|、|/|&|vs|VS|x|X)([\u4e00-\u9fff]{1,8}?(?:大道|路|街))")
SEC_RE = re.compile(r"([\u4e00-\u9fff]{2,6}?段)")
CN_DIGIT = "一二三四五六七八九"
REMARK_KW = {
    "rm_resettle": "安置", "rm_discount": "減收", "rm_superficies": "地上權",
    "rm_raw": "毛胚", "rm_joint": "合建", "rm_terrace": "露台|露臺", "rm_deco": "裝潢",
    "rm_furn": "傢俱|家具|家電", "rm_rel": "親友|特殊關係|員工|關係人", "rm_rain": "雨遮",
    "rm_change": "更正|撤銷|重新申報", "rm_park": "車位", "rm_shop": "店面|店鋪|商業",
}


def norm_text(s):
    s = unicodedata.normalize("NFKC", str(s))
    s = re.sub(r"\s+", "", s).replace("臺", "台")
    return re.sub(r"(?<!\d)([1-9])段", lambda m: CN_DIGIT[int(m.group(1)) - 1] + "段", s)


def strip_admin(rest, dist):
    for _ in range(4):
        m = CITY_RE.match(rest)
        if m:
            rest = rest[m.end():]
            continue
        if dist and rest.startswith(dist):
            rest = rest[len(dist):]
            continue
        m = ADMIN_RE.match(rest)
        if m and (m.group(0)[-1] in "區鄉鎮" or len(m.group(0)) == 3):
            rest = rest[m.end():]
            continue
        break
    return rest


def parse_rest(r):
    r = VILLAGE_RE.sub("", r, count=1)
    r = re.sub(r"^\d+鄰", "", r)
    road = seg = road2 = sec = None
    after = r
    m = ROAD_RE.match(r)
    if m:
        road, seg = m.group(1), m.group(2) or ""
        after = r[m.end():]
        m2 = ROAD2_RE.search(after)
        if m2 and m2.group(1) != road:
            road2 = m2.group(1)
    ms = SEC_RE.search(after)
    if ms:
        s = ms.group(1)
        if not re.search(r"[路街巷弄]", s) and not re.fullmatch(r"[一二三四五六七八九十]+段", s):
            sec = s
    return road, seg, road2, sec


def v5_base(df):
    f = pd.DataFrame(index=df.index)
    addr = df["土地位置建物門牌"].map(norm_text)
    dist = df["鄉鎮市區"].fillna("").astype(str)
    cities, rests = [], []
    for a, d in zip(addr, dist):
        m = CITY_RE.match(a)
        cities.append(OLD_CITY.get(m.group(1), m.group(1)) if m else None)
        rests.append(strip_admin(a[m.end():] if m else a, d))
    parsed = [parse_rest(r) for r in rests]
    f["_city"], f["_dist"] = cities, dist.values
    f["_road"] = [p[0] for p in parsed]
    f["_seg"] = [p[1] for p in parsed]
    f["_road2"] = [p[2] for p in parsed]
    f["_sec"] = [p[3] for p in parsed]
    rest = pd.Series(rests, index=df.index)
    f["is_cross"] = rest.str.contains(r"交叉|交岔|交口|路口|與|及|vs|VS|、").astype("int8")
    f["is_lot"] = rest.str.contains("地號").astype("int8")
    f["has_number"] = rest.str.contains(r"\d+號").astype("int8")
    proj = df["建案名稱"].fillna("").astype(str).str.strip()
    f["_proj"] = proj.where(~proj.isin(["", "nan", "無", "空白", "None"]))
    bn = df["棟及號"].fillna("").astype(str).str.strip()
    f["_bld"] = bn.str.extract(r"^(.+?)棟")[0].fillna(bn.str.split("-").str[0])
    d = num(df, "交易年月日")
    d = d.where((d > 1000000) & (d < 1200000))
    f["mi"] = (d // 10000) * 12 + (d // 100) % 100
    f["month"] = (d // 100) % 100
    n = df["交易筆棟數"].astype(str).str.extract(r"土地(\d+)建物(\d+)車位(\d+)").astype(float)
    f["n_land"], f["n_bld"], f["n_park"] = n[0], n[1], n[2]
    fl = df["移轉層次"].map(parse_floor_cn)
    f["floor"] = [a for a, _, _ in fl]
    f["floor_max"] = [b for _, b, _ in fl]
    f["n_floor"] = [c for _, _, c in fl]
    f["floor_all"] = df["移轉層次"].astype(str).str.contains("全").astype("int8")
    f["total_floor"] = num(df, "總樓層數")
    f["floor_ratio"] = f["floor"] / f["total_floor"].replace(0, np.nan)
    f["is_top"] = (f["floor_max"] == f["total_floor"]).astype("int8")
    a_b = num(df, "建物移轉總面積平方公尺").fillna(0)
    a_l = num(df, "土地移轉總面積平方公尺").fillna(0)
    a_p = num(df, "車位移轉總面積平方公尺").fillna(0)
    f["a_bld"], f["a_land"], f["a_park"] = a_b, a_l, a_p
    f["a_main"] = (a_b - a_p).clip(lower=0)
    f["log_a_bld"] = np.log1p(a_b)
    f["land_ratio"] = a_l / a_b.replace(0, np.nan)
    f["park_ratio"] = a_p / a_b.replace(0, np.nan)
    for c in ["房", "廳", "衛"]:
        f[c] = num(df, f"建物現況格局-{c}")
    f["a_per_room"] = f["a_main"] / f["房"].replace(0, np.nan)
    rm = df["備註"].fillna("").astype(str)
    for k, pat in REMARK_KW.items():
        f[k] = rm.str.contains(pat).astype("int8")
    f["has_remark"] = (rm != "").astype("int8")
    disc = rm.str.extract(r"原房屋價款(\d+)元.*?減收後之房屋價款為(\d+)元").astype(float)
    f["disc_ratio"] = disc[1] / disc[0]
    f["disc_amt"] = disc[0] - disc[1]
    f["cancel"] = df["解約情形"].notna().astype("int8") if "解約情形" in df else 0
    for c in CAT_COLS[2:]:
        f[c] = df[c].astype(str) if c in df else "nan"
    f["area_eff"] = area_eff(df)
    return f


def add_keys(f):
    kd = f["_city"].fillna("?") + f["_dist"]

    def kk(v):
        return np.where(v.notna(), kd + "|" + v.fillna("").astype(str), None)

    f["k_all"] = "all"
    f["k_city"] = f["_city"].fillna("?")
    f["k_dist"] = kd
    f["k_road"] = kk(f["_road"])
    f["k_seg"] = np.where(f["_road"].notna(), f["k_road"].astype(str) + f["_seg"].fillna(""), None)
    f["k_road2"] = kk(f["_road2"])
    f["k_sec"] = kk(f["_sec"])
    f["k_proj"] = kk(f["_proj"])
    f["k_bld"] = np.where(f["k_proj"].notna() & f["_bld"].fillna("").ne(""),
                          f["k_proj"].astype(str) + "|" + f["_bld"].fillna(""), None)
    f["k_dtype"] = kd + "|" + f["建物型態"]
    fb = pd.cut(f["total_floor"], [-1, 5, 10, 15, 20, 30, 999], labels=False).astype(str)
    f["k_dfl"] = kd + "|" + fb
    return f


def add_project_feats(f):
    g = f.dropna(subset=["k_proj"]).groupby("k_proj")
    agg = pd.DataFrame({
        "proj_units": g.size(),
        "proj_mean_area": g["a_bld"].mean(),
        "proj_max_floor": g["total_floor"].max(),
        "proj_first_mi": g["mi"].min(),
        "proj_park_rate": g["n_park"].apply(lambda s: (s > 0).mean()),
    })
    f = f.join(agg, on="k_proj")
    f["proj_age"] = f["mi"] - f["proj_first_mi"]
    f["area_vs_proj"] = f["a_bld"] / f["proj_mean_area"]
    f["dist_units"] = f.groupby("k_dist")["k_dist"].transform("size")
    return f


def add_new_keys(f):
    new = f["proj_age"].le(6) & f["k_proj"].notna()
    f["k_citynew"] = np.where(new, f["k_city"], None)
    f["k_distnew"] = np.where(new, f["k_dist"], None)
    f["k_dtypenew"] = np.where(new, f["k_dtype"], None)
    f["k_roadnew"] = np.where(new & f["k_road"].notna(), f["k_road"], None)
    return f


def v5_prepare(full, test):
    t0 = time.time()
    ff, ft = v5_base(full), v5_base(test)
    ft["mi"] = ft["mi"].fillna(ft["mi"].median())
    n = len(ff)
    allf = pd.concat([ff, ft], ignore_index=True)
    mp = allf.dropna(subset=["_city"]).groupby("_dist")["_city"].agg(lambda s: s.value_counts().index[0])
    allf["_city"] = allf["_city"].fillna(allf["_dist"].map(mp))
    allf = add_new_keys(add_project_feats(add_keys(allf)))
    ff = allf.iloc[:n].reset_index(drop=True)
    ft = allf.iloc[n:].reset_index(drop=True)
    ff["總價元"] = full["總價元"].astype(float).values
    ff = ff.dropna(subset=["mi"]).reset_index(drop=True)
    print(f"[v5 特徵] 完成 ({time.time() - t0:.0f}s)")
    return ff, ft


SPEC1 = [
    ("all_24", "k_all", 24, 0, None),
    ("all_6", "k_all", 6, 0, "all_24"),
    ("city", "k_city", None, 50, None),
    ("city_12", "k_city", 12, 50, "city"),
    ("dist", "k_dist", None, 20, "city"),
    ("dist_12", "k_dist", 12, 20, "dist"),
    ("dist_6", "k_dist", 6, 20, "dist_12"),
    ("dtype", "k_dtype", None, 20, "dist"),
    ("dtype_12", "k_dtype", 12, 20, "dist_12"),
    ("dfl", "k_dfl", None, 20, "dist"),
    ("dfl_12", "k_dfl", 12, 20, "dist_12"),
    ("sec", "k_sec", None, 10, "dist_12"),
    ("road", "k_road", None, 10, "dist_12"),
    ("road_12", "k_road", 12, 10, "road"),
    ("seg", "k_seg", None, 10, "road_12"),
    ("road2", "k_road2", None, 10, "dist_12"),
    ("citynew_12", "k_citynew>k_city", 12, 30, "city_12"),
    ("distnew_12", "k_distnew>k_dist", 12, 10, "citynew_12"),
    ("distnew_24", "k_distnew>k_dist", 24, 10, "dist_12"),
    ("dtypenew_12", "k_dtypenew>k_dtype", 12, 10, "distnew_12"),
    ("roadnew_24", "k_roadnew>k_road", 24, 5, "distnew_24"),
]
SPEC2 = [
    ("proj", "k_proj", None, 3, "local"),
    ("proj_6", "k_proj", 6, 3, "proj"),
    ("bld", "k_bld", None, 3, "proj"),
]
STALE = {"proj", "road", "dist"}


def te_block(fit, y, q, cutoff):
    cutoff = np.asarray(cutoff, float)
    gmean = float(y.mean())
    cache = {}
    out = pd.DataFrame(index=q.index)

    def run(spec):
        for name, key, win, m, parent in spec:
            kf, kq = key.split(">") if ">" in key else (key, key)
            if kf not in cache:
                cache[kf] = _cum_stats(fit[kf], fit["mi"], y)
            s, n, last = _lookup(cache[kf], q[kq], cutoff)
            if win:
                s0, n0, _ = _lookup(cache[kf], q[kq], cutoff - win)
                s, n = s - s0, n - n0
            prior = out[f"te_{parent}"].to_numpy() if parent else gmean
            with np.errstate(invalid="ignore", divide="ignore"):
                out[f"te_{name}"] = np.where(n > 0, (s + m * prior) / (n + m), prior)
            out[f"cnt_{name}"] = n
            if name in STALE:
                out[f"stale_{name}"] = np.where(n > 0, q["mi"].to_numpy() - last, np.nan)

    run(SPEC1)
    out["te_local"] = np.where(out["cnt_seg"] > 0, out["te_seg"],
                               np.where(out["cnt_sec"] > 0, out["te_sec"], out["te_dist_12"]))
    run(SPEC2)
    out["gap"] = np.clip(q["mi"].to_numpy() - cutoff, 1, None)
    out["te_all_trend"] = out["te_all_6"] - out["te_all_24"]
    out["te_dist_trend"] = out["te_dist_6"] - out["te_dist"]
    out["te_proj_prem"] = out["te_proj"] - out["te_local"]
    out["te_new_prem_dist"] = out["te_distnew_12"] - out["te_dist_12"]
    out["te_new_prem_road"] = out["te_roadnew_24"] - out["te_road_12"]
    return out


V5_DROP = {"總價元", "area_eff", "mi", "proj_first_mi"}
KEEP_K = {"k_city", "k_dist"}


def v5_build(ff, ft):
    t0 = time.time()
    y = np.log(ff["總價元"] / ff["area_eff"])
    lo, hi = y.quantile([0.002, 0.998])
    keep = y.between(lo, hi).to_numpy()
    ff, y = ff[keep].reset_index(drop=True), y[keep].reset_index(drop=True)
    rng = np.random.default_rng(0)
    mi = ff["mi"].to_numpy()
    lag = rng.integers(1, 13, size=len(ff))
    enc_fit = te_block(ff, y, ff, mi - lag)
    enc_app = te_block(ff, y, ft, np.full(len(ft), np.nanmax(mi)))
    cats = {c: list(ff[c].astype(str).value_counts().index[:400]) for c in CAT_COLS}

    def mat(f, enc):
        cols = [c for c in f.columns
                if c not in V5_DROP and (not c.startswith(("_", "k_")) or c in KEEP_K)]
        X = pd.concat([f[cols].reset_index(drop=True), enc.reset_index(drop=True)], axis=1)
        for c in CAT_COLS:
            X[c] = pd.Categorical(X[c].astype(str), categories=cats[c])
        return X.loc[:, ~X.columns.duplicated()]

    X_fit, X_app = mat(ff, enc_fit), mat(ft, enc_app)
    print(f"  v5 歷史特徵 {time.time() - t0:.0f}s, 特徵數 {X_fit.shape[1]}")
    return X_fit, y, X_app, ff


# =========================================================
# 訓練與輸出
# =========================================================
def lgb_fit_predict(X_fit, y, X_app, params, n_iter, w=None, cats="auto"):
    m = lgb.LGBMRegressor(n_estimators=n_iter, random_state=0, **params)
    m.fit(X_fit, y, sample_weight=w, categorical_feature=cats)
    return m.predict(X_app)


def hgb_fit_predict(X_fit, y, X_app, cat_cols):
    Zf, Za = X_fit.copy(), X_app.copy()
    for c in cat_cols:
        for Z in (Zf, Za):
            codes = Z[c].cat.codes.astype(float)
            codes[(codes < 0) | (codes >= 250)] = np.nan
            Z[c] = codes
    useful = [c for c in Zf.columns if Zf[c].nunique(dropna=True) > 1]
    Zf, Za = Zf[useful], Za[useful]
    m = HistGradientBoostingRegressor(
        loss="absolute_error", learning_rate=0.1, max_iter=OLD_HGB_MAX_ITER,
        max_leaf_nodes=127, min_samples_leaf=40, l2_regularization=1.0,
        early_stopping=True, validation_fraction=0.1, n_iter_no_change=100,
        categorical_features=[c in cat_cols for c in Zf.columns], random_state=0)
    m.fit(Zf, y)
    print(f"    (HGB 實際 {m.n_iter_} 棵樹)")
    return m.predict(Za)


def save(path, ids, pred):
    pd.DataFrame({"編號": ids, "總價元": np.round(pred, 2)}).to_csv(
        path, index=False, encoding="utf-8-sig")


def load_part(path, ids):
    s = pd.read_csv(path, encoding="utf-8-sig")
    return pd.DataFrame({"編號": ids}).merge(s, on="編號", how="left")["總價元"].to_numpy()


def apply_resettle(test, pred):
    rm = test["備註"].fillna("").astype(str)
    mask = rm.str.contains("安置").to_numpy()
    a_b = num(test, "建物移轉總面積平方公尺").fillna(0)
    a_p = num(test, "車位移轉總面積平方公尺").fillna(0)
    n_p = test["交易筆棟數"].astype(str).str.extract(r"車位(\d+)")[0].astype(float).fillna(0)
    d = rm.str.extract(r"原房屋價款(\d+)元.*?減收後之房屋價款為(\d+)元").astype(float)
    disc = (d[0] - d[1]).fillna(0)
    rule = ((a_b - a_p).clip(lower=0) / 3.305785 * PER_PING + n_p * PARK - disc).to_numpy()
    out = pred.copy()
    out[mask] = rule[mask]
    print(f"  安置住宅規則套用 {mask.sum():,} 筆（每坪 {PER_PING:,}）")
    return out


def main():
    t_all = time.time()
    PARTS_DIR.mkdir(exist_ok=True)
    full, test = load()
    ids = test["編號"].values
    area = area_eff(test)
    preds = {}

    def need(name):
        if MEMBER_WEIGHTS.get(name, 0) <= 0:
            return False
        part = PARTS_DIR / f"{name}.csv"
        if REUSE_PARTS and part.exists():
            preds[name] = load_part(part, ids)
            print(f"  [{name}] 讀取既有預測")
            return False
        return True

    def done(name, logp, t0):
        p = np.exp(logp) * area
        preds[name] = p
        save(PARTS_DIR / f"{name}.csv", ids, p)
        print(f"  [{name}] 完成 {time.time() - t0:.0f}s")

    # ---------- A. 舊版特徵 ----------
    todo_old = [n for n in ["old_hgb", "old_lgb", "old_l2", "old_lgb_s7"] if need(n)]
    if todo_old:
        of, ot = old_prepare(full, test)
        if any(n in todo_old for n in ["old_hgb", "old_lgb", "old_l2"]):
            Xo_fit, yo, Xo_te = old_build(of, ot, seed=42)
            if "old_lgb" in todo_old:
                t0 = time.time()
                done("old_lgb", lgb_fit_predict(Xo_fit, yo, Xo_te, OLD_LGB_PARAMS,
                                                OLD_LGB_ITER, cats=O_CAT), t0)
            if "old_l2" in todo_old:
                t0 = time.time()
                p2 = dict(OLD_LGB_PARAMS, objective="regression")
                done("old_l2", lgb_fit_predict(Xo_fit, yo, Xo_te, p2, 1500, cats=O_CAT), t0)
            if "old_hgb" in todo_old:
                t0 = time.time()
                done("old_hgb", hgb_fit_predict(Xo_fit, yo, Xo_te, O_CAT), t0)
            del Xo_fit, Xo_te
        if "old_lgb_s7" in todo_old:
            t0 = time.time()
            Xo_fit, yo, Xo_te = old_build(of, ot, seed=7)
            done("old_lgb_s7", lgb_fit_predict(Xo_fit, yo, Xo_te, OLD_LGB_PARAMS,
                                               OLD_LGB_ITER, cats=O_CAT), t0)
            del Xo_fit, Xo_te

    # ---------- B. v5 特徵 ----------
    todo_v5 = [v for v in V5_VARIANTS if need(v["name"])]
    if todo_v5:
        ff, ft = v5_prepare(full, test)
        X_fit, y, X_te, f_k = v5_build(ff, ft)
        for v in todo_v5:
            t0 = time.time()
            Xf, Xa = X_fit, X_te
            if not v["kdist_cat"]:
                Xf, Xa = X_fit.drop(columns=["k_dist"]), X_te.drop(columns=["k_dist"])
            w = None
            if v["weight"]:
                tot = f_k["總價元"].to_numpy()
                w = np.clip(tot / np.median(tot), 0.1, 10)
            params = dict(V5_PARAMS, objective=v["objective"])
            done(v["name"], lgb_fit_predict(Xf, y, Xa, params, v["n_iter"], w=w), t0)

    # ---------- 混合 ----------
    print("\n[混合]")
    tot_w, log_sum = 0.0, np.zeros(len(ids))
    for name, w in MEMBER_WEIGHTS.items():
        if w > 0 and name in preds:
            log_sum += w * np.log(preds[name])
            tot_w += w
            print(f"  {name:<11} 權重 {w:.3f}")
    pred = np.exp(log_sum / tot_w)
    save(OUT_RAW, ids, pred)
    save(OUT_RULE, ids, apply_resettle(test, pred))
    print(f"  寫出 {OUT_RAW.name}、{OUT_RULE.name}")
    print(f"\n完成，共 {time.time() - t_all:.0f}s")


if __name__ == "__main__":
    main()