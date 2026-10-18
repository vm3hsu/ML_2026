import numpy as np
import pandas as pd

# ---------------- 設定 ----------------
# 未套規則的原始預測檔 : 權重（權重會自動正規化）
MEMBERS = {
    "v7_parts/old_hgb.csv": 0.13,
    "v7_parts/old_lgb.csv": 0.17,
    "v7_parts/old_lgb_s7.csv": 0.17,
    "v7_parts/old_l2.csv": 0.05,
    "v7_parts/v5_l1_w.csv": 0.10,
    "v7_parts/v5_l1_nok.csv": 0.19,
    "v7_parts/v5_l2.csv": 0.19,
}
PER_PING = 180_000
OUT = "submission-v8-w2.csv"
PER_PING = 180_000                  # 安置住宅每坪單價（不含車位）
PARK = 1_000_000
# OUT = "submission-blend.csv"
TEST = "test-reindex-test-v2.2.xlsx"
# --------------------------------------

te = pd.read_excel(TEST)
te.columns = [str(c).replace(" ", "").replace("\u3000", "") for c in te.columns]
base = te[["編號"]].copy()

total_w, log_sum = 0.0, np.zeros(len(base))
for path, w in MEMBERS.items():
    s = pd.read_csv(path, encoding="utf-8-sig")
    p = base.merge(s, on="編號", how="left")["總價元"].to_numpy()
    assert np.all(np.isfinite(p) & (p > 0)), f"{path} 有缺值或非正值"
    log_sum += w * np.log(p)
    total_w += w
pred = np.exp(log_sum / total_w)

# 安置住宅規則
rm = te["備註"].fillna("").astype(str)
mask = rm.str.contains("安置").to_numpy()
a_b = pd.to_numeric(te["建物移轉總面積平方公尺"], errors="coerce").fillna(0)
a_p = pd.to_numeric(te["車位移轉總面積平方公尺"], errors="coerce").fillna(0)
n_p = te["交易筆棟數"].astype(str).str.extract(r"車位(\d+)")[0].astype(float).fillna(0)
d = rm.str.extract(r"原房屋價款(\d+)元.*?減收後之房屋價款為(\d+)元").astype(float)
disc = (d[0] - d[1]).fillna(0)
rule = ((a_b - a_p).clip(lower=0) / 3.305785 * PER_PING + n_p * PARK - disc).to_numpy()
pred[mask] = rule[mask]

pd.DataFrame({"編號": base["編號"], "總價元": np.round(pred, 2)}).to_csv(
    OUT, index=False, encoding="utf-8-sig")
print(f"寫出 {OUT}，安置住宅 {mask.sum():,} 筆，成員: {MEMBERS}")