import numpy as np
import pandas as pd

from common import LOGY, HORMUZ, REDSEA, HORMUZ_DONORS_NOCHOKE, REDSEA_DONORS_CLEAN, sc_fit

H = 26          # 사건 후 효과를 평균낼 기간(주)
MIN_PRE = 40    # 가중치를 학습하려면 사건 전 최소 40주는 있어야 함


def att_at(treated, donors, t0, h=H, min_pre=MIN_PRE):
    """날짜 t0를 사건일로 가정하고 h주 동안의 평균 격차(ATT, 로그)를 계산함.

    t0 이전 데이터가 min_pre주 미만이거나 t0 이후 데이터가 부족하면 None.
    """
    idx = LOGY.index
    pre = idx < t0
    post = (idx >= t0) & (idx < t0 + pd.Timedelta(weeks=h))
    if pre.sum() < min_pre or post.sum() < h - 2:
        return None
    # t0 이전 데이터만으로 가중치를 학습 (미래 정보를 쓰지 않음)
    w, a = sc_fit(LOGY.loc[pre, treated], LOGY.loc[pre, donors])
    gap = LOGY[treated] - (LOGY[donors].values @ w + a)
    return float(gap[post].mean())


def placebo_in_time(label, case, donors):
    """진짜 사건일 효과를 수많은 가짜 사건일 효과들과 비교해 p값을 냄."""
    treated, event = case["treated"], case["event"]
    real = att_at(treated, donors, event)              # 진짜 사건의 효과
    # 가짜 사건일 후보: 진짜 사건 h주 전까지의 모든 주 (그 뒤는 진짜 사건과 겹치므로 제외)
    fake_dates = [t for t in LOGY.index if t < event - pd.Timedelta(weeks=H)]
    fakes = [att_at(treated, donors, t) for t in fake_dates]
    fakes = np.array([f for f in fakes if f is not None])   # 학습기간 부족한 날짜는 버림

    if len(fakes) == 0:
        print(f"{label}: 사건 전 기간이 짧아 가짜 사건일을 만들 수 없음 → 검정 불가")
        return
    # p값: 가짜 중 '진짜 이상으로 큰(절댓값)' 것의 수. +1은 진짜 자신 포함.
    p = (np.sum(np.abs(fakes) >= abs(real)) + 1) / (len(fakes) + 1)
    print(f"{label}\n"
          f"   진짜 효과 ATT={real:.3f} (약 {np.exp(real)-1:+.0%})\n"
          f"   가짜 사건일 수={len(fakes)}\n"
          f"   가짜 효과 |ATT| 최대={np.abs(fakes).max():.3f}, 95번째 백분위수={np.percentile(np.abs(fakes), 95):.3f}\n"
          f"   p = {p:.4f}  (가능한 최소 = 1/{len(fakes)+1})\n")


if __name__ == "__main__":
    placebo_in_time("호르무즈 KMEI (26주 효과)", HORMUZ, HORMUZ_DONORS_NOCHOKE)
    placebo_in_time("홍해 KNEI", REDSEA, REDSEA_DONORS_CLEAN)
