import numpy as np
import pandas as pd

from common import RAW

# 항로별 '사건 구간' (post_end까지 포함) — 이 구간 안의 발동은 '탐지', 밖의 발동은 '오탐'으로 셈
EVENT_WINDOW = {
    "KMEI": (pd.Timestamp("2026-03-01"), pd.Timestamp("2026-08-31")),
    "KNEI": (pd.Timestamp("2023-11-19"), pd.Timestamp("2024-06-30")),
}


def backtest(route):
    """한 항로에 대해 주간·4주누적 변화율 기준으로 트리거 발동 횟수를 셈."""
    s = RAW[route]
    a, b = EVENT_WINDOW[route]
    for label, change in [("주간 변화율", s.pct_change()), ("4주 누적 변화율", s.pct_change(4))]:
        change = change.dropna()
        in_event = (change.index >= a) & (change.index <= b)   # 사건 구간이면 True

        # 평상시 분포: '사건 구간을 제외한' 주들의 상승률 90/95/99 백분위수
        calm = change[~in_event]
        pct = [f"{np.percentile(calm, p):.0%}" for p in (90, 95, 99)]
        print(f"\n{route} {label}: 관측 {len(change)}주, 평상시 90/95/99 백분위수 = {pct}")

        for th in (0.10, 0.15, 0.20, 0.30, 0.40, 0.50):
            fired = change >= th                                # 기준값 이상이면 발동
            hit = fired & in_event                              # 사건 구간 안 발동 = 탐지
            false = fired & ~in_event                           # 사건 구간 밖 발동 = 오탐
            first = change.index[hit][0].date() if hit.any() else None   # 첫 발동일
            print(f"   기준 +{th:.0%}: 발동 {fired.sum():2d}주 | 사건구간 {hit.sum():2d}주 "
                  f"(첫 발동 {first}) | 사건 밖(오탐) {false.sum():2d}주")


if __name__ == "__main__":
    print("=== 트리거 백테스트 ===")
    backtest("KMEI")
    backtest("KNEI")
    print("\n해석 요약")
    print(" - 중동향 주간 +20~50%: 사건 구간에서만 발동하고 오탐 0 → 기준을 넓게 잡아도 안정적")
