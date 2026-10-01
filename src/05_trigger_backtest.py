"""
05_trigger_backtest.py — 보험 트리거 기준값 백테스트 (1차 버전, 아직 GPD·베이지안·Kaiko식은 없음)

이 스크립트가 왜 필요한가
  "운임이 전주(또는 4주) 대비 X% 이상 오르면 보험금 지급"이라는 트리거를 과거 데이터에
  적용해서 ① 진짜 사건 구간에서 몇 주 발동했는지(탐지) ② 사건이 아닌 구간에서 몇 주
  발동했는지(오탐)를 기준값 X별로 셈. 정확한 기준값은 정할 수 없지만 '합리적 범위'를 보려는 것.

핵심 아이디어
  사건이 2개뿐이라 사건 표본으로는 기준값을 못 정함.
  대신 194주 전체의 '평상시 변동 분포'를 씀 → 평상시 99번째 백분위수 바깥으로 튀면 발동.
  이러면 사건 개수와 상관없이 오탐 빈도를 통계적으로 통제할 수 있음.

지금 버전의 한계 — 중간발표 노션 6장(향후계획서)에 '확장' 대상으로 올라간 이유
  여기서는 '평상시 백분위수'만 보고 고정 임계값(+10~50%)을 그냥 다 돌려본 수준이고,
  5장에서 설계한 GPD·베이지안 보정·Kaiko식 항로별 가중은 아직 반영 안 돼 있음.

주의
  - 이 분석은 오탐률은 통제하지만 '탐지율'은 사건 2개라 검증하지 못함.
  - 유럽향의 '사건 밖 발동'에는 홍해 위기가 2024.6 이후에도 이어진 구간이 포함돼 있을 수 있음
    → 진짜 오탐인지는 별도 확인 필요.

실행: python3 05_trigger_backtest.py
"""
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
