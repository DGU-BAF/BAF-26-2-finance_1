"""
02_synthetic_control.py — 합성통제법(Synthetic Control)으로 해협 위기의 '순수 효과' 추정

이 스크립트가 왜 필요한가
  01_did.py(DID)는 통제 항로 묶음을 사람이 미리 정해서 평균을 냄. 근데 그 묶음을 어떻게
  고르느냐에 따라 결과가 꽤 흔들림(특히 홍해). 합성통제법은 "어떤 항로를 얼마나 섞어야
  사건 전 처치 항로 움직임을 가장 잘 흉내내는가"를 데이터가 직접 찾게 해서, 통제군을
  사람이 자의적으로 고르는 문제를 줄이려는 것. 가중치(w)는 common.py의 sc_fit이 최적화로 찾음.

이 스크립트가 만드는 결과물
  - run_case(): 중간발표 노션 3-1절 표(DID vs 합성통제법 수치, 호르무즈 +106~119% 등)와
    3-2절 "항로간 플라세보" 표(호르무즈 p=0.077, 홍해 p=0.286)의 근거가 되는 숫자를 출력함.
  - run_all_routes(): 13개 항로를 전부 번갈아 '처치'라고 가정했을 때 호르무즈·홍해가
    실제로 효과 순위 1등인지 확인하는 보조 검증용(3-2절 각주 성격).

실행: python3 02_synthetic_control.py
"""
import numpy as np
import pandas as pd

from common import (LOGY, ROUTES, HORMUZ, REDSEA, HORMUZ_DONORS_ALL, HORMUZ_DONORS_NOCHOKE,
                    REDSEA_DONORS_CLEAN, REDSEA_DONORS_WIDE, synthetic_control)


def rmse(x):
    """제곱평균제곱근: 오차의 '평균적인 크기'."""
    return float(np.sqrt((np.asarray(x) ** 2).mean()))


def run_case(label, case, donors, pre_start=None):
    """합성통제 한 가지 설정을 돌리고, 항로 간 플라세보까지 계산해 출력함."""
    treated, event, post_end = case["treated"], case["event"], case["post_end"]
    idx = LOGY.index
    post = (idx >= event) & (idx <= post_end)         # 사건 후 기간 (효과를 평균낼 구간)

    # ── 실제 처치 항로 ──
    gap, w, pre = synthetic_control(treated, donors, event, pre_start)
    pre_rmse = rmse(gap[pre])                          # 사건 전 적합 오차
    att = gap[post].mean()                             # 사건 후 평균 격차 = 효과(로그)
    # RMSPE 비율 = (사건 후 오차 크기) ÷ (사건 전 오차 크기)
    #   사건 전에 잘 맞던 항로가 사건 후 크게 벗어날수록 값이 커짐.
    ratio_real = rmse(gap[post]) / max(pre_rmse, 1e-9)

    # ── 항로 간 플라세보: 통제 후보를 하나씩 '가짜 처치'로 ──
    ratios, atts = [], []
    for fake in donors:
        others = [d for d in donors if d != fake]      # 가짜 처치 항로를 뺀 나머지가 통제 후보
        g, _, pre_f = synthetic_control(fake, others, event, pre_start)
        ratios.append(rmse(g[post]) / max(rmse(g[pre_f]), 1e-9))
        atts.append(g[post].mean())

    n = len(ratios) + 1                                # 실제 1개 + 가짜들
    # p값 = (실제 이상으로 큰 가짜 수 + 1) ÷ (전체 수). +1은 '실제 자신'을 셈에 포함하는 표준 방식.
    p_ratio = (np.sum(np.array(ratios) >= ratio_real) + 1) / n
    p_att = (np.sum(np.abs(atts) >= abs(att)) + 1) / n

    wtxt = {k: round(float(v), 2) for k, v in zip(donors, w) if v > 0.01}
    print(f"{label}\n"
          f"   preRMSE={pre_rmse:.3f}  ATT(log)={att:.3f}  효과={np.exp(att)-1:+.1%}  "
          f"피크={np.exp(gap[post].max())-1:+.1%}\n"
          f"   RMSPE비율 p={p_ratio:.3f}  ATT순위 p={p_att:.3f}  (항로 수 n={n}, 최소가능 p={1/n:.3f})\n"
          f"   가중치(1% 이상): {wtxt}\n")


def run_all_routes(label, case, post_end=None):
    """모든 항로를 번갈아 '처치'로 놓고 효과를 비교함 (전 항로 상호 플라세보)."""
    event = case["event"]
    end = post_end or case["post_end"]
    idx = LOGY.index
    post = (idx >= event) & (idx <= end)
    res = {}
    for tr in ROUTES:
        donors = [r for r in ROUTES if r != tr]
        gap, _, pre = synthetic_control(tr, donors, event)
        res[tr] = (gap[post].mean(), rmse(gap[pre]))
    s = sorted(res.items(), key=lambda kv: -kv[1][0])
    print(label)
    print("   " + "  ".join(f"{k}:{np.exp(v[0])-1:+.0%}(pre{v[1]:.2f})" for k, v in s) + "\n")


if __name__ == "__main__":
    print("=== 합성통제법: 호르무즈 (처치=KMEI 중동향) ===\n")
    run_case("[A] 통제 후보 = 처치 항로 제외 전부", HORMUZ, HORMUZ_DONORS_ALL)
    run_case("[B] 통제 후보 = 초크포인트 관련 항로도 제외", HORMUZ, HORMUZ_DONORS_NOCHOKE)
    # 사건 전 기간에 홍해 사태(2023.11~2024.6)가 섞여 있으므로, 홍해 이후만 학습에 씀
    run_case("[C] [B] + 학습기간을 2024-07 이후로 제한", HORMUZ, HORMUZ_DONORS_NOCHOKE,
             pre_start=pd.Timestamp("2024-07-01"))

    print("=== 합성통제법: 홍해 (처치=KNEI 유럽향) ===\n")
    run_case("[D] 통제 후보 = 수에즈와 무관한 깨끗한 6개", REDSEA, REDSEA_DONORS_CLEAN)
    run_case("[E] 통제 후보 = 조금 넓힌 10개", REDSEA, REDSEA_DONORS_WIDE)

    print("=== 모든 항로를 번갈아 '처치'로 놓은 비교 (전 항로 상호 플라세보) ===\n")
    run_all_routes("호르무즈 (효과 큰 순서, 괄호=사건 전 적합오차)", HORMUZ)
    run_all_routes("홍해", REDSEA)
