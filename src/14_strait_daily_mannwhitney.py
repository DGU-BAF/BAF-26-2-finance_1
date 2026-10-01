"""
14_strait_daily_mannwhitney.py — 일별 통항량으로 '배가 실제로 우회·급감했는지' 통계 검증

이 스크립트가 왜 필요한가
  지금까지(01~13번)는 KCCI '운임' 데이터로 사건 효과를 봤음. 운임이 오른 게 정말 해협
  위기 때문인지는 운임 데이터만으로 완전히 구분이 안 되므로, 운임과 **출처가 다른** 물리적
  관측 데이터(IMF PortWatch 해협별 일별 통과 선박 수, 위성 AIS 추적 기반)로 "진짜로 배가
  줄었는가"를 직접 확인함 — 운임 데이터의 교차검증.

이 스크립트가 만드는 결과물
  중간발표 노션 3-3절 "물리적 통항량으로 우회 운항 통계적 검증" 표(바브엘만데브 p=3.5×10⁻¹⁹,
  희망봉 p=8.1×10⁻⁹, 호르무즈 p=5.8×10⁻²⁸)와, 같은 절의 일별 통항량 그래프 2장.

사건 전/후 12주(84일) 구간. 중간발표 노션 페이지 Step1(2-2절)의 정의와 동일하게 맞춤

실행: python3 14_strait_daily_mannwhitney.py
"""

from datetime import date

import matplotlib

matplotlib.use("Agg")  # 화면 없는 서버 환경에서도 그림을 파일로 저장하기 위한 비대화형 백엔드
import matplotlib.pyplot as plt
import numpy as np
import openpyxl
from matplotlib import font_manager
from scipy import stats

from common import BASE

XLSX = BASE / "data" / "운임비" / "해협리스크_교차검증" / "해협리스크_운임_비교데이터_v3.xlsx"
OUT_DIR = BASE / "analysis" / "운임분석" / "output"

# 사건 전/후 12주(84일) 구간. 중간발표 노션 페이지 Step1(2-2절)의 정의와 동일하게 맞춤
WINDOWS = {
    "홍해": {
        "사건일": date(2023, 11, 19),
        "평시": (date(2023, 8, 28), date(2023, 11, 13)),
        "사건후": (date(2023, 11, 20), date(2024, 2, 5)),
    },
    "호르무즈": {
        "사건일": date(2026, 3, 1),
        "평시": (date(2025, 12, 8), date(2026, 2, 23)),
        "사건후": (date(2026, 3, 3), date(2026, 5, 18)),
    },
}


def load_daily():
    """'해협_일별통과' 시트를 (날짜, 해협명) → 값 조회가 쉬운 레코드 리스트로 읽어옴.
    data_only=True로 열어야 수식이 아니라 계산된 값이 그대로 들어옴."""
    wb = openpyxl.load_workbook(XLSX, data_only=True)
    ws = wb["해협_일별통과"]
    headers = [c.value for c in ws[1]]
    recs = []
    for row in ws.iter_rows(min_row=2, values_only=True):
        d = dict(zip(headers, row))
        v = d["date"]
        # 셀이 문자열("2023-01-01")로 들어오는 경우와 datetime으로 들어오는 경우를 둘 다 처리함
        if isinstance(v, str):
            y, m, day = v.split("-")
            d["date"] = date(int(y), int(m), int(day))
        elif hasattr(v, "date"):
            d["date"] = v.date()
        recs.append(d)
    return recs


def window_values(recs, port, start, end, col="n_container"):
    """특정 해협(port)의 [start, end] 구간 일별 값을 오름차순 정렬된 (날짜 리스트, 값 배열)로 반환함."""
    pts = [(r["date"], r[col]) for r in recs if r["portname"] == port and start <= r["date"] <= end]
    pts.sort()
    dates = [p[0] for p in pts]
    vals = np.array([p[1] for p in pts])
    return dates, vals


def rank_biserial(x, y, u_stat):
    """Mann-Whitney U 통계량으로 효과크기(rank-biserial correlation r)를 계산함.
    |r|이 1에 가까울수록 두 그룹의 분포가 거의 안 겹친다는 뜻(효과가 큼)."""
    return 1 - (2 * u_stat) / (len(x) * len(y))


def run_test(recs, label, port, col="n_container"):
    """한 해협에 대해 평시 vs 사건후 구간을 비교하는 Mann-Whitney U 검정을 실행하고 결과를 출력함."""
    w = WINDOWS[label]
    _, pre = window_values(recs, port, *w["평시"], col=col)
    _, post = window_values(recs, port, *w["사건후"], col=col)

    # alternative="two-sided": 늘었는지 줄었는지 방향을 미리 정하지 않고 "다른가"만 검정
    u_stat, p_val = stats.mannwhitneyu(pre, post, alternative="two-sided")
    r_eff = rank_biserial(pre, post, u_stat)

    print(f"--- {port} ({label} 기준 창) ---")
    print(f"  평시   n={len(pre):3d}  중앙값={np.median(pre):5.1f}  평균={pre.mean():5.1f}  "
          f"IQR=[{np.percentile(pre, 25):.1f}, {np.percentile(pre, 75):.1f}]")
    print(f"  사건후 n={len(post):3d}  중앙값={np.median(post):5.1f}  평균={post.mean():5.1f}  "
          f"IQR=[{np.percentile(post, 25):.1f}, {np.percentile(post, 75):.1f}]")
    print(f"  Mann-Whitney U p값 = {p_val:.6g}   효과크기(rank-biserial r) = {r_eff:.3f}\n")
    return pre, post, p_val, r_eff


def find_korean_font():
    """그래프에 한글이 깨지지 않게 시스템에 설치된 한글 폰트를 찾아 적용함."""
    for name in ["AppleGothic", "Apple SD Gothic Neo", "NanumGothic"]:
        try:
            font_manager.findfont(name, fallback_to_default=False)
            plt.rcParams["font.family"] = name
            return name
        except Exception:
            continue
    return None


def plot_timeline(recs, port_a, label_a, port_b, label_b, start, end, event_day, title, out_path):
    """두 해협(주로 진입로 vs 우회로)의 일별 통항량을 한 그래프에 겹쳐 그리고 사건일에 수직선을 그음."""
    dates_a, vals_a = window_values(recs, port_a, start, end)
    dates_b, vals_b = window_values(recs, port_b, start, end)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(dates_a, vals_a, label=label_a, color="crimson")
    if port_b:
        ax.plot(dates_b, vals_b, label=label_b, color="steelblue")
    ax.axvline(event_day, color="black", linestyle="--", linewidth=1, label=f"사건일({event_day})")
    ax.set_title(title)
    ax.set_ylabel("일별 통과 선박 수(척)")
    ax.legend()
    fig.autofmt_xdate()
    plt.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"  그래프 저장: {out_path}")


if __name__ == "__main__":
    find_korean_font()
    recs = load_daily()

    print("=" * 70)
    print("사건 전후 12주 일별 통항량 분포 비교 (Mann-Whitney U, 비모수 검정)")
    print("=" * 70)
    print("t-검정이 아니라 비모수 검정을 쓴 이유: 일별 통과선박수는 정수 카운트라 0 근처에")
    print("몰려있고(특히 호르무즈 사건후는 중앙값 0) 정규분포를 가정하기 어려움\n")

    run_test(recs, "홍해", "Bab el-Mandeb Strait")
    run_test(recs, "홍해", "Cape of Good Hope")
    run_test(recs, "호르무즈", "Strait of Hormuz")

    print("=" * 70)
    print("시계열 그래프 저장")
    print("=" * 70)
    plot_timeline(
        recs,
        "Bab el-Mandeb Strait", "바브엘만데브(홍해 진입로)",
        "Cape of Good Hope", "희망봉(우회로)",
        date(2023, 8, 28), date(2024, 2, 5), date(2023, 11, 19),
        "홍해 사건 전후 일별 컨테이너선 통항량",
        OUT_DIR / "홍해_바브엘만데브_희망봉_일별통항량.png",
    )
    plot_timeline(
        recs,
        "Strait of Hormuz", "호르무즈 해협",
        None, None,
        date(2025, 12, 8), date(2026, 5, 18), date(2026, 3, 1),
        "호르무즈 해협 사건 전후 일별 컨테이너선 통항량",
        OUT_DIR / "호르무즈_일별통항량.png",
    )
