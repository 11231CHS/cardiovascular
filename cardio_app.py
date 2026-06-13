import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import urllib.request
import warnings
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import plotly.graph_objects as go

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# 0. 페이지 설정
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CardioCluster AI",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────────────────
# 1. 한글 폰트 (matplotlib)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def setup_korean_font():
    font_path = "NanumGothic.ttf"
    if not os.path.exists(font_path):
        urllib.request.urlretrieve(
            "https://github.com/google/fonts/raw/main/ofl/nanumgothic/NanumGothic-Regular.ttf",
            font_path,
        )
    fm.fontManager.addfont(font_path)
    prop = fm.FontProperties(fname=font_path)
    name = prop.get_name()
    plt.rcParams["font.family"] = name
    plt.rcParams["axes.unicode_minus"] = False
    return name

font_name = setup_korean_font()

# ─────────────────────────────────────────────────────────────────────────────
# 2. 모델 / 데이터 로드
#    Python 3.7은 pickle protocol 5를 못 읽으므로,
#    로드 실패 시 pickle5 패키지로 재시도 → protocol=4로 덮어씀
# ─────────────────────────────────────────────────────────────────────────────
def _safe_load(path):
    """protocol 5 pkl을 Python 3.7에서 읽는 방법:
       pickle5 패키지를 설치해 로드한 뒤 protocol=4로 재저장(1회만 실행)."""
    try:
        return joblib.load(path)
    except ValueError:
        # pickle5 설치 시도 (없으면 자동 설치)
        import subprocess, sys
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "pickle5", "-q"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        import pickle5  # noqa: F401 — 설치 확인용
        import pickle5 as pkl5
        with open(path, "rb") as f:
            obj = pkl5.load(f)
        # protocol=4 로 덮어써서 다음 실행부터는 정상 로드
        import pickle
        with open(path, "wb") as f:
            pickle.dump(obj, f, protocol=4)
        return obj

@st.cache_resource
def load_artifacts():
    scaler = _safe_load("scaler.pkl")
    model  = _safe_load("cardio_model.pkl")
    return scaler, model

@st.cache_data
def load_data():
    return pd.read_csv("cardio.csv")

# ─────────────────────────────────────────────────────────────────────────────
# 3. 군집 메타
# ─────────────────────────────────────────────────────────────────────────────
CLUSTER_META = {
    0: {
        "name": "중고령 음주 관리군",
        "emoji": "🍺",
        "accent": "#D97706",          # amber-600
        "bg":     "#FFFBEB",          # amber-50
        "border": "#FDE68A",          # amber-200
        "text":   "#78350F",          # amber-900
        "badge_bg": "#FEF3C7",
        "risk":   "음주 위험",
        "desc":   "나이가 많고 과체중·비만은 아니나, 30일 중 음주 일수(평균 24일)가 압도적으로 높은 폭음·애주가 그룹입니다.",
        "advice": [
            ("🍻 음주 빈도 줄이기",      "주 2~3회 이하, 한 번에 2잔 이내로 조절하세요."),
            ("🏃 유산소 운동 병행",       "걷기·수영 등 중강도 운동을 주 150분 이상 실천하세요."),
            ("🩺 정기 간 기능 검사",      "6~12개월마다 AST/ALT, γ-GTP 수치를 확인하세요."),
            ("💧 충분한 수분 섭취",       "음주 후 물을 충분히 마셔 심혈관 부담을 줄이세요."),
        ],
    },
    1: {
        "name": "청년 건강 청정군",
        "emoji": "✨",
        "accent": "#059669",
        "bg":     "#ECFDF5",
        "border": "#A7F3D0",
        "text":   "#064E3B",
        "badge_bg": "#D1FAE5",
        "risk":   "낮은 위험",
        "desc":   "나이가 매우 젊고(20~30대 위주), BMI와 음주량이 낮으며 전반적인 건강 상태가 매우 좋은 건강 그룹입니다.",
        "advice": [
            ("🥗 건강한 식습관 유지",     "가공식품·초가공 탄수화물 섭취를 최소화하세요."),
            ("🏋️ 근력 운동 추가",        "주 2회 이상 근력 운동으로 기초대사량을 높이세요."),
            ("😴 수면 규칙화",           "매일 7~8시간 숙면으로 대사·심혈관 건강을 지키세요."),
            ("🔍 연 1회 건강검진",        "혈압·혈당·콜레스테롤을 매년 확인하세요."),
        ],
    },
    2: {
        "name": "초고도비만 대사증후군 고위험군",
        "emoji": "⚠️",
        "accent": "#DC2626",
        "bg":     "#FEF2F2",
        "border": "#FECACA",
        "text":   "#7F1D1D",
        "badge_bg": "#FEE2E2",
        "risk":   "매우 높은 위험",
        "desc":   "나이는 중장년층이며, BMI가 매우 높고(평균 37 이상으로 고도비만) 전반적 건강 점수가 가장 낮은 고위험 그룹입니다.",
        "advice": [
            ("🏥 즉시 전문의 상담",       "BMI 35 이상은 의료적 개입이 필요합니다. 내분비·비만 전문의를 찾아가세요."),
            ("🥦 칼로리 조절 식단",       "단백질·식이섬유 위주로 전환해 주 0.5kg 내외 감량을 목표로 하세요."),
            ("🚶 저강도 운동부터",        "수중 걷기, 자전거 등 저충격 운동부터 시작하세요."),
            ("📉 대사지표 모니터링",      "혈압·공복혈당·중성지방·HDL을 3개월마다 점검하세요."),
        ],
    },
    3: {
        "name": "고령 만성질환 취약군",
        "emoji": "🏥",
        "accent": "#7C3AED",
        "bg":     "#F5F3FF",
        "border": "#DDD6FE",
        "text":   "#3B0764",
        "badge_bg": "#EDE9FE",
        "risk":   "만성질환 위험",
        "desc":   "나이가 많고(60대 이상 위주), 음주량은 적으나 전반적인 건강 점수가 낮은 노인 취약 그룹입니다.",
        "advice": [
            ("💊 복약 관리 철저히",       "만성질환 약은 빠짐없이 복용하고 임의로 중단하지 마세요."),
            ("🧘 낙상 예방 운동",         "한 발 서기, 태극권 등 균형 감각 훈련을 매일 실천하세요."),
            ("🥩 단백질 충분히 섭취",     "근감소증 예방을 위해 매 끼니 단백질을 반드시 포함하세요."),
            ("👨‍👩‍👧 사회적 연결 유지",   "가족·지역사회 활동에 참여해 인지·심혈관 건강을 지키세요."),
        ],
    },
}

AGE_LABELS = {
    1:"18-24", 2:"25-29", 3:"30-34", 4:"35-39", 5:"40-44",
    6:"45-49", 7:"50-54", 8:"55-59", 9:"60-64", 10:"65-69",
    11:"70-74", 12:"75-79", 13:"80+",
}

CHART_COLORS = {
    0: "#D97706", 1: "#059669", 2: "#DC2626", 3: "#7C3AED",
}

def age_to_category(age):
    if   age <= 24: return 1
    elif age <= 29: return 2
    elif age <= 34: return 3
    elif age <= 39: return 4
    elif age <= 44: return 5
    elif age <= 49: return 6
    elif age <= 54: return 7
    elif age <= 59: return 8
    elif age <= 64: return 9
    elif age <= 69: return 10
    elif age <= 74: return 11
    elif age <= 79: return 12
    else:           return 13

# ─────────────────────────────────────────────────────────────────────────────
# 4. 전역 CSS — 라이트 모드 / 세리프+산세리프 / Streamlit 기본 UI 숨김
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── 구글 폰트 ── */
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@400;600;700&family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;1,9..40,300&display=swap');

/* ── Streamlit 기본 UI 제거 ── */
#MainMenu, footer, header,
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="collapsedControl"],
[data-testid="stSidebarNav"]          { display: none !important; }

/* ── 전체 배경·폰트 (라이트) ── */
html, body,
[data-testid="stAppViewContainer"],
[data-testid="stAppViewContainer"] > .main,
[data-testid="stMain"] > div         {
    background-color: #F8F7F4 !important;
    color: #1C1917 !important;
    font-family: 'DM Sans', sans-serif !important;
}
.block-container { padding: 2.5rem 3.5rem 4rem !important; max-width: 1280px; }

/* ── 상단 얇은 액센트 바 ── */
[data-testid="stAppViewContainer"]::before {
    content: "";
    display: block;
    position: fixed;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, #D97706 0%, #DC2626 35%, #7C3AED 70%, #059669 100%);
    z-index: 9999;
}

/* ══════════════════════════════════
   카드 시스템
══════════════════════════════════ */
.card {
    background: #FFFFFF;
    border: 1px solid #E7E5E0;
    border-radius: 16px;
    padding: 28px 32px;
    margin-bottom: 16px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06), 0 4px 16px rgba(0,0,0,0.04);
}
.card-sm {
    background: #FFFFFF;
    border: 1px solid #E7E5E0;
    border-radius: 12px;
    padding: 16px 20px;
    margin-bottom: 10px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
}

/* ══════════════════════════════════
   타이포그래피
══════════════════════════════════ */
.eyebrow {
    font-size: 0.67rem;
    font-weight: 600;
    letter-spacing: 2.5px;
    text-transform: uppercase;
    color: #A8A29E;
    margin-bottom: 6px;
}
.display-title {
    font-family: 'Cormorant Garamond', serif;
    font-size: 3.8rem;
    font-weight: 700;
    color: #1C1917;
    letter-spacing: -1.5px;
    line-height: 1.05;
    margin: 0 0 10px;
}
.section-title {
    font-family: 'Cormorant Garamond', serif;
    font-size: 1.65rem;
    font-weight: 600;
    color: #1C1917;
    margin-bottom: 4px;
    line-height: 1.2;
}
.body-lg  { font-size: 1rem;   color: #57534E; line-height: 1.65; }
.body-sm  { font-size: 0.83rem; color: #78716C; line-height: 1.55; }
.caption  { font-size: 0.72rem; color: #A8A29E; letter-spacing: 0.3px; }

/* ══════════════════════════════════
   메트릭 카드
══════════════════════════════════ */
.metric-box {
    background: #FFFFFF;
    border: 1px solid #E7E5E0;
    border-radius: 12px;
    padding: 18px 16px 14px;
    text-align: center;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
}
.metric-num {
    font-family: 'Cormorant Garamond', serif;
    font-size: 2.2rem;
    font-weight: 700;
    color: #1C1917;
    line-height: 1;
    margin-bottom: 2px;
}
.metric-lbl {
    font-size: 0.67rem;
    font-weight: 600;
    letter-spacing: 1.8px;
    text-transform: uppercase;
    color: #A8A29E;
}
.metric-sub {
    font-size: 0.75rem;
    color: #78716C;
    margin-top: 3px;
}

/* ══════════════════════════════════
   배지
══════════════════════════════════ */
.badge {
    display: inline-block;
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    border-radius: 100px;
    padding: 3px 11px;
    margin-bottom: 10px;
}

/* ══════════════════════════════════
   결과 배너
══════════════════════════════════ */
.result-banner {
    border-radius: 16px;
    padding: 26px 30px;
    margin-bottom: 18px;
    border-width: 1px;
    border-style: solid;
}
.result-name {
    font-family: 'Cormorant Garamond', serif;
    font-size: 1.9rem;
    font-weight: 700;
    margin-bottom: 6px;
}

/* ══════════════════════════════════
   온보딩 군집 카드
══════════════════════════════════ */
.cluster-card {
    border-radius: 14px;
    padding: 20px 22px;
    border-width: 1px;
    border-style: solid;
    margin-bottom: 12px;
}
.cluster-name {
    font-weight: 600;
    font-size: 0.92rem;
    margin-bottom: 5px;
}
.cluster-desc { font-size: 0.79rem; line-height: 1.5; }

/* ══════════════════════════════════
   조언 카드
══════════════════════════════════ */
.advice-item {
    background: #FAFAF9;
    border: 1px solid #E7E5E0;
    border-left-width: 3px;
    border-radius: 10px;
    padding: 13px 16px;
    margin-bottom: 9px;
}
.advice-head { font-weight: 600; font-size: 0.87rem; color: #1C1917; margin-bottom: 3px; }
.advice-body { font-size: 0.81rem; color: #78716C; line-height: 1.5; }

/* ══════════════════════════════════
   분포 바
══════════════════════════════════ */
.dist-row {
    background: #FFFFFF;
    border: 1px solid #E7E5E0;
    border-radius: 10px;
    padding: 12px 16px;
    margin-bottom: 8px;
}
.dist-bar-bg  { background: #F5F5F4; border-radius: 4px; height: 7px; margin-top: 6px; overflow: hidden; }
.dist-bar-fill { height: 7px; border-radius: 4px; }

/* ══════════════════════════════════
   Streamlit 위젯 스타일
══════════════════════════════════ */
[data-testid="stNumberInput"] input {
    background: #FFFFFF !important;
    border: 1px solid #D6D3D1 !important;
    color: #1C1917 !important;
    border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif !important;
}
[data-testid="stNumberInput"] input:focus {
    border-color: #7C3AED !important;
    box-shadow: 0 0 0 3px rgba(124,58,237,0.12) !important;
}
div[data-baseweb="select"] > div {
    background: #FFFFFF !important;
    border-color: #D6D3D1 !important;
    border-radius: 8px !important;
    color: #1C1917 !important;
}
[data-testid="stButton"] > button {
    background: #1C1917 !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    padding: 13px 0 !important;
    width: 100% !important;
    letter-spacing: 0.2px !important;
    transition: background 0.18s ease !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.12) !important;
}
[data-testid="stButton"] > button:hover {
    background: #44403C !important;
}
/* 보조 버튼 (두 번째) */
[data-testid="stButton"]:nth-child(2) > button {
    background: #F5F5F4 !important;
    color: #44403C !important;
    box-shadow: none !important;
    border: 1px solid #E7E5E0 !important;
}
[data-testid="stButton"]:nth-child(2) > button:hover {
    background: #E7E5E0 !important;
}

/* 슬라이더 */
[data-testid="stSlider"] > div > div > div > div {
    background: #1C1917 !important;
}

/* 인풋 레이블 */
.input-lbl {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 1.2px;
    text-transform: uppercase;
    color: #78716C;
    margin-bottom: 4px;
    margin-top: 2px;
}

/* 구분선 */
hr { border-color: #E7E5E0 !important; margin: 28px 0 !important; }

/* 탭 */
[data-testid="stTabs"] [data-testid="stTab"] {
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    color: #78716C !important;
}
[data-testid="stTabs"] [aria-selected="true"] {
    color: #1C1917 !important;
    border-bottom-color: #1C1917 !important;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# 5. 세션 상태
# ─────────────────────────────────────────────────────────────────────────────
if "page" not in st.session_state:
    st.session_state["page"] = "landing"
if "result" not in st.session_state:
    st.session_state["result"] = None

def go_to(page):
    st.session_state["page"] = page
    # Python 3.7 호환: experimental_rerun 사용
    try:
        st.rerun()
    except AttributeError:
        st.experimental_rerun()

# ══════════════════════════════════════════════════════════════════════════════
# ▌LANDING
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state["page"] == "landing":

    left, divider, right = st.columns([1.05, 0.04, 0.91])

    with left:
        st.markdown('<div style="padding-top:6vh"></div>', unsafe_allow_html=True)
        st.markdown('<p class="eyebrow">🫀 Cardiovascular Risk Intelligence</p>', unsafe_allow_html=True)
        st.markdown('<h1 class="display-title">CardioCluster<br/>AI</h1>', unsafe_allow_html=True)
        st.markdown(
            '<p class="body-lg">머신러닝 K-Means 군집화로<br/>'
            '심혈관 건강 위험 프로파일을 분석합니다.</p>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<p class="caption" style="margin-top:6px; margin-bottom:32px;">'
            '나이 · BMI · 전반적 건강상태 · 음주량 — 4가지 지표 기반</p>',
            unsafe_allow_html=True,
        )
        if st.button("분석 시작하기  →", key="land_start"):
            go_to("input")

        # 하단 통계 바
        st.markdown('<div style="margin-top:48px; display:flex; gap:32px;">', unsafe_allow_html=True)
        for stat_val, stat_lbl in [("4","건강 군집"), ("K-Means","모델"), ("4개","핵심 지표")]:
            st.markdown(
                f'<div style="display:inline-block; margin-right:36px;">'
                f'  <span style="font-family:\'Cormorant Garamond\',serif;'
                f'    font-size:1.9rem; font-weight:700; color:#1C1917;">{stat_val}</span>'
                f'  <span class="caption" style="margin-left:6px;">{stat_lbl}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
        st.markdown('</div>', unsafe_allow_html=True)

    with divider:
        st.markdown(
            '<div style="width:1px; background:#E7E5E0; height:72vh; margin:auto;"></div>',
            unsafe_allow_html=True,
        )

    with right:
        st.markdown('<div style="padding-top:6vh"></div>', unsafe_allow_html=True)
        st.markdown('<p class="eyebrow">4가지 건강 군집</p>', unsafe_allow_html=True)
        for cid, m in CLUSTER_META.items():
            st.markdown(
                f'<div class="cluster-card" style="background:{m["bg"]}; border-color:{m["border"]};">'
                f'  <div class="cluster-name" style="color:{m["text"]};">'
                f'      {m["emoji"]} &nbsp;군집 {cid} &nbsp;·&nbsp; {m["name"]}'
                f'  </div>'
                f'  <div class="cluster-desc" style="color:{m["text"]}99;">{m["desc"]}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

# ══════════════════════════════════════════════════════════════════════════════
# ▌INPUT
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state["page"] == "input":

    st.markdown('<p class="eyebrow">🫀 CardioCluster AI</p>', unsafe_allow_html=True)
    st.markdown('<h1 class="display-title" style="font-size:2.5rem;">환자 정보 입력</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="body-lg" style="margin-bottom:28px;">'
        'AI가 4가지 지표를 바탕으로 건강 군집을 예측합니다.</p>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="card">', unsafe_allow_html=True)

    r1c1, r1c2, r1c3 = st.columns(3, gap="large")

    # ── 열 1 ──
    with r1c1:
        st.markdown('<p class="input-lbl">나이 (세)</p>', unsafe_allow_html=True)
        age_raw = st.number_input("나이", min_value=18, max_value=100, value=40,
                                  label_visibility="collapsed")
        st.markdown('<p class="input-lbl" style="margin-top:14px;">키 (cm)</p>', unsafe_allow_html=True)
        height_in = st.number_input("키", min_value=100.0, max_value=230.0,
                                    value=170.0, step=0.1, label_visibility="collapsed")

    # ── 열 2 ──
    with r1c2:
        st.markdown('<p class="input-lbl">몸무게 (kg)</p>', unsafe_allow_html=True)
        weight_in = st.number_input("몸무게", min_value=30.0, max_value=200.0,
                                    value=70.0, step=0.1, label_visibility="collapsed")

        bmi_val = weight_in / ((height_in / 100) ** 2)
        if   bmi_val < 18.5: bmi_lbl, bmi_clr = "저체중",    "#3B82F6"
        elif bmi_val < 23:   bmi_lbl, bmi_clr = "정상",      "#059669"
        elif bmi_val < 25:   bmi_lbl, bmi_clr = "과체중",    "#D97706"
        elif bmi_val < 30:   bmi_lbl, bmi_clr = "비만 1단계","#EA580C"
        else:                bmi_lbl, bmi_clr = "고도비만",  "#DC2626"

        st.markdown(
            f'<div class="metric-box" style="margin-top:10px; border-top:3px solid {bmi_clr};">'
            f'  <div class="metric-num" style="color:{bmi_clr};">{bmi_val:.1f}</div>'
            f'  <div class="metric-lbl">BMI</div>'
            f'  <div class="metric-sub">{bmi_lbl}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # ── 열 3 ──
    with r1c3:
        st.markdown('<p class="input-lbl">전반적 건강 상태(자신이 생각하기에)</p>', unsafe_allow_html=True)
        health_opts = {
            "1 – Poor (매우 나쁨)": 1,
            "2 – Fair (나쁨)": 2,
            "3 – Good (보통)": 3,
            "4 – Very Good (좋음)": 4,
            "5 – Excellent (매우 좋음)": 5,
        }
        health_sel = st.selectbox("건강상태", list(health_opts.keys()),
                                  index=2, label_visibility="collapsed")
        health_in = health_opts[health_sel]

        st.markdown('<p class="input-lbl" style="margin-top:14px;">지난 30일 음주 일수</p>',
                    unsafe_allow_html=True)
        alc_in = st.slider("음주 일수", 0, 30, 5, label_visibility="collapsed")
        st.markdown(
            f'<p class="caption" style="margin-top:2px;">'
            f'선택: <b style="color:#1C1917;">{alc_in}일</b> / 30일</p>',
            unsafe_allow_html=True,
        )

    st.markdown('</div>', unsafe_allow_html=True)

    btn_c1, btn_c2, _ = st.columns([1, 1, 2], gap="small")
    with btn_c1:
        predict_clicked = st.button("🔍  군집 예측하기", key="predict")
    with btn_c2:
        back_clicked = st.button("← 처음으로", key="back_land")

    if back_clicked:
        go_to("landing")

    if predict_clicked:
        try:
            scaler, model = load_artifacts()
        except Exception as e:
            st.error(f"모델 로드 실패: {e}\n\n→ convert_pkl.py 를 먼저 실행하세요.")
            st.stop()

        age_cat = age_to_category(age_raw)
        new_df  = pd.DataFrame(
            [[age_cat, bmi_val, health_in, alc_in]],
            columns=["나이범주", "BMI", "전반적건강상태", "음주량"],
        )
        scaled  = scaler.transform(new_df)
        pred    = int(model.predict(scaled)[0])

        try:
            df   = load_data()
            dist = df["cluster_4"].value_counts(normalize=True).sort_index() \
                   if "cluster_4" in df.columns else {i: 0.25 for i in range(4)}
        except Exception:
            df   = pd.DataFrame()
            dist = {i: 0.25 for i in range(4)}

        st.session_state["result"] = {
            "cluster": pred,
            "age_raw": age_raw, "age_cat": age_cat,
            "bmi": bmi_val, "bmi_lbl": bmi_lbl, "bmi_clr": bmi_clr,
            "health": health_in, "alc": alc_in,
            "dist": dist, "df": df,
        }
        go_to("result")

# ══════════════════════════════════════════════════════════════════════════════
# ▌RESULT
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state["page"] == "result":

    r   = st.session_state["result"]
    cid = r["cluster"]
    m   = CLUSTER_META[cid]

    st.markdown('<p class="eyebrow">🫀 CardioCluster AI — 분석 결과</p>', unsafe_allow_html=True)

    # ── 상단 메트릭 4종 ──────────────────────────────────────────────────────
    mc = st.columns(4, gap="medium")
    metrics_data = [
        ("나이", f"{int(r['age_raw'])}세",
         f"범주 {r['age_cat']} · {AGE_LABELS[r['age_cat']]}"),
        ("BMI", f"{r['bmi']:.1f}",
         r["bmi_lbl"]),
        ("건강 점수", f"{r['health']}/5",
         ["–","Poor","Fair","Good","Very Good","Excellent"][r["health"]]),
        ("음주 일수", f"{r['alc']}일",
         "/ 30일 기준"),
    ]
    for col, (lbl, val, sub) in zip(mc, metrics_data):
        with col:
            st.markdown(
                f'<div class="metric-box">'
                f'  <div class="metric-num">{val}</div>'
                f'  <div class="metric-lbl">{lbl}</div>'
                f'  <div class="metric-sub">{sub}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.write("")

    # ── 결과 배너 ────────────────────────────────────────────────────────────
    st.markdown(
        f'<div class="result-banner" style="background:{m["bg"]}; border-color:{m["border"]}; color:{m["text"]};">'
        f'  <span class="badge" style="background:{m["badge_bg"]}; color:{m["text"]};">{m["risk"]}</span>'
        f'  <div class="result-name">{m["emoji"]} &nbsp;군집 {cid} &nbsp;·&nbsp; {m["name"]}</div>'
        f'  <p class="body-sm" style="color:{m["text"]}bb; margin:0;">{m["desc"]}</p>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ── 2열 레이아웃 ─────────────────────────────────────────────────────────
    left_col, right_col = st.columns([1, 1], gap="large")

    with left_col:
        st.markdown('<p class="eyebrow">맞춤 건강 조언</p>', unsafe_allow_html=True)
        st.markdown(f'<p class="section-title">군집 {cid} 권장 관리</p>', unsafe_allow_html=True)
        for title, body in m["advice"]:
            st.markdown(
                f'<div class="advice-item" style="border-left-color:{m["accent"]};">'
                f'  <div class="advice-head">{title}</div>'
                f'  <div class="advice-body">{body}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    with right_col:
        st.markdown('<p class="eyebrow">군집 분포</p>', unsafe_allow_html=True)
        st.markdown('<p class="section-title">전체 데이터 기준</p>', unsafe_allow_html=True)
        dist = r["dist"]
        for i in range(4):
            pct  = float(dist.get(i, 0.25)) * 100
            mm   = CLUSTER_META[i]
            is_me = (i == cid)
            st.markdown(
                f'<div class="dist-row" style="'
                f'{"border-color:"+mm["accent"]+"; background:"+mm["bg"]+";" if is_me else ""}">'
                f'  <div style="display:flex; justify-content:space-between; align-items:center;">'
                f'    <span style="font-size:0.85rem; color:#1C1917; '
                f'          font-weight:{"600" if is_me else "400"};">'
                f'      {"▶ " if is_me else ""}{mm["emoji"]} {mm["name"]}'
                f'    </span>'
                f'    <span style="font-size:0.85rem; color:{mm["accent"]}; font-weight:700;">'
                f'      {pct:.1f}%'
                f'    </span>'
                f'  </div>'
                f'  <div class="dist-bar-bg">'
                f'    <div class="dist-bar-fill" style="width:{min(pct,100):.1f}%; background:{mm["accent"]};"></div>'
                f'  </div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.markdown("<hr/>", unsafe_allow_html=True)

    # ── 시각화 ───────────────────────────────────────────────────────────────
    st.markdown('<p class="eyebrow">시각화</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-title">군집화 산점도 &amp; 분포</p>', unsafe_allow_html=True)

    df = r["df"]
    PLOTLY_LIGHT = dict(
        template="plotly_white",
        paper_bgcolor="#F8F7F4",
        plot_bgcolor="#FFFFFF",
        font=dict(family="DM Sans, sans-serif", color="#1C1917"),
        margin=dict(l=10, r=10, t=36, b=10),
        height=390,
    )

    if not df.empty and "cluster_4" in df.columns:
        tab1, tab2, tab3 = st.tabs(["📊 BMI × 나이범주", "📈 BMI 분포", "🍺 음주량 × BMI"])

        with tab1:
            df_s = df.sample(n=min(8000, len(df)), random_state=42)
            fig = go.Figure()
            for cn in sorted(df_s["cluster_4"].unique()):
                sub = df_s[df_s["cluster_4"] == cn]
                fig.add_trace(go.Scatter(
                    x=sub["BMI"], y=sub["나이범주"], mode="markers",
                    marker=dict(color=CHART_COLORS.get(int(cn),"#888"),
                                size=4, opacity=0.45, line=dict(width=0)),
                    name=f"군집 {cn} {CLUSTER_META[int(cn)]['name']}",
                ))
            fig.add_trace(go.Scatter(
                x=[r["bmi"]], y=[r["age_cat"]], mode="markers",
                marker=dict(symbol="x", size=17, color=m["accent"],
                            line=dict(color="#1C1917", width=2)),
                name=f"현재 환자 (군집 {cid})",
            ))
            fig.update_layout(
                xaxis_title="BMI", yaxis_title="나이 범주",
                legend=dict(orientation="h", y=-0.2, x=0.5, xanchor="center", font_size=11),
                **PLOTLY_LIGHT
            )
            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            fig2 = go.Figure()
            for cn in sorted(df["cluster_4"].unique()):
                sub = df[df["cluster_4"] == cn]
                fig2.add_trace(go.Violin(
                    x=sub["BMI"], name=f"군집 {cn}",
                    line_color=CHART_COLORS.get(int(cn), "#888"),
                    fillcolor=CHART_COLORS.get(int(cn), "#888"),
                    opacity=0.45, meanline_visible=True,
                ))
            fig2.add_vline(x=r["bmi"], line_dash="dash",
                           line_color=m["accent"], line_width=2,
                           annotation_text=f"현재 BMI {r['bmi']:.1f}",
                           annotation_font_color=m["accent"])
            fig2.update_layout(violinmode="overlay", xaxis_title="BMI", **PLOTLY_LIGHT)
            st.plotly_chart(fig2, use_container_width=True)

        with tab3:
            if "음주량" in df.columns:
                df_s2 = df.sample(n=min(6000, len(df)), random_state=7)
                fig3 = go.Figure()
                for cn in sorted(df_s2["cluster_4"].unique()):
                    sub = df_s2[df_s2["cluster_4"] == cn]
                    fig3.add_trace(go.Scatter(
                        x=sub["음주량"], y=sub["BMI"], mode="markers",
                        marker=dict(color=CHART_COLORS.get(int(cn), "#888"),
                                    size=4, opacity=0.4),
                        name=f"군집 {cn} {CLUSTER_META[int(cn)]['name']}",
                    ))
                fig3.add_trace(go.Scatter(
                    x=[r["alc"]], y=[r["bmi"]], mode="markers",
                    marker=dict(symbol="x", size=17, color=m["accent"],
                                line=dict(color="#1C1917", width=2)),
                    name=f"현재 환자 (군집 {cid})",
                ))
                fig3.update_layout(
                    xaxis_title="음주량 (일/30일)", yaxis_title="BMI",
                    legend=dict(orientation="h", y=-0.2, x=0.5, xanchor="center", font_size=11),
                    **PLOTLY_LIGHT
                )
                st.plotly_chart(fig3, use_container_width=True)
            else:
                st.info("데이터에 음주량 컬럼이 없습니다.")
    else:
        st.info("cluster_4 컬럼이 없어 시각화를 건너뜁니다.")

    # ── 하단 버튼 ────────────────────────────────────────────────────────────
    st.write("")
    b1, b2, _ = st.columns([1, 1, 2], gap="small")
    with b1:
        if st.button("← 다시 입력", key="re_input"):
            go_to("input")
    with b2:
        if st.button("🏠 처음으로", key="re_land"):
            go_to("landing")
    
