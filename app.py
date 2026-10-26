
import streamlit as st
import pandas as pd
import numpy as np

from pathlib import Path
from io import BytesIO
import zipfile

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle


# =========================================================
# CARWISE - SMART CAR FINDER
# =========================================================

st.set_page_config(
    page_title="CarWise | Smart Car Finder",
    page_icon="🚗",
    layout="wide",
)

DATA_FILE = Path(__file__).parent / "cars.csv"


# =========================================================
# PAGE STYLING
# =========================================================

st.markdown("""
<style>
    :root {
        --cw-navy: #102744;
        --cw-blue: #1769d2;
        --cw-pale: #eef5ff;
        --cw-border: #dce7f5;
        --cw-text: #172b4d;
        --cw-muted: #61718a;
    }
    .stApp {
        background: linear-gradient(180deg, #f4f8ff 0%, #f8faff 55%, #eef4fb 100%);
        color: var(--cw-text);
    }
    .block-container {
        padding-top: 1.15rem;
        padding-bottom: 2rem;
        max-width: 1500px;
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #102744 0%, #16385f 100%);
        border-right: 1px solid #25466e;
    }
    [data-testid="stSidebar"] * { color: #f4f8ff; }
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        color: #e5efff;
    }
    [data-testid="stSidebar"] [data-baseweb="select"] > div,
    [data-testid="stSidebar"] [data-baseweb="input"] > div {
        background: #ffffff;
        border-radius: 9px;
    }
    [data-testid="stSidebar"] [data-baseweb="select"] span,
    [data-testid="stSidebar"] [data-baseweb="input"] input {
        color: #142a49;
    }
    .hero {
        position: relative;
        overflow: hidden;
        background: linear-gradient(115deg, #0d2745 0%, #174477 58%, #2876c8 100%);
        color: white;
        padding: 30px 32px;
        border-radius: 20px;
        margin: 0 0 18px 0;
        border: 1px solid rgba(255,255,255,.18);
        box-shadow: 0 12px 30px rgba(22, 59, 105, .16);
    }
    .hero:after {
        content: "🚘";
        position: absolute;
        right: 4%;
        bottom: -24px;
        font-size: 118px;
        opacity: .16;
        transform: rotate(-5deg);
        pointer-events: none;
    }
    .hero h1 { color: white !important; font-size: clamp(27px, 3vw, 39px); margin: 0 0 8px 0; }
    .hero p { color: #e3efff !important; font-size: 16px; margin: 0; max-width: 760px; }
    h1, h2, h3 { color: #142d52; letter-spacing: -.02em; }
    h2, h3 { margin-top: .35rem; }
    div[data-testid="stMetric"] {
        background: rgba(255,255,255,.94);
        padding: 15px 16px;
        border: 1px solid var(--cw-border);
        border-radius: 15px;
        box-shadow: 0 4px 14px rgba(25, 65, 115, .045);
    }
    div[data-testid="stMetricLabel"] { color: var(--cw-muted); }
    div[data-testid="stMetricValue"] { color: #153b70; font-weight: 750; }
    div[data-testid="stVerticalBlockBorderWrapper"] {
        border-color: var(--cw-border);
        border-radius: 14px;
    }
    div.stButton > button, div.stDownloadButton > button {
        border-radius: 10px;
        font-weight: 650;
        border: 1px solid #cbdcf2;
        transition: all .15s ease;
    }
    div.stButton > button[kind="primary"] {
        background: linear-gradient(100deg, #1769d2, #2788ed);
        border: none;
        color: white;
    }
    div.stButton > button:hover, div.stDownloadButton > button:hover {
        border-color: #1769d2;
        color: #1458a7;
        box-shadow: 0 4px 12px rgba(23, 105, 210, .12);
    }
    div[data-testid="stAlert"] { border-radius: 12px; }
    hr { border-color: #dce7f5; }
    [data-testid="stDataFrame"], [data-testid="stTable"] { border-radius: 12px; overflow: hidden; }
    @media (max-width: 760px) {
        .block-container { padding: .8rem .7rem 1.5rem .7rem; }
        .hero { padding: 22px 20px; }
        .hero:after { font-size: 74px; right: 1%; }
    }
</style>
""", unsafe_allow_html=True)


# =========================================================
# LOAD AND VALIDATE DATA
# =========================================================

@st.cache_data
def load_cars(file_path, modified_time):
    df = pd.read_csv(file_path)
    df.columns = [str(column).strip().lower() for column in df.columns]

    required_columns = [
        "name",
        "price",
        "fuel",
        "transmission",
        "seats",
        "type",
        "mileage",
        "safety",
    ]

    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing columns in cars.csv: "
            + ", ".join(missing_columns)
        )

    if "maintenance" not in df.columns:
        df["maintenance"] = 3

    if "image_url" not in df.columns:
        df["image_url"] = ""

    for column in ["name", "fuel", "transmission", "type"]:
        df[column] = (
            df[column]
            .fillna("Unknown")
            .astype(str)
            .str.strip()
        )

    for column in [
        "price", "seats", "mileage", "safety", "maintenance"
    ]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(
        subset=[
            "name", "price", "seats", "mileage",
            "safety", "maintenance"
        ]
    ).copy()

    df = df[
        (df["price"] > 0)
        & (df["seats"] > 0)
        & (df["mileage"] >= 0)
    ].copy()

    df["safety"] = df["safety"].clip(0, 5)
    df["maintenance"] = df["maintenance"].clip(1, 5)
    df["image_url"] = df["image_url"].fillna("").astype(str).str.strip()

    return df.reset_index(drop=True)


if not DATA_FILE.exists():
    st.error("cars.csv was not found beside app.py.")
    st.stop()

try:
    cars = load_cars(str(DATA_FILE), DATA_FILE.stat().st_mtime)
except Exception as error:
    st.error(f"Could not load cars.csv: {error}")
    st.stop()

if cars.empty:
    st.error("No valid car records were found in cars.csv.")
    st.stop()


# =========================================================
# RECOMMENDATION ENGINE
# =========================================================

categorical_features = ["fuel", "transmission", "type"]

numerical_features = [
    "price", "seats", "mileage", "safety", "maintenance"
]


@st.cache_resource
def build_recommendation_engine(data):
    feature_transformer = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_features,
            ),
            (
                "numerical",
                MinMaxScaler(),
                numerical_features,
            ),
        ]
    )

    feature_matrix = feature_transformer.fit_transform(
        data[categorical_features + numerical_features]
    )

    return feature_transformer, feature_matrix


try:
    transformer, car_feature_matrix = build_recommendation_engine(cars)
except Exception as error:
    st.error(f"Could not prepare the recommendation engine: {error}")
    st.stop()


def money(value):
    return f"₹{float(value):,.2f} lakh"


def calculate_ownership_score(car, budget, min_mileage, min_safety, max_maintenance,
                              fuel, transmission, seats, car_type):
    """Transparent rule-based suitability score out of 100 (not a prediction)."""
    budget_score = max(0.0, min(100.0, (1 - float(car["price"]) / max(float(budget), 0.01)) * 100 + 35))
    mileage_score = 100.0 if min_mileage <= 0 else max(0.0, min(100.0, float(car["mileage"]) / float(min_mileage) * 70))
    maintenance_score = max(0.0, min(100.0, (6 - float(car["maintenance"])) / 5 * 100))
    safety_score = max(0.0, min(100.0, float(car["safety"]) / 5 * 100))

    checks = []
    if fuel == "Any" or str(car["fuel"]).lower() == str(fuel).lower():
        checks.append(1.0)
    else:
        checks.append(0.0)
    if transmission == "Any" or str(car["transmission"]).lower() == str(transmission).lower():
        checks.append(1.0)
    else:
        checks.append(0.0)
    if seats == "Any" or float(car["seats"]) >= float(seats):
        checks.append(1.0)
    else:
        checks.append(0.0)
    if car_type == "Any" or str(car["type"]).lower() == str(car_type).lower():
        checks.append(1.0)
    else:
        checks.append(0.0)
    requirement_score = sum(checks) / len(checks) * 100

    total = (budget_score * 0.30 + mileage_score * 0.25 +
             maintenance_score * 0.20 + safety_score * 0.15 +
             requirement_score * 0.10)
    return round(max(0.0, min(100.0, total)), 1)


def get_match_score_breakdown(
    car, budget, fuel, transmission, seats, car_type, min_mileage, min_safety
):
    """Return the exact rule-based components used by the recommendation score."""
    budget_ratio = float(car["price"]) / max(float(budget), 0.01)
    if budget_ratio <= 0.75:
        budget_points = 15.0
    else:
        budget_points = max(8.0, 15.0 - ((budget_ratio - 0.75) / 0.25) * 7.0)

    fuel_points = 15.0 if fuel == "Any" or str(car["fuel"]).lower() == str(fuel).lower() else 0.0
    transmission_points = 10.0 if transmission == "Any" or str(car["transmission"]).lower() == str(transmission).lower() else 0.0
    type_points = 15.0 if car_type == "Any" or str(car["type"]).lower() == str(car_type).lower() else 0.0
    seat_points = 10.0 if float(car["seats"]) >= float(seats) else 0.0
    mileage_points = 15.0 if float(min_mileage) <= 0 else min(15.0, max(0.0, float(car["mileage"]) / float(min_mileage) * 15.0))
    safety_points = 15.0 if float(min_safety) <= 0 else min(15.0, max(0.0, float(car["safety"]) / float(min_safety) * 15.0))
    maintenance_points = max(0.0, (6.0 - float(car["maintenance"])) / 5.0 * 5.0)

    components = [
        ("Budget fit", budget_points, 15),
        ("Fuel preference", fuel_points, 15),
        ("Transmission preference", transmission_points, 10),
        ("Vehicle type preference", type_points, 15),
        ("Seat capacity", seat_points, 10),
        ("Mileage target", mileage_points, 15),
        ("Safety target", safety_points, 15),
        ("Maintenance level", maintenance_points, 5),
    ]
    return pd.DataFrame(components, columns=["Factor", "Points earned", "Maximum points"]).assign(
        **{"Points earned": lambda frame: frame["Points earned"].round(1),
           "Maximum points": lambda frame: frame["Maximum points"].astype(int),
           "Percent of factor": lambda frame: (frame["Points earned"] / frame["Maximum points"] * 100).round(1)}
    )


def get_match_reasons(
    car,
    budget,
    fuel,
    transmission,
    seats,
    car_type,
    min_mileage,
    min_safety,
    max_maintenance,
):
    reasons = []

    if car["price"] <= budget:
        reasons.append("Within your budget")

    if fuel != "Any" and car["fuel"].lower() == fuel.lower():
        reasons.append(f"Matches {fuel} fuel preference")

    if (
        transmission != "Any"
        and car["transmission"].lower() == transmission.lower()
    ):
        reasons.append(f"Matches {transmission} transmission")

    if car["seats"] >= seats:
        reasons.append(f"Has at least {seats} seats")

    if car_type != "Any" and car["type"].lower() == car_type.lower():
        reasons.append(f"Matches {car_type} vehicle type")

    if car["mileage"] >= min_mileage:
        reasons.append("Meets mileage target")

    if car["safety"] >= min_safety:
        reasons.append("Meets safety-rating target")

    if car["maintenance"] <= max_maintenance:
        reasons.append("Meets maintenance-level limit")

    if not reasons:
        reasons.append("Ranked using your budget and selected preferences")

    return "; ".join(reasons)


def recommend_cars(
    data,
    budget,
    fuel,
    transmission,
    seats,
    car_type,
    min_mileage,
    min_safety,
    max_maintenance,
):
    eligible = data[
        (data["price"] <= budget)
        & (data["maintenance"] <= max_maintenance)
    ].copy()

    if eligible.empty:
        return eligible

    # Use one transparent 0-100 score across all recommendation cards.
    # The score explains preference fit; it is not a prediction of ownership quality.
    def score_one_car(car):
        score = 0.0

        # Budget fit: cars comfortably below the limit score slightly higher.
        budget_ratio = float(car["price"]) / max(float(budget), 0.01)
        if budget_ratio <= 0.75:
            score += 15
        else:
            score += max(8.0, 15.0 - ((budget_ratio - 0.75) / 0.25) * 7.0)

        # Exact preferences; selecting Any means no penalty or bonus.
        score += 15 if fuel == "Any" or str(car["fuel"]).lower() == fuel.lower() else 0
        score += 10 if transmission == "Any" or str(car["transmission"]).lower() == transmission.lower() else 0
        score += 15 if car_type == "Any" or str(car["type"]).lower() == car_type.lower() else 0

        # Seat capacity: meeting or exceeding the requested number is acceptable.
        score += 10 if float(car["seats"]) >= float(seats) else 0

        # Mileage: full marks when the target is met; partial marks below target.
        if float(min_mileage) <= 0:
            score += 15
        else:
            score += min(15.0, max(0.0, float(car["mileage"]) / float(min_mileage) * 15.0))

        # Safety rating: score against the requested minimum, capped at full marks.
        if float(min_safety) <= 0:
            score += 15
        else:
            score += min(15.0, max(0.0, float(car["safety"]) / float(min_safety) * 15.0))

        # Prefer lower-maintenance cars within the user's allowed maintenance level.
        score += max(0.0, (6.0 - float(car["maintenance"])) / 5.0 * 5.0)
        return round(min(100.0, score), 1)

    eligible["match_score"] = eligible.apply(score_one_car, axis=1)

    eligible["match_reasons"] = eligible.apply(
        lambda row: get_match_reasons(
            row,
            budget,
            fuel,
            transmission,
            seats,
            car_type,
            min_mileage,
            min_safety,
            max_maintenance,
        ),
        axis=1,
    )

    return eligible.sort_values(
        ["match_score", "safety", "mileage", "price"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)


def get_highlights(
    car,
    budget,
    seats,
    min_mileage,
    min_safety,
    max_maintenance,
):
    pros = []
    checks = []

    if car["price"] <= budget:
        pros.append("Listed price is within your selected budget.")
    else:
        checks.append("Listed price exceeds your selected budget.")

    if car["seats"] >= seats:
        pros.append(f"Meets the requirement for {seats} or more seats.")
    else:
        checks.append("Does not meet your selected seat requirement.")

    if car["mileage"] >= min_mileage:
        pros.append("Meets your selected mileage target.")
    else:
        checks.append("Falls below your selected mileage target.")

    if car["safety"] >= min_safety:
        pros.append("Meets your selected safety-rating target.")
    else:
        checks.append("Falls below your selected safety-rating target.")

    if car["maintenance"] <= max_maintenance:
        pros.append("Meets your selected maintenance-level limit.")
    else:
        checks.append("Exceeds your selected maintenance-level limit.")

    if not checks:
        checks.append(
            "Verify current specifications and running costs before buying."
        )

    return pros, checks


# =========================================================
# PDF REPORT
# =========================================================

def build_pdf_report(
    car,
    budget,
    score,
    reasons,
    pros,
    checks,
    comparison,
):
    buffer = BytesIO()

    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=18 * mm,
        bottomMargin=18 * mm,
        title="CarWise Recommendation Report",
        author="CarWise",
    )

    styles = getSampleStyleSheet()

    styles.add(ParagraphStyle(
        name="CarWiseTitle",
        parent=styles["Title"],
        fontSize=22,
        leading=28,
        textColor=colors.HexColor("#17376b"),
        alignment=TA_CENTER,
        spaceAfter=10,
    ))

    styles.add(ParagraphStyle(
        name="CarWiseSection",
        parent=styles["Heading2"],
        fontSize=13,
        leading=17,
        textColor=colors.HexColor("#17376b"),
        spaceBefore=12,
        spaceAfter=6,
    ))

    story = [
        Paragraph("CARWISE", styles["CarWiseTitle"]),
        Paragraph(
            "Smart Car Recommendation Report",
            styles["Heading2"]
        ),
        Paragraph(
            "A summary of your selected car and preferences.",
            styles["BodyText"]
        ),
        Spacer(1, 10),
        Paragraph("Recommended Car", styles["CarWiseSection"]),
    ]

    remaining = float(budget) - float(car["price"])

    specifications = [
        ["Specification", "Value"],
        ["Car name", str(car["name"])],
        ["Listed price", money(car["price"])],
        ["Selected budget", money(budget)],
        ["Budget remaining", money(remaining)],
        ["Preference match score", f"{score:.1f}%"],
        ["Fuel type", str(car["fuel"])],
        ["Transmission", str(car["transmission"])],
        ["Seats", f"{car['seats']:g}"],
        ["Vehicle type", str(car["type"])],
        ["Mileage in dataset", f"{car['mileage']:g}"],
        ["Safety rating in dataset", f"{car['safety']:g}/5"],
        ["Maintenance level in dataset", f"{car['maintenance']:g}/5"],
    ]

    specification_table = Table(
        specifications,
        colWidths=[55 * mm, 105 * mm],
        repeatRows=1,
    )

    specification_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#17376b")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#d6deeb")),
        ("BACKGROUND", (0, 1), (0, -1), colors.HexColor("#eef3fb")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 7),
        ("RIGHTPADDING", (0, 0), (-1, -1), 7),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))

    story.append(specification_table)
    story.append(
        Paragraph("Why It Was Recommended", styles["CarWiseSection"])
    )

    for reason in reasons:
        story.append(Paragraph("• " + str(reason), styles["BodyText"]))

    story.append(Paragraph("Positive Points", styles["CarWiseSection"]))

    for item in pros:
        story.append(Paragraph("• " + str(item), styles["BodyText"]))

    story.append(Paragraph("Points to Check", styles["CarWiseSection"]))

    for item in checks:
        story.append(Paragraph("• " + str(item), styles["BodyText"]))

    if comparison is not None and not comparison.empty:
        story.append(
            Paragraph("Selected Car Comparison", styles["CarWiseSection"])
        )

        comparison_data = [
            ["Car", "Price (lakh)", "Mileage", "Safety / 5"]
        ]

        for _, row in comparison.iterrows():
            comparison_data.append([
                str(row["name"]),
                f"{row['price']:.2f}",
                f"{row['mileage']:g}",
                f"{row['safety']:g}",
            ])

        comparison_table = Table(
            comparison_data,
            colWidths=[65 * mm, 35 * mm, 30 * mm, 30 * mm],
            repeatRows=1,
        )

        comparison_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#17376b")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#d6deeb")),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ]))

        story.append(comparison_table)

    story.append(Spacer(1, 12))
    story.append(Paragraph(
        "Disclaimer: The match score represents similarity to the selected "
        "preferences. It is not prediction accuracy or a guarantee of the "
        "best purchase. Dataset values may not reflect current specifications "
        "or prices. Verify information before buying.",
        styles["BodyText"],
    ))

    document.build(story)
    buffer.seek(0)
    return buffer.getvalue()


def csv_download(dataframe, label, filename, key):
    st.download_button(
        label=label,
        data=dataframe.to_csv(index=False).encode("utf-8"),
        file_name=filename,
        mime="text/csv",
        key=key,
    )


# =========================================================
# HEADER
# =========================================================

st.markdown("""
<div class="hero">
    <h1>🚗 Welcome to CarWise</h1>
    <p>Your Smart Car Recommendation &amp; Ownership Guide</p>
    <p style="margin-top:12px;font-size:14px;max-width:760px;">
        Find cars for your budget, compare options, estimate ownership costs,
        and make a more confident buying decision.
    </p>
</div>
""", unsafe_allow_html=True)

st.caption(
    "Personalized recommendations • Car comparison • Budget planning • Ownership-cost insights"
)

# Quick-access dashboard cards (links jump to the corresponding sections below)
st.markdown("""
<style>
    .cw-quick-grid { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:12px; margin: 12px 0 20px 0; }
    .cw-quick-card { display:block; min-height:108px; padding:17px 16px; border:1px solid #dce7f5; border-radius:15px;
        background:linear-gradient(145deg,rgba(255,255,255,.98),rgba(241,247,255,.96));
        box-shadow:0 5px 16px rgba(25,65,115,.055); color:#142d52 !important; text-decoration:none !important; }
    .cw-quick-card:hover { border-color:#7db5f5; box-shadow:0 8px 20px rgba(23,105,210,.12); transform:translateY(-1px); }
    .cw-quick-icon { display:inline-flex; width:34px; height:34px; align-items:center; justify-content:center; border-radius:10px;
        background:#e2efff; color:#1769d2; font-size:20px; margin-bottom:8px; }
    .cw-quick-title { font-weight:750; font-size:15px; margin-bottom:5px; }
    .cw-quick-desc { font-size:12px; line-height:1.45; color:#61718a; }
    @media (max-width:900px) { .cw-quick-grid { grid-template-columns:repeat(2,minmax(0,1fr)); } }
    @media (max-width:520px) { .cw-quick-grid { grid-template-columns:1fr; } }
</style>
<div class="cw-quick-grid">
  <a class="cw-quick-card" href="#top-recommendations"><div class="cw-quick-icon">🚘</div><div class="cw-quick-title">Find Your Car</div><div class="cw-quick-desc">See your best matches and ranked recommendations.</div></a>
  <a class="cw-quick-card" href="#smart-car-alternatives"><div class="cw-quick-icon">🔄</div><div class="cw-quick-title">Smart Alternatives</div><div class="cw-quick-desc">Explore other options that fit your needs.</div></a>
  <a class="cw-quick-card" href="#five-year-ownership-cost-calculator"><div class="cw-quick-icon">🧮</div><div class="cw-quick-title">Ownership Cost</div><div class="cw-quick-desc">Estimate costs across five years of ownership.</div></a>
  <a class="cw-quick-card" href="#smart-car-budget-planner"><div class="cw-quick-icon">💰</div><div class="cw-quick-title">Plan Your Budget</div><div class="cw-quick-desc">Check affordability before making a decision.</div></a>
</div>
""", unsafe_allow_html=True)


# =========================================================
# SIDEBAR FILTERS
# =========================================================

st.sidebar.title("🔎 Find Your Car")

if st.sidebar.button("↻ Reset all filters", use_container_width=True):
    for key in [
        "budget_filter",
        "fuel_filter",
        "transmission_filter",
        "seats_filter",
        "type_filter",
        "mileage_filter",
        "safety_filter",
        "maintenance_filter",
    ]:
        st.session_state.pop(key, None)
    st.rerun()

max_budget = max(1.0, float(cars["price"].max()))
min_budget = max(0.5, float(cars["price"].min()))
min_budget = min(min_budget, max_budget)

budget_default = min(max(15.0, min_budget), max_budget)

budget = st.sidebar.slider(
    "Maximum budget (₹ lakh)",
    float(min_budget),
    float(max_budget),
    float(budget_default),
    step=0.5 if max_budget - min_budget >= 0.5 else 0.1,
    key="budget_filter",
)

fuel_options = sorted(cars["fuel"].unique().tolist())
transmission_options = sorted(cars["transmission"].unique().tolist())
type_options = sorted(cars["type"].unique().tolist())

fuel = st.sidebar.selectbox(
    "Fuel type",
    ["Any"] + fuel_options,
    key="fuel_filter",
)

transmission = st.sidebar.selectbox(
    "Transmission",
    ["Any"] + transmission_options,
    key="transmission_filter",
)

seat_options = sorted(set(
    [2, 4, 5, 6, 7, 8] + cars["seats"].astype(int).tolist()
))

seats = st.sidebar.selectbox(
    "Minimum seats required",
    seat_options,
    index=seat_options.index(5) if 5 in seat_options else 0,
    key="seats_filter",
)

car_type = st.sidebar.selectbox(
    "Vehicle type",
    ["Any"] + type_options,
    key="type_filter",
)

min_mileage = st.sidebar.slider(
    "Minimum mileage",
    0.0,
    max(1.0, float(cars["mileage"].max())),
    0.0,
    step=1.0,
    key="mileage_filter",
)

min_safety = st.sidebar.slider(
    "Minimum safety rating",
    0.0,
    5.0,
    0.0,
    step=0.5,
    key="safety_filter",
)

max_maintenance = st.sidebar.slider(
    "Maximum maintenance level",
    1,
    5,
    5,
    step=1,
    key="maintenance_filter",
)

st.sidebar.caption(
    "Maintenance scale: 1 = lower listed level, 5 = higher listed level."
)


# =========================================================
# GENERATE RECOMMENDATIONS
# =========================================================

eligible = recommend_cars(
    cars,
    budget,
    fuel,
    transmission,
    seats,
    car_type,
    min_mileage,
    min_safety,
    max_maintenance,
)


# =========================================================
# DASHBOARD METRICS
# =========================================================

m1, m2, m3, m4 = st.columns(4)

m1.metric("Cars in dataset", len(cars))
m2.metric("Within budget", int((cars["price"] <= budget).sum()))
m3.metric("Eligible matches", len(eligible))
m4.metric(
    "Best match",
    f"{eligible.iloc[0]['match_score']:.1f}%"
    if not eligible.empty else "N/A",
)

st.divider()
st.header("🏆 Best Recommendation")

selected_for_report = None
selected_score = 0.0
selected_reasons = []
selected_pros = []
selected_checks = []

if eligible.empty:
    st.warning(
        "No cars meet both your budget and maintenance limits. "
        "Try increasing your budget or allowing a higher maintenance level."
    )

else:
    best = eligible.iloc[0]
    selected_for_report = best
    selected_score = float(best["match_score"])

    selected_reasons = [
        reason for reason in best["match_reasons"].split("; ") if reason
    ]

    selected_pros, selected_checks = get_highlights(
        best,
        budget,
        seats,
        min_mileage,
        min_safety,
        max_maintenance,
    )

    left, right = st.columns([1, 1.4])

    with left:
        if str(best["image_url"]).startswith("https://"):
            st.image(best["image_url"], use_container_width=True)

        st.subheader(best["name"])
        st.metric("Preference match", f"{selected_score:.1f}%")
        st.progress(min(1.0, selected_score / 100))
        st.markdown(f"### {money(best['price'])}")
        st.metric("Budget remaining", money(budget - best["price"]))

        ownership_score = calculate_ownership_score(
            best, budget, min_mileage, min_safety, max_maintenance,
            fuel, transmission, seats, car_type
        )
        st.metric("Smart Ownership Score", f"{ownership_score}/100")
        st.progress(ownership_score / 100)
        if ownership_score >= 80:
            st.success("Strong overall suitability based on your selected preferences.")
        elif ownership_score >= 60:
            st.info("A reasonable option. Review the cost and requirement breakdown before deciding.")
        else:
            st.warning("Review this car carefully; some factors may not suit your needs.")
        st.caption("Rule-based estimate using price, mileage, listed maintenance, safety and preference match. Not a guarantee of real-world ownership costs.")

    with right:
        st.subheader("Car specifications")
        st.write(f"**Fuel:** {best['fuel']}")
        st.write(f"**Transmission:** {best['transmission']}")
        st.write(f"**Seats:** {best['seats']:g}")
        st.write(f"**Vehicle type:** {best['type']}")
        st.write(f"**Mileage:** {best['mileage']:g}")
        st.write(f"**Safety rating:** {best['safety']:g}/5")
        st.write(f"**Maintenance level:** {best['maintenance']:g}/5")

        st.markdown("#### Why it matches")
        for reason in selected_reasons:
            st.write(f"✅ {reason}")

        with st.expander("🔍 Explain this match score", expanded=True):
            score_breakdown = get_match_score_breakdown(
                best, budget, fuel, transmission, seats, car_type, min_mileage, min_safety
            )
            st.caption(
                "The score is a transparent rule-based fit indicator. It is not a probability, "
                "a prediction of reliability, or a guarantee of satisfaction."
            )
            st.dataframe(
                score_breakdown,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Points earned": st.column_config.NumberColumn(format="%.1f"),
                    "Maximum points": st.column_config.NumberColumn(format="%d"),
                    "Percent of factor": st.column_config.ProgressColumn(
                        min_value=0, max_value=100, format="%.1f%%"
                    ),
                },
            )
            st.write(
                f"**Total:** {score_breakdown['Points earned'].sum():.1f} / "
                f"{score_breakdown['Maximum points'].sum()} points. "
                "A selected ‘Any’ preference earns full points because it is not being used to filter cars."
            )

    st.divider()
    st.subheader("📄 Your Recommendation Report")
    st.write(
        "Download a PDF containing the recommended car, specifications, "
        "match reasons and points to check."
    )

    report_comparison = None

    if "comparison_names" in st.session_state:
        names = st.session_state["comparison_names"]
        if len(names) >= 2:
            report_comparison = cars[cars["name"].isin(names)].copy()

    try:
        pdf_bytes = build_pdf_report(
            selected_for_report,
            budget,
            selected_score,
            selected_reasons,
            selected_pros,
            selected_checks,
            report_comparison,
        )

        st.download_button(
            label="⬇️ Download PDF Recommendation Report",
            data=pdf_bytes,
            file_name="CarWise_Recommendation_Report.pdf",
            mime="application/pdf",
            key="download_pdf_report",
            use_container_width=True,
        )

    except Exception as error:
        st.error(f"Could not generate the PDF report: {error}")


# =========================================================
# TOP THREE RECOMMENDATIONS
# =========================================================

st.divider()
st.header("⭐ Top Recommendations")

if eligible.empty:
    st.info("No eligible cars were found with the current filters.")
else:
    top = eligible.head(3)
    columns = st.columns(len(top))

    for index, (_, car) in enumerate(top.iterrows()):
        with columns[index]:
            if str(car["image_url"]).startswith("https://"):
                st.image(car["image_url"], use_container_width=True)

            st.subheader(f"{index + 1}. {car['name']}")
            st.metric("Match", f"{car['match_score']:.1f}%")
            st.write(f"**Price:** {money(car['price'])}")
            st.write(f"Fuel: {car['fuel']}")
            st.write(f"Transmission: {car['transmission']}")
            st.write(f"Seats: {car['seats']:g}")
            st.write(f"Mileage: {car['mileage']:g}")

    shortlist_export = eligible.copy()
    shortlist_export["ownership_score"] = shortlist_export.apply(
        lambda row: calculate_ownership_score(
            row, budget, min_mileage, min_safety, max_maintenance,
            fuel, transmission, seats, car_type
        ),
        axis=1,
    )
    export_columns = [
        column for column in [
            "name", "price", "fuel", "transmission", "seats", "type",
            "mileage", "safety", "maintenance", "match_score",
            "ownership_score", "match_reasons"
        ] if column in shortlist_export.columns
    ]
    shortlist_export = shortlist_export[export_columns].rename(columns={
        "name": "Car", "price": "Price (₹ lakh)", "fuel": "Fuel",
        "transmission": "Transmission", "seats": "Seats", "type": "Vehicle type",
        "mileage": "Mileage", "safety": "Safety / 5",
        "maintenance": "Maintenance / 5", "match_score": "Preference match / 100",
        "ownership_score": "Ownership suitability / 100", "match_reasons": "Why it matched"
    })
    st.download_button(
        "⬇️ Download ranked shortlist with score explanations (CSV)",
        data=shortlist_export.to_csv(index=False).encode("utf-8-sig"),
        file_name="CarWise_Ranked_Shortlist.csv",
        mime="text/csv",
        key="download_ranked_shortlist_explained",
        use_container_width=True,
    )
    csv_download(
        eligible,
        "⬇️ Download recommendations",
        "carwise_recommendations.csv",
        "download_recommendations",
    )


# =========================================================
# SMART ALTERNATIVES
# =========================================================

st.divider()
st.header("🔁 Smart Car Alternatives")
st.write(
    "Explore other cars that fit your budget and are similar to your top choice. "
    "If no car passes every filter, this section relaxes the maintenance limit "
    "to show possible alternatives rather than leaving you with no options."
)

if not eligible.empty:
    anchor = eligible.iloc[0]
    candidate_pool = cars[
        (cars["price"] <= budget)
        & (cars["name"].str.lower() != str(anchor["name"]).lower())
    ].copy()
    alternative_intro = f"Alternatives similar to {anchor['name']}"
else:
    anchor = {
        "fuel": fuel,
        "transmission": transmission,
        "type": car_type,
        "seats": seats,
        "price": budget * 0.75,
        "mileage": min_mileage,
        "safety": min_safety,
        "maintenance": max_maintenance,
        "name": "your selected preferences",
    }
    candidate_pool = cars[cars["price"] <= budget].copy()
    alternative_intro = "Budget-friendly alternatives with relaxed maintenance filtering"

if candidate_pool.empty:
    st.info("No alternative cars are listed within your current budget. Try increasing the budget or expanding the dataset.")
else:
    def alternative_score(car, reference):
        score = 0.0
        # Similarity to the anchor car or to the user's preferences.
        if str(reference["fuel"]).lower() == "any" or str(car["fuel"]).lower() == str(reference["fuel"]).lower():
            score += 20
        if str(reference["transmission"]).lower() == "any" or str(car["transmission"]).lower() == str(reference["transmission"]).lower():
            score += 20
        if str(reference["type"]).lower() == "any" or str(car["type"]).lower() == str(reference["type"]).lower():
            score += 15
        if float(car["seats"]) >= float(reference["seats"]):
            score += 5
        if float(car["mileage"]) >= float(reference["mileage"]):
            score += 5
        if float(car["safety"]) >= float(reference["safety"]):
            score += 5

        ref_price = max(float(reference["price"]), 0.1)
        price_similarity = max(0.0, 1.0 - abs(float(car["price"]) - ref_price) / max(ref_price, float(budget), 0.1))
        score += price_similarity * 20

        # Prefer cars with lower listed maintenance while respecting the user's limit when possible.
        if float(car["maintenance"]) <= float(max_maintenance):
            score += 10
        else:
            score += max(0.0, 10.0 - (float(car["maintenance"]) - float(max_maintenance)) * 3)
        return round(min(100.0, score), 1)

    candidate_pool["alternative_score"] = candidate_pool.apply(
        lambda row: alternative_score(row, anchor), axis=1
    )
    candidate_pool["budget_status"] = np.where(
        candidate_pool["price"] <= budget, "Within budget", "Over budget"
    )
    alternatives = candidate_pool.sort_values(
        ["alternative_score", "safety", "mileage", "price"],
        ascending=[False, False, False, True],
    ).drop_duplicates(subset=["name"], keep="first").head(3)

    st.caption(alternative_intro)
    alt_columns = st.columns(len(alternatives))
    for idx, (_, alt) in enumerate(alternatives.iterrows()):
        with alt_columns[idx]:
            if str(alt["image_url"]).startswith("https://"):
                st.image(alt["image_url"], use_container_width=True)
            st.subheader(alt["name"])
            st.metric("Alternative score", f"{alt['alternative_score']:.1f}/100")
            st.write(f"**Price:** {money(alt['price'])}")
            st.write(f"**Budget remaining:** {money(budget - alt['price'])}")
            st.write(f"**Fuel:** {alt['fuel']}")
            st.write(f"**Transmission:** {alt['transmission']}")
            st.write(f"**Mileage:** {alt['mileage']:g}")
            st.write(f"**Safety:** {alt['safety']:g}/5")
            if float(alt["maintenance"]) > float(max_maintenance):
                st.warning("Listed maintenance level exceeds your selected limit.")
            else:
                st.success("Fits your selected maintenance limit.")
    st.caption(
        "Alternative score is a rule-based similarity estimate, not a quality guarantee. "
        "Confirm variant, on-road price, current specifications and service costs before buying."
    )


# =========================================================
# FIVE-YEAR OWNERSHIP COST CALCULATOR
# =========================================================

st.divider()
st.header("💰 Five-Year Ownership Cost Calculator")
st.write(
    "Compare estimated purchase, energy, maintenance and insurance costs. "
    "Edit the assumptions to suit your situation."
)

with st.expander("⚙️ Set your cost assumptions", expanded=True):
    assumption_col1, assumption_col2 = st.columns(2)

    with assumption_col1:
        annual_km = st.slider(
            "Estimated driving per year (km)",
            min_value=3000,
            max_value=50000,
            value=12000,
            step=1000,
            key="ownership_annual_km",
        )

        ownership_years = st.selectbox(
            "Ownership period (years)",
            [3, 4, 5, 6, 7, 8],
            index=2,
            key="ownership_years",
        )

        annual_insurance_rate = st.slider(
            "Estimated insurance per year (% of listed price)",
            min_value=1.0,
            max_value=5.0,
            value=2.5,
            step=0.25,
            key="ownership_insurance_rate",
        )

    with assumption_col2:
        petrol_price = st.number_input(
            "Petrol price (₹/litre)",
            min_value=1.0,
            max_value=300.0,
            value=105.0,
            step=1.0,
            key="ownership_petrol_price",
        )

        diesel_price = st.number_input(
            "Diesel price (₹/litre)",
            min_value=1.0,
            max_value=300.0,
            value=92.0,
            step=1.0,
            key="ownership_diesel_price",
        )

        cng_price = st.number_input(
            "CNG price (₹/kg)",
            min_value=1.0,
            max_value=300.0,
            value=85.0,
            step=1.0,
            key="ownership_cng_price",
        )

        electricity_price = st.number_input(
            "Electricity price (₹/kWh)",
            min_value=1.0,
            max_value=100.0,
            value=10.0,
            step=0.5,
            key="ownership_electricity_price",
        )

    st.caption(
        "These are editable example rates, not live local prices. Dataset "
        "mileage is treated as km/litre for petrol/diesel, km/kg for CNG, "
        "and km/kWh for electric cars. Verify the units for each vehicle."
    )

energy_prices = {
    "petrol": float(petrol_price),
    "diesel": float(diesel_price),
    "cng": float(cng_price),
    "ev": float(electricity_price),
    "electric": float(electricity_price),
    "electric vehicle": float(electricity_price),
}

maintenance_cost_by_level = {
    1: 8000.0,
    2: 12000.0,
    3: 18000.0,
    4: 25000.0,
    5: 35000.0,
}


def estimate_ownership_cost(
    car_row,
    yearly_km,
    years,
    energy_rates,
    insurance_rate,
):
    fuel_key = str(car_row["fuel"]).strip().lower()

    if "petrol" in fuel_key:
        energy_price = energy_rates["petrol"]
    elif "diesel" in fuel_key:
        energy_price = energy_rates["diesel"]
    elif "cng" in fuel_key:
        energy_price = energy_rates["cng"]
    elif any(term in fuel_key for term in ["electric", "ev"]):
        energy_price = energy_rates["electric"]
    else:
        energy_price = None

    efficiency = float(car_row["mileage"])

    if energy_price is None or efficiency <= 0:
        energy_cost = np.nan
    else:
        energy_cost = (
            float(yearly_km) / efficiency
        ) * energy_price * years

    maintenance_level = int(round(float(car_row["maintenance"])))

    annual_maintenance = maintenance_cost_by_level.get(
        min(5, max(1, maintenance_level)),
        18000.0,
    )

    maintenance_total = annual_maintenance * years

    purchase_cost = float(car_row["price"]) * 100000

    insurance_total = (
        purchase_cost * (insurance_rate / 100) * years
    )

    if pd.isna(energy_cost):
        total_cost = np.nan
        monthly_average = np.nan
    else:
        total_cost = (
            purchase_cost
            + energy_cost
            + maintenance_total
            + insurance_total
        )

        monthly_average = (
            energy_cost + maintenance_total + insurance_total
        ) / (years * 12)

    return {
        "Purchase price (₹)": purchase_cost,
        "Estimated energy cost (₹)": energy_cost,
        "Estimated maintenance (₹)": maintenance_total,
        "Estimated insurance (₹)": insurance_total,
        f"Estimated total over {years} years (₹)": total_cost,
        "Estimated running cost/month (₹)": monthly_average,
    }


cost_candidates = cars.copy()

if not eligible.empty:
    cost_candidates = eligible.head(10).copy()
    st.caption(
        "Showing up to 10 currently eligible recommended cars. "
        "Change the sidebar filters to change this shortlist."
    )
else:
    st.caption(
        "No cars meet the current recommendation filters, "
        "so the calculator shows the dataset catalogue."
    )

cost_rows = []

for _, cost_car in cost_candidates.iterrows():
    estimate = estimate_ownership_cost(
        cost_car,
        annual_km,
        ownership_years,
        energy_prices,
        annual_insurance_rate,
    )

    row = {
        "Car": cost_car["name"],
        "Fuel type": cost_car["fuel"],
        "Listed price (₹ lakh)": float(cost_car["price"]),
        "Dataset mileage": float(cost_car["mileage"]),
    }

    row.update(estimate)
    cost_rows.append(row)

ownership_table = pd.DataFrame(cost_rows)

total_column = f"Estimated total over {ownership_years} years (₹)"

valid_totals = ownership_table.dropna(subset=[total_column])

if ownership_table.empty:
    st.info("No cars are available to calculate costs.")

elif valid_totals.empty:
    st.warning(
        "A cost estimate could not be calculated. Check that the fuel type "
        "is recognized and mileage is greater than zero."
    )

else:
    lowest_cost = valid_totals.loc[
        valid_totals[total_column].idxmin()
    ]

    best_col1, best_col2, best_col3 = st.columns(3)

    best_col1.metric(
        "Lowest estimated total",
        f"₹{lowest_cost[total_column]:,.0f}",
    )

    best_col2.metric(
        "Car in this estimate",
        str(lowest_cost["Car"]),
    )

    best_col3.metric(
        "Estimated monthly running cost",
        f"₹{lowest_cost['Estimated running cost/month (₹)']:,.0f}",
    )

    st.dataframe(
        ownership_table.style.format({
            "Listed price (₹ lakh)": "₹{:.2f} lakh",
            "Dataset mileage": "{:.1f}",
            "Purchase price (₹)": "₹{:,.0f}",
            "Estimated energy cost (₹)": "₹{:,.0f}",
            "Estimated maintenance (₹)": "₹{:,.0f}",
            "Estimated insurance (₹)": "₹{:,.0f}",
            total_column: "₹{:,.0f}",
            "Estimated running cost/month (₹)": "₹{:,.0f}",
        }, na_rep="Unavailable"),
        use_container_width=True,
        hide_index=True,
    )

    csv_download(
        ownership_table,
        "⬇️ Download ownership cost estimates",
        "carwise_ownership_cost_estimates.csv",
        "download_ownership_costs",
    )

st.warning(
    "Estimate only: excludes loan interest, depreciation, registration, "
    "taxes, parking, repairs, tyres and resale value. Insurance and maintenance "
    "are simplified assumptions, not quotes. Verify fuel/mileage units and "
    "replace sample rates with current local values before making a purchase."
)


# =========================================================
# DETAILED CAR EXPLORER
# =========================================================

st.divider()
st.header("🔍 Detailed Car Explorer")

car_names = cars["name"].tolist()

detail_name = st.selectbox(
    "Choose a car",
    car_names,
    key="detail_car_select",
)

detail = cars[cars["name"] == detail_name].iloc[0]

if str(detail["image_url"]).startswith("https://"):
    st.image(detail["image_url"], use_container_width=True)

d1, d2, d3 = st.columns(3)

d1.metric("Price", money(detail["price"]))
d2.metric("Mileage", f"{detail['mileage']:g}")
d3.metric("Safety rating", f"{detail['safety']:g}/5")

st.write(f"**Fuel:** {detail['fuel']}")
st.write(f"**Transmission:** {detail['transmission']}")
st.write(f"**Seats:** {detail['seats']:g}")
st.write(f"**Vehicle type:** {detail['type']}")
st.write(f"**Maintenance level:** {detail['maintenance']:g}/5")

detail_pros, detail_checks = get_highlights(
    detail,
    budget,
    seats,
    min_mileage,
    min_safety,
    max_maintenance,
)

pcol, ccol = st.columns(2)

with pcol:
    st.subheader("✅ Positive points")
    for item in detail_pros:
        st.write(f"• {item}")

with ccol:
    st.subheader("⚠️ Points to check")
    for item in detail_checks:
        st.write(f"• {item}")


# =========================================================
# SMART BUDGET PLANNER
# =========================================================

st.divider()
st.header("📊 Smart Car Budget Planner")
st.write(
    "Estimate a car budget using your monthly income, expenses, savings and "
    "loan assumptions. This is a planning tool, not a loan approval or financial advice."
)

with st.expander("Enter your budget details", expanded=False):
    budget_col1, budget_col2 = st.columns(2)
    with budget_col1:
        monthly_income = st.number_input(
            "Monthly take-home income (₹)", min_value=0, max_value=10000000,
            value=50000, step=5000, key="planner_monthly_income"
        )
        monthly_expenses = st.number_input(
            "Monthly essential expenses and existing EMIs (₹)", min_value=0,
            max_value=10000000, value=25000, step=1000,
            key="planner_monthly_expenses"
        )
        current_savings = st.number_input(
            "Savings available for down payment (₹)", min_value=0,
            max_value=100000000, value=150000, step=10000,
            key="planner_savings"
        )
    with budget_col2:
        safe_share = st.slider(
            "Share of remaining income for car EMI (%)", min_value=5,
            max_value=40, value=20, step=5, key="planner_safe_share"
        )
        interest_rate = st.number_input(
            "Assumed annual loan interest rate (%)", min_value=0.0,
            max_value=40.0, value=9.5, step=0.5, key="planner_interest_rate"
        )
        loan_years = st.selectbox(
            "Loan tenure (years)", [1, 2, 3, 4, 5, 6, 7], index=4,
            key="planner_loan_years"
        )
        reserve = st.number_input(
            "Savings to keep aside (₹)", min_value=0,
            max_value=100000000, value=50000, step=10000,
            key="planner_reserve"
        )

    st.caption(
        "The suggested EMI is a conservative planning estimate. Actual affordability "
        "also depends on household needs, insurance, fuel, maintenance, taxes and lender terms."
    )

monthly_surplus = max(0.0, float(monthly_income) - float(monthly_expenses))
comfortable_emi = monthly_surplus * float(safe_share) / 100.0
monthly_rate = float(interest_rate) / 1200.0
number_of_payments = int(loan_years) * 12
if monthly_rate == 0:
    loan_principal_estimate = comfortable_emi * number_of_payments
else:
    factor = (1 + monthly_rate) ** number_of_payments
    loan_principal_estimate = comfortable_emi * (factor - 1) / (monthly_rate * factor)
usable_down_payment = max(0.0, float(current_savings) - float(reserve))
indicative_budget = max(0.0, loan_principal_estimate + usable_down_payment)

budget_metric1, budget_metric2, budget_metric3 = st.columns(3)
budget_metric1.metric("Monthly surplus", money(monthly_surplus))
budget_metric2.metric("Indicative comfortable EMI", money(comfortable_emi))
budget_metric3.metric("Indicative car budget", money(indicative_budget))

if monthly_income <= 0:
    st.info("Enter your monthly take-home income in the planner to get a meaningful estimate.")
elif monthly_surplus <= 0:
    st.warning("Your entered expenses equal or exceed your income. Consider reviewing your budget before taking on a car payment.")
else:
    st.progress(min(1.0, comfortable_emi / monthly_surplus) if monthly_surplus else 0.0)
    st.caption(
        f"The estimate uses {safe_share}% of your entered monthly surplus for EMI, "
        f"a {loan_years}-year tenure, and {interest_rate:.1f}% assumed annual interest. "
        "It excludes running costs from the EMI estimate."
    )

if "cars" in globals() and isinstance(cars, pd.DataFrame) and not cars.empty and "price" in cars.columns:
    budget_cars = cars[pd.to_numeric(cars["price"], errors="coerce") <= indicative_budget]
    if not budget_cars.empty:
        st.subheader("Cars within your indicative budget")
        budget_show_cols = [c for c in ["name", "price", "fuel", "transmission", "mileage", "seats"] if c in budget_cars.columns]
        st.dataframe(
            budget_cars[budget_show_cols].sort_values("price").head(10),
            use_container_width=True, hide_index=True
        )
    else:
        st.info("No cars in the current dataset fall within this indicative budget. You can adjust the assumptions or review the available dataset.")


# =========================================================
# SMART CAR BUYING ADVISOR
# =========================================================

st.divider()
st.header("🧠 Smart Car Buying Advisor")
st.caption(
    "A practical budget check and pre-purchase checklist to help you "
    "review a car before making a decision."
)

advisor_car_names = sorted(cars["name"].dropna().astype(str).unique().tolist())
if advisor_car_names:
    advisor_default = 0
    if "recommended" in globals() and isinstance(recommended, pd.DataFrame) and not recommended.empty:
        first_recommended_name = str(recommended.iloc[0]["name"])
        if first_recommended_name in advisor_car_names:
            advisor_default = advisor_car_names.index(first_recommended_name)

    advisor_car_name = st.selectbox(
        "Choose a car to review",
        advisor_car_names,
        index=advisor_default,
        key="advisor_car_name",
    )
    advisor_rows = cars[cars["name"].astype(str) == advisor_car_name]
    advisor_car = advisor_rows.iloc[0]

    advisor_price = float(advisor_car["price"])
    advisor_remaining = float(budget) - advisor_price
    advisor_budget_used = (advisor_price / float(budget) * 100) if float(budget) > 0 else 0

    st.subheader("💰 Budget health check")
    budget_cols = st.columns(3)
    budget_cols[0].metric("Listed price", money(advisor_price))
    budget_cols[1].metric("Your selected budget", money(budget))
    budget_cols[2].metric("Budget remaining", money(advisor_remaining))
    st.progress(min(max(advisor_budget_used / 100, 0.0), 1.0))
    st.caption(f"This car uses {advisor_budget_used:.0f}% of your selected budget.")

    if advisor_price > float(budget):
        st.error("This car is over your selected budget. Consider a lower-priced option or adjust your budget only if it is genuinely affordable.")
    elif advisor_budget_used >= 90:
        st.warning("This car uses at least 90% of your budget. Keep extra money aside for insurance, registration, taxes, and unexpected costs.")
    elif advisor_budget_used >= 75:
        st.info("This car uses a large part of your budget. Check the on-road price and ongoing costs before deciding.")
    else:
        st.success("The listed price leaves some room in your selected budget. Remember that the on-road price will usually be higher.")

    st.subheader("🎯 Personalised advice")
    advice_items = []
    if advisor_price > float(budget):
        advice_items.append("Look at cars within budget before comparing features.")
    else:
        advice_items.append("Confirm the on-road price with an authorised dealer; the listed price may not include all charges.")
    if float(advisor_car["mileage"]) >= float(min_mileage):
        advice_items.append("The dataset mileage meets your current target; check real-world mileage or EV range for your driving conditions.")
    else:
        advice_items.append("The dataset mileage is below your selected target; consider whether your daily running makes this acceptable.")
    if float(advisor_car["safety"]) >= float(min_safety):
        advice_items.append("The dataset safety score meets your filter; verify the exact variant and current independent safety information.")
    else:
        advice_items.append("The dataset safety score is below your target; compare safer alternatives and verify the exact variant.")
    if float(advisor_car["seats"]) < float(seats):
        advice_items.append("This car may not meet your seating requirement.")
    if float(advisor_car["maintenance"]) > float(max_maintenance):
        advice_items.append("Its dataset maintenance level is above your limit; research service costs and parts availability.")
    for advice in advice_items:
        st.write(f"• {advice}")

    st.subheader("📝 Pre-purchase checklist")
    st.write("Tick each item after you have verified it. Progress is kept for this Streamlit session.")
    checklist_items = [
        "Confirm the final on-road price in writing",
        "Compare insurance quotes and renewal costs",
        "Check warranty terms and exclusions",
        "Review service history and maintenance costs",
        "Verify the exact variant and safety equipment",
        "Take a proper test drive",
        "Arrange an independent inspection for a used car",
        "Check documents, registration and ownership details (used car)",
        "Compare at least two alternatives before deciding",
    ]
    completed_items = []
    for item in checklist_items:
        checked = st.checkbox(item, key=f"advisor_check_{item}")
        if checked:
            completed_items.append(item)
    checklist_progress = len(completed_items) / len(checklist_items)
    st.progress(checklist_progress)
    st.caption(f"Checklist completed: {len(completed_items)} of {len(checklist_items)}")

    checklist_export = pd.DataFrame({
        "Car": [advisor_car_name] * len(checklist_items),
        "Pre-purchase check": checklist_items,
        "Status": ["Completed" if item in completed_items else "Pending" for item in checklist_items],
    })
    st.download_button(
        label="⬇️ Download pre-purchase checklist (CSV)",
        data=checklist_export.to_csv(index=False).encode("utf-8"),
        file_name="CarWise_PrePurchase_Checklist.csv",
        mime="text/csv",
        key="download_pre_purchase_checklist",
        use_container_width=True,
    )

    if len(completed_items) == len(checklist_items):
        st.success("All checklist items are marked complete. Keep receipts and verification records before purchase.")
    else:
        st.info(f"You still have {len(checklist_items) - len(completed_items)} checklist items to verify before making a decision.")

    st.caption("CarWise provides shortlist guidance using the dataset available in this project. Prices, mileage, safety and maintenance values may be sample or outdated data; verify them independently before making a purchase.")
else:
    st.info("Add car records to cars.csv to use the Smart Car Buying Advisor.")


# =========================================================
# CAR COMPARISON
# =========================================================

st.divider()
st.header("⚖️ Compare Cars")

comparison_names = st.multiselect(
    "Select 2 or 3 cars",
    options=car_names,
    max_selections=3,
    key="comparison_names",
)

if len(comparison_names) < 2:
    st.info("Choose at least two cars to compare.")

else:
    comparison = cars[cars["name"].isin(comparison_names)].copy()

    order = {
        name: index for index, name in enumerate(comparison_names)
    }

    comparison["_order"] = comparison["name"].map(order)
    comparison = comparison.sort_values("_order").drop(columns="_order")

    comparison_columns = [
        "name", "price", "fuel", "transmission",
        "seats", "type", "mileage", "safety", "maintenance"
    ]

    comparison["ownership_score"] = comparison.apply(
        lambda row: calculate_ownership_score(
            row, budget, min_mileage, min_safety, max_maintenance,
            fuel, transmission, seats, car_type
        ), axis=1
    )
    comparison_columns.append("ownership_score")

    display_comparison = comparison[comparison_columns].rename(
        columns={
            "name": "Car",
            "price": "Price (₹ lakh)",
            "fuel": "Fuel",
            "transmission": "Transmission",
            "seats": "Seats",
            "type": "Vehicle type",
            "mileage": "Mileage",
            "safety": "Safety / 5",
            "maintenance": "Maintenance / 5",
            "ownership_score": "Smart Ownership Score / 100",
        }
    )

    st.dataframe(
        display_comparison,
        use_container_width=True,
        hide_index=True,
    )

    cheapest = comparison.loc[comparison["price"].idxmin()]
    highest_mileage = comparison.loc[comparison["mileage"].idxmax()]
    highest_safety = comparison.loc[comparison["safety"].idxmax()]

    st.info(
        f"Lowest listed price: {cheapest['name']} — "
        f"{money(cheapest['price'])}"
    )

    st.info(
        f"Highest listed mileage: {highest_mileage['name']} — "
        f"{highest_mileage['mileage']:g}"
    )

    st.info(
        f"Highest dataset safety rating: {highest_safety['name']} — "
        f"{highest_safety['safety']:g}/5"
    )

    csv_download(
        display_comparison,
        "⬇️ Download comparison",
        "carwise_comparison.csv",
        "download_comparison",
    )


# =========================================================
# FULL CAR CATALOGUE
# =========================================================

st.divider()
st.header("📚 Car Catalogue")

search = st.text_input("Search car name", key="catalogue_search")

catalogue_fuel = st.selectbox(
    "Catalogue fuel filter",
    ["All"] + fuel_options,
    key="catalogue_fuel_filter",
)

catalogue = cars.copy()

if search.strip():
    catalogue = catalogue[
        catalogue["name"].str.contains(
            search.strip(),
            case=False,
            na=False,
        )
    ]

if catalogue_fuel != "All":
    catalogue = catalogue[
        catalogue["fuel"].str.lower() == catalogue_fuel.lower()
    ]

sort_choice = st.selectbox(
    "Sort by",
    [
        "Price: low to high",
        "Price: high to low",
        "Mileage: high to low",
        "Safety: high to low",
        "Name: A to Z",
    ],
)

sort_options = {
    "Price: low to high": ("price", True),
    "Price: high to low": ("price", False),
    "Mileage: high to low": ("mileage", False),
    "Safety: high to low": ("safety", False),
    "Name: A to Z": ("name", True),
}

sort_column, ascending = sort_options[sort_choice]

catalogue = catalogue.sort_values(
    sort_column,
    ascending=ascending,
)

st.caption(f"Showing {len(catalogue)} of {len(cars)} cars.")

catalogue_display = catalogue[
    [
        "name", "price", "fuel", "transmission",
        "seats", "type", "mileage", "safety", "maintenance"
    ]
].rename(columns={
    "name": "Car",
    "price": "Price (₹ lakh)",
    "fuel": "Fuel",
    "transmission": "Transmission",
    "seats": "Seats",
    "type": "Vehicle type",
    "mileage": "Mileage",
    "safety": "Safety / 5",
    "maintenance": "Maintenance / 5",
})

st.dataframe(
    catalogue_display,
    use_container_width=True,
    hide_index=True,
)

csv_download(
    catalogue_display,
    "⬇️ Download catalogue",
    "carwise_catalogue.csv",
    "download_catalogue",
)


# =========================================================
# DATASET HEALTH & QUALITY CENTRE
# =========================================================

st.divider()
st.header("🧪 Dataset Health & Quality Centre")
st.caption(
    "A quick audit of the car records currently loaded by CarWise. "
    "This checks the available dataset; it does not independently verify "
    "prices or specifications with manufacturers."
)

quality_cars = cars.copy()
quality_numeric = [
    column for column in ["price", "seats", "mileage", "safety", "maintenance"]
    if column in quality_cars.columns
]

exact_duplicate_count = int(quality_cars.duplicated().sum())
identity_columns = [
    column for column in ["name", "fuel", "transmission", "type"]
    if column in quality_cars.columns
]
if identity_columns:
    possible_duplicate_mask = quality_cars.duplicated(
        subset=identity_columns, keep=False
    )
    possible_duplicate_rows = quality_cars.loc[possible_duplicate_mask].sort_values(
        identity_columns
    )
else:
    possible_duplicate_rows = quality_cars.iloc[0:0].copy()

missing_by_column = quality_cars.isna().sum()
missing_cell_count = int(missing_by_column.sum())
invalid_checks = {
    "Price must be greater than 0": (
        (quality_cars["price"] <= 0) if "price" in quality_cars.columns
        else pd.Series(False, index=quality_cars.index)
    ),
    "Seats must be greater than 0": (
        (quality_cars["seats"] <= 0) if "seats" in quality_cars.columns
        else pd.Series(False, index=quality_cars.index)
    ),
    "Mileage cannot be negative": (
        (quality_cars["mileage"] < 0) if "mileage" in quality_cars.columns
        else pd.Series(False, index=quality_cars.index)
    ),
    "Safety rating must be between 1 and 5": (
        ~quality_cars["safety"].between(1, 5) if "safety" in quality_cars.columns
        else pd.Series(False, index=quality_cars.index)
    ),
    "Maintenance rating must be between 1 and 5": (
        ~quality_cars["maintenance"].between(1, 5) if "maintenance" in quality_cars.columns
        else pd.Series(False, index=quality_cars.index)
    ),
}
invalid_count_by_rule = {
    rule: int(mask.fillna(True).sum())
    for rule, mask in invalid_checks.items()
}
invalid_record_count = int(
    pd.concat([mask.rename(rule) for rule, mask in invalid_checks.items()], axis=1)
    .any(axis=1).sum()
) if invalid_checks else 0

quality_col1, quality_col2, quality_col3, quality_col4 = st.columns(4)
quality_col1.metric("Loaded car records", f"{len(quality_cars):,}")
quality_col2.metric("Exact duplicate rows", f"{exact_duplicate_count:,}")
quality_col3.metric("Missing cells", f"{missing_cell_count:,}")
quality_col4.metric("Records with range issues", f"{invalid_record_count:,}")

with st.expander("View detailed dataset audit", expanded=False):
    st.subheader("Missing values by column")
    missing_report = pd.DataFrame({
        "Column": missing_by_column.index,
        "Missing values": missing_by_column.values,
        "Missing percentage": [
            round((count / len(quality_cars) * 100), 2) if len(quality_cars) else 0
            for count in missing_by_column.values
        ],
    })
    st.dataframe(missing_report, use_container_width=True, hide_index=True)

    st.subheader("Range validation")
    range_report = pd.DataFrame({
        "Validation rule": list(invalid_count_by_rule.keys()),
        "Records flagged": list(invalid_count_by_rule.values()),
    })
    st.dataframe(range_report, use_container_width=True, hide_index=True)

    st.subheader("Possible repeated model configurations")
    st.caption(
        "These records share the same name, fuel, transmission and vehicle type. "
        "They are flagged for review, not automatically deleted, because different "
        "variants can legitimately share these details."
    )
    if possible_duplicate_rows.empty:
        st.success("No repeated model configurations found by this check.")
    else:
        duplicate_columns = [
            column for column in [
                "name", "price", "fuel", "transmission", "seats", "type",
                "mileage", "safety", "maintenance"
            ] if column in possible_duplicate_rows.columns
        ]
        st.dataframe(
            possible_duplicate_rows[duplicate_columns],
            use_container_width=True,
            hide_index=True,
        )
        st.download_button(
            "⬇️ Download repeated configurations (CSV)",
            data=possible_duplicate_rows[duplicate_columns].to_csv(index=False).encode("utf-8"),
            file_name="CarWise_Possible_Duplicates.csv",
            mime="text/csv",
            key="download_possible_duplicates",
            use_container_width=True,
        )

    issue_rows = []
    for rule, count in invalid_count_by_rule.items():
        if count:
            issue_rows.append({"Issue": rule, "Affected records": count})
    for column, count in missing_by_column.items():
        if count:
            issue_rows.append({"Issue": f"Missing values in {column}", "Affected records": int(count)})
    if exact_duplicate_count:
        issue_rows.append({"Issue": "Exact duplicate rows", "Affected records": exact_duplicate_count})
    audit_export = pd.DataFrame(issue_rows, columns=["Issue", "Affected records"])
    st.download_button(
        "⬇️ Download dataset audit report (CSV)",
        data=audit_export.to_csv(index=False).encode("utf-8"),
        file_name="CarWise_Dataset_Audit.csv",
        mime="text/csv",
        key="download_dataset_audit",
        use_container_width=True,
    )

    if exact_duplicate_count == 0 and missing_cell_count == 0 and invalid_record_count == 0:
        st.success("No exact duplicates, missing cells, or checked range issues were detected in the loaded dataset.")
    else:
        st.info("Review the audit results before expanding the dataset. Records are not automatically removed by this report.")


# =========================================================
# RECOMMENDATION FEEDBACK CENTRE
# =========================================================

st.divider()
st.header("💬 CarWise Recommendation Feedback")
st.write(
    "Record whether the recommendations were useful. This creates an exportable feedback log "
    "for future evaluation; it does not automatically retrain the recommendation engine."
)

if "carwise_feedback_log" not in st.session_state:
    st.session_state.carwise_feedback_log = []

feedback_car_options = cars["name"].astype(str).drop_duplicates().tolist()
if selected_for_report is not None:
    current_recommended_name = str(selected_for_report.get("name", feedback_car_options[0] if feedback_car_options else ""))
else:
    current_recommended_name = feedback_car_options[0] if feedback_car_options else ""

with st.form("carwise_feedback_form", clear_on_submit=True):
    feedback_car = st.selectbox(
        "Which car are you giving feedback about?",
        options=feedback_car_options,
        index=feedback_car_options.index(current_recommended_name) if current_recommended_name in feedback_car_options else 0,
        key="feedback_car_choice",
    )
    feedback_rating = st.select_slider(
        "How useful was this recommendation?",
        options=["Not useful", "Somewhat useful", "Useful", "Very useful"],
        value="Useful",
    )
    feedback_reason = st.selectbox(
        "Main reason",
        ["Matches my budget", "Matches my requirements", "Good value", "Missing important options", "Price or specifications seem inaccurate", "Other"],
    )
    feedback_notes = st.text_area(
        "Optional comments",
        placeholder="Example: I need more boot space or the on-road price is different.",
        max_chars=500,
    )
    feedback_submitted = st.form_submit_button("Save Feedback", use_container_width=True)

if feedback_submitted:
    st.session_state.carwise_feedback_log.append({
        "timestamp_utc": pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d %H:%M:%S UTC"),
        "car_name": feedback_car,
        "usefulness": feedback_rating,
        "reason": feedback_reason,
        "comments": feedback_notes.strip(),
    })
    st.success("Feedback saved for this active session.")

feedback_df = pd.DataFrame(
    st.session_state.carwise_feedback_log,
    columns=["timestamp_utc", "car_name", "usefulness", "reason", "comments"],
)
feedback_left, feedback_right = st.columns([2, 1])
with feedback_left:
    st.metric("Feedback entries this session", len(feedback_df))
    if not feedback_df.empty:
        st.dataframe(feedback_df.tail(5), use_container_width=True, hide_index=True)
    else:
        st.caption("No feedback has been saved in this session yet.")
with feedback_right:
    st.download_button(
        "⬇️ Download Feedback Log (CSV)",
        data=feedback_df.to_csv(index=False).encode("utf-8-sig"),
        file_name="CarWise_Recommendation_Feedback.csv",
        mime="text/csv",
        key="download_feedback_log",
        use_container_width=True,
    )
    st.caption("Feedback is kept in the current Streamlit session and is not sent to a server database.")


# =========================================================
# ONE-CLICK CARWISE BUYER PACK
# =========================================================

st.divider()
st.header("📦 One-Click CarWise Buyer Pack")
st.write(
    "Download your key CarWise results together in one ZIP file: recommendations, "
    "the complete catalogue, a pre-purchase checklist, a dataset audit, and your "
    "PDF recommendation report when a suitable car is available."
)

buyer_pack_buffer = BytesIO()

# Build a concise audit summary from the records currently loaded by CarWise.
required_audit_columns = [
    "name", "price", "fuel", "transmission", "seats", "type",
    "mileage", "safety", "maintenance"
]
missing_audit = {
    column: int(cars[column].isna().sum())
    for column in required_audit_columns
    if column in cars.columns
}
exact_duplicates = int(cars.duplicated().sum())
possible_duplicates = cars.duplicated(
    subset=[column for column in ["name", "fuel", "transmission", "type"] if column in cars.columns],
    keep=False,
).sum() if all(column in cars.columns for column in ["name", "fuel", "transmission", "type"]) else 0

buyer_audit_rows = [
    {"Check": "Loaded car records", "Result": len(cars)},
    {"Check": "Exact duplicate rows", "Result": exact_duplicates},
    {"Check": "Possible repeated model configurations", "Result": int(possible_duplicates)},
    {"Check": "Records without HTTPS image URL", "Result": int((~cars["image_url"].astype(str).str.startswith("https://")).sum()) if "image_url" in cars.columns else len(cars)},
]
for column, count in missing_audit.items():
    buyer_audit_rows.append({"Check": f"Missing values: {column}", "Result": count})

buyer_checklist_items = [
    "Confirm exact model and variant",
    "Verify current on-road price and taxes",
    "Take a test drive",
    "Check service history and warranty",
    "Verify safety features for the exact variant",
    "Compare insurance quotes",
    "Estimate fuel or charging cost",
    "Inspect tyres, documents and vehicle condition",
    "Check availability of service centres and spare parts",
]
buyer_checklist_template = pd.DataFrame({
    "Pre-purchase check": buyer_checklist_items,
    "Status": ["Pending"] * len(buyer_checklist_items),
    "Notes": [""] * len(buyer_checklist_items),
})

with zipfile.ZipFile(buyer_pack_buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as buyer_zip:
    recommendations_export = eligible.copy() if "eligible" in globals() else pd.DataFrame()
    if not recommendations_export.empty:
        buyer_zip.writestr(
            "CarWise_Recommendations.csv",
            recommendations_export.to_csv(index=False).encode("utf-8-sig"),
        )
    else:
        buyer_zip.writestr(
            "CarWise_Recommendations.csv",
            "No eligible recommendations for the current filters.\n",
        )

    catalogue_export = cars.copy()
    buyer_zip.writestr(
        "CarWise_Full_Catalogue.csv",
        catalogue_export.to_csv(index=False).encode("utf-8-sig"),
    )
    buyer_zip.writestr(
        "CarWise_PrePurchase_Checklist_Template.csv",
        buyer_checklist_template.to_csv(index=False).encode("utf-8-sig"),
    )
    buyer_zip.writestr(
        "CarWise_Dataset_Audit_Summary.csv",
        pd.DataFrame(buyer_audit_rows).to_csv(index=False).encode("utf-8-sig"),
    )
    buyer_zip.writestr(
        "CarWise_Recommendation_Feedback.csv",
        feedback_df.to_csv(index=False).encode("utf-8-sig"),
    )

    if selected_for_report is not None:
        try:
            buyer_pdf = build_pdf_report(
                selected_for_report,
                budget,
                selected_score,
                selected_reasons,
                selected_pros,
                selected_checks,
                None,
            )
            buyer_zip.writestr("CarWise_Recommendation_Report.pdf", buyer_pdf)
        except Exception:
            buyer_zip.writestr(
                "PDF_Report_Note.txt",
                "The PDF report could not be generated for this run. Use the PDF download in the Best Recommendation section if available.\n",
            )
    else:
        buyer_zip.writestr(
            "PDF_Report_Note.txt",
            "No eligible car was available for the current filters, so a recommendation PDF was not included.\n",
        )

st.download_button(
    label="⬇️ Download Complete CarWise Buyer Pack (ZIP)",
    data=buyer_pack_buffer.getvalue(),
    file_name="CarWise_Complete_Buyer_Pack.zip",
    mime="application/zip",
    key="download_complete_buyer_pack",
    use_container_width=True,
)
st.caption(
    "The ZIP is generated from the current app dataset and filters. Checklist entries start as pending; "
    "the audit is a basic data summary, not independent verification of car specifications or prices."
)


# =========================================================
# PROJECT INFORMATION
# =========================================================

st.divider()
st.header("ℹ️ About CarWise")

with st.expander("Project overview"):
    st.markdown("""
    **Objective:** Help users shortlist cars based on budget and preferences.

    **Technologies:** Python, Streamlit, Pandas, scikit-learn and ReportLab.

    **Recommendation workflow:**
    1. Load and validate the car dataset.
    2. Encode categorical car features.
    3. Scale numerical features.
    4. Convert user preferences into a feature profile.
    5. Calculate content-based similarity.
    6. Rank eligible cars and display recommendations.
    7. Allow users to compare cars and download reports.

    **Ownership-cost calculator:** Estimates purchase price, energy,
    maintenance and insurance costs using user-adjustable assumptions.

    **Important limitation:** This is a content-based similarity
    recommendation prototype, not a supervised model trained on real
    customer choices. The match score is not prediction accuracy.
    Results depend on the quality and size of the dataset.
    """)

st.divider()
st.caption(
    "CarWise | Smart Car Finder | Built with Python, Streamlit and scikit-learn"
)
