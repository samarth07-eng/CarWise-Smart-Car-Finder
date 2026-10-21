
import streamlit as st
import pandas as pd
import numpy as np

from pathlib import Path
from io import BytesIO

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
# CARWISE - MACHINE-LEARNING CAR RECOMMENDATION SYSTEM
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
    .stApp {
        background-color: #f5f7fb;
    }

    .block-container {
        padding-top: 1.5rem;
        max-width: 1400px;
    }

    .hero {
        background: linear-gradient(120deg, #10234f, #2459a9);
        color: white;
        padding: 28px;
        border-radius: 18px;
        margin-bottom: 20px;
    }

    .hero h1 {
        color: white;
        font-size: 36px;
    }

    .hero p {
        color: #e4ecff;
    }

    div[data-testid="stMetric"] {
        background: white;
        padding: 12px;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
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

    categorical_columns = [
        "name", "fuel", "transmission", "type"
    ]

    for column in categorical_columns:
        df[column] = (
            df[column]
            .fillna("Unknown")
            .astype(str)
            .str.strip()
        )

    numerical_columns = [
        "price", "seats", "mileage", "safety", "maintenance"
    ]

    for column in numerical_columns:
        df[column] = pd.to_numeric(
            df[column], errors="coerce"
        )

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

    df["image_url"] = (
        df["image_url"].fillna("").astype(str).str.strip()
    )

    return df.reset_index(drop=True)


if not DATA_FILE.exists():
    st.error("cars.csv was not found beside app.py.")
    st.stop()

try:
    cars = load_cars(
        str(DATA_FILE),
        DATA_FILE.stat().st_mtime
    )
except Exception as error:
    st.error(f"Could not load cars.csv: {error}")
    st.stop()

if cars.empty:
    st.error("No valid car records were found in cars.csv.")
    st.stop()


# =========================================================
# MACHINE-LEARNING RECOMMENDATION ENGINE
# =========================================================

categorical_features = ["fuel", "transmission", "type"]

numerical_features = [
    "price", "seats", "mileage", "safety", "maintenance"
]

@st.cache_resource
def build_recommendation_engine(data):
    """
    Learn feature representations from the available car dataset.

    OneHotEncoder handles categorical features.
    MinMaxScaler scales numerical features.
    Cosine similarity compares a user's preferences with each car.
    """

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
    """
    First apply strict budget and maintenance limits.
    Then rank eligible cars using content-based similarity.
    """

    eligible = data[
        (data["price"] <= budget)
        & (data["maintenance"] <= max_maintenance)
    ].copy()

    if eligible.empty:
        return eligible

    # Build a preference profile for the user's requested car.
    # A target price below the maximum budget gives affordable
    # cars a chance to rank well instead of rewarding only
    # cars that cost almost the entire budget.

    preferred_price = budget * 0.75

    query = pd.DataFrame([{
        "fuel": fuel if fuel != "Any" else "Unknown",
        "transmission": (
            transmission if transmission != "Any" else "Unknown"
        ),
        "type": car_type if car_type != "Any" else "Unknown",
        "price": preferred_price,
        "seats": seats,
        "mileage": min_mileage,
        "safety": min_safety,
        "maintenance": max_maintenance,
    }])

    query_matrix = transformer.transform(query)

    # Similarity is calculated against all cars and then
    # applied to the eligible cars only.
    similarities = cosine_similarity(
        query_matrix,
        car_feature_matrix
    )[0]

    eligible_indices = eligible.index.to_list()

    eligible["similarity_score"] = [
        float(similarities[index]) * 100
        for index in eligible_indices
    ]

    # Small preference bonuses help account for the user's
    # minimum mileage and safety targets.
    eligible["target_bonus"] = 0.0

    eligible.loc[
        eligible["mileage"] >= min_mileage,
        "target_bonus"
    ] += 5

    eligible.loc[
        eligible["safety"] >= min_safety,
        "target_bonus"
    ] += 5

    if fuel != "Any":
        eligible.loc[
            eligible["fuel"].str.lower() == fuel.lower(),
            "target_bonus"
        ] += 5

    if transmission != "Any":
        eligible.loc[
            eligible["transmission"].str.lower()
            == transmission.lower(),
            "target_bonus"
        ] += 5

    if car_type != "Any":
        eligible.loc[
            eligible["type"].str.lower() == car_type.lower(),
            "target_bonus"
        ] += 5

    eligible["match_score"] = (
        eligible["similarity_score"] * 0.8
        + eligible["target_bonus"]
    ).clip(0, 100).round(1)

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
        reasons.append("Ranked by similarity to your preferences")

    return "; ".join(reasons)


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
        ["Similarity-based match score", f"{score:.1f}%"],
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
        story.append(Paragraph(
            "• " + str(reason),
            styles["BodyText"]
        ))

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
    <h1>🚗 CarWise — Smart Car Finder</h1>
    <p>Find suitable cars, compare specifications and download a report.</p>
</div>
""", unsafe_allow_html=True)

st.caption(
    "Recommendation engine: feature encoding, numerical scaling "
    "and content-based similarity ranking."
)


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
    step=(
        0.5 if max_budget - min_budget >= 0.5 else 0.1
    ),
    key="budget_filter",
)

fuel_options = sorted(cars["fuel"].unique().tolist())
transmission_options = sorted(
    cars["transmission"].unique().tolist()
)
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
    [2, 4, 5, 6, 7, 8]
    + cars["seats"].astype(int).tolist()
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
    "Maintenance scale: 1 = lower listed level, "
    "5 = higher listed level."
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
        reason
        for reason in best["match_reasons"].split("; ")
        if reason
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
            st.image(
                best["image_url"],
                use_container_width=True,
            )

        st.subheader(best["name"])
        st.metric("Similarity-based match", f"{selected_score:.1f}%")
        st.progress(min(1.0, selected_score / 100))
        st.markdown(f"### {money(best['price'])}")
        st.metric(
            "Budget remaining",
            money(budget - best["price"]),
        )

    with right:
        st.subheader("Car specifications")
        st.write(f"**Fuel:** {best['fuel']}")
        st.write(f"**Transmission:** {best['transmission']}")
        st.write(f"**Seats:** {best['seats']:g}")
        st.write(f"**Vehicle type:** {best['type']}")
        st.write(f"**Mileage:** {best['mileage']:g}")
        st.write(f"**Safety rating:** {best['safety']:g}/5")
        st.write(
            f"**Maintenance level:** {best['maintenance']:g}/5"
        )

        st.markdown("#### Why it matches")

        for reason in selected_reasons:
            st.write(f"✅ {reason}")

    # =====================================================
    # PDF REPORT DOWNLOAD
    # =====================================================

    st.divider()
    st.subheader("📄 Your Recommendation Report")

    st.write(
        "Download a PDF containing the recommended car, "
        "specifications, match reasons and points to check."
    )

    report_comparison = None

    if "comparison_names" in st.session_state:
        names = st.session_state["comparison_names"]

        if len(names) >= 2:
            report_comparison = cars[
                cars["name"].isin(names)
            ].copy()

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
                st.image(
                    car["image_url"],
                    use_container_width=True,
                )

            st.subheader(f"{index + 1}. {car['name']}")
            st.metric("Match", f"{car['match_score']:.1f}%")
            st.write(f"**Price:** {money(car['price'])}")
            st.write(f"Fuel: {car['fuel']}")
            st.write(f"Transmission: {car['transmission']}")
            st.write(f"Seats: {car['seats']:g}")
            st.write(f"Mileage: {car['mileage']:g}")

    csv_download(
        eligible,
        "⬇️ Download recommendations",
        "carwise_recommendations.csv",
        "download_recommendations",
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
    st.image(
        detail["image_url"],
        use_container_width=True,
    )

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
    comparison = cars[
        cars["name"].isin(comparison_names)
    ].copy()

    order = {
        name: index
        for index, name in enumerate(comparison_names)
    }

    comparison["_order"] = comparison["name"].map(order)
    comparison = comparison.sort_values("_order").drop(
        columns="_order"
    )

    comparison_columns = [
        "name", "price", "fuel", "transmission",
        "seats", "type", "mileage", "safety", "maintenance"
    ]

    display_comparison = comparison[
        comparison_columns
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
        display_comparison,
        use_container_width=True,
        hide_index=True,
    )

    cheapest = comparison.loc[comparison["price"].idxmin()]
    highest_mileage = comparison.loc[
        comparison["mileage"].idxmax()
    ]
    highest_safety = comparison.loc[
        comparison["safety"].idxmax()
    ]

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

search = st.text_input(
    "Search car name",
    key="catalogue_search",
)

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
        catalogue["fuel"].str.lower()
        == catalogue_fuel.lower()
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
# PROJECT INFORMATION
# =========================================================

st.divider()
st.header("ℹ️ About CarWise")

with st.expander("Project overview"):
    st.markdown("""
    **Objective:** Help users shortlist cars based on budget and preferences.

    **Technologies:** Python, Streamlit, Pandas, scikit-learn and ReportLab.

    **Machine-learning workflow:**
    1. Load and validate the car dataset.
    2. Encode categorical car features.
    3. Scale numerical features.
    4. Convert the user's preferences into a feature profile.
    5. Calculate content-based similarity.
    6. Rank eligible cars and display recommendations.
    7. Allow users to compare cars and download reports.

    **Important limitation:** This is a content-based similarity
    recommendation prototype, not a supervised model trained on
    real customer choices. The match score is not prediction accuracy.
    Results depend on the quality and size of the dataset.
    """)

st.divider()
st.caption(
    "CarWise | Smart Car Finder | Built with Python, Streamlit and scikit-learn"
)
