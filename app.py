import streamlit as st
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
import os

st.set_page_config(layout="wide")
st.title("🛡️ HighStakes Elite Command Center")

SIMS = 10000

# ==============================
# MODEL HANDLING
# ==============================

def create_dummy_models():
    np.random.seed(42)
    n_samples = 1000
    X = np.random.rand(n_samples, 21)  # Increased features
    
    X[:, 0] *= 10   # h_sot
    X[:, 1] *= 5    # h_bc
    X[:, 2] *= 3    # h_gpg
    X[:, 3] *= 3    # h_con
    X[:, 6] *= 10   # a_sot
    X[:, 7] *= 5    # a_bc
    X[:, 8] *= 3    # a_gpg
    X[:, 9] *= 3    # a_con
    X[:, 19] *= 20  # h_position (1-20)
    X[:, 20] *= 20  # a_position (1-20)
    
    y_home = (
        X[:, 0] * 0.15 +
        X[:, 1] * 0.4 +
        X[:, 2] * 0.3 +
        (3 - X[:, 9]) * 0.2 +
        X[:, 12] * 0.5 +
        (21 - X[:, 19]) * 0.05 +  # Better position = more goals
        np.random.normal(0, 0.3, n_samples)
    ).clip(0.1, 4.0)
    
    y_away = (
        X[:, 6] * 0.12 +
        X[:, 7] * 0.35 +
        X[:, 8] * 0.3 +
        (3 - X[:, 3]) * 0.15 +
        X[:, 13] * 0.4 +
        (21 - X[:, 20]) * 0.04 +
        np.random.normal(0, 0.25, n_samples) - 0.3
    ).clip(0.1, 3.5)
    
    y_corners_h = (X[:, 0] * 0.8 + X[:, 2] * 2 + np.random.normal(4, 1.5, n_samples)).clip(1, 15)
    y_corners_a = (X[:, 6] * 0.7 + X[:, 8] * 1.8 + np.random.normal(3.5, 1.5, n_samples)).clip(1, 12)
    
    home_model = RandomForestRegressor(n_estimators=50, random_state=42)
    away_model = RandomForestRegressor(n_estimators=50, random_state=42)
    corner_home_model = RandomForestRegressor(n_estimators=30, random_state=42)
    corner_away_model = RandomForestRegressor(n_estimators=30, random_state=42)
    
    home_model.fit(X, y_home)
    away_model.fit(X, y_away)
    corner_home_model.fit(X, y_corners_h)
    corner_away_model.fit(X, y_corners_a)
    
    joblib.dump(home_model, "home_model.pkl")
    joblib.dump(away_model, "away_model.pkl")
    joblib.dump(corner_home_model, "corner_home_model.pkl")
    joblib.dump(corner_away_model, "corner_away_model.pkl")
    
    return home_model, away_model, corner_home_model, corner_away_model

def load_models():
    model_files = ["home_model.pkl", "away_model.pkl", "corner_home_model.pkl", "corner_away_model.pkl"]
    
    if all(os.path.exists(f) for f in model_files):
        try:
            return (
                joblib.load("home_model.pkl"),
                joblib.load("away_model.pkl"),
                joblib.load("corner_home_model.pkl"),
                joblib.load("corner_away_model.pkl")
            )
        except:
            st.warning("Corrupted model files detected. Recreating...")
            return create_dummy_models()
    else:
        st.info("🤖 Creating prediction models for the first time...")
        return create_dummy_models()

home_model, away_model, corner_home_model, corner_away_model = load_models()

# ==============================
# MAPS
# ==============================
league_map = {
    "EPL": 1.0, "LA_LIGA": 0.92, "SERIE_A": 0.88, "BUNDESLIGA": 1.05,
    "LIGUE_1": 0.95, "UCL": 1.1, "EUROPA": 1.02, "MLS": 1.15
}

intensity_map = {
    "Friendly": 0.75, "League": 1.0, "Derby": 1.15,
    "Cup Final": 1.25, "Relegation": 1.2, "Title": 1.3
}

# NEW: H2H as dropdown instead of slider
h2h_map = {
    "Strong Home H2H": 0.25,
    "Slight Home H2H": 0.12,
    "Neutral": 0.0,
    "Slight Away H2H": -0.12,
    "Strong Away H2H": -0.25
}

# NEW: Lineup strength impact
lineup_map = {
    "Full A-Team": 1.0,
    "Mostly A-Team (1-2 changes)": 0.95,
    "Mixed (Rotation)": 0.85,
    "Mostly B-Team": 0.75,
    "Full B-Team/Youth": 0.65
}

def form_score(f):
    if not f:
        return 1.0
    pts = sum({'W': 3, 'D': 1, 'L': 0}.get(c, 0) for c in f.upper())
    return 0.5 + (pts / (len(f) * 3)) * 0.8

# ==============================
# 🏥 INJURY SYSTEM
# ==============================

injury_severity = {
    "None": 0.0, "Knock": 0.05, "Muscle Fatigue": 0.07,
    "Hamstring": 0.15, "Ankle Sprain": 0.12,
    "Knee Injury": 0.18, "Groin": 0.13,
    "Back Injury": 0.10, "ACL": 0.35, "Broken Bone": 0.35
}

player_importance = {
    "Key Player": 1.3, "Starter": 1.0,
    "Rotation": 0.6, "Bench": 0.3
}

position_weight = {
    "Forward": 1.2, "Midfielder": 1.0,
    "Defender": 0.9, "Goalkeeper": 1.1
}

def injury_input_block(team_name, key):
    st.markdown(f"### {team_name} Injuries")
    
    attack_impact = 0
    defense_impact = 0
    
    count = st.number_input(f"{team_name} Injured Players", 0, 10, 2, key=f"{key}_count")
    
    for i in range(count):
        c1, c2, c3, c4, c5 = st.columns(5)
        
        name = c1.text_input("Name", key=f"{key}_n{i}")
        pos = c2.selectbox("Pos", list(position_weight.keys()), key=f"{key}_p{i}")
        sev = c3.selectbox("Injury", list(injury_severity.keys()), key=f"{key}_s{i}")
        imp = c4.selectbox("Role", list(player_importance.keys()), key=f"{key}_i{i}")
        start = c5.checkbox("XI", key=f"{key}_x{i}")
        
        if name:
            base = injury_severity[sev] * player_importance[imp] * position_weight[pos]
            if start:
                base *= 1.1
            
            if pos == "Forward":
                attack_impact += base
            elif pos in ["Defender", "Goalkeeper"]:
                defense_impact += base
            else:
                attack_impact += base * 0.6
                defense_impact += base * 0.4
    
    return min(attack_impact, 0.6), min(defense_impact, 0.6)

# ==============================
# 🧠 PLAYER MEMORY
# ==============================
if "player_db" not in st.session_state:
    st.session_state.player_db = {}

# ==============================
# INPUT UI
# ==============================
with st.form("match"):
    col1, col2 = st.columns(2)
    
    with col1:
        h_name = st.text_input("Home Team")
        h_sot = st.number_input("SoT", 0.0, 10.0, 5.0)
        h_bc = st.number_input("Big Chances", 0.0, 5.0, 2.0)
        h_gpg = st.number_input("Goals/Game", 0.0, 5.0, 1.8)
        h_con = st.number_input("Conceded", 0.0, 5.0, 1.0)
        h_cs = st.number_input("Clean Sheets", 0.0, 1.0, 0.3)
        h_pos = st.number_input("Possession", 0.0, 100.0, 55.0)
        h_form = st.text_input("Form", "WWDLW")
        # NEW: League position
        h_position = st.number_input("League Position (Home)", 1, 20, 5)
    
    with col2:
        a_name = st.text_input("Away Team")
        a_sot = st.number_input("SoT ", 0.0, 10.0, 4.0)
        a_bc = st.number_input("Big Chances ", 0.0, 5.0, 1.5)
        a_gpg = st.number_input("Goals/Game ", 0.0, 5.0, 1.2)
        a_con = st.number_input("Conceded ", 0.0, 5.0, 1.3)
        a_cs = st.number_input("Clean Sheets ", 0.0, 1.0, 0.2)
        a_pos = st.number_input("Possession ", 0.0, 100.0, 45.0)
        a_form = st.text_input("Form ", "LDWLL")
        # NEW: League position
        a_position = st.number_input("League Position (Away)", 1, 20, 12)
    
    # Match context
    league = st.selectbox("League", list(league_map.keys()))
    intensity = st.selectbox("Match Type", list(intensity_map.keys()))
    
    # CHANGED: H2H dropdown instead of slider
    h2h = st.selectbox("Head-to-Head History", list(h2h_map.keys()))
    
    # NEW: Lineup strength selectors
    col_lineup1, col_lineup2 = st.columns(2)
    with col_lineup1:
        h_lineup = st.selectbox("Home Lineup Strength", list(lineup_map.keys()))
    with col_lineup2:
        a_lineup = st.selectbox("Away Lineup Strength", list(lineup_map.keys()))
    
    # INJURIES
    st.markdown("## 🏥 Injury System")
    
    coli1, coli2 = st.columns(2)
    
    with coli1:
        h_att_inj, h_def_inj = injury_input_block("Home", "h")
    
    with coli2:
        a_att_inj, a_def_inj = injury_input_block("Away", "a")
    
    st.markdown("### Odds")
    bk1 = st.number_input("Home Odds", 1.0, 10.0, 1.8)
    bkx = st.number_input("Draw Odds", 1.0, 10.0, 3.5)
    bk2 = st.number_input("Away Odds", 1.0, 10.0, 4.5)
    bk_btts = st.number_input("BTTS Odds", 1.0, 10.0, 1.7)
    bk_o25 = st.number_input("Over 2.5 Odds", 1.0, 10.0, 1.6)
    bk_o15 = st.number_input("Over 1.5 Odds", 1.0, 10.0, 1.3)
    bk_o35 = st.number_input("Over 3.5 Odds", 1.0, 10.0, 2.5)
    bk_corners = st.number_input("Corners >9.5 Odds", 1.0, 10.0, 1.9)
    bk_3plus = st.number_input("3+ Goals Streak (Yes) Odds", 1.0, 10.0, 1.6)
    
    run = st.form_submit_button("RUN SIM")

# ==============================
# RUN ENGINE
# ==============================
if run:
    # Apply lineup strength adjustments
    h_lineup_factor = lineup_map[h_lineup]
    a_lineup_factor = lineup_map[a_lineup]
    
    features = np.array([[
        h_sot * h_lineup_factor,  # Adjust stats by lineup quality
        h_bc * h_lineup_factor,
        h_gpg * h_lineup_factor,
        h_con * (2 - h_lineup_factor),  # Weaker lineup = more conceded
        h_cs * h_lineup_factor,
        h_pos * h_lineup_factor,
        a_sot * a_lineup_factor,
        a_bc * a_lineup_factor,
        a_gpg * a_lineup_factor,
        a_con * (2 - a_lineup_factor),
        a_cs * a_lineup_factor,
        a_pos * a_lineup_factor,
        form_score(h_form) * h_lineup_factor,  # Form also affected by lineup
        form_score(a_form) * a_lineup_factor,
        (h_att_inj + h_def_inj),
        (a_att_inj + a_def_inj),
        league_map[league],
        intensity_map[intensity],
        h2h_map[h2h],  # Use mapped value
        h_position,    # NEW: League position
        a_position     # NEW: League position
    ]])
    
    h_xg = home_model.predict(features)[0]
    a_xg = away_model.predict(features)[0]
    
    # Apply defensive injury boost (opponent's defense weakened)
    h_xg *= (1 + a_def_inj)
    a_xg *= (1 + h_def_inj)
    
    # Apply league and intensity
    h_xg *= league_map[league] * intensity_map[intensity]
    a_xg *= league_map[league] * intensity_map[intensity]
    
    # Apply H2H adjustment
    h_xg += h2h_map[h2h]
    a_xg -= h2h_map[h2h]
    
    # Apply standing/position adjustment (better position = better performance)
    position_diff = (21 - h_position) - (21 - a_position)  # Higher number = better
    h_xg += position_diff * 0.03
    a_xg -= position_diff * 0.03
    
    # Ensure positive xG
    h_xg = max(0.1, h_xg)
    a_xg = max(0.1, a_xg)
    
    h_sim = np.random.poisson(h_xg, SIMS)
    a_sim = np.random.poisson(a_xg, SIMS)
    total = h_sim + a_sim
    
    corners = np.random.poisson(
        corner_home_model.predict(features)[0] +
        corner_away_model.predict(features)[0], SIMS
    )
    
    streak_3plus = total >= 3
    
    df = pd.DataFrame({
        "Market": ["Home", "Draw", "Away", "BTTS", "O2.5", "O1.5", "O3.5", "Corners>9.5", "3+ Goals Streak"],
        "Prob": [
            np.mean(h_sim > a_sim),
            np.mean(h_sim == a_sim),
            np.mean(h_sim < a_sim),
            np.mean((h_sim > 0) & (a_sim > 0)),
            np.mean(total > 2.5),
            np.mean(total > 1.5),
            np.mean(total > 3.5),
            np.mean(corners > 9.5),
            np.mean(streak_3plus)
        ],
        "Odds": [bk1, bkx, bk2, bk_btts, bk_o25, bk_o15, bk_o35, bk_corners, bk_3plus]
    })
    
    df["Edge %"] = (df["Prob"] * df["Odds"] - 1) * 100
    
    # Display results
    st.subheader("📊 Prediction Results")
    
    # Color coding for edges
    def color_edge(val):
        if val > 5:
            return 'background-color: #006400; color: white'  # Dark green
        elif val > 0:
            return 'background-color: #90EE90'  # Light green
        elif val > -5:
            return 'background-color: #FFB6C1'  # Light red
        else:
            return 'background-color: #8B0000; color: white'  # Dark red
    
    styled_df = df.style.format({"Prob": "{:.1%}", "Edge %": "{:.1f}%"})\
                       .applymap(color_edge, subset=['Edge %'])
    
    st.dataframe(styled_df)
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Home xG", f"{h_xg:.2f}")
    col2.metric("Away xG", f"{a_xg:.2f}")
    col3.metric("Total xG", f"{h_xg + a_xg:.2f}")
    col4.metric("Model Confidence", f"{abs(h_xg - a_xg):.2f}")
    
    # Context info
    st.caption(f"Lineup Impact: Home {h_lineup_factor:.0%} | Away {a_lineup_factor:.0%}")
    st.caption(f"Standing Impact: Position {h_position} vs {a_position}")
    
    # Most likely scorelines
    st.subheader("🔮 Most Likely Scorelines")
    scores = pd.DataFrame({
        'Home': h_sim,
        'Away': a_sim
    })
    score_counts = scores.value_counts().head(5)
    score_df = pd.DataFrame({
        'Score': [f"{h}-{a}" for h, a in score_counts.index],
        'Probability': [f"{c/SIMS:.1%}" for c in score_counts.values]
    })
    st.table(score_df)
    
    # 3+ Goals Streak specific
    st.subheader("🎯 3+ Goals Streak Analysis")
    col1, col2, col3 = st.columns(3)
    col1.metric("3+ Goals Probability", f"{np.mean(streak_3plus):.1%}")
    col2.metric("Odds", f"{bk_3plus:.2f}")
    col3.metric("Edge", f"{(np.mean(streak_3plus) * bk_3plus - 1) * 100:.1f}%", 
                delta="Value!" if (np.mean(streak_3plus) * bk_3plus - 1) > 0 else "No value")
