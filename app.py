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
# MODEL HANDLING (Auto-create if missing)
# ==============================

def create_dummy_models():
    """Create realistic dummy models when .pkl files don't exist"""
    np.random.seed(42)
    
    # Create synthetic training data based on football statistics relationships
    n_samples = 1000
    
    # Generate realistic features
    X = np.random.rand(n_samples, 19)
    # Scale features to realistic ranges
    X[:, 0] *= 10   # h_sot (0-10)
    X[:, 1] *= 5    # h_bc (0-5)
    X[:, 2] *= 3    # h_gpg (0-3)
    X[:, 3] *= 3    # h_con (0-3)
    X[:, 6] *= 10   # a_sot (0-10)
    X[:, 7] *= 5    # a_bc (0-5)
    X[:, 8] *= 3    # a_gpg (0-3)
    X[:, 9] *= 3    # a_con (0-3)
    
    # Home goals: based on shots on target, big chances, form, opponent defense
    y_home = (
        X[:, 0] * 0.15 +      # SoT contribution
        X[:, 1] * 0.4 +       # Big chances (high weight)
        X[:, 2] * 0.3 +       # Historical goals
        (3 - X[:, 9]) * 0.2 + # Opponent defense weakness
        X[:, 12] * 0.5 +      # Home form
        np.random.normal(0, 0.3, n_samples)
    ).clip(0.1, 4.0)
    
    # Away goals: similar but away disadvantage
    y_away = (
        X[:, 6] * 0.12 +
        X[:, 7] * 0.35 +
        X[:, 8] * 0.3 +
        (3 - X[:, 3]) * 0.15 +
        X[:, 13] * 0.4 +
        np.random.normal(0, 0.25, n_samples) - 0.3  # Away penalty
    ).clip(0.1, 3.5)
    
    # Corners models
    y_corners_h = (X[:, 0] * 0.8 + X[:, 2] * 2 + np.random.normal(4, 1.5, n_samples)).clip(1, 15)
    y_corners_a = (X[:, 6] * 0.7 + X[:, 8] * 1.8 + np.random.normal(3.5, 1.5, n_samples)).clip(1, 12)
    
    # Train models
    home_model = RandomForestRegressor(n_estimators=50, random_state=42)
    away_model = RandomForestRegressor(n_estimators=50, random_state=42)
    corner_home_model = RandomForestRegressor(n_estimators=30, random_state=42)
    corner_away_model = RandomForestRegressor(n_estimators=30, random_state=42)
    
    home_model.fit(X, y_home)
    away_model.fit(X, y_away)
    corner_home_model.fit(X, y_corners_h)
    corner_away_model.fit(X, y_corners_a)
    
    # Save them
    joblib.dump(home_model, "home_model.pkl")
    joblib.dump(away_model, "away_model.pkl")
    joblib.dump(corner_home_model, "corner_home_model.pkl")
    joblib.dump(corner_away_model, "corner_away_model.pkl")
    
    return home_model, away_model, corner_home_model, corner_away_model

def load_models():
    """Load existing or create new models"""
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

# Load or create models
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

def form_score(f):
    if not f:
        return 1.0
    pts = sum({'W': 3, 'D': 1, 'L': 0}.get(c, 0) for c in f.upper())
    return 0.5 + (pts / (len(f) * 3)) * 0.8

# ==============================
# 🏥 INJURY SYSTEM (ADVANCED)
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

def save_player(name, pos, role):
    st.session_state.player_db[name] = {"pos": pos, "role": role}

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
    
    with col2:
        a_name = st.text_input("Away Team")
        a_sot = st.number_input("SoT ", 0.0, 10.0, 4.0)
        a_bc = st.number_input("Big Chances ", 0.0, 5.0, 1.5)
        a_gpg = st.number_input("Goals/Game ", 0.0, 5.0, 1.2)
        a_con = st.number_input("Conceded ", 0.0, 5.0, 1.3)
        a_cs = st.number_input("Clean Sheets ", 0.0, 1.0, 0.2)
        a_pos = st.number_input("Possession ", 0.0, 100.0, 45.0)
        a_form = st.text_input("Form ", "LDWLL")
    
    league = st.selectbox("League", list(league_map.keys()))
    intensity = st.selectbox("Match Type", list(intensity_map.keys()))
    h2h = st.slider("H2H", -0.3, 0.3, 0.0)
    
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
    
    run = st.form_submit_button("RUN SIM")

# ==============================
# RUN ENGINE
# ==============================
if run:
    features = np.array([[
        h_sot, h_bc, h_gpg, h_con, h_cs, h_pos,
        a_sot, a_bc, a_gpg, a_con, a_cs, a_pos,
        form_score(h_form), form_score(a_form),
        (h_att_inj + h_def_inj), (a_att_inj + a_def_inj),
        league_map[league], intensity_map[intensity],
        h2h
    ]])
    
    h_xg = home_model.predict(features)[0]
    a_xg = away_model.predict(features)[0]
    
    # 🔥 DEFENSIVE INJURY BOOST
    h_xg *= (1 + a_def_inj)
    a_xg *= (1 + h_def_inj)
    
    # Apply league and intensity adjustments
    h_xg *= league_map[league] * intensity_map[intensity]
    a_xg *= league_map[league] * intensity_map[intensity]
    
    # H2H adjustment
    h_xg += h2h
    a_xg -= h2h
    
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
    
    df = pd.DataFrame({
        "Market": ["Home", "Draw", "Away", "BTTS", "O2.5", "O1.5", "O3.5", "Corners>9.5"],
        "Prob": [
            np.mean(h_sim > a_sim),
            np.mean(h_sim == a_sim),
            np.mean(h_sim < a_sim),
            np.mean((h_sim > 0) & (a_sim > 0)),
            np.mean(total > 2.5),
            np.mean(total > 1.5),
            np.mean(total > 3.5),
            np.mean(corners > 9.5)
        ],
        "Odds": [bk1, bkx, bk2, 1.7, 1.6, 1.3, 2.5, 1.9]
    })
    
    df["Edge %"] = (df["Prob"] * df["Odds"] - 1) * 100
    
    # Display results
    st.subheader("📊 Prediction Results")
    st.dataframe(df.style.format({"Prob": "{:.1%}", "Edge %": "{:.1f}%"}))
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Home xG", f"{h_xg:.2f}")
    col2.metric("Away xG", f"{a_xg:.2f}")
    col3.metric("Confidence", f"{abs(h_xg - a_xg):.2f}")
    
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
    df["Edge %"] = ((df["Prob"] * df["Odds"]) - 1) * 100

    st.dataframe(df.style.format({"Prob":"{:.1%}","Edge %":"{:.1f}%"}))

    # CONFIDENCE
    confidence = abs(h_xg - a_xg)
    st.metric("Model Confidence", f"{confidence:.2f}")
