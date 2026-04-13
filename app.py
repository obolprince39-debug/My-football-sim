import streamlit as st
import numpy as np
import pandas as pd
import joblib

st.set_page_config(layout="wide")
st.title("🛡️ HighStakes Elite Engine")

SIMS = 10000

# LOAD MODELS
home_model = joblib.load("home_model.pkl")
away_model = joblib.load("away_model.pkl")
corner_home_model = joblib.load("corner_home_model.pkl")
corner_away_model = joblib.load("corner_away_model.pkl")

# MAPS
league_map = {
    "EPL":1.0,"LA_LIGA":0.92,"SERIE_A":0.88,"BUNDESLIGA":1.05,
    "LIGUE_1":0.95,"UCL":1.1,"EUROPA":1.02,"MLS":1.15
}

intensity_map = {
    "Friendly":0.75,"League":1.0,"Derby":1.15,
    "Cup Final":1.25,"Relegation":1.2,"Title":1.3
}

def form_score(f):
    pts = sum({'W':3,'D':1,'L':0}.get(c,0) for c in f)
    return 0.5 + (pts/(len(f)*3))*0.8 if f else 1.0

# INPUT
with st.form("form"):
    col1,col2 = st.columns(2)

    with col1:
        h_name = st.text_input("Home Team")
        h_sot = st.number_input("SoT",0.0,10.0,5.0)
        h_bc = st.number_input("Big Chances",0.0,5.0,2.0)
        h_gpg = st.number_input("Goals/Game",0.0,5.0,1.8)
        h_con = st.number_input("Conceded",0.0,5.0,1.0)
        h_cs = st.number_input("Clean Sheets",0.0,1.0,0.3)
        h_pos = st.number_input("Possession",0.0,100.0,55.0)
        h_form = st.text_input("Form","WWDLW")
        h_inj = st.slider("Injury",0.0,0.6,0.1)

    with col2:
        a_name = st.text_input("Away Team")
        a_sot = st.number_input("SoT ",0.0,10.0,4.0)
        a_bc = st.number_input("Big Chances ",0.0,5.0,1.5)
        a_gpg = st.number_input("Goals/Game ",0.0,5.0,1.2)
        a_con = st.number_input("Conceded ",0.0,5.0,1.3)
        a_cs = st.number_input("Clean Sheets ",0.0,1.0,0.2)
        a_pos = st.number_input("Possession ",0.0,100.0,45.0)
        a_form = st.text_input("Form ","LDWLL")
        a_inj = st.slider("Injury ",0.0,0.6,0.1)

    league = st.selectbox("League", list(league_map.keys()))
    intensity = st.selectbox("Match Type", list(intensity_map.keys()))
    h2h = st.slider("H2H", -0.3,0.3,0.0)

    st.markdown("### Odds")
    bk1 = st.number_input("Home",1.0,10.0,1.8)
    bkx = st.number_input("Draw",1.0,10.0,3.5)
    bk2 = st.number_input("Away",1.0,10.0,4.5)

    run = st.form_submit_button("RUN")

# RUN
if run:
    features = np.array([[ 
        h_sot,h_bc,h_gpg,h_con,h_cs,h_pos,
        a_sot,a_bc,a_gpg,a_con,a_cs,a_pos,
        form_score(h_form),form_score(a_form),
        h_inj,a_inj,
        league_map[league],intensity_map[intensity],
        h2h
    ]])

    h_xg = home_model.predict(features)[0]
    a_xg = away_model.predict(features)[0]

    h_sim = np.random.poisson(h_xg, SIMS)
    a_sim = np.random.poisson(a_xg, SIMS)
    total = h_sim + a_sim

    # CORNERS
    h_corner = corner_home_model.predict(features)[0]
    a_corner = corner_away_model.predict(features)[0]
    corner_sim = np.random.poisson(h_corner+a_corner, SIMS)

    # MARKETS
    df = pd.DataFrame({
        "Market":[
            "Home Win","Draw","Away Win",
            "BTTS","Over2.5","Over1.5","Over3.5",
            "Corners Over 9.5"
        ],
        "Prob":[
            np.mean(h_sim>a_sim),
            np.mean(h_sim==a_sim),
            np.mean(h_sim<a_sim),
            np.mean((h_sim>0)&(a_sim>0)),
            np.mean(total>2.5),
            np.mean(total>1.5),
            np.mean(total>3.5),
            np.mean(corner_sim>9.5)
        ],
        "Odds":[bk1,bkx,bk2,1.7,1.6,1.3,2.5,1.9]
    })

    df["Edge %"] = ((df["Prob"] * df["Odds"]) - 1) * 100

    st.dataframe(df.style.format({"Prob":"{:.1%}","Edge %":"{:.1f}%"}))

    # CONFIDENCE
    confidence = abs(h_xg - a_xg)
    st.metric("Model Confidence", f"{confidence:.2f}")
