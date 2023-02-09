import pandas as pd
import plotly.graph_objs as go
import streamlit as st

try:
    df = pd.read_csv("5_days_forecast.csv")
    data = df.iloc[:, 0:-1].dropna()
    data.index = data.iloc[:, 0]

    data_pred = df.iloc[:, [0, -1]].dropna()
    data_pred.index = data_pred.iloc[:, 0]

    #!-------------- Steramlit viz --------------!#
    # ---------------------------------------------

    rows = st.columns(2)
    rows[0].table(data[["Close"]])
    rows[1].table(data_pred["Prediction - Next 5 days"])

    fig = go.Figure()
    fig.update_layout(xaxis=dict(showgrid=False), yaxis=dict(showgrid=False))

    fig.add_scattergl(x=data.index, y=data["Close"], line={"color": "blue"}, name="Ground Truth")
    fig.add_scattergl(
        x=data_pred.index, y=data_pred["Prediction - Next 5 days"], line={"color": "red"}, name="Prediction"
    )
    st.plotly_chart(fig)

    # st.code(f"// Next price fetch will occur at: {schedule.next_run()}", language="bash")

except Exception:
    st.markdown("Please execute run.py to start fetching data.", unsafe_allow_html=False)
