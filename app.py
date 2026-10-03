"""
app.py
======
Interactive Dash GUI for the chiller FDD system (thesis Chapter 3.10:
Online 24/7 Monitoring and GUI Development).

Layout:
  - Sidebar: dataset selector (EnergyPlus 338-Ton vs RP-1043 90-Ton),
    live/manual mode toggle, manual sensor-input form.
  - Main area: KPI row (COP, Power, Power Residual, primary diagnosis),
    7 fault-probability gauges with the 0.50 alarm threshold marked,
    a rolling trend chart of calibrated probabilities, and a
    diagnosis/alarm banner.

Data source:
  - "Simulated live stream" mode generates a synthetic but physically
    plausible sensor stream (random walk around the dataset's healthy
    baseline, with an operator-triggered fault injection) via
    dcc.Interval, standing in for the real MQTT feed
    (topic hvac/sensor_detailed) until that broker is wired in.
  - "Manual" mode lets you type in one raw reading and see the bank's
    output for that single instant - useful for validating the pipeline
    against known RP-1043 benchmark rows.

Run:
    pip install -r requirements.txt
    python app.py
  then open http://127.0.0.1:8050
"""

import random
from collections import deque

import dash
from dash import dcc, html, Input, Output, State, ctx
import plotly.graph_objects as go

from fdd_pipeline import (
    FDDModelBank, FAULTS, FAULT_LABELS, DATASETS, DATASET_LABELS,
    BASELINE, ALARM_THRESHOLD,
)
from feature_engineering import RawReading

bank = FDDModelBank()

HISTORY_LEN = 60
history = {ds: {f: deque([0.0] * HISTORY_LEN, maxlen=HISTORY_LEN) for f in FAULTS} for ds in DATASETS}

FAULT_COLORS = {
    "nc": "#4C78A8", "oc": "#F58518", "rl": "#54A24B", "cf": "#B279A2",
    "eo": "#E45756", "fwe": "#72B7B2", "fwc": "#EECA3B",
}

app = dash.Dash(__name__)
server = app.server
app.title = "Chiller FDD Monitor"


def gauge(fault: str, probability: float):
    color = FAULT_COLORS.get(fault, "#888")
    alarm = probability >= ALARM_THRESHOLD
    return go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100,
        number={"suffix": "%", "font": {"size": 22}},
        title={"text": FAULT_LABELS[fault], "font": {"size": 12}},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "#D8232A" if alarm else color},
            "threshold": {"line": {"color": "#D8232A", "width": 3}, "value": ALARM_THRESHOLD * 100},
            "steps": [
                {"range": [0, 50], "color": "#EFEFEF"},
                {"range": [50, 100], "color": "#FBE3E3"},
            ],
        },
    )).update_layout(height=180, margin=dict(l=15, r=15, t=40, b=5))


def trend_figure(dataset: str):
    fig = go.Figure()
    for f in FAULTS:
        fig.add_trace(go.Scatter(
            y=list(history[dataset][f]), mode="lines", name=FAULT_LABELS[f].split(" (")[1][:-1],
            line=dict(color=FAULT_COLORS.get(f, "#888"), width=1.6),
        ))
    fig.add_hline(y=ALARM_THRESHOLD, line_dash="dash", line_color="#D8232A",
                   annotation_text="alarm threshold (0.50)")
    fig.update_layout(
        height=300, margin=dict(l=40, r=10, t=30, b=30),
        yaxis=dict(title="Calibrated fault probability", range=[0, 1]),
        xaxis=dict(title="Samples (most recent 60)"),
        legend=dict(orientation="h", y=-0.25),
        title="Fault-probability trend",
    )
    return fig


def sidebar():
    return html.Div([
        html.H2("Chiller FDD Monitor", style={"marginBottom": "4px"}),
        html.P("Physics-informed Gordon-Ng + RBF-SVM | 7-fault bank",
               style={"color": "#777", "fontSize": "13px", "marginTop": 0}),

        html.Div([
            html.Label("Dataset / chiller"),
            dcc.Dropdown(
                id="dataset-select",
                options=[{"label": DATASET_LABELS[d], "value": d} for d in DATASETS],
                value="sql", clearable=False,
            ),
        ], style={"marginBottom": "16px"}),

        html.Div([
            html.Label("Mode"),
            dcc.RadioItems(
                id="mode-select",
                options=[{"label": "Simulated live stream", "value": "live"},
                         {"label": "Manual reading", "value": "manual"}],
                value="live", labelStyle={"display": "block", "marginBottom": "4px"},
            ),
        ], style={"marginBottom": "16px"}),

        html.Div(id="live-controls", children=[
            html.Label("Inject fault into stream"),
            dcc.Dropdown(
                id="inject-fault",
                options=[{"label": "None (healthy)", "value": "none"}] +
                        [{"label": FAULT_LABELS[f], "value": f} for f in FAULTS],
                value="none", clearable=False,
            ),
            html.Label("Severity", style={"marginTop": "8px"}),
            dcc.Slider(id="severity", min=0, max=50, step=10, value=20,
                       marks={i: f"{i}%" for i in range(0, 51, 10)}),
        ]),

        html.Div(id="manual-controls", style={"display": "none"}, children=[
            html.Label("Compressor power P (kW)"),
            dcc.Input(id="m-power", type="number", value=300, style={"width": "100%"}),
            html.Label("Evaporator cooling rate Qevap (kW)", style={"marginTop": "6px"}),
            dcc.Input(id="m-qevap", type="number", value=1100, style={"width": "100%"}),
            html.Label("Chilled water / evaporator INLET temp (°C)", style={"marginTop": "6px"}),
            dcc.Input(id="m-tevi", type="number", value=12.2, style={"width": "100%"}),
            html.Label("Chilled water / evaporator OUTLET temp (°C)", style={"marginTop": "6px"}),
            dcc.Input(id="m-teo", type="number", value=6.7, style={"width": "100%"}),
            html.Label("Condenser water INLET temp (°C) — RP-1043 only",
                       id="m-tci-label", style={"marginTop": "6px"}),
            dcc.Input(id="m-tci", type="number", value=29.4, style={"width": "100%"}),
            html.Button("Evaluate", id="m-evaluate", n_clicks=0, style={"marginTop": "10px", "width": "100%"}),
        ]),

        html.Hr(),
        html.Div(id="model-status", style={"fontSize": "12px", "color": "#777"}),
    ], style={
        "width": "300px", "padding": "20px", "backgroundColor": "#FAFAF8",
        "borderRight": "1px solid #E4E0D8", "height": "100vh", "overflowY": "auto",
        "position": "fixed", "left": 0, "top": 0,
    })


app.layout = html.Div([
    sidebar(),
    html.Div([
        html.Div(id="caveat-banner", children=[
            html.B("⚠ Power_Residual / Power_Residual_norm are not reliable for hand-entered or simulated readings. "),
            "The 12-feature formulas now match your real training script exactly (verified line-by-line). "
            "But the Gordon-Ng physics model itself is numerically ill-conditioned for the EnergyPlus/SQL dataset: "
            "its two inputs (x1, x2) are nearly identical for every row, because the extraction script sets the "
            "condenser-inlet proxy T_ci equal to the evaporator-inlet temperature T_ei (no real condenser-water "
            "variable was queried from the SQL file). That makes the fitted coefficients huge and almost "
            "perfectly cancelling (47425.38 / -47424.56) — the regression's condition number is ~1.3 million, "
            "so it only predicts sensibly for inputs lying almost exactly on the original training data's "
            "correlation structure; any hand-entered or simulated reading pushes it off that manifold and the "
            "residual explodes. For RP-1043, no physics model was ever saved by the training script at all "
            "(only *_sql versions were joblib.dump'd) — this app currently reuses the SQL one as a stand-in, "
            "which is wrong. All other 10 features and all 7 SVM classifiers are correct and usable as-is."
        ], style={
            "backgroundColor": "#FFF4E5", "border": "1px solid #F0B429", "borderRadius": "6px",
            "padding": "10px 14px", "fontSize": "13px", "color": "#7A5B00", "marginBottom": "16px",
            "lineHeight": "1.5",
        }),

        html.Div(id="kpi-row", style={"display": "flex", "gap": "12px", "marginBottom": "16px"}),

        html.Div(id="gauge-grid", style={
            "display": "grid", "gridTemplateColumns": "repeat(4, 1fr)", "gap": "6px",
        }),

        dcc.Graph(id="trend-chart"),

        html.Div(id="diagnosis-banner", style={
            "padding": "14px", "borderRadius": "6px", "fontSize": "15px", "fontWeight": "600",
            "marginTop": "10px",
        }),

        dcc.Interval(id="tick", interval=1500, n_intervals=0),
        dcc.Store(id="latest-result"),
    ], style={"marginLeft": "320px", "padding": "24px"}),
], style={"fontFamily": "Inter, Arial, sans-serif", "backgroundColor": "#FFFDF9"})


@app.callback(
    Output("live-controls", "style"), Output("manual-controls", "style"),
    Input("mode-select", "value"),
)
def toggle_mode(mode):
    if mode == "live":
        return {"display": "block"}, {"display": "none"}
    return {"display": "none"}, {"display": "block"}


def simulate_reading(dataset: str, fault: str, severity_pct: float) -> RawReading:
    base = BASELINE[dataset]
    p_nom = base["power_nominal_kw"]
    cop_nom = base["cop_nominal"]

    noise = lambda scale: random.uniform(-scale, scale)
    p = p_nom * (1 + noise(0.02))
    qevap = p * cop_nom * (1 + noise(0.02))
    tci = 29.4 + noise(1.5)
    tevi = 12.2 + noise(0.5)
    teo = 6.7 + noise(0.2)

    sev = severity_pct / 100.0
    if fault in ("oc", "nc", "cf", "fwc"):
        p *= (1 + 0.5 * sev)       # these faults raise condensing pressure -> more power
        tci += 2.0 * sev
    elif fault == "eo":
        p *= (1 + 0.3 * sev)
    elif fault == "rl":
        p *= (1 - 0.1 * sev)       # thesis: RL can show negative residual on small chillers
        qevap *= (1 - 0.1 * sev)
    elif fault == "fwe":
        teo += 2.0 * sev           # dominant effect is thermal, not electrical
        tevi += 1.0 * sev

    return RawReading(power_kw=p, qevap_kw=qevap, t_evap_in_c=tevi, t_evap_out_c=teo,
                       dataset=dataset, t_cond_in_c=tci)


@app.callback(
    Output("kpi-row", "children"),
    Output("gauge-grid", "children"),
    Output("trend-chart", "figure"),
    Output("diagnosis-banner", "children"),
    Output("diagnosis-banner", "style"),
    Output("model-status", "children"),
    Input("tick", "n_intervals"),
    Input("m-evaluate", "n_clicks"),
    State("dataset-select", "value"),
    State("mode-select", "value"),
    State("inject-fault", "value"),
    State("severity", "value"),
    State("m-power", "value"), State("m-qevap", "value"),
    State("m-tci", "value"),
    State("m-tevi", "value"), State("m-teo", "value"),
)
def update(n_intervals, n_clicks, dataset, mode, inject_fault, severity,
           m_power, m_qevap, m_tci, m_tevi, m_teo):

    if mode == "manual":
        reading = RawReading(
            power_kw=m_power or 300, qevap_kw=m_qevap or 1100,
            t_evap_in_c=m_tevi or 12.2, t_evap_out_c=m_teo or 6.7,
            dataset=dataset, t_cond_in_c=m_tci or 29.4,
        )
    else:
        reading = simulate_reading(dataset, inject_fault, severity)

    results = bank.diagnose(dataset, reading)
    for f in FAULTS:
        history[dataset][f].append(results.get(f, {}).get("probability", 0.0))

    cop = reading.qevap_kw / reading.power_kw
    power_residual = results[FAULTS[0]]["power_residual"] if results else float("nan")

    def kpi(label, value):
        return html.Div([
            html.Div(label, style={"fontSize": "12px", "color": "#888"}),
            html.Div(value, style={"fontSize": "20px", "fontWeight": "700"}),
        ], style={
            "flex": 1, "backgroundColor": "#fff", "border": "1px solid #E4E0D8",
            "borderRadius": "8px", "padding": "12px 16px",
        })

    kpis = [
        kpi("Compressor Power", f"{reading.power_kw:.1f} kW"),
        kpi("Evaporator Cooling Rate", f"{reading.qevap_kw:.0f} kW"),
        kpi("COP", f"{cop:.3f}"),
        kpi("Power Residual", f"{power_residual:.2f} kW"),
    ]

    gauges = [dcc.Graph(figure=gauge(f, results.get(f, {}).get("probability", 0.0)),
                         config={"displayModeBar": False})
              for f in FAULTS]

    primary_fault, primary_p = bank.primary_diagnosis(results)
    if primary_fault:
        banner_text = f"⚠ ALARM: {FAULT_LABELS[primary_fault]} — calibrated probability {primary_p:.1%} (≥ {ALARM_THRESHOLD:.0%} threshold)"
        banner_style = {"backgroundColor": "#FBE3E3", "color": "#A6241A", "border": "1px solid #D8232A"}
    else:
        banner_text = "✓ Normal — no fault probability exceeds the 0.50 alarm threshold"
        banner_style = {"backgroundColor": "#E9F5E9", "color": "#1E6B2E", "border": "1px solid #54A24B"}
    banner_style.update({"padding": "14px", "borderRadius": "6px", "fontSize": "15px",
                          "fontWeight": "600", "marginTop": "10px"})

    status = f"Loaded {len(bank.available_faults)}/7 fault models. Missing files: {len(bank.missing)}."

    return kpis, gauges, trend_figure(dataset), banner_text, banner_style, status


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=8050)
