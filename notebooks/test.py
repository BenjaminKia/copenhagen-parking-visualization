import dash
from dash import Dash, dcc, html
from dash.dependencies import Input, Output

import pandas as pd
import numpy as np
import plotly.graph_objs as go

# ---------------------------------------------------------
# 1. Load and prepare data
# ---------------------------------------------------------
df = pd.read_csv("data/cleaned_df.csv")

# Drop unnamed index column if present
if "Unnamed: 0" in df.columns:
    df = df.drop(columns=["Unnamed: 0"])

# Ensure numeric coords
df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
df["lng"] = pd.to_numeric(df["lng"], errors="coerce")
df = df.dropna(subset=["lat", "lng"])

# Make sure year / month are numeric
df["year"] = pd.to_numeric(df["year"], errors="coerce").astype(int)
df["month"] = pd.to_numeric(df["month"], errors="coerce").astype(int)

# Dropdown options
years = sorted(df["year"].dropna().unique())
months = sorted(df["month"].dropna().unique())

# Time-of-day config
TIME_CONFIG = {
    "12": {
        "label": "12:00",
        "occ_col": "belaegning_kl_12_pct",
        "cap_col": "lovlig_p_kl_12",
    },
    "17": {
        "label": "17:00",
        "occ_col": "belaegning_kl_17_pct",
        "cap_col": "lovlig_p_kl_17",
    },
    "22": {
        "label": "22:00",
        "occ_col": "belaegning_kl_22_pct",
        "cap_col": "lovlig_p_kl_22",
    },
}

# ---------------------------------------------------------
# 2. Capacity calculations (rows -> year/vej niveau)
# ---------------------------------------------------------
capacity_cols = [cfg["cap_col"] for cfg in TIME_CONFIG.values()]

# "capacity" = max lovlig_p over de tre tidspunkter, pr. række
df["capacity"] = df[capacity_cols].max(axis=1, skipna=True)

# Drop rows uden kapacitet
df_cap = df.dropna(subset=["capacity"])

# Én kapacitetsværdi pr. (year, vej_id, vejnavn):
# max over alle tællinger i det år
cap_street_year = (
    df_cap.groupby(["year", "vej_id", "vejnavn"])["capacity"].max().reset_index()
)

# Aggregér til total pladser i byen pr. år (til linjeplottet)
cap_by_year = (
    cap_street_year.groupby("year")["capacity"]
    .sum()
    .reset_index()
    .rename(columns={"capacity": "total_spots"})
)
cap_by_year["year"] = cap_by_year["year"].astype(int)


def make_capacity_figure():
    if cap_by_year.empty:
        return go.Figure()

    fig = go.Figure(
        go.Scatter(
            x=cap_by_year["year"],
            y=cap_by_year["total_spots"],
            mode="lines+markers",
            line=dict(width=3),
            marker=dict(size=8),
            hovertemplate="Year %{x}<br>Total spots: %{y:,}<extra></extra>",
            name="Total parking spots",
        )
    )

    first_year = int(cap_by_year["year"].iloc[0])
    last_year = int(cap_by_year["year"].iloc[-1])
    first_val = int(cap_by_year["total_spots"].iloc[0])
    last_val = int(cap_by_year["total_spots"].iloc[-1])
    delta = last_val - first_val
    pct = (delta / first_val * 100) if first_val > 0 else None

    subtitle = (
        f"Growth from {first_year} to {last_year}: +{delta:,} spots ({pct:.0f}%)"
        if pct is not None
        else ""
    )

    fig.update_layout(
        title=(
            "Total monitored parking spots in Copenhagen over time"
            + (f"<br><sup>{subtitle}</sup>" if subtitle else "")
        ),
        xaxis_title="Year",
        yaxis_title="Number of parking spots",
        hovermode="x unified",
        margin=dict(l=40, r=20, t=80, b=40),
    )
    return fig


# ---------------------------------------------------------
# 3. Dash app + layout
# ---------------------------------------------------------
external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]
app = Dash(__name__, external_stylesheets=external_stylesheets)
app.title = "Copenhagen Parking Occupancy Growth"
server = app.server

MAPBOX_TOKEN = "pk.eyJ1IjoicGxvdGx5bWFwYm94IiwiYSI6ImNrOWJqb2F4djBnMjEzbG50amg0dnJieG4ifQ.Zme1-Uzoi75IaFbieBDl3A"

app.layout = html.Div(
    className="row",
    children=[
        # LEFT COLUMN: filters
        html.Div(
            className="four columns div-user-controls",
            style={"padding": "20px"},
            children=[
                html.H2("Copenhagen Parking"),
                html.P(
                    "Explore parking occupancy by time, year, month and occupancy filters."
                ),
                html.Label("Time of day"),
                dcc.Dropdown(
                    id="time-dropdown",
                    options=[
                        {"label": cfg["label"], "value": t}
                        for t, cfg in TIME_CONFIG.items()
                    ],
                    value="12",
                    clearable=False,
                ),
                html.Br(),
                html.Label("Year"),
                dcc.Dropdown(
                    id="year-dropdown",
                    options=[{"label": int(y), "value": int(y)} for y in years],
                    value=int(years[0]),
                    clearable=False,
                ),
                html.Br(),
                html.Label("Month"),
                dcc.Dropdown(
                    id="month-dropdown",
                    options=[{"label": int(m), "value": int(m)} for m in months],
                    value=int(months[0]),
                    clearable=False,
                ),
                html.Br(),
                html.Label("Max occupancy (%)"),
                dcc.Slider(
                    id="occ-slider",
                    min=0,
                    max=100,
                    step=5,
                    value=100,
                    marks={i: f"{i}%" for i in range(0, 101, 20)},
                ),
                html.Br(),
                html.Div(id="summary-text", style={"marginTop": "10px"}),
            ],
        ),
        # RIGHT COLUMN: map + time series + trends
        html.Div(
            className="eight columns",
            style={"padding": "20px"},
            children=[
                dcc.Graph(
                    id="map-graph",
                    style={"height": "60vh"},
                    config={"displaylogo": False},
                ),
                html.Br(),
                html.H4("Street occupancy over time"),
                dcc.Graph(
                    id="street-timeseries",
                    style={"height": "55vh"},
                    config={"displaylogo": False},
                ),
                html.Br(),
                html.H4("Parking spots over time"),
                dcc.Graph(
                    id="capacity-trend",
                    figure=make_capacity_figure(),
                    style={"height": "35vh"},
                    config={"displaylogo": False},
                ),
                html.Br(),
                html.H4("Year-over-year change in parking capacity"),
                dcc.Graph(id="capacity-change-table"),
            ],
        ),
    ],
)


# ---------------------------------------------------------
# 4. Year-over-year capacity changes (simple, only by year)
# ---------------------------------------------------------
def compute_year_changes(selected_year: int) -> pd.DataFrame:
    """
    capacity_change = capacity(selected_year) - capacity(previous_year)
    hvor 'capacity' er max antal pladser pr. vej i det år.
    """

    # Sørg for at vi arbejder med ints
    selected_year = int(selected_year)
    years_sorted = sorted(int(y) for y in cap_street_year["year"].unique())

    # Første år -> ingen sammenligning
    if selected_year == years_sorted[0]:
        return pd.DataFrame(columns=["vejnavn", "capacity_change"])

    prev_year = years_sorted[years_sorted.index(selected_year) - 1]

    # Kapacitet i valgt år
    curr = cap_street_year[cap_street_year["year"] == selected_year][
        ["vej_id", "vejnavn", "capacity"]
    ].rename(columns={"capacity": "capacity_curr"})

    # Kapacitet i året før
    prev = cap_street_year[cap_street_year["year"] == prev_year][
        ["vej_id", "capacity"]
    ].rename(columns={"capacity": "capacity_prev"})

    # Venstre join: alle veje i selected_year
    merged = curr.merge(prev, on="vej_id", how="left")

    # Veje, der ikke fandtes året før, får 0 som tidligere kapacitet
    merged["capacity_prev"] = merged["capacity_prev"].fillna(0)

    # Ændring som heltal
    merged["capacity_change"] = (
        (merged["capacity_curr"] - merged["capacity_prev"]).round().astype(int)
    )

    # Største stigninger øverst (sæt ascending=True for closures øverst)
    merged = merged.sort_values("capacity_change", ascending=False)

    return merged[["vejnavn", "capacity_change"]]


# ---------------------------------------------------------
# 5. Callbacks
# ---------------------------------------------------------
@app.callback(
    [Output("map-graph", "figure"), Output("summary-text", "children")],
    [
        Input("time-dropdown", "value"),
        Input("year-dropdown", "value"),
        Input("month-dropdown", "value"),
        Input("occ-slider", "value"),
    ],
)
def update_map(selected_time, selected_year, selected_month, max_occ):
    cfg = TIME_CONFIG[selected_time]
    occ_col = cfg["occ_col"]
    cap_col = cfg["cap_col"]

    dff = df.copy()
    if selected_year is not None:
        dff = dff[dff["year"] == selected_year]
    if selected_month is not None:
        dff = dff[dff["month"] == selected_month]

    dff = dff.dropna(subset=[occ_col, cap_col])
    if dff.empty:
        fig = go.Figure()
        fig.update_layout(
            title="No data for selected filters",
            mapbox=dict(
                accesstoken=MAPBOX_TOKEN,
                style="carto-positron",
                zoom=10,
                center=dict(lat=55.6761, lon=12.5683),
            ),
        )
        return fig, "No data available for these filters."

    occ = dff[occ_col].astype(float).clip(lower=0, upper=100)
    capacity = dff[cap_col].astype(float)

    # max occupancy filter
    mask = occ <= max_occ
    dff = dff[mask]
    occ = occ[mask]
    capacity = capacity[mask]

    if dff.empty:
        fig = go.Figure()
        fig.update_layout(
            title="No streets match the filters",
            mapbox=dict(
                accesstoken=MAPBOX_TOKEN,
                style="carto-positron",
                zoom=11,
                center=dict(lat=55.6761, lon=12.5683),
            ),
        )
        return fig, "No streets match the filters."

    min_cap = capacity.min()
    max_cap = capacity.max()
    if max_cap == min_cap:
        sizes = np.full_like(capacity, 10)
    else:
        sizes = 5 + 25 * (capacity - min_cap) / (max_cap - min_cap)

    hover_text = (
        "Vej: "
        + dff["vejnavn"].astype(str)
        + "<br>Vej ID: "
        + dff["vej_id"].astype(str)
        + f"<br>Time: {cfg['label']}"
        + "<br>Occupancy: "
        + occ.round(1).astype(str)
        + " %"
        + "<br>Capacity: "
        + capacity.astype(int).astype(str)
        + " spots"
    )

    fig = go.Figure(
        go.Scattermapbox(
            lat=dff["lat"],
            lon=dff["lng"],
            mode="markers",
            marker=dict(
                size=sizes,
                sizemode="diameter",
                color=occ,
                cmin=0,
                cmax=100,
                colorscale="Viridis",
                colorbar=dict(title="Occupancy (%)"),
                opacity=0.7,
            ),
            text=hover_text,
            hoverinfo="text",
            customdata=dff["vej_id"],
        )
    )

    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        mapbox=dict(
            accesstoken=MAPBOX_TOKEN,
            style="carto-positron",
            zoom=11,
            center=dict(lat=55.69, lon=12.5683),
        ),
        uirevision="parking-map",
    )

    summary = (
        f"{len(dff)} observations | Time: {cfg['label']} | "
        f"Year: {selected_year} | Month: {selected_month}"
    )

    return fig, summary


@app.callback(
    Output("street-timeseries", "figure"),
    Input("map-graph", "clickData"),
)
def update_street_timeseries(clickData):
    if clickData is None:
        fig = go.Figure()
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Occupancy (%)",
            yaxis=dict(range=[0, 100]),
            annotations=[
                dict(
                    x=0.5,
                    y=0.5,
                    xref="paper",
                    yref="paper",
                    text="Click a street on the map to see its history",
                    showarrow=False,
                )
            ],
        )
        return fig

    vej_id = clickData["points"][0]["customdata"]
    dff_street = df[df["vej_id"] == vej_id].copy()
    if dff_street.empty:
        fig = go.Figure()
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Occupancy (%)",
            yaxis=dict(range=[0, 100]),
            annotations=[
                dict(
                    x=0.5,
                    y=0.5,
                    xref="paper",
                    yref="paper",
                    text=f"No data for vej_id={vej_id}",
                    showarrow=False,
                )
            ],
        )
        return fig

    dff_street["date"] = pd.to_datetime(
        dict(year=dff_street["year"], month=dff_street["month"], day=1)
    )
    dff_street = dff_street.sort_values("date")
    dff_street["event_number"] = dff_street.groupby(["year", "month"]).cumcount() + 1

    symbol_map = {1: "circle", 2: "diamond"}
    dff_street["marker_symbol"] = (
        dff_street["event_number"].map(symbol_map).fillna("circle")
    )

    traces = []
    max_y = 0
    for t in ["12", "17", "22"]:
        occ_col = TIME_CONFIG[t]["occ_col"]
        label = TIME_CONFIG[t]["label"]

        if occ_col in dff_street.columns:
            y = dff_street[occ_col].astype(float)
            traces.append(
                go.Scatter(
                    x=dff_street["date"],
                    y=y,
                    mode="lines+markers",
                    name=label,
                    marker=dict(
                        symbol=dff_street["marker_symbol"],
                        size=8,
                    ),
                    customdata=dff_street["event_number"],
                    hovertemplate=(
                        "%{x|%b %Y}<br>"
                        "Occupancy: %{y:.0f}%<br>"
                        "Measurement: %{customdata}"
                        "<extra></extra>"
                    ),
                )
            )
            max_y = max(max_y, y.max())

    max_y = float(max_y)
    max_y = max(100, max_y)
    max_y = max_y * 1.05

    street_name = (
        dff_street["vejnavn"].iloc[0] if "vejnavn" in dff_street.columns else ""
    )
    title = f"Occupancy over time - {street_name} (vej_id={vej_id})"

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Occupancy (%)",
        yaxis=dict(range=[-10, max_y]),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
    )
    return fig


@app.callback(
    Output("capacity-change-table", "figure"),
    Input("year-dropdown", "value"),
)
def update_capacity_table(selected_year):
    changes = compute_year_changes(selected_year)

    if changes.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No previous-year comparison available for this year.",
            x=0.5,
            y=0.5,
            showarrow=False,
            xref="paper",
            yref="paper",
        )
        fig.update_layout(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            margin=dict(l=0, r=0, t=40, b=0),
        )
        return fig

    formatted_change = changes["capacity_change"].apply(lambda x: f"{int(x):+d}")

    fig = go.Figure(
        data=[
            go.Table(
                header=dict(
                    values=["Street name", "Capacity change vs. previous year"],
                    fill_color="lightgrey",
                    align="left",
                ),
                cells=dict(
                    values=[changes["vejnavn"], formatted_change],
                    align="left",
                ),
            )
        ]
    )
    fig.update_layout(height=600, margin=dict(l=0, r=0, t=30, b=0))
    return fig


if __name__ == "__main__":
    app.run(debug=True)
