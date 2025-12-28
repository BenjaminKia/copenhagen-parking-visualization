from dash import Dash, dcc, html
from dash.dependencies import Input, Output

import pandas as pd
import numpy as np
import plotly.graph_objs as go

# Load and prepare data

df = pd.read_csv("data/cleaned_df.csv")

if "Unnamed: 0" in df.columns:
    df = df.drop(columns=["Unnamed: 0"])

df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
df["lng"] = pd.to_numeric(df["lng"], errors="coerce")
df = df.dropna(subset=["lat", "lng"])

# NOTE: Data Quality Issue
# ========================
# Some vej_id values map to multiple vejnavn (street names) across years.
# Example: vej_id 1013064000000 includes both "Hovmålvej" (139 spots) and "Landehjælpsvej" (9 spots)
# This is likely due to road name changes or data collection methodology changes.
#
# IMPORTANT: All aggregations must group by vej_id ONLY (not by vejnavn) to avoid
# mixing capacity values from different street segments.
#
# Affected aggregations must use: groupby("vej_id") or groupby(["year", "vej_id"])
# NOT: groupby(["vej_id", "vejnavn"]) or groupby(["year", "vej_id", "vejnavn"])

# Dropdown options
years = sorted(df["year"].dropna().unique())
months = sorted(df["month"].dropna().unique())

# map time of day to relevant columns
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

# Get belaegnings columns - to get capacity
capacity_cols = [cfg["cap_col"] for cfg in TIME_CONFIG.values()]

# Take max
df["capacity"] = df[capacity_cols].max(axis=1, skipna=True)

# Drop nas
df_cap = df.dropna(subset=["capacity"])

# Get max capacity by year and street (grouped by vej_id only to avoid vejnavn duplicates)
cap_by_street_year = df_cap.groupby(["year", "vej_id"])["capacity"].max().reset_index()

# Sum across streets
cap_by_year = (
    cap_by_street_year.groupby("year")["capacity"]
    .sum()
    .reset_index()
    .rename(columns={"capacity": "total_spots"})
)

# Make numeric
cap_by_year["year"] = cap_by_year["year"].astype(int)


def make_capacity_figure():
    if cap_by_year.empty:
        return go.Figure()

    fig = go.Figure(
        go.Scatter(
            x=cap_by_year["year"],
            y=cap_by_year["total_spots"],
            mode="lines+markers",
            line=dict(color=COLOR_SCHEME["primary"], width=3),
            marker=dict(
                size=10,
                color=COLOR_SCHEME["primary"],
                line=dict(width=2, color="white"),
            ),
            fill="tozeroy",
            fillcolor="rgba(46, 134, 171, 0.1)",
            hovertemplate="<b>Year %{x}</b><br>Total spots: %{y:,.0f}<extra></extra>",
            name="Total parking spots",
        )
    )

    fig.update_layout(
        title=None,
        xaxis_title="Year",
        yaxis_title="Number of parking spots",
        hovermode="x unified",
        margin=dict(l=50, r=20, t=20, b=50),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color=COLOR_SCHEME["text"]),
        xaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor="rgba(0,0,0,0.05)",
            zeroline=False,
        ),
        yaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor="rgba(0,0,0,0.05)",
            zeroline=False,
        ),
        showlegend=False,
    )
    return fig


# Create Dash app
external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]
app = Dash(__name__, external_stylesheets=external_stylesheets)
app.title = "Copenhagen Parking Occupancy Growth"
server = app.server

# Use your own Mapbox token here
MAPBOX_TOKEN = "pk.eyJ1IjoicGxvdGx5bWFwYm94IiwiYSI6ImNrOWJqb2F4djBnMjEzbG50amg0dnJieG4ifQ.Zme1-Uzoi75IaFbieBDl3A"

# Color scheme - professional and accessible
COLOR_SCHEME = {
    "primary": "#2E86AB",  # Copenhagen blue
    "accent": "#A23B72",  # Accent purple
    "success": "#06A77D",  # Green for gains
    "danger": "#D62246",  # Red for losses
    "neutral": "#F8F9FA",  # Light background
    "text": "#2C3E50",  # Dark text
    "text_light": "#7F8C8D",  # Light text
    "border": "#E1E8ED",  # Subtle borders
}

# 1 app layout
app.layout = html.Div(
    style={
        "fontFamily": "'Segoe UI', Tahoma, Geneva, Verdana, sans-serif",
        "backgroundColor": "#FFFFFF",
        "color": COLOR_SCHEME["text"],
        "margin": "0",
        "padding": "0",
    },
    children=[
        # Header
        html.Div(
            style={
                "backgroundColor": COLOR_SCHEME["primary"],
                "color": "white",
                "padding": "40px 30px",
                "marginBottom": "0",
                "boxShadow": "0 2px 12px rgba(0,0,0,0.08)",
            },
            children=[
                html.H1(
                    "🅿 Copenhagen Parking Occupancy Explorer",
                    style={
                        "margin": "0 0 12px 0",
                        "fontSize": "36px",
                        "fontWeight": "700",
                    },
                ),
                html.P(
                    "Analyze parking availability by time of day, date, and location across Copenhagen",
                    style={
                        "margin": "0",
                        "opacity": "0.95",
                        "fontSize": "16px",
                        "fontWeight": "300",
                    },
                ),
            ],
        ),
        # Filter bar (dark horizontal) - sticky header so controls follow on scroll
        html.Div(
            style={
                "backgroundColor": "#2C3E50",
                "padding": "20px 30px",
                "display": "flex",
                "gap": "30px",
                "alignItems": "center",
                "boxShadow": "0 2px 8px rgba(0,0,0,0.12)",
                "flexWrap": "wrap",
                "position": "sticky",
                "top": "0",
                "zIndex": "1000",
            },
            children=[
                # Year
                html.Div(
                    style={"display": "flex", "gap": "10px", "alignItems": "center"},
                    children=[
                        html.Label(
                            "Year:",
                            style={
                                "color": "white",
                                "fontWeight": "600",
                                "margin": "0",
                                "fontSize": "13px",
                            },
                        ),
                        dcc.Dropdown(
                            id="year-dropdown",
                            options=[{"label": int(y), "value": int(y)} for y in years],
                            # Default to 2013 if present, otherwise fallback to earliest available year
                            value=(2013 if 2013 in years else int(years[0])),
                            clearable=False,
                            style={"width": "100px"},
                        ),
                    ],
                ),
                # Month
                html.Div(
                    style={"display": "flex", "gap": "10px", "alignItems": "center"},
                    children=[
                        html.Label(
                            "Month:",
                            style={
                                "color": "white",
                                "fontWeight": "600",
                                "margin": "0",
                                "fontSize": "13px",
                            },
                        ),
                        dcc.Dropdown(
                            id="month-dropdown",
                            options=[
                                {"label": int(m), "value": int(m)} for m in months
                            ],
                            value=int(months[0]),
                            clearable=False,
                            style={"width": "100px"},
                        ),
                    ],
                ),
                # Time of day
                html.Div(
                    style={"display": "flex", "gap": "10px", "alignItems": "center"},
                    children=[
                        html.Label(
                            "Time:",
                            style={
                                "color": "white",
                                "fontWeight": "600",
                                "margin": "0",
                                "fontSize": "13px",
                            },
                        ),
                        dcc.Dropdown(
                            id="time-dropdown",
                            options=[
                                {"label": cfg["label"], "value": t}
                                for t, cfg in TIME_CONFIG.items()
                            ],
                            value="12",
                            clearable=False,
                            style={
                                "width": "140px",
                                "backgroundColor": "white",
                            },
                        ),
                    ],
                ),
                # Occupancy filter
                html.Div(
                    style={
                        "display": "flex",
                        "gap": "10px",
                        "alignItems": "center",
                        "flex": "1",
                        "minWidth": "300px",
                    },
                    children=[
                        html.Label(
                            "Max Occupancy:",
                            style={
                                "color": "white",
                                "fontWeight": "600",
                                "margin": "0",
                                "fontSize": "13px",
                                "whiteSpace": "nowrap",
                            },
                        ),
                        html.Div(
                            style={"flex": "1"},
                            children=[
                                dcc.Slider(
                                    id="occ-slider",
                                    min=0,
                                    max=100,
                                    step=5,
                                    value=100,
                                    marks={i: f"{i}%" for i in range(0, 101, 20)},
                                    tooltip={
                                        "placement": "bottom",
                                        "always_visible": True,
                                    },
                                ),
                            ],
                        ),
                    ],
                ),
            ],
        ),
        # Summary stats bar
        html.Div(
            id="summary-text",
            style={
                "backgroundColor": "#ECF0F1",
                "padding": "15px 30px",
                "borderBottom": f"1px solid {COLOR_SCHEME['border']}",
                "fontSize": "13px",
                "color": COLOR_SCHEME["text"],
            },
        ),
        # Vertical layout - big map on top, three visuals in a row below; allow scrolling
        html.Div(
            style={
                "maxWidth": "100%",
            },
            children=[
                # Map header
                html.Div(
                    style={
                        "padding": "15px 20px",
                        "borderBottom": f"1px solid {COLOR_SCHEME['border']}",
                        "backgroundColor": "#F8F9FA",
                    },
                    children=[
                        html.H3(
                            "Parking Occupancy Map",
                            style={
                                "color": COLOR_SCHEME["primary"],
                                "margin": "0 0 5px 0",
                                "fontSize": "16px",
                                "fontWeight": "600",
                            },
                        ),
                        html.P(
                            "Click on a street to see its history",
                            style={
                                "fontSize": "13px",
                                "color": COLOR_SCHEME["text_light"],
                                "margin": "0",
                            },
                        ),
                    ],
                ),
                # Large map (fills viewport height to start)
                dcc.Graph(
                    id="map-graph",
                    style={"height": "70vh", "width": "100%", "margin": "0"},
                    config={
                        "displaylogo": False,
                        "modeBarButtonsToRemove": ["lasso2d", "select2d"],
                    },
                ),
                # Bottom grid: 2x2 for charts (fourth empty placeholder)
                html.Div(
                    style={
                        "display": "grid",
                        "gridTemplateColumns": "1fr 1fr",
                        "gridAutoRows": "minmax(300px, auto)",
                        "gap": "0",
                        "paddingTop": "20px",
                    },
                    children=[
                        # Cell 1: Street timeseries
                        html.Div(
                            style={
                                "borderRight": f"1px solid {COLOR_SCHEME['border']}",
                                "borderBottom": f"1px solid {COLOR_SCHEME['border']}",
                                "overflow": "hidden",
                                "display": "flex",
                                "flexDirection": "column",
                            },
                            children=[
                                html.Div(
                                    style={
                                        "padding": "15px 20px",
                                        "backgroundColor": "#F8F9FA",
                                    },
                                    children=[
                                        html.H3(
                                            "Street Occupancy Trend",
                                            style={
                                                "color": COLOR_SCHEME["primary"],
                                                "margin": "0 0 5px 0",
                                                "fontSize": "14px",
                                                "fontWeight": "600",
                                            },
                                        )
                                    ],
                                ),
                                dcc.Graph(
                                    id="street-timeseries",
                                    style={"height": "45vh", "margin": "0"},
                                    config={"displaylogo": False},
                                ),
                            ],
                        ),
                        # Cell 2: Total capacity trend
                        html.Div(
                            style={
                                "borderBottom": f"1px solid {COLOR_SCHEME['border']}",
                                "overflow": "hidden",
                                "display": "flex",
                                "flexDirection": "column",
                            },
                            children=[
                                html.Div(
                                    style={
                                        "padding": "15px 20px",
                                        "backgroundColor": "#F8F9FA",
                                    },
                                    children=[
                                        html.H3(
                                            "Total Spots",
                                            style={
                                                "color": COLOR_SCHEME["primary"],
                                                "margin": "0 0 5px 0",
                                                "fontSize": "14px",
                                                "fontWeight": "600",
                                            },
                                        )
                                    ],
                                ),
                                dcc.Graph(
                                    id="capacity-trend",
                                    figure=make_capacity_figure(),
                                    style={"height": "45vh", "margin": "0"},
                                    config={"displaylogo": False},
                                ),
                            ],
                        ),
                        # Cell 3: Breakdown
                        html.Div(
                            style={
                                "borderRight": f"1px solid {COLOR_SCHEME['border']}",
                                "overflow": "hidden",
                                "display": "flex",
                                "flexDirection": "column",
                            },
                            children=[
                                html.Div(
                                    style={
                                        "padding": "15px 20px",
                                        "backgroundColor": "#F8F9FA",
                                    },
                                    children=[
                                        html.H3(
                                            "Street Capacity Changes",
                                            id="breakdown-title",
                                            style={
                                                "color": COLOR_SCHEME["primary"],
                                                "margin": "0 0 5px 0",
                                                "fontSize": "14px",
                                                "fontWeight": "600",
                                            },
                                        )
                                    ],
                                ),
                                dcc.Graph(
                                    id="capacity-breakdown-chart",
                                    style={"height": "45vh", "margin": "0"},
                                    config={"displaylogo": False},
                                ),
                            ],
                        ),
                        # Cell 4: Empty placeholder
                        html.Div(
                            style={
                                "overflow": "hidden",
                                "display": "flex",
                                "alignItems": "center",
                                "justifyContent": "center",
                            },
                            children=[
                                html.Div(
                                    "(Reserved for future visualization)",
                                    style={
                                        "color": COLOR_SCHEME["text_light"],
                                        "fontSize": "14px",
                                    },
                                )
                            ],
                        ),
                    ],
                ),
            ],
        ),
    ],
)


def get_year_breakdown(selected_year):
    """Get capacity change per street (vej_id) between consecutive years.

    Aggregates capacity at the yearly level (ignores month and time selections).
    Compares selected_year vs previous_year capacity by vej_id.
    """
    years_sorted = sorted(df["year"].dropna().unique())

    if selected_year == years_sorted[0] or len(years_sorted) < 2:
        return None

    prev_year = years_sorted[years_sorted.index(selected_year) - 1]

    # cap_by_street_year already contains max capacity per (year, vej_id)
    curr = (
        cap_by_street_year[cap_by_street_year["year"] == selected_year]
        .copy()
        .rename(columns={"capacity": "capacity_curr"})
    )

    prev = (
        cap_by_street_year[cap_by_street_year["year"] == prev_year]
        .copy()
        .rename(columns={"capacity": "capacity_prev"})
    )

    merged = curr.merge(prev, on="vej_id", how="inner")

    # Attach a representative street name
    vejnames = df.groupby("vej_id")["vejnavn"].first().reset_index()
    merged = merged.merge(vejnames, on="vej_id", how="left")

    merged["change"] = merged["capacity_curr"] - merged["capacity_prev"]
    merged = merged[["vejnavn", "capacity_prev", "capacity_curr", "change"]]

    return prev_year, merged


# callbacks
@app.callback(
    [Output("map-graph", "figure"), Output("summary-text", "children")],
    [
        Input("time-dropdown", "value"),
        Input("year-dropdown", "value"),
        Input("month-dropdown", "value"),
        Input("occ-slider", "value"),
    ],
)

# 4 update-functions

def update_map(selected_time, selected_year, selected_month, max_occ):
    cfg = TIME_CONFIG[selected_time]
    occ_col = cfg["occ_col"]
    cap_col = cfg["cap_col"]

    # Filter by year + month
    dff = df.copy()
    if selected_year is not None:
        dff = dff[dff["year"] == selected_year]
    if selected_month is not None:
        dff = dff[dff["month"] == selected_month]

    # Keep only rows that have data for this time of day
    dff = dff.dropna(subset=[occ_col, cap_col])
    if dff.empty:
        # Return an empty figure with a message
        fig = go.Figure()
        fig.update_layout(
            title="No data for selected filters",
            mapbox=dict(
                accesstoken=MAPBOX_TOKEN,
                style="carto-positron",
                zoom=10,
                center=dict(lat=55.6761, lon=12.5683),  # Copenhagen
            ),
        )
        return fig, "No data available for these filters."

    # Occupancy clipped to [0, 100] for color scale
    occ = dff[occ_col].astype(float).clip(lower=0, upper=100)
    capacity = dff[cap_col].astype(float)

    # Apply max occupancy filter
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

    # Scale capacities to reasonable marker sizes (e.g. 5–30)
    min_cap = capacity.min()
    max_cap = capacity.max()
    if max_cap == min_cap:
        sizes = np.full_like(capacity, 10)  # fallback
    else:
        sizes = 5 + 25 * (capacity - min_cap) / (max_cap - min_cap)

    # Build hover text
    hover_text = (
        "Street: "
        + dff["vejnavn"].astype(str)
        # + "<br>Street ID: "
        # + dff["vej_id"].astype(str)
        + f"<br>Time: {cfg['label']}"
        + "<br>Occupancy: "
        + occ.round(1).astype(str)
        + " %"
        + "<br>Capacity: "
        + capacity.astype(int).astype(str)
        + " spots"
    )
    # 2: fig
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
                colorscale=[
                    [0, "#2ecc71"],
                    [0.5, "#f1c40f"],
                    [1, "#e74c3c"],
                ],
                colorbar=dict(title="Occupancy (%)", ticks="outside"),
                opacity=0.8,
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
            center=dict(lat=55.69, lon=12.5683),  # Copenhagen center
        ),
        uirevision="parking-map",
    )

    summary = (
        f"{len(dff)} observations | "
        f"Time: {cfg['label']} | "
        f"Year: {selected_year} | Month: {selected_month}"
    )

    return fig, summary


@app.callback(
    Output("street-timeseries", "figure"),
    Input("map-graph", "clickData"),
)
def update_street_timeseries(clickData):
    # If nothing clicked yet
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

    # Get vej_id from the clicked point
    vej_id = clickData["points"][0]["customdata"]

    # Filter all data for this street
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

    # Aggregate measurements within the same month: take the max occupancy
    occ_cols = [cfg["occ_col"] for cfg in TIME_CONFIG.values()]
    cap_cols = [cfg["cap_col"] for cfg in TIME_CONFIG.values()]

    # For each (year, month) keep the max of occupancy/capacity and first of metadata
    agg_dict = {col: "max" for col in occ_cols + cap_cols}
    agg_dict.update(
        {"vejnavn": "first", "vej_id": "first", "lat": "first", "lng": "first"}
    )

    dff_street = (
        dff_street.groupby(["year", "month"], as_index=False)
        .agg(agg_dict)
        .sort_values(["year", "month"])
    )

    # Create date from year + month
    dff_street["date"] = pd.to_datetime(
        dict(year=dff_street["year"], month=dff_street["month"], day=1)
    )

    # build time series (12, 17, 22)
    traces = []
    max_y = 0  # to scale y-axis depending on occupancy of that street
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
                    marker=dict(size=8),
                    hovertemplate=("%{x|%b %Y}<br>Occupancy: %{y:.0f}%<extra></extra>"),
                )
            )
            max_y = max(max_y, y.max())

    max_y = float(max_y)
    # Allow the timeseries to exceed 100% (do not force a 100% lower cap)
    # Keep a small floor so very-small values still render nicely and
    # add visual padding.
    max_y = max(max_y, 10)
    max_y = max_y * 1.05  # visual padding

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
    [
        Output("breakdown-title", "children"),
        Output("breakdown-title", "style"),
        Output("capacity-breakdown-chart", "figure"),
    ],
    [
        Input("capacity-trend", "clickData"),
        Input("year-dropdown", "value"),
    ],
)
def update_breakdown(clickData, selected_year):
    # Determine which year to use: clicked year from graph or selected year from dropdown
    if clickData is not None:
        try:
            clicked_year = int(clickData["points"][0]["x"])
        except (KeyError, ValueError, IndexError, TypeError):
            clicked_year = selected_year
    else:
        clicked_year = selected_year

    breakdown_result = get_year_breakdown(clicked_year)
    if breakdown_result is None:
        empty_fig = go.Figure()
        # If the selected year is 2012, show a clear message in the chart
        if clicked_year == 2012:
            empty_fig.update_layout(
                xaxis_title="Change in parking spots",
                yaxis_title="",
                annotations=[
                    dict(
                        x=0.5,
                        y=0.5,
                        xref="paper",
                        yref="paper",
                        text="No data before 2012 to compare with",
                        showarrow=False,
                        font=dict(size=14, color=COLOR_SCHEME["text_light"]),
                    )
                ],
                margin=dict(l=200, r=20, t=20, b=40),
            )
            title = f"Street capacity changes: {clicked_year}"
            # Match H3 title style used in layout
            h3_style = {
                "display": "block",
                "color": COLOR_SCHEME["primary"],
                "fontSize": "14px",
                "fontWeight": "600",
            }
            return title, h3_style, empty_fig

        return "", {"display": "none"}, empty_fig

    prev_year, merged = breakdown_result

    # Get top 10 losers and gainers
    losers = merged[merged["change"] < 0].nsmallest(10, "change")
    gainers = merged[merged["change"] > 0].nlargest(10, "change")

    # Combine and sort
    combined = pd.concat([losers, gainers]).sort_values("change")

    if combined.empty:
        empty_fig = go.Figure()
        return "", {"display": "none"}, empty_fig

    # Create horizontal bar chart
    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            y=combined["vejnavn"],
            x=combined["change"],
            orientation="h",
            marker=dict(
                color=combined["change"],
                colorscale="RdYlGn",
                cmid=0,
                showscale=False,
            ),
            customdata=combined[["capacity_prev", "capacity_curr"]],
            hovertemplate="<b>%{y}</b><br>Change: %{x:+.0f} spots<br>"
            + "Previous: %{customdata[0]:.0f} → Current: %{customdata[1]:.0f}<extra></extra>",
        )
    )

    fig.update_layout(
        xaxis_title="Change in parking spots",
        yaxis_title="",
        margin=dict(l=200, r=20, t=20, b=40),
        hovermode="closest",
    )

    title = f"Street capacity changes: {prev_year} → {clicked_year}"
    h3_style = {
        "display": "block",
        "color": COLOR_SCHEME["primary"],
        "fontSize": "14px",
        "fontWeight": "600",
    }
    return title, h3_style, fig


if __name__ == "__main__":
    app.run(debug=True)
