import sys
import numpy as np
from dash import Dash, html, dcc, callback, Output, Input, State
from dash.exceptions import PreventUpdate
import kmeans

# import correlate
import read as file_read
import plotting
import utils
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
from time import time
import csv
import os


# sys.settrace("call")

# import argparse

# parser = argparse.ArgumentParser()
# parser.add_argument("--metal_id", dest="metal_type", type=int, help="define metal ID")
# args = parser.parse_args()

# if args.metal_type is None:
metal_type = 1
port = 8080
# metal_type = 1
#
# else:
# metal_type = args.metal_type

# env = os.environ
# for key, value in env.items():
#     print(key, value)
atomic_masses = {
    1: 28.085,
    2: 28.085,
    3: 15.999,
    4: 1.008,
    5: 12.011,
    6: 1.008,
    7: 26.9815,
}
atoms = {1: "Si", 2: "Si", 3: "O", 4: "H", 5: "C", 6: "H", 7: "Al"}
Ntypes = 1  # number of molecule types
reference_molecule = 1  # a ref molecule to obtain Natom_per_molecule

app = Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])


non_data_vars = [
    "molecule_id",
    "id",
    "timestep",
    "id",
    "type",
    "phase",
    "mask",
    "lower_mask",
]  # remove from analysis
pos_vars = [
    "q",
    "lt",
    "ld",
    "speed",
    "ke",
    "mass",
]  # exclude pos version from dropdown

higher_vars = [
    "|z|",
    "|vz|",
    "ke",
    "lt",
    "speed",
    "dz",
    "displacement",
    "pe",
]  # highter indicates gas
lower_vars = [
    "ld",
    "coordination",
]  # lower indicates gas


df = {}
text_color = "white"

file_in = html.Div(
    [
        dbc.Label("Data File:", style={"display": "inline-block", "color": text_color}),
        dcc.Input(
            value="",
            id="file-in",
            type="text",
            placeholder="Enter File Path...",
            debounce=True,
            style={
                "background-color": "rgb(25, 25, 25)",
                "color": "rgb(222, 226, 230, 1)",
                "margin-right": "5px",
                "margin-left": "10px",
                "width": "20%",
                "display": "inline-block",
            },
        ),
        dbc.Label("Nt:", style={"display": "inline-block", "color": text_color}),
        dcc.Input(
            value="",
            id="Nt-lim",
            type="number",
            min=0,
            step=10,
            placeholder="Limit",
            debounce=False,
            style={
                "background-color": "rgb(25, 25, 25)",
                "color": "rgb(222, 226, 230, 1)",
                "margin-right": "10px",
                "margin-left": "5px",
                "width": "65px",
                "display": "inline-block",
            },
        ),
        html.Div(
            html.Button(
                children="Read",
                id="read-data",
                n_clicks=0,
                className="write-data",
                style={"height": "100%", "textAlign": "center"},
            ),
            style={
                "height": "34px",
                "display": "inline-block",
                "marginRight": "10px",
            },
        ),
        dbc.Label("Mode:", style={"display": "inline-block", "color": text_color}),
        dcc.RadioItems(
            options=["View", "Train", "Classify"],
            value="View",
            id="mode",
            className="radio-group",
            inline=True,
            style={
                "margin-left": "10px",
                "margin-right": "0px",
                "display": "inline-block",
                "color": text_color,
            },
            labelStyle={"margin-right": "10px"},
            inputStyle={"margin-right": "5px"},
        ),
        dcc.Checklist(
            options=[
                {"value": "Use External Centroids", "label": "Use External Centroids"}
            ],
            value=[],
            id="use-external",
            style={
                "margin-left": "3px",
                "display": "inline-block",
                "margin-right": "0px",
                "font-size": "8",
                "color": text_color,
            },
            inputStyle={"margin-right": "5px"},
        ),
        dcc.Input(
            value="",
            id="centroids-path",
            type="text",
            placeholder="Enter Centroid Path...",
            debounce=True,
            style={
                "background-color": "rgb(25, 25, 25)",
                "color": "rgb(222, 226, 230, 1)",
                "margin-right": "10px",
                "margin-left": "10px",
                "width": "20%",
                "display": "inline-block",
            },
        ),
        html.Div(
            html.Button(
                children="Read Centroids",
                id="read-centroids",
                n_clicks=0,
                className="write-data",
                style={"height": "100%", "textAlign": "center"},
            ),
            style={
                "height": "34px",
                "display": "inline-block",
                "marginRight": "5px",
            },
        ),
        html.Div(
            html.Button(
                "Write Centroids",
                id="write-centroids",
                n_clicks=0,
                style={
                    "height": "100%",
                    "textAlign": "center",
                },
                className="write-data",
            ),
            style={
                "height": "34px",
                "display": "inline-block",
                "margin-left": "5px",
            },
        ),
        html.Div(
            html.Button(
                "Write to Dump",
                id="write-dump",
                n_clicks=0,
                style={
                    "height": "100%",
                    "textAlign": "center",
                },
                className="dump",
            ),
            style={
                "height": "34px",
                "display": "inline-block",
                "margin-left": "20px",
            },
        ),
    ],
    style={
        "margin-top": "10px",
        "margin-right": "0px",
        "margin-left": "0px",
        "width": "100%",
    },
)

distribution_options = html.Div(
    [
        dbc.Label(
            "Distribution Mode:",
            style={"display": "inline-block", "color": text_color},
        ),
        dcc.RadioItems(
            ["Atom", "Molecule"],
            value="Atom",
            id="distribution-mode",
            inline=True,
            style={
                "margin-left": "10px",
                "margin-right": "10px",
                "display": "inline-block",
                "color": text_color,
            },
            labelStyle={"margin-right": "5px"},
            inputStyle={"margin-right": "5px"},
        ),
        html.Div(
            dcc.Dropdown(
                [],
                None,
                id="distribution-var",
                placeholder="Select Var",
                style={"width": "100%", "height": "100%", "color": text_color},
            ),
            style={
                "width": "10%",
                "display": "inline-block",
                "vertical-align": "middle",
                "height": "34px",
                "min-width": "100px",
            },
        ),
        dcc.Checklist(
            options=[{"value": "Abs", "label": "Abs"}],
            value=[],
            id="abs",
            style={
                "margin-left": "10px",
                "display": "inline-block",
                "margin-right": "0px",
                "color": text_color,
            },
            inputStyle={"margin-right": "5px"},
        ),
        dcc.Checklist(
            options=[{"value": "Normalize", "label": "Normalize"}],
            value=[],
            id="normalize-dist",
            style={
                "margin-left": "10px",
                "display": "inline-block",
                "margin-right": "0px",
                "color": text_color,
            },
            inputStyle={"margin-right": "5px"},
        ),
        dcc.Checklist(
            options=[{"value": "Log", "label": "Log"}],
            value=[],
            id="log",
            style={
                "margin-left": "10px",
                "display": "inline-block",
                "margin-right": "0px",
                "color": text_color,
            },
            inputStyle={"margin-right": "5px"},
        ),
        dbc.Label(
            "Color By:",
            style={
                "display": "inline-block",
                "margin-left": "10px",
                "margin-right": "3px",
                "color": text_color,
            },
        ),
        html.Div(
            dcc.Dropdown(
                options=[
                    {"value": "Phase", "label": "Phase"},
                    {"value": "Atom Type", "label": "Atom Type"},
                ],
                value="",
                id="color-by-distribution",
                placeholder="Select",
                style={"width": "100%", "height": "100%", "color": text_color},
            ),
            style={
                "display": "inline-block",
                "width": "15%",
                "margin-right": "0px",
                "margin-left": "10px",
                "vertical-align": "middle",
                "height": "34px",
                "min-width": "120px",
            },
        ),
    ],
    style={
        "width": "100%",
        "margin-right": "0px",
        "margin-left": "0px",
        "margin-top": "0px",
        "margin-bottom": "0px",
        "vertical-align": "middle",
    },
)

distribution = html.Div(
    [
        dcc.Graph(
            id="distribution-graph",
            style={
                "width": "100%",
                "height": "85vh",
            },
            mathjax=True,
        ),
        html.Div(
            html.Div(
                [html.Button("Save", id="save")],
                style={
                    "width": "100%",
                    "display": "flex",
                    "align-items": "center",
                    "justify-content": "center",
                },
            ),
            style={"width": "100%", "marginTop": "20px"},
        ),
    ],
    style={
        "width": "100%",
        "margin-right": "0px",
        "margin-left": "0px",
        "margin-top": "0px",
        "margin-bottom": "0px",
    },
)

cluster_options = html.Div(
    [
        dbc.Label(
            "View Mode:",
            style={
                "display": "inline-block",
                "margin-left": "0px",
                "color": text_color,
            },
        ),
        dcc.RadioItems(
            ["Atom", "Molecule"],
            value="Atom",
            id="scatter-mode",
            className="radio-group",
            inline=True,
            style={
                "margin-right": "0px",
                "display": "inline-block",
                "padding-left": "5px",
                "color": text_color,
            },
            labelStyle={"margin-right": "5px"},
            inputStyle={"margin-right": "5px"},
        ),
        dcc.Checklist(
            options=[{"value": "Show Traces", "label": "Show Traces"}],
            value=[],
            id="trace",
            style={
                "margin-left": "5px",
                "display": "inline-block",
                "margin-right": "5px",
                "color": text_color,
            },
            inputStyle={"margin-right": "5px"},
        ),
        dbc.Label(
            "Color By:",
            style={
                "display": "inline-block",
                "margin-left": "5px",
                "color": text_color,
            },
        ),
        html.Div(
            dcc.Dropdown(
                options=[
                    {"value": "phase", "label": "Phase"},
                    {"value": "molecule_id", "label": "Molecule ID"},
                    {"value": "type", "label": "Atom Type"},
                ],
                value="type",
                id="color-by-scatter",
                placeholder="Select",
                style={"width": "100%", "height": "100%", "color": text_color},
            ),
            style={
                "display": "inline-block",
                "width": "15%",
                "margin-right": "0px",
                "margin-left": "10px",
                "vertical-align": "middle",
                "height": "34px",
                "min-width": "120px",
            },
        ),
        html.Div(
            dcc.Dropdown(
                options=[],
                value=[],
                id="cluster-vars",
                multi=True,
                placeholder="Select Cluster Variables",
                style={"width": "100%", "height": "100%", "color": text_color},
            ),
            style={
                "display": "inline-block",
                "width": "30%",
                "margin-right": "10px",
                "margin-left": "10px",
                "vertical-align": "middle",
                "height": "34px",
                "min-width": "150px",
            },
        ),
    ],
    style={
        "width": "100%",
        "margin-top": "0px",
        "margin-bottom": "0px",
        "margin-right": "0px",
        "margin-left": "0px",
        "vertical-align": "middle",
    },
)

scatter = html.Div(
    [
        dcc.Graph(
            id="scatter-graph",
            style={
                "width": "100%",
                "height": "85vh",
                "marginLeft": "0px",
                "marginRight": "0px",
            },
        ),
        html.Div(
            [
                html.Div(
                    [
                        html.Button("←", id="back"),
                        html.Button("→ ", id="forward"),
                    ],
                ),
                html.Div(
                    dcc.Slider(
                        min=0,
                        max=1,
                        step=0.01,
                        value=0,
                        id="timestep",
                        marks=None,
                    ),
                    style={"width": "90%"},
                ),
            ],
            style={
                "display": "flex",
            },
        ),
        html.Div(
            html.Div(
                [html.Button("Save", id="save")],
                style={
                    "width": "100%",
                    "display": "flex",
                    "align-items": "center",
                    "justify-content": "center",
                },
            ),
            style={"width": "100%", "marginTop": "-10px"},
        ),
    ],
    style={
        "width": "100%",
        "margin-left": "0px",
        "margin-right": "0px",
    },
)


app.layout = html.Div(
    [
        dbc.Row(
            dbc.Col(
                file_in,
                width=12,
                className="g-0",
            ),
            className="g-0",
        ),
        dbc.Row(
            [
                dbc.Col(
                    distribution_options,
                    width=6,
                    className="g-0",
                    style={
                        "margin-top": "8px",
                        "margin-bottom": "4px",
                        "padding-right": "5px",
                    },
                ),
                dbc.Col(
                    cluster_options,
                    width=6,
                    className="g-0",
                    style={
                        "margin-top": "8px",
                        "margin-bottom": "4px",
                        "padding-left": "5px",
                    },
                ),
            ],
            className="g-0",
        ),
        dbc.Row(
            [
                dbc.Col(
                    dbc.Row(distribution, className="g-0"),
                    width=6,
                    className="g-0",
                    style={
                        "padding-right": "5px",
                        "margin-right": "0px",
                        "margin-left": "0px",
                    },
                ),
                dbc.Col(
                    dbc.Row(scatter, className="g-0"),
                    width=6,
                    className="g-0",
                    style={
                        "padding-left": "5px",
                        "margin-right": "0px",
                        "margin-left": "0px",
                    },
                ),
            ],
            className="g-0",
        ),
    ],
)


@callback(
    Output("cluster-vars", "options"),
    Output("distribution-var", "options"),
    Output("timestep", "min"),
    Output("timestep", "max"),
    Output("timestep", "step"),
    Output("timestep", "marks"),
    # Output("scatter-graph", "figure", allow_duplicate=True),
    # Output("distribution-graph", "figure", allow_duplicate=True),
    Output("mode", "value"),  # start reset
    Output("scatter-mode", "value"),
    Output("color-by-scatter", "value", allow_duplicate=True),
    Output("color-by-scatter", "options", allow_duplicate=True),
    Output("distribution-var", "value"),
    Output("abs", "value"),
    Output("normalize-dist", "value"),
    Output("log", "value", allow_duplicate=True),
    Output("timestep", "value"),
    Output("distribution-mode", "value"),  # end reset
    Output("Nt-lim", "value"),
    Input("read-data", "n_clicks"),
    State("file-in", "value"),
    State("mode", "value"),
    State("scatter-mode", "value"),
    State("color-by-scatter", "value"),
    State("color-by-scatter", "options"),
    State("distribution-var", "value"),
    State("abs", "value"),
    State("normalize-dist", "value"),
    State("log", "value"),
    State("log", "options"),
    State("color-by-distribution", "value"),
    State("color-by-distribution", "options"),
    State("timestep", "value"),
    State("distribution-mode", "value"),
    State("Nt-lim", "value"),
    prevent_initial_call=True,
)
def update_data(
    read,
    file_in,
    mode,
    scatter_mode,
    color_by_scatter,
    color_by_scatter_options,
    distribution_var,
    abs,
    normalize,
    log,
    log_options,
    color_by_distribution,
    color_by_distribution_options,
    t,
    distribution_mode,
    Nt_lim,
):
    if read:
        file_in = file_in.strip()
        file_in = file_in.strip('"')
        df_ = file_read.process_data(
            file_in, atomic_masses, Nt_lim=Nt_lim, mode="mmap", metal_type=metal_type
        )
        global df
        df = df_
        df["atom"]
        df_molecule = df["molecule"]
        Nt = df["Nt"]
        vars = list(df_molecule.keys())
        cluster_vars = []
        distribution_vars = []
        for var in list(vars):
            if var not in non_data_vars:
                distribution_vars.append(var.split("_")[-1])
                if var == "z":
                    distribution_vars.append("|" + var + "|")

            if var not in pos_vars + non_data_vars:
                cluster_vars.append(var)
                cluster_vars.append("|" + var + "|")

            elif var not in non_data_vars:
                cluster_vars.append(var)

        cluster_vars = np.array(cluster_vars)
        distribution_vars = np.array(distribution_vars)

        df["cluster_vars"] = cluster_vars
        df["distribution_vars"] = distribution_vars
        df["external_centroids"] = None

        print("\nDone\n")

        timesteps = df["timesteps"]
        marks, step, t_min, t_max = plotting.get_marks(timesteps)
        color_by_options = [{"value": var, "label": var} for var in distribution_vars]

        return (
            cluster_vars,
            distribution_vars,
            t_min,
            t_max,
            step,
            marks,
            mode,
            scatter_mode,
            color_by_scatter,
            color_by_options,
            distribution_var,
            [] if abs is None else abs,
            [] if normalize is None else normalize,
            [] if log is None else log,
            t,
            distribution_mode,
            Nt,
        )

    raise PreventUpdate


@callback(
    Output("use-external", "options"),
    Output("use-external", "labelStyle"),
    Output("write-centroids", "disabled"),
    Output("read-centroids", "disabled"),
    Output("centroids-path", "disabled"),
    Output("centroids-path", "style"),
    Output("cluster-vars", "disabled"),
    Output("cluster-vars", "style"),
    Output("write-dump", "disabled"),
    Input("mode", "value"),
    Input("use-external", "value"),
    State("use-external", "options"),
    State("centroids-path", "style"),
    State("cluster-vars", "style"),
)
def update_mode(
    mode,
    use_external_value,
    use_external_options,
    centroids_path_style,
    cluster_vars_style,
):
    # print("update mode")

    use_external_label_style = {}
    use_external_options[0]["disabled"] = False

    write_centroids_disable = False
    read_centroids_disable = False
    centroids_path_disable = False
    cluster_vars_disabled = False
    write_dump_disable = False

    if mode.lower() == "view":
        use_external_options[0]["disabled"] = True
        use_external_label_style["opacity"] = 0.6
        use_external_label_style["cursor"] = "not-allowed"
        read_centroids_disable = True
        centroids_path_disable = True
        centroids_path_style["opacity"] = 0.6
        centroids_path_style["cursor"] = "not-allowed"
        cluster_vars_disabled = True
        cluster_vars_style["opacity"] = 0.6
        cluster_vars_style["cursor"] = "not-allowed"
        write_centroids_disable = True
        write_dump_disable = True

    elif mode.lower() == "train":
        use_external_label_style["opacity"] = 0.6
        use_external_label_style["cursor"] = "not-allowed"
        use_external_options[0]["disabled"] = True
        centroids_path_style["opacity"] = 1
        centroids_path_style["cursor"] = "text"
        cluster_vars_style["opacity"] = 1
        cluster_vars_style["cursor"] = "default"
        read_centroids_disable = True

    elif mode.lower() == "classify":
        write_centroids_disable = True
        use_external_options[0]["disabled"] = False
        use_external_label_style["opacity"] = 1
        use_external_label_style["cursor"] = "default"
        centroids_path_style["opacity"] = 1
        centroids_path_style["cursor"] = "text"

        if use_external_value:
            cluster_vars_disabled = True
            cluster_vars_style["opacity"] = 0.6
            cluster_vars_style["cursor"] = "not-allowed"

        else:
            read_centroids_disable = True
            cluster_vars_style["opacity"] = 1
            cluster_vars_style["cursor"] = "default"

    return (
        use_external_options,
        use_external_label_style,
        write_centroids_disable,
        read_centroids_disable,
        centroids_path_disable,
        centroids_path_style,
        cluster_vars_disabled,
        cluster_vars_style,
        write_dump_disable,
    )


@callback(
    Output("cluster-vars", "value"),
    Input("read-centroids", "n_clicks"),
    State("centroids-path", "value"),
)
def read_centroids(
    read,
    path,
):
    global df

    path = path.strip()
    path = path.strip('"')
    # vars, centroids = read_centroids(path)
    # df['external_centroids'] = centroids

    if path:
        print("\nReading Centroids...\n")

        with open(path, mode="r") as f:
            reader = csv.reader(f)
            cluster_vars = next(reader)
            centroids = next(reader)

        df["external_centroids"] = np.array(centroids, dtype=np.float64)

        print("\nDone\n")

        return cluster_vars

    else:
        raise PreventUpdate


@callback(
    Input("write-centroids", "n_clicks"),
    State("centroids-path", "value"),
    State("cluster-vars", "value"),
)
def write_centroids(write, path, cluster_vars):
    global df
    # df['centroids'] = ...
    # write_centroids(df['centroids'], path, cluster_vars)

    if path and cluster_vars:
        print("\nWriting Centroids...\n")

        with open(path, mode="w+") as f:
            writer = csv.writer(f)
            writer.writerow(cluster_vars)

            _, _, centroids = kmeans.classify_phase(df, cluster_vars, k=2)

            writer.writerow(centroids)

        print("\nDone\n")

        return None

    else:
        raise PreventUpdate


@callback(Input("write-dump", "n_clicks"))
def write_dump(write):
    pass


@callback(
    Output("scatter-graph", "figure"),
    Output("color-by-scatter", "value"),
    Output("color-by-scatter", "options"),
    Input("cluster-vars", "value"),
    Input("timestep", "value"),
    Input("mode", "value"),
    Input("scatter-mode", "value"),
    Input("color-by-scatter", "value"),
    Input("color-by-scatter", "options"),
    Input("use-external", "value"),
    State("timestep", "min"),
)
def update_scatter(
    cluster_vars, t, mode, scatter_mode, color_by, color_by_options, use_external, t0
):
    global df
    # color_by_options[0]["disabled"] = False

    color = None

    if not df:
        fig = go.Figure()
        fig.update_layout(
            margin=dict(l=20, r=20, t=20, b=20),
            uirevision=True,
            template="plotly_dark",
        )
        return fig, color_by, color_by_options

    if not cluster_vars:
        # color_by_options[0]["disabled"] = True

        if color_by == "phase":
            color_by = ""

    data = []

    t = int((t - t0) / df["dt"])
    bounds = df["bounds"]
    df_atom = df["atom"]
    df_molecule = df["molecule"]
    scatter_mode = scatter_mode.lower()
    # print(list(df.keys()))
    data = df[scatter_mode]

    if scatter_mode == "molecule":
        color_by_options[2]["disabled"] = True

        if color_by == "Atom Type":
            color_by = ""

    x = data["x"][t]
    y = data["y"][t]
    z = data["z"][t]

    if color_by == "Atom Type":
        color_by = "type"
        color = data[color_by][t]

    elif color_by == "Molecule ID":
        if scatter_mode.lower() == "atom":
            color_by = "molecule_id"
            color = data[color_by][t]
        else:
            color_by = "id"
            color = data[color_by][t]

    elif color_by == "Phase":
        if cluster_vars:
            if use_external and df["external_centroids"] is not None:
                molecule_phase, atom_phase, centroids = kmeans.classify_phase(
                    df,
                    cluster_vars,
                    t=t,
                    k=2,
                    external_centroids=df["external_centroids"],
                )

            else:
                molecule_phase, atom_phase, centroids = kmeans.classify_phase(
                    df, cluster_vars, t=t, k=2
                )

            df_molecule["phase_t"] = molecule_phase
            df_atom["phase_t"] = atom_phase[df["non_metal"]]

            if scatter_mode == "atom":
                color = atom_phase

            else:
                color = molecule_phase

        else:
            color = data["phase"][t]
            # color determined by dump

    marker_dict = {"size": 7}
    if color is not None:
        marker_dict["color"] = color

    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=x,
                y=y,
                z=z,
                mode="markers",
                marker=marker_dict,
            )
        ],
    )
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        uirevision=True,
        scene=dict(
            xaxis=dict(range=[bounds[0, 0], bounds[0, 1]]),
            yaxis=dict(range=[bounds[1, 0], bounds[1, 1]]),
            zaxis=dict(range=[bounds[2, 0], bounds[2, 1]]),
        ),
        template="plotly_dark",
    )

    # @callback(
    #     Output("graph", "figure", allow_duplicate=True),
    #     Input("timestep", "value"),
    #     prevent_initial_call="initial_duplicate",
    # )
    # def update_slider(t):
    #     title = plotting.get_title(df, var, t)

    #     fig = scatter_var(
    #         df,
    #         var,
    #         t,
    #         mode=mode,
    #         title=title,
    #         traces=traces,
    #         ntraces=ntraces,
    #         reversescale=reversescale,
    #         metal=metal,
    #         metal_dynamics=metal_dynamics,
    #         top=top,
    #         bottom=bottom,
    #         step=True,
    #     )

    #     return fig

    # @callback(
    #     Output("graph", "figure", allow_duplicate=True),
    #     Output("timestep", "value", allow_duplicate=True),
    #     Input("forward", "n_clicks"),
    #     State("timestep", "value"),
    #     prevent_initial_call=True,
    # )
    # def step_forward(forward, t):
    #     if forward:
    #         t = min(t + 1, Nt - 1)

    #     title = get_title(df, var, t)

    #     fig = scatter_var(
    #         df,
    #         var,
    #         t,
    #         mode=mode,
    #         title=title,
    #         traces=traces,
    #         ntraces=ntraces,
    #         reversescale=reversescale,
    #         metal=metal,
    #         metal_dynamics=metal_dynamics,
    #         top=top,
    #         bottom=bottom,
    #         step=True,
    #     )

    #     return fig, t

    # @callback(
    #     Output("graph", "figure", allow_duplicate=True),
    #     Output("timestep", "value", allow_duplicate=True),
    #     Input("back", "n_clicks"),
    #     State("timestep", "value"),
    #     prevent_initial_call=True,
    # )
    # def step_backward(back, t):
    #     if back:
    #         if traces:
    #             t = max(t - 1, 0, ntraces)
    #         else:
    #             t = max(t - 1, 0)
    #     title = get_title(df, var, t)

    #     fig = scatter_var(
    #         df,
    #         var,
    #         t,
    #         mode=mode,
    #         title=title,
    #         traces=traces,
    #         ntraces=ntraces,
    #         reversescale=reversescale,
    #         metal=metal,
    #         metal_dynamics=metal_dynamics,
    #         top=top,
    #         bottom=bottom,
    #         step=True,
    #     )

    #     return fig, t

    # @callback(
    #     Input("save", "n_clicks"),
    #     State("graph", "relayoutData"),
    #     State("timestep", "value"),
    # )
    # def save_fig(save, relayout_data, t):
    #     if save:
    #         t = (t - df["timesteps"][0]) // df["dt"]
    #         title = get_title(df, var, t)
    #         fig = scatter_var(
    #             df,
    #             var,
    #             t,
    #             mode=mode,
    #             title=title,
    #             traces=traces,
    #             ntraces=ntraces,
    #             reversescale=reversescale,
    #             metal=metal,
    #             metal_dynamics=metal_dynamics,
    #             top=top,
    #             bottom=bottom,
    #             step=False,
    #         )
    #         if relayout_data is not None and "scene.camera" in relayout_data:
    #             camera = relayout_data["scene.camera"]
    #             fig.update_layout(scene_camera=camera)

    #         fig.write_image(f"{var}_{t}.png", scale=3)
    #     return None

    return fig, color_by, color_by_options


@callback(
    Output("distribution-graph", "figure"),
    Output("log", "options"),
    Output("log", "value"),
    Output("log", "labelStyle"),
    Output("color-by-distribution", "options"),
    Output("color-by-distribution", "value"),
    Input("distribution-mode", "value"),
    Input("distribution-var", "value"),
    Input("timestep", "value"),
    Input("abs", "value"),
    Input("normalize-dist", "value"),
    Input("log", "value"),
    State("log", "options"),
    Input("color-by-distribution", "value"),
    State("color-by-distribution", "options"),
    State("cluster-vars", "value"),
    Input("color-by-scatter", "value"),
    State("timestep", "min"),
)
def update_distribution(
    mode,
    var,
    t,
    abs_val,
    normalize,
    log,
    log_options,
    color_by,
    color_by_options,
    cluster_vars,
    color_by_scatter,
    t0,
):
    # print("update distribution")
    global df
    log_value = ["Log"] if log else []
    log_label_style = {}
    log_options[0]["disabled"] = False
    if not color_by:
        color_by = ""
    color_by_options[0]["disabled"] = False
    color = None
    if not df or var is None:
        fig = go.Figure()
        fig.update_layout(
            margin=dict(l=20, r=20, t=20, b=20),
            uirevision=True,
            # transition={"duration": 3000, "easing": "cubic-in-out"},
            template="plotly_dark",
        )
        return (
            fig,
            log_options,
            log_value,
            log_label_style,
            color_by_options,
            color_by,
        )

    t = int((t - t0) / df["dt"])
    xlabel = "\\text{" + var + "}"

    if mode.lower() == "atom":
        data = df["atom"]
        x = data[var][t][df["non_metal"]]
        nbins = 100

    else:
        data = df["molecule"]
        x = data[var][t]
        nbins = 10

        color_by_options[1]["disabled"] = True

        if color_by == "Atom Type":
            color_by = ""

    # deal with phase written into dump in atom_to_molecuele
    if "phase" not in list(df["molecule"].keys()) and "phase_t" not in list(
        data.keys()
    ):
        color_by_options[0]["disabled"] = True

        if color_by == "Phase":
            color_by = ""

    if color_by:
        if color_by.lower() == "phase":
            if cluster_vars:
                color = data["phase_t"]

            else:
                color = data["phase"][t][df["non_metal"]]

        elif color_by.lower() == "atom type":
            color = data["type"][t][df["non_metal"]]

    if abs_val:
        x = np.abs(x)
        xlabel = "\\lvert" + xlabel + "\\rvert"

    if normalize:
        x = utils.normalize_arr(x)
        xlabel += "_{\\text{norm}}"

    if x.min() < 0:
        log_options[0]["disabled"] = True
        log_value = []
        log_label_style["opacity"] = 0.6
        log_label_style["cursor"] = "not-allowed"

    elif log:
        x[x == 0.0] = 1e-6
        x = np.log10(x)
        xlabel = "\\log (" + xlabel + ")"

    xlabel = "$" + xlabel + "$"

    marker_dict = {"color": "rgb(0,0,100)"}

    fig = go.Figure()

    if color_by.lower() == "phase":
        mask1 = color == 0
        mask2 = color == 1

        x1 = x[mask1]
        x2 = x[mask2]
        counts1, bins1 = np.histogram(x1, bins=nbins)
        bins1 = 0.5 * (bins1[:-1] + bins1[1:])

        counts2, bins2 = np.histogram(x2, bins=nbins)
        bins2 = 0.5 * (bins2[:-1] + bins2[1:])
        norm = mask1.sum() + mask2.sum()

        fig.add_trace(go.Bar(x=bins1, y=counts1 / norm, name="0"))
        fig.add_trace(go.Bar(x=bins2, y=counts2 / norm, name="1"))

        # fig.add_trace(go.Histogram(x=x1, name="0", histnorm="probability"))
        # fig.add_trace(go.Histogram(x = x2, name = "1", histnorm= 'probability'))
        fig.update_traces(opacity=0.7, marker_line_width=0)
        fig.update_layout(barmode="overlay", bargap=0.0, bargroupgap=0.0)

        # x1 = x[mask1]
        # x2 = x[mask2]
        # upper = np.max([x1.max(), x2.max()])
        # print(upper)
        # fig.update_xaxes(range = [0, upper])

        # fig.add_trace(go.Histogram(x = x[mask1], name = "0", bingroup=1))
        # fig.add_trace(go.Histogram(x = x[mask2], name = "1", bingroup=1))
        # fig.update_traces(opacity = 0.7)
        # fig.update_layout(barmode = "stack", bargap = 0, bargroupgap = 0)
    else:
        fig = go.Figure(
            data=[
                go.Histogram(
                    x=x,
                    name=var,
                    histnorm="probability",
                    marker=marker_dict,
                )
            ]
        )

    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        # uirevision=True,
        # transition={"duration": 3000, "easing": "cubic-in-out"},
        template="plotly_dark",
        xaxis_title={"text": xlabel, "standoff": 30, "font.size": 18},
        yaxis_title={"text": "Percent", "standoff": 20, "font.size": 18},
    )

    return (
        fig,
        log_options,
        log_value,
        log_label_style,
        color_by_options,
        color_by,
    )


if __name__ == "__main__":
    app.run(
        debug=True,
        port="8080",
        # dev_tools_hot_reload=True,
        # dev_tools_hot_reload_interval=5,
        # dev_tools_hot_reload_max_retry=10,
        # dev_tools_hot_reload_watch_interval=5,
        # use_reloader=False,
    )

#    1444 manager dave something
