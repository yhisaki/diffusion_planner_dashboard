"""Ego state charts: the XY path and selectable time series of ego state fields."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from diffusion_planner.data.dimensions import (
    EGO_STEERING_INDEX,
    EGO_VELOCITY_INDEX,
    EGO_YAW_RATE_INDEX,
)
from diffusion_planner.data.transforms.pose_augmentation import (
    EGO_STATE_AUGMENTATION_RECORDED_STATE_KEY,
)

TIME_STEP_S = 0.1

_DASHES = ("solid", "dot", "dash", "dashdot")

XY_PLOT = "XY path"
# Time-series plots: name -> value of each ego state row.
_TIME_SERIES: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "X [m]": lambda values: values[:, 0],
    "Y [m]": lambda values: values[:, 1],
    "Yaw [deg]": lambda values: np.rad2deg(np.arctan2(values[:, 3], values[:, 2])),
    "Speed [m/s]": lambda values: values[:, EGO_VELOCITY_INDEX],
    "Steering tire angle [deg]": lambda values: np.rad2deg(
        values[:, EGO_STEERING_INDEX]
    ),
    "Yaw rate [deg/s]": lambda values: np.rad2deg(values[:, EGO_YAW_RATE_INDEX]),
}
_DEFAULT_PLOTS = ("Speed [m/s]", "Steering tire angle [deg]")


def render_ego_state_chart(
    frames: Mapping[str, Mapping[str, Any]], *, key_prefix: str = "ego-state-chart"
) -> None:
    """Plot the ego past and future states selected in a collapsed section.

    ``frames`` maps a label to a frame; several frames are overlaid with
    different line dashes.
    """
    with st.expander("Ego State", expanded=False):
        selected = st.multiselect(
            "Plots",
            (XY_PLOT, *_TIME_SERIES),
            default=_DEFAULT_PLOTS,
            key=f"{key_prefix}::plots",
        )
        if XY_PLOT in selected:
            st.plotly_chart(
                _xy_figure(frames), width="stretch", key=f"{key_prefix}::xy"
            )
        names = [name for name in selected if name in _TIME_SERIES]
        if names:
            st.plotly_chart(
                _time_series_figure(frames, names),
                width="stretch",
                key=f"{key_prefix}::time-series",
            )


def _xy_figure(frames: Mapping[str, Mapping[str, Any]]) -> go.Figure:
    """The ego path in the recorded ego frame."""
    figure = go.Figure()
    for frame_index, (label, frame_data) in enumerate(frames.items()):
        dash = _DASHES[frame_index % len(_DASHES)]
        for name, _, values, color in _sequences(frame_data):
            figure.add_trace(
                go.Scatter(
                    x=values[:, 0],
                    y=values[:, 1],
                    mode="lines+markers",
                    marker={"size": 4},
                    line={"color": color, "dash": dash},
                    name=_trace_name(label, name, frames),
                )
            )
    figure.update_xaxes(title_text="X [m]")
    figure.update_yaxes(title_text="Y [m]", scaleanchor="x", scaleratio=1)
    figure.update_layout(height=500, margin={"t": 40, "b": 40})
    return figure


def _to_recorded_frame(states: np.ndarray, frame_data: Mapping[str, Any]) -> np.ndarray:
    """Express ego states of an ego-state-augmented frame in the recorded ego frame.

    The augmented frame stores the recorded ego pose in its own coordinates; a frame
    without it is already in the recorded ego frame. Position and heading are frame
    dependent; speed, steering angle, and yaw rate are not.
    """
    recorded = frame_data.get(EGO_STATE_AUGMENTATION_RECORDED_STATE_KEY)
    if recorded is None:
        return states
    recorded = np.asarray(recorded, dtype=np.float64)
    cos, sin = recorded[2:4] / max(float(np.linalg.norm(recorded[2:4])), 1e-6)
    rotation = np.array(((cos, sin), (-sin, cos)))
    result = np.array(states, dtype=np.float64, copy=True)
    result[:, :2] = (result[:, :2] - recorded[:2]) @ rotation.T
    result[:, 2:4] = result[:, 2:4] @ rotation.T
    return result


def _time_series_figure(
    frames: Mapping[str, Mapping[str, Any]], names: list[str]
) -> go.Figure:
    """One row per selected field over the past and future horizons."""
    figure = make_subplots(
        rows=len(names),
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08 if len(names) > 1 else 0.0,
        subplot_titles=names,
    )
    for frame_index, (label, frame_data) in enumerate(frames.items()):
        dash = _DASHES[frame_index % len(_DASHES)]
        for name, time, values, color in _sequences(frame_data):
            trace_name = _trace_name(label, name, frames)
            for row, field in enumerate(names, start=1):
                figure.add_trace(
                    go.Scatter(
                        x=time,
                        y=_TIME_SERIES[field](values),
                        mode="lines+markers",
                        marker={"size": 4},
                        line={"color": color, "dash": dash},
                        name=trace_name,
                        legendgroup=trace_name,
                        showlegend=row == 1,
                    ),
                    row=row,
                    col=1,
                )

    figure.add_vline(x=0.0, line_dash="dash", line_color="gray")
    figure.update_xaxes(title_text="Time from frame [s]", row=len(names), col=1)
    figure.update_layout(height=250 * len(names), margin={"t": 40, "b": 40})
    return figure


def _trace_name(label: str, name: str, frames: Mapping[str, Mapping[str, Any]]) -> str:
    return f"{label} {name}" if len(frames) > 1 else name


def _sequences(
    frame_data: Mapping[str, Any],
) -> list[tuple[str, np.ndarray, np.ndarray, str]]:
    """Past and future ego states with their times relative to the frame, expressed in
    the recorded ego frame so augmented and original frames share one coordinate
    system."""
    sequences = []
    past = frame_data.get("ego_agent_past")
    if past is not None and len(past) > 0:
        past = _to_recorded_frame(np.asarray(past), frame_data)
        past_time = (np.arange(len(past)) - (len(past) - 1)) * TIME_STEP_S
        sequences.append(("past", past_time, past, "#1f77b4"))
    future = frame_data.get("ego_agent_future")
    if future is not None and len(future) > 0:
        future = _to_recorded_frame(np.asarray(future), frame_data)
        future_time = (np.arange(len(future)) + 1) * TIME_STEP_S
        sequences.append(("future", future_time, future, "#d62728"))
    return sequences
