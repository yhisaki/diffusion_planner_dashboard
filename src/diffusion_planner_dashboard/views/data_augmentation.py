"""Inspect training data augmentation on one H5 frame."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import streamlit as st
from omegaconf import OmegaConf

import diffusion_planner
from diffusion_planner.data import (
    PlannerEgoStateAugmentation,
    PlannerFixStopPoint,
    PlannerILQRRefinement,
    PlannerStartDecisionAugmentation,
)
from diffusion_planner.visualizer import plot_frame
from diffusion_planner_dashboard.services import (
    FrameIndex,
    FrameIndexRow,
    FrameLoader,
    inspect_augmentation,
    load_frame_index,
)
from diffusion_planner_dashboard.ui.ego_state_chart import render_ego_state_chart
from diffusion_planner_dashboard.ui.metadata import (
    render_index_summary,
    render_row_metadata,
)
from diffusion_planner_dashboard.ui.settings import (
    render_data_source_settings,
    render_frame_selector,
    render_plot_options,
)
from diffusion_planner_dashboard.ui.tensor_inspector import render_tensor_table

# The training transform configs of the workspace this dashboard runs from.
TRANSFORM_CONFIGS = (
    Path(diffusion_planner.__file__).parents[4]
    / "configs"
    / "train"
    / "dataloader"
    / "transforms"
)


@st.cache_data(show_spinner=False)
def _cached_index(path: str, modification_time_ns: int) -> FrameIndex:
    del modification_time_ns
    return load_frame_index(path)


@st.cache_resource
def _frame_loader() -> FrameLoader:
    return FrameLoader()


@st.cache_data(max_entries=64, show_spinner="Reading frame data from H5...")
def _cached_frame(
    h5_path: str,
    frame_index: int,
    frame_time_ns: int,
    modification_time_ns: int,
):
    del modification_time_ns
    row = FrameIndexRow(0, h5_path, frame_index, frame_time_ns, {})
    return _frame_loader().load(row)


@dataclass(frozen=True)
class ILQRSettings:
    """User-adjustable iLQR settings for the augmentation inspector."""

    longitudinal_position_weight: float
    lateral_position_weight: float
    heading_weight: float
    terminal_weight_scale: float
    velocity_weight: float
    steering_weight: float
    acceleration_weight: float
    steering_rate_weight: float
    velocity_max: float
    steering_limit_rad: float
    acceleration_bounds: tuple[float, float]
    steering_rate_limit_rps: float
    max_iterations: int


@dataclass(frozen=True)
class AugmentationPipelineSettings:
    """Controls for the training-order augmentation preview pipeline."""

    apply_start_decision: bool
    start_stop_speed_threshold: float
    start_max_shift_steps: int
    apply_fix_stop_point: bool
    fix_stop_speed_threshold: float
    fix_stop_stopped_ego_search_start_index: int
    apply_ego_state: bool
    refinement: str
    longitudinal_offset: float
    lateral_offset: float
    yaw_offset: float
    ego_speed_scale: float
    steering_offset: float
    ilqr: ILQRSettings


def _render_augmentation_settings() -> AugmentationPipelineSettings:
    st.sidebar.subheader("Augmentation")
    st.sidebar.caption("Training order")
    apply_start_decision = st.sidebar.checkbox("1. Start Decision", value=False)
    apply_fix_stop_point = st.sidebar.checkbox("2. Fix Stop Point", value=True)
    apply_ego_state = st.sidebar.checkbox("3. Ego State Augmentation", value=True)
    refinement = str(
        st.sidebar.radio(
            "4. Refinement", ("iLQR", "Frenet", "None"), index=0, horizontal=True
        )
    )
    with st.sidebar.expander("Start Decision parameters", expanded=False):
        start_stop_speed_threshold = float(
            st.number_input(
                "Start stop-speed threshold [m/s]", 0.0, value=0.1, step=0.05
            )
        )
        start_max_shift_steps = int(
            st.number_input("Start maximum shift steps", 1, 30, 10, step=1)
        )
    longitudinal_offset = float(
        st.sidebar.slider(
            "Longitudinal offset [m]",
            min_value=-5.0,
            max_value=5.0,
            value=0.0,
            step=0.1,
        )
    )
    lateral_offset = float(
        st.sidebar.slider(
            "Lateral offset [m]",
            min_value=-5.0,
            max_value=5.0,
            value=0.0,
            step=0.1,
        )
    )
    yaw_offset_degrees = float(
        st.sidebar.slider(
            "Yaw offset [deg]",
            min_value=-30.0,
            max_value=30.0,
            value=0.0,
            step=0.5,
        )
    )
    ego_speed_scale = float(
        st.sidebar.slider(
            "Ego history speed scale",
            min_value=0.0,
            max_value=2.0,
            value=1.0,
            step=0.01,
        )
    )
    steering_offset_degrees = float(
        st.sidebar.slider(
            "Ego steering offset [deg]",
            min_value=-10.0,
            max_value=10.0,
            value=0.0,
            step=0.1,
        )
    )
    with st.sidebar.expander("iLQR parameters", expanded=False):
        st.caption("Trajectory tracking weights")
        q_longitudinal = float(
            st.number_input("Longitudinal position weight", 0.0, value=1.0, step=0.1)
        )
        q_lateral = float(
            st.number_input("Lateral position weight", 0.0, value=1.0, step=0.1)
        )
        q_heading = float(st.number_input("Heading weight", 0.0, value=0.5, step=0.1))
        terminal_weight_scale = float(
            st.number_input("Terminal weight scale", 0.0, value=10.0, step=1.0)
        )
        velocity_weight = float(
            st.number_input("Velocity tracking weight", 0.0, value=0.2, step=0.1)
        )
        steering_weight = float(
            st.number_input("Steering tracking weight", 0.0, value=0.1, step=0.1)
        )
        st.caption("Input weights")
        acceleration_weight = float(
            st.number_input("Acceleration weight", 0.0, value=1.0, step=0.1)
        )
        steering_rate_weight = float(
            st.number_input("Steering rate weight", 0.0, value=10.0, step=1.0)
        )
        st.caption("Constraints and solver")
        velocity_max = float(
            st.number_input("Maximum velocity [m/s]", 0.1, value=30.0, step=1.0)
        )
        steering_limit_rad = math.radians(
            float(
                st.number_input(
                    "Steering limit [deg]", 0.1, 89.0, math.degrees(0.7), step=1.0
                )
            )
        )
        acceleration_min = float(
            st.number_input("Minimum acceleration [m/s^2]", max_value=0.0, value=-4.0)
        )
        acceleration_max = float(
            st.number_input("Maximum acceleration [m/s^2]", 0.0, value=3.0)
        )
        steering_rate_limit_rps = float(
            st.number_input("Steering rate limit [rad/s]", 0.01, value=1.0, step=0.1)
        )
        max_iterations = int(st.number_input("Maximum iterations", 1, 100, 15, step=1))
    with st.sidebar.expander("Fix stop point", expanded=False):
        fix_stop_speed_threshold = float(
            st.number_input(
                "Stop-point speed threshold [m/s]",
                0.0,
                value=0.1,
                step=0.05,
                help="Hold the ego pose and zero its motion from the first future point at or below this speed.",
            )
        )
        fix_stop_stopped_ego_search_start_index = int(
            st.number_input(
                "Stopped-ego search start index",
                min_value=0,
                value=0,
                step=1,
                help=(
                    "When the ego is currently stopped, ignore low-speed future "
                    "points before this index so a planned departure is preserved."
                ),
            )
        )
    return AugmentationPipelineSettings(
        apply_start_decision=apply_start_decision,
        start_stop_speed_threshold=start_stop_speed_threshold,
        start_max_shift_steps=start_max_shift_steps,
        apply_fix_stop_point=apply_fix_stop_point,
        fix_stop_speed_threshold=fix_stop_speed_threshold,
        fix_stop_stopped_ego_search_start_index=(
            fix_stop_stopped_ego_search_start_index
        ),
        apply_ego_state=apply_ego_state,
        refinement=refinement,
        longitudinal_offset=longitudinal_offset,
        lateral_offset=lateral_offset,
        yaw_offset=math.radians(yaw_offset_degrees),
        ego_speed_scale=ego_speed_scale,
        steering_offset=math.radians(steering_offset_degrees),
        ilqr=ILQRSettings(
            longitudinal_position_weight=q_longitudinal,
            lateral_position_weight=q_lateral,
            heading_weight=q_heading,
            terminal_weight_scale=terminal_weight_scale,
            velocity_weight=velocity_weight,
            steering_weight=steering_weight,
            acceleration_weight=acceleration_weight,
            steering_rate_weight=steering_rate_weight,
            velocity_max=velocity_max,
            steering_limit_rad=steering_limit_rad,
            acceleration_bounds=(acceleration_min, acceleration_max),
            steering_rate_limit_rps=steering_rate_limit_rps,
            max_iterations=max_iterations,
        ),
    )


def _refinement(
    settings: AugmentationPipelineSettings,
) -> Callable[[dict[str, Any]], dict[str, Any]] | None:
    """The selected refinement: iLQR from the sidebar, Frenet from the training
    configuration."""
    if settings.refinement == "Frenet":
        config = OmegaConf.load(TRANSFORM_CONFIGS / "frenet_refinement.yaml")
        return hydra.utils.instantiate(config)
    if settings.refinement != "iLQR":
        return None
    ilqr = settings.ilqr
    return PlannerILQRRefinement(
        longitudinal_position_weight=ilqr.longitudinal_position_weight,
        lateral_position_weight=ilqr.lateral_position_weight,
        heading_weight=ilqr.heading_weight,
        terminal_weight_scale=ilqr.terminal_weight_scale,
        velocity_weight=ilqr.velocity_weight,
        steering_weight=ilqr.steering_weight,
        acceleration_weight=ilqr.acceleration_weight,
        steering_rate_weight=ilqr.steering_rate_weight,
        velocity_bounds=(0.0, ilqr.velocity_max),
        steering_limit_rad=ilqr.steering_limit_rad,
        acceleration_bounds=ilqr.acceleration_bounds,
        steering_rate_limit_rps=ilqr.steering_rate_limit_rps,
        max_iterations=ilqr.max_iterations,
    )


def _augment_frame(
    frame_data: dict[str, Any],
    settings: AugmentationPipelineSettings,
) -> dict[str, Any]:
    start_decision = PlannerStartDecisionAugmentation(
        probability=1.0,
        stop_speed_threshold=settings.start_stop_speed_threshold,
        max_shift_steps=settings.start_max_shift_steps,
    )
    fix_stop_point = PlannerFixStopPoint(
        stop_speed_threshold=settings.fix_stop_speed_threshold,
        stopped_ego_search_start_index=(
            settings.fix_stop_stopped_ego_search_start_index
        ),
    )
    ego_state_augmentation = PlannerEgoStateAugmentation(
        normal_case={
            "longitudinal_offset": {"mean": settings.longitudinal_offset, "std": 0.0},
            "lateral_offset": {"mean": settings.lateral_offset, "std": 0.0},
            "yaw_offset": {"mean": settings.yaw_offset, "std": 0.0},
            "speed_scale": {"mean": settings.ego_speed_scale, "std": 0.0},
            "steering_offset": {"mean": settings.steering_offset, "std": 0.0},
        },
        curve={
            "steering_threshold": 0.17453292519943295,
            "longitudinal_offset": {"mean": settings.longitudinal_offset, "std": 0.0},
            "lateral_offset": {"mean": settings.lateral_offset, "std": 0.0},
            "yaw_offset": {"mean": settings.yaw_offset, "std": 0.0},
            "speed_scale": {"mean": settings.ego_speed_scale, "std": 0.0},
            "steering_offset": {"mean": settings.steering_offset, "std": 0.0},
        },
        stopped_in_intersection={
            "stopped_speed_threshold": 0.1,
            "longitudinal_offset": {"mean": settings.longitudinal_offset, "std": 0.0},
            "lateral_offset": {"mean": settings.lateral_offset, "std": 0.0},
            "yaw_offset": {"mean": settings.yaw_offset, "std": 0.0},
            "speed_scale": {"mean": settings.ego_speed_scale, "std": 0.0},
            "steering_offset": {"mean": settings.steering_offset, "std": 0.0},
        },
    )
    refinement = _refinement(settings)
    output = dict(frame_data)
    if settings.apply_start_decision:
        output = start_decision(output)
    if settings.apply_fix_stop_point:
        output = fix_stop_point(output)
    if settings.apply_ego_state:
        output = ego_state_augmentation(output)
    if refinement is not None:
        output = refinement(output)
    return output


def _difference_frame(
    original: dict[str, Any], augmented: dict[str, Any]
) -> dict[str, np.ndarray]:
    difference: dict[str, np.ndarray] = {}
    for key, original_value in original.items():
        original_array = np.asarray(original_value)
        augmented_array = np.asarray(augmented[key])
        if np.issubdtype(original_array.dtype, np.number):
            difference[key] = augmented_array.astype(np.float64) - original_array
        else:
            difference[key] = np.zeros(original_array.shape, dtype=np.float64)
    return difference


def render_data_augmentation() -> None:
    """Render original and deterministically augmented frame data."""
    st.title("Data Augmentation")
    source_path_text = render_data_source_settings()
    settings = _render_augmentation_settings()
    if source_path_text is None:
        st.info("Configure an H5 file or frame-index Parquet from the sidebar.")
        return

    source_path = Path(source_path_text).expanduser()
    try:
        source_modification_time_ns = source_path.stat().st_mtime_ns
        index = _cached_index(str(source_path), source_modification_time_ns)
    except (OSError, RuntimeError, ValueError) as error:
        st.error(str(error))
        return

    render_index_summary(index)
    row = render_frame_selector(index)
    render_row_metadata(row)
    options = render_plot_options()

    try:
        h5_modification_time_ns = Path(row.h5_path).stat().st_mtime_ns
        original = _cached_frame(
            row.h5_path,
            row.frame_index,
            row.frame_time_ns,
            h5_modification_time_ns,
        )
        augmented = _augment_frame(original, settings)
    except Exception as error:
        st.exception(error)
        return

    enabled = [
        name
        for name, applied in (
            ("Start Decision", settings.apply_start_decision),
            ("Fix Stop Point", settings.apply_fix_stop_point),
            ("Ego State", settings.apply_ego_state),
            (settings.refinement, settings.refinement != "None"),
        )
        if applied
    ]
    st.caption(
        "Training-order pipeline: Start Decision → Fix Stop Point → Ego State → "
        f"iLQR or Frenet. Enabled: {', '.join(enabled) if enabled else 'none'}."
    )
    original_column, augmented_column = st.columns(2)
    chart_identity = f"{index.path}::{row.index}::{settings}"
    with original_column:
        st.subheader("Original")
        original_figure = plot_frame(
            original, options=replace(options, title="Original frame")
        )
        original_figure.update_layout(uirevision=f"original::{chart_identity}")
        st.plotly_chart(
            original_figure,
            width="stretch",
            height=700,
            key=f"augmentation-original::{chart_identity}",
            config={"responsive": True, "scrollZoom": True},
        )
    with augmented_column:
        st.subheader("Augmented")
        augmented_figure = plot_frame(
            augmented, options=replace(options, title="Augmented frame")
        )
        augmented_figure.update_layout(uirevision=f"augmented::{chart_identity}")
        st.plotly_chart(
            augmented_figure,
            width="stretch",
            height=700,
            key=f"augmentation-augmented::{chart_identity}",
            config={"responsive": True, "scrollZoom": True},
        )

    inspection = inspect_augmentation(
        original,
        augmented,
        nonrigid_keys={"ego_agent_past"} if settings.apply_start_decision else (),
    )
    failures = sum(not bool(row_data["valid"]) for row_data in inspection)
    metric_columns = st.columns(3)
    metric_columns[0].metric("Checked tensors", len(inspection))
    metric_columns[1].metric("Failed checks", failures)
    metric_columns[2].metric(
        "Augmented ego current",
        np.array2string(np.asarray(augmented["ego_agent_past"])[-1, :4], precision=3),
    )
    st.subheader("Validation")
    if failures:
        st.error(f"{failures} augmentation invariant check(s) failed.")
    else:
        st.success("All augmentation invariant checks passed.")
    st.dataframe(inspection, width="stretch", hide_index=True)

    difference = _difference_frame(original, augmented)
    with st.expander("Tensor Inspector", expanded=False):
        original_tab, augmented_tab, difference_tab = st.tabs(
            ("Original tensors", "Augmented tensors", "Difference")
        )
        with original_tab:
            render_tensor_table(original, key_prefix="augmentation-original-inspector")
        with augmented_tab:
            render_tensor_table(
                augmented, key_prefix="augmentation-augmented-inspector"
            )
        with difference_tab:
            render_tensor_table(
                difference, key_prefix="augmentation-difference-inspector"
            )
    render_ego_state_chart(
        {"original": original, "augmented": augmented},
        key_prefix="augmentation-ego-state-chart",
    )
