from typing import Optional

import numpy as np
import numpy.typing as npt

import plotly.express as px
import plotly.graph_objects as go

from movement_primitive_diffusion.datasets.trajectory_dataset import TrajectoryDataset
from movement_primitive_diffusion.workspaces.obstacle_avoidance.obstacle_avoidance_env import (
    ObstacleAvoidanceScene,
    Mode,
)


def plotly_trajectories(
    trajs: list[np.ndarray],
    traj_labels: Optional[list[str]] = None,
    title: Optional[str] = None,
    scene: Optional[ObstacleAvoidanceScene] = None,
    init_eef_pos: npt.ArrayLike = np.array([0.525, -0.28]),
    rod_radius: float = 0.01,
    obs_lvl_radii: dict[int, float] = {1: 0.03, 2: 0.025, 3: 0.025},
    ws_limits_low: npt.ArrayLike = np.array([0.293, -0.3]),
    ws_limits_high: npt.ArrayLike = np.array([0.707, 0.38]),
) -> go.Figure:
    """trajs: list of arrays with dim (timesteps, xy_pos)"""
    num_trajs = len(trajs)

    if traj_labels is not None:
        assert (
            len(traj_labels) == num_trajs
        ), "Number of labels must match number of trajectories"

    # plot trajectories
    fig = px.scatter(
        data_frame={
            "trajectory": [
                (idx if traj_labels is None else traj_labels[idx])
                for i in range(num_trajs)
                for idx in trajs[i].shape[0] * [i]
            ],
            "t": [t for i in range(num_trajs) for t in range(trajs[i].shape[0])],
            "x": [x for i in range(num_trajs) for x in trajs[i][:, 0]],
            "y": [y for i in range(num_trajs) for y in trajs[i][:, 1]],
            "rod": [
                2.01 * rod_radius
                for i in range(num_trajs)
                for j in range(trajs[i].shape[0])
            ],
        },
        x="y",
        y="x",
        color="trajectory",
        size="rod",
        hover_data=["t"],
    )
    fig.update_traces(marker=dict(line_color="black"))

    # plot obstacles
    shapes = []
    if scene is None:
        scene = ObstacleAvoidanceScene()
    for lvl in [1, 2, 3]:
        obs = scene.get_level_positions(lvl)
        obs_radius = obs_lvl_radii[lvl]
        for i in range(obs.shape[0]):
            shapes.append(
                {
                    "type": "circle",
                    "xref": "x",
                    "yref": "y",
                    "x0": obs[i, 1] - obs_radius,
                    "y0": obs[i, 0] - obs_radius,
                    "x1": obs[i, 1] + obs_radius,
                    "y1": obs[i, 0] + obs_radius,
                    "fillcolor": "red",
                    "line_color": "red",
                }
            )
    fig.update_layout(shapes=shapes)

    # plot finish line
    fig.add_vline(scene.get_goal()[1], line_color="green")

    fig.update_layout(
        title=title,
        xaxis_title="y",
        yaxis_title="x",
        showlegend=True,
        legend=dict(itemsizing="constant"),
    )

    fig.update_xaxes(
        range=(ws_limits_low[1] - 2 * rod_radius, ws_limits_high[1] + 2 * rod_radius),
        constrain="domain",
        tick0=init_eef_pos[1],
        dtick=2 * rod_radius,
        zeroline=False,
    )

    # equal aspect ratio and reversed y axis
    fig.update_yaxes(
        range=(ws_limits_high[0] + 2 * rod_radius, ws_limits_low[0] - 2 * rod_radius),
        scaleanchor="x",
        scaleratio=1,
        tick0=init_eef_pos[0],
        dtick=2 * rod_radius,
        zeroline=False,
    )

    return fig


def plotly_dataset(
    dataset: TrajectoryDataset,
    position_key: str,
    trajcetories: Optional[list[int]] = None,
    title: Optional[str] = None,
    scene: Optional[ObstacleAvoidanceScene] = None,
    init_eef_pos: npt.ArrayLike = np.array([0.525, -0.28]),
    rod_radius: float = 0.01,
    obs_lvl_radii: dict[int, float] = {1: 0.03, 2: 0.025, 3: 0.025},
    ws_limits_low: npt.ArrayLike = np.array([0.293, -0.3]),
    ws_limits_high: npt.ArrayLike = np.array([0.707, 0.38]),
):
    trajs = []
    traj_labels = []
    indices = range(len(dataset)) if trajcetories is None else trajcetories
    for idx in indices:
        trajs.append(dataset[idx][position_key].detach().cpu().numpy())
        traj_labels.append(dataset.get_trajectory_dir(idx).name)

    return plotly_trajectories(
        trajs=trajs,
        traj_labels=traj_labels,
        title=title,
        scene=scene,
        init_eef_pos=init_eef_pos,
        rod_radius=rod_radius,
        obs_lvl_radii=obs_lvl_radii,
        ws_limits_low=ws_limits_low,
        ws_limits_high=ws_limits_high,
    )

def plotly_trajectory_modes(
    traj_modes: list[Mode],
    title: Optional[str] = None,
) -> go.Figure:
    levels = [[] for _ in range(3)]
    for mode in traj_modes:
        e = mode.get_encoding()

        l1 = e[0:2].astype(int).tolist()
        match l1: 
            case [0, 0]:
                levels[0].append("-|-")
            case [1, 0]:
                levels[0].append("x|-")
            case [0, 1]:
                levels[0].append("-|x")
            case _:
                raise ValueError(f"Invalid encoding {l1 = }")

        l2 = e[2:5].astype(int).tolist()
        match l2: 
            case [0, 0, 0]:
                levels[1].append("-|-|-")
            case [1, 0, 0]:
                levels[1].append("x|-|-")
            case [0, 1, 0]:
                levels[1].append("-|x|-")
            case [0, 0, 1]:
                levels[1].append("-|-|x")
            case _:
                raise ValueError(f"Invalid encoding {l2 = }")

        l3 = e[5:9].astype(int).tolist()
        match l3: 
            case [0, 0, 0, 0]:
                levels[2].append("-|-|-|-")
            case [1, 0, 0, 0]:
                levels[2].append("x|-|-|-")
            case [0, 1, 0, 0]:
                levels[2].append("-|x|-|-")
            case [0, 0, 1, 0]:
                levels[2].append("-|-|x|-")
            case [0, 0, 0, 1]:
                levels[2].append("-|-|-|x")
            case _:
                raise ValueError(f"Invalid encoding {l3 = }")

    fig = px.parallel_categories(
        data_frame={
            "level1": levels[0],
            "level2": levels[1],
            "level3": levels[2],
            "mode": [mode.decode() for mode in traj_modes],
        },
        color="mode",
        dimensions=["level1","level2","level3"],
        labels={"level1": "Level 1","level2": "Level 2", "level3": "Level 3"},
        color_continuous_scale=px.colors.sequential.Viridis,
    )
    fig.update_traces(dimensions=[
        {
            "categoryorder": "array",
            "categoryarray": ["-|-", "x|-", "-|x"],
        },
        {
            "categoryorder": "array",
            "categoryarray": ["-|-|-", "x|-|-", "-|x|-", "-|-|x",],
        },
        {
            "categoryorder": "array",
            "categoryarray": ["-|-|-|-", "x|-|-|-", "-|x|-|-", "-|-|x|-", "-|-|-|x"],
        },
    ])
    fig.update_layout(title=title)

    return fig

if __name__ == "__main__":
    from pathlib import Path
    import git

    #modes = []
    #success_mask = []
    #for mode in range(5*24):
    #    encoding = np.zeros(9, dtype=int)
    #    idx = np.random.choice([0, 1], size=1)
    #    encoding[idx] = np.random.choice([0, 1], size=1)
    #    if encoding[idx] == 0:
    #        success_mask.append(False)
    #        modes.append(Mode.from_encoding(encoding))
    #        continue
    #    idx = np.random.choice([2, 3, 4], size=1)
    #    encoding[idx] = np.random.choice([0, 1], size=1)
    #    if encoding[idx] == 0:
    #        success_mask.append(False)
    #        modes.append(Mode.from_encoding(encoding))
    #        continue
    #    idx = np.random.choice([5, 6, 7, 8], size=1)
    #    encoding[idx] = np.random.choice([0, 1], size=1)
    #    modes.append(Mode.from_encoding(encoding))
    #    success_mask.append(np.sum(encoding) == 3)

    #fig = plotly_trajectory_modes(
    #    traj_modes=modes,
    #)
    #fig.show()

    dataset_hz = 30

    git_repo = git.Repo(".", search_parent_directories=True)
    git_root = git_repo.git.rev_parse("--show-toplevel")
    trajectories_root = Path(git_root) / "data" / f"obstacle_avoidance_trajectories_{dataset_hz}_hz"

    dataset = TrajectoryDataset(
        trajectory_dirs=[path for path in trajectories_root.iterdir() if path.is_dir()],
        keys=["action", "agent_pos", "agent_vel"],
        dt=1.0 / dataset_hz,
    )

    fig = plotly_dataset(
        dataset=dataset,
        position_key="agent_pos",
        trajcetories=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    )
    fig.show()
