"""ARC-AGI 3 puzzle solver module."""
from soma_mythos_ehra.arc3.adapter import ARC3Task, load_tasks_from_dir
from soma_mythos_ehra.arc3.objects import extract_objects, connected_component_labeling, build_feature_map
from soma_mythos_ehra.arc3.solver import ARC3Solver, ARC3Config, solve_task
from soma_mythos_ehra.arc3.transforms import (
    TransformType,
    apply_transform,
    apply_sequence,
)
