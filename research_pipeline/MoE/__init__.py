"""E5：E2 代表因子选组后，在所选完整因子群内等权组合。"""

from .e5_experiment import (
    build_factor_group_table,
    expand_selected_groups_to_equal_weights,
    run_e5_experiment,
    select_top_representative_group,
)
from .e6_experiment import run_e6_experiment, select_active_inner_weights

__all__ = [
    "build_factor_group_table",
    "expand_selected_groups_to_equal_weights",
    "run_e5_experiment",
    "run_e6_experiment",
    "select_active_inner_weights",
    "select_top_representative_group",
]
