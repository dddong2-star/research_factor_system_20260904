import importlib
import pytest


@pytest.mark.parametrize(
    "module_name",
    [
        "research_pipeline.research_system",
        "research_pipeline.MoE",
        "research_pipeline.factor_moe",
        "research_pipeline.MoE.run_e5",
        "research_pipeline.MoE.run_e6",
    ],
)
def test_old_packages_cannot_be_imported(module_name):
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(module_name)
