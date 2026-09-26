"""Registration uses the pinned VectorDBBench CLI's normal run path."""
from typing import Annotated, Unpack
import click
from ....cli.cli import CommonTypedDict, cli, click_parameter_decorators_from_typed_dict, run
from .. import DB


class KTANNTypedDict(CommonTypedDict):
    socket_path: Annotated[str, click.option("--socket-path", required=True)]
    dataset_identity: Annotated[str, click.option("--dataset-identity", required=True)]
    companion_dir: Annotated[str, click.option("--companion-dir", required=True)]
    leaf_budget: Annotated[int | None, click.option("--leaf-budget", type=int, default=None)]
    leaf_beam: Annotated[int | None, click.option("--leaf-beam", type=int, default=None)]


@cli.command()
@click_parameter_decorators_from_typed_dict(KTANNTypedDict)
def KTANN(**parameters: Unpack[KTANNTypedDict]):
    if parameters["case_type"] not in {"Performance768D1M", "PerformanceCustomDataset"}:
        raise click.UsageError("KTANN bridge supports Cohere 1M and unfiltered custom ANN performance cases only")
    from .config import KTANNConfig, KTANNCaseConfig
    run(db=DB.KTANN,
        db_config=KTANNConfig(db_label=parameters["db_label"], socket_path=parameters["socket_path"],
                              dataset=parameters["dataset_identity"], companion_dir=parameters["companion_dir"]),
        db_case_config=KTANNCaseConfig(leaf_budget=parameters["leaf_budget"], leaf_beam=parameters["leaf_beam"]),
        **parameters)
