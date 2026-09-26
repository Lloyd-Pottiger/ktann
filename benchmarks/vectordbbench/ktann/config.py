"""Configuration for the private, explicitly launched KTANN benchmark bridge."""
from pydantic import BaseModel, Field
from ..api import DBConfig, DBCaseConfig, MetricType


class KTANNConfig(DBConfig):
    socket_path: str
    dataset: str
    companion_dir: str

    def to_dict(self) -> dict:
        return self.model_dump(include={"socket_path", "dataset", "companion_dir"})


class KTANNCaseConfig(BaseModel, DBCaseConfig):
    metric_type: MetricType | None = None
    leaf_budget: int | None = Field(default=None, ge=1, le=1048576)
    leaf_beam: int | None = Field(default=None, ge=1, le=16384)

    def index_param(self) -> dict:
        return {"max_partition_entries": 128, "write_beam_size": 8}

    def search_param(self) -> dict:
        return {"leaf_budget": self.leaf_budget, "leaf_beam": self.leaf_beam}
