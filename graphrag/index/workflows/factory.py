# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Encapsulates pipeline construction and selection."""

from typing import ClassVar

from graphrag.config.enums import IndexingMethod
from graphrag.config.models.graph_rag_config import GraphRagConfig
from graphrag.index.typing.pipeline import Pipeline
from graphrag.index.typing.workflow import WorkflowFunction


class PipelineFactory:
    """A factory class for workflow pipelines."""

    workflows: ClassVar[dict[str, WorkflowFunction]] = {}
    pipelines: ClassVar[dict[str, list[str]]] = {}

    @classmethod
    def register(cls, name: str, workflow: WorkflowFunction):
        """Register a custom workflow function."""
        cls.workflows[name] = workflow

    @classmethod
    def register_all(cls, workflows: dict[str, WorkflowFunction]):
        """Register a dict of custom workflow functions."""
        for name, workflow in workflows.items():
            cls.register(name, workflow)

    @classmethod
    def register_pipeline(cls, name: str, workflows: list[str]):
        """Register a new pipeline method as a list of workflow names."""
        cls.pipelines[name] = workflows

    @classmethod
    def create_pipeline(
        cls,
        config: GraphRagConfig,
        method: IndexingMethod | str = IndexingMethod.Standard,
    ) -> Pipeline:
        """Create a pipeline generator."""
        workflows = config.workflows or cls.pipelines.get(method, [])
        return Pipeline([(name, cls.workflows[name]) for name in workflows])


# --- Register default implementations ---
_standard_workflows = [
    #执行完 这个工作流会在output生成两个文档 document.parquet 和 text_units_parquet
    "create_base_text_units",
    # 这个工作流会将这两个文档的字段内容进行合并，即把`text_units.parquet`文件中的字段合并到`documents.parquet`文件中
    # 会在原来的文档的基础上增加一个 text_unit_ids字段
    "create_final_documents",
    # "extract_graph",
    # "finalize_graph",
    # "extract_covariates",
    # "create_communities",
    # "create_final_text_units",
    # "create_community_reports",
    # "generate_text_embeddings",
]
_fast_workflows = [
    "create_base_text_units",
    "create_final_documents",
    "extract_graph_nlp",
    "prune_graph",
    "finalize_graph",
    "create_communities",
    "create_final_text_units",
    "create_community_reports_text",
    "generate_text_embeddings",
]
_update_workflows = [
    "update_final_documents",
    "update_entities_relationships",
    "update_text_units",
    "update_covariates",
    "update_communities",
    "update_community_reports",
    "update_text_embeddings",
    "update_clean_state",
]
PipelineFactory.register_pipeline(
    IndexingMethod.Standard, ["load_input_documents", *_standard_workflows]
)
PipelineFactory.register_pipeline(
    IndexingMethod.Fast, ["load_input_documents", *_fast_workflows]
)
PipelineFactory.register_pipeline(
    IndexingMethod.StandardUpdate,
    ["load_update_documents", *_standard_workflows, *_update_workflows],
)
PipelineFactory.register_pipeline(
    IndexingMethod.FastUpdate,
    ["load_update_documents", *_fast_workflows, *_update_workflows],
)
