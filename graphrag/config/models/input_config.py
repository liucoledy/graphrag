# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Parameterization settings for the default configuration."""

from pydantic import BaseModel, Field

import graphrag.config.defaults as defs
from graphrag.config.defaults import graphrag_config_defaults
from graphrag.config.enums import InputFileType
from graphrag.config.models.storage_config import StorageConfig


class InputConfig(BaseModel):
    """The default configuration section for Input."""

    storage: StorageConfig = Field(
        description="The storage configuration to use for reading input documents.",
        default=StorageConfig(
            base_dir=graphrag_config_defaults.input.storage.base_dir,
        ),
    )
    """
     文件类型目前版本支持3中文件类型  txt、json、csv 可以手动扩展文件类型
    """
    file_type: InputFileType = Field(
        description="The input file type to use.",
        default=graphrag_config_defaults.input.file_type,
    )
    encoding: str = Field(
        description="The input file encoding to use.",
        default=defs.graphrag_config_defaults.input.encoding,
    )
    """
        如果input（base_dir）文件夹下面有很多文件 可以根据正则来匹配文件库里面的文件
        如以下的配置 file_pattern: '^(?P<source>[^/]+)_(?P<year>\d{4})-(?P<month>\d{2})-(?P<day>\d{2})\.txt' 
                    file_filter:
                        year: '2025'
                        month: '03'
                        day: '05'
        就会加载 source_2025-03-05.txt 这个文件 所以 file_pattern 往往是和 file_filter一起搭配使用
    """
    file_pattern: str = Field(
        description="The input file pattern to use.",
        default=graphrag_config_defaults.input.file_pattern,
    )
    file_filter: dict[str, str] | None = Field(
        description="The optional file filter for the input files.",
        default=graphrag_config_defaults.input.file_filter,
    )


    """
     最终输出的DataFrame必须包含 id 、 text、 title三列
     用户如果指定了 text_column 就使用用户指定的 text_column作为 text列
     用户如果指定了 title_column 就使用用户指定的 title_column作为 title列
     
    if config.text_column is not None and "text" not in documents.columns:
        if config.text_column not in documents.columns:
            log.warning(
                "text_column %s not found in csv file %s",
                config.text_column,
                path,
            )
        else:
            documents["text"] = documents.apply(lambda x: x[config.text_column], axis=1)
      指定的列必须在 document中存在 所以下面两个配置字段 就是用来指定那两个列是 text和title的       
    """
    text_column: str = Field(
        description="The input text column to use.",
        default=graphrag_config_defaults.input.text_column,
    )
    title_column: str | None = Field(
        description="The input title column to use.",
        default=graphrag_config_defaults.input.title_column,
    )
    metadata: list[str] | None = Field(
        description="The document attribute columns to use.",
        default=graphrag_config_defaults.input.metadata,
    )
