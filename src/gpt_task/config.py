import os
from typing import List, Tuple, Type

from pydantic import BaseModel
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource
)


class ModelConfig(BaseModel):
    id: str


class PreloadedModelsConfig(BaseModel):
    base: List[ModelConfig] | None


class ModelsDirConfig(BaseModel):
    huggingface: str


class DataDirConfig(BaseModel):
    models: ModelsDirConfig


class ProxyConfig(BaseModel):
    host: str = ""
    port: int = 8080
    username: str = ""
    password: str = ""


class Config(BaseSettings):
    preloaded_models: PreloadedModelsConfig = PreloadedModelsConfig(base=[])
    data_dir: DataDirConfig | None = None
    proxy: ProxyConfig | None = None
    local_files_only: bool = False

    model_config = SettingsConfigDict(
        env_nested_delimiter="__",
        yaml_file=os.getenv("GPT_TASK_CONFIG", "config.yml"),
        env_prefix="gpt_",
        extra="ignore"
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: Type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> Tuple[PydanticBaseSettingsSource, ...]:
        return (
            init_settings,
            YamlConfigSettingsSource(settings_cls),
            env_settings,
            dotenv_settings,
            file_secret_settings,
        )


_default_config: Config | None = None


def get_config() -> Config:
    global _default_config

    if _default_config is None:
        _default_config = Config()  # type: ignore

    return _default_config
