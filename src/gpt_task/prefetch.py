from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from fnmatch import fnmatch

from huggingface_hub import model_info, snapshot_download

from .config import Config, ProxyConfig, get_config

_logger = logging.getLogger(__name__)


def get_requests_proxy_url(proxy: ProxyConfig | None) -> str | None:
    if proxy is not None and proxy.host != "":

        if "://" in proxy.host:
            scheme, host = proxy.host.split("://", 2)
        else:
            scheme, host = "", proxy.host

        proxy_str = ""
        if scheme != "":
            proxy_str += f"{scheme}://"

        if proxy.username != "":
            proxy_str += f"{proxy.username}"

            if proxy.password != "":
                proxy_str += f":{proxy.password}"

            proxy_str += f"@"

        proxy_str += f"{host}:{proxy.port}"

        return proxy_str
    else:
        return None


@contextmanager
def requests_proxy_session(proxy: ProxyConfig | None):
    proxy_url = get_requests_proxy_url(proxy)
    if proxy_url is not None:
        origin_http_proxy = os.environ.get("HTTP_PROXY", None)
        origin_https_proxy = os.environ.get("HTTPS_PROXY", None)
        os.environ["HTTP_PROXY"] = proxy_url
        os.environ["HTTPS_PROXY"] = proxy_url
        try:
            yield {
                "http": proxy_url,
                "https": proxy_url,
            }
        finally:
            if origin_http_proxy is not None:
                os.environ["HTTP_PROXY"] = origin_http_proxy
            else:
                os.environ.pop("HTTP_PROXY")
            if origin_https_proxy is not None:
                os.environ["HTTPS_PROXY"] = origin_https_proxy
            else:
                os.environ.pop("HTTPS_PROXY")
    else:
        yield None


def download_model(
    model_name: str,
    hf_model_cache_dir: str | None = None,
    proxy: ProxyConfig | None = None,
    revision: str | None = None,
):
    with requests_proxy_session(proxy=proxy) as proxies:
        call_args = {
            "cache_dir": hf_model_cache_dir,
            "resume_download": True,
            "proxies": proxies,
        }
        if revision is not None:
            call_args["revision"] = revision

        info = model_info(model_name, revision=revision)
        siblings = info.siblings

        ignore_patterns = ["**/*"]

        possible_weight_ext = [
            "*.safetensors*",
            "*.bin*",
            "*.msgpack*",
            "*.h5*",
            "*.tflite*",
            "*.ot*",
        ]
        if siblings is not None:
            filenames = [sibling.rfilename for sibling in siblings]
            for i, ext in enumerate(possible_weight_ext):
                if any(fnmatch(filename, ext) for filename in filenames):
                    ignore_patterns.extend(possible_weight_ext[i + 1 :])
                    break

        snapshot_download(
            repo_id=model_name,
            repo_type="model",
            ignore_patterns=ignore_patterns,
            **call_args,
        )


def prefetch_models(config: Config | None = None):
    if config is None:
        config = get_config()

    if config.preloaded_models.base is not None:
        for model_config in config.preloaded_models.base:
            _logger.info(f"Preloading base models: {model_config.id}")
            call_args = {}
            if config.data_dir is not None:
                call_args["hf_model_cache_dir"] = config.data_dir.models.huggingface
            if config.proxy is not None:
                call_args["proxy"] = config.proxy

            download_model(model_config.id, **call_args)
            _logger.info(f"Successfully preloaded model: {model_config.id}")
