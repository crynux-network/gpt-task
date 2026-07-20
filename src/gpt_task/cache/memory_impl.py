from typing import Any, Dict, Callable

import torch


class MemoryModelCache(object):
    def __init__(self, max_size: int = 1) -> None:
        self.max_size = max_size

        self._cache: Dict[str, Any] = {}

    def load(self, key: str, model_loader: Callable[[], Any]):
        if key in self._cache:
            return self._cache[key]
        else:
            if len(self._cache) >= self.max_size:
                keys = list(self._cache.keys())
                t = self._cache.pop(keys[0])
                del t
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            model = model_loader()
            self._cache[key] = model
            return model

    def clear(self) -> None:
        self._cache.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
