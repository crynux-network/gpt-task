from typing import Protocol, Dict, Any, TypeVar, Callable

T = TypeVar("T")


class ModelCache(Protocol[T]):
    def load(self, key: str, model_loader: Callable[[], T]) -> T:
        ...

    def clear(self) -> None:
        ...
