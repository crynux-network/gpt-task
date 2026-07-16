from contextlib import contextmanager
from functools import wraps
from typing import Callable, Iterable, Type, TypeVar

import torch.cuda
from huggingface_hub.utils import (GatedRepoError, LocalEntryNotFoundError,
                                   RepositoryNotFoundError,
                                   RevisionNotFoundError)
from pydantic import ValidationError
from requests import ConnectionError, HTTPError
from typing_extensions import ParamSpec

__all__ = [
    "wrap_error",
    "error_context",
    "TaskArgsInvalid",
    "ModelInvalid",
    "ModelDownloadError",
    "ModelNotDownloaded",
    "TaskExecutionError",
]


class TaskArgsInvalid(ValueError):
    def __str__(self) -> str:
        return "Task args invalid"


class ModelInvalid(ValueError):
    def __str__(self) -> str:
        return "Task model invalid"


class ModelDownloadError(ValueError):
    def __str__(self) -> str:
        return "Task model download error"


class ModelNotDownloaded(ValueError):
    def __str__(self) -> str:
        return "Task model not downloaded"


class TaskExecutionError(ValueError):
    def __str__(self) -> str:
        return "Task execution error"


def travel_exc(e: BaseException):
    queue = [e]
    exc_set = set(queue)

    while len(queue) > 0:
        exc = queue.pop(0)
        yield exc
        if exc.__cause__ is not None and exc.__cause__ not in exc_set:
            queue.append(exc.__cause__)
            exc_set.add(exc.__cause__)
        if exc.__context__ is not None and exc.__context__ not in exc_set:
            queue.append(exc.__context__)
            exc_set.add(exc.__context__)


def match_exception(e: Exception, targets: Iterable[Type[Exception]]) -> bool:
    for exc in travel_exc(e):
        if any(isinstance(exc, target) for target in targets):
            return True
    return False


@contextmanager
def error_context(local_files_only: bool = False):
    try:
        yield
    except ValidationError as e:
        raise TaskArgsInvalid from e
    except EnvironmentError as e:
        # In local_files_only mode every model load failure caused by a
        # missing local cache entry is a not-downloaded model; invalid model
        # identifiers are indistinguishable from a cache miss offline.
        if local_files_only:
            raise ModelNotDownloaded from e
        if match_exception(e, [LocalEntryNotFoundError]):
            raise ModelDownloadError from e
        elif match_exception(
            e,
            [
                RepositoryNotFoundError,
                RevisionNotFoundError,
                GatedRepoError,
            ],
        ):
            raise ModelInvalid from e
        elif match_exception(e, [HTTPError, ConnectionError]):
            raise ModelDownloadError from e
        else:
            raise TaskExecutionError from e
    except torch.cuda.OutOfMemoryError:
        raise
    except RuntimeError as e:
        if "out of memory" in str(e):
            raise
        else:
            raise TaskExecutionError from e
    except Exception as e:
        raise TaskExecutionError from e


T = TypeVar("T")
P = ParamSpec("P")


def wrap_error(f: Callable[P, T]) -> Callable[P, T]:
    @wraps(f)
    def inner(*args: P.args, **kwargs: P.kwargs) -> T:
        with error_context():
            return f(*args, **kwargs)

    return inner
