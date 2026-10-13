from typing import Protocol, Sequence

from gpt_task import models

from .errors import ExecutionModelFileError


class ExecutionModelFile(Protocol):
    repository: str
    revision: str
    format: str


def _is_commit_sha(revision: str) -> bool:
    if len(revision) != 40:
        return False
    try:
        int(revision, 16)
    except ValueError:
        return False
    return True


def apply_execution_model_files(
    args: models.GPTTaskArgs, model_files: Sequence[ExecutionModelFile]
) -> models.GPTTaskArgs:
    """Return task args that load the exact revision allocated for the model.

    The allocation is an independent load constraint. Branch or tag values in
    TaskArgs do not override it. A TaskArgs commit SHA that differs from the
    allocation is a conflict.
    """
    entries = [entry for entry in model_files if entry.repository == args.model]
    if not entries:
        raise ExecutionModelFileError(
            f"No execution model file is allocated for model {args.model}"
        )

    revisions = {entry.revision for entry in entries}
    if len(revisions) != 1:
        raise ExecutionModelFileError(
            f"Conflicting execution model file revisions for model {args.model}"
        )
    revision = revisions.pop()
    if revision == "":
        raise ExecutionModelFileError(
            f"Execution model file of model {args.model} has no revision"
        )
    if any(entry.format != "" for entry in entries):
        raise ExecutionModelFileError(
            f"Unsupported execution model file format for model {args.model}"
        )
    if (
        args.revision is not None
        and _is_commit_sha(args.revision)
        and args.revision != revision
    ):
        raise ExecutionModelFileError(
            f"Task revision {args.revision} conflicts with the allocated "
            f"revision {revision} of model {args.model}"
        )
    return args.model_copy(update={"revision": revision})
