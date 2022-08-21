# This Software (Dioptra) is being made available as a public service by the
# National Institute of Standards and Technology (NIST), an Agency of the United
# States Department of Commerce. This software was developed in part by employees of
# NIST and in part by NIST contractors. Copyright in portions of this software that
# were developed by NIST contractors has been licensed or assigned to NIST. Pursuant
# to Title 17 United States Code Section 105, works of NIST employees are not
# subject to copyright protection in the United States. However, NIST may hold
# international copyright in software created by its employees and domestic
# copyright (or licensing rights) in portions of software that were assigned or
# licensed to NIST. To the extent that NIST holds copyright in this software, it is
# being made available under the Creative Commons Attribution 4.0 International
# license (CC BY 4.0). The disclaimers of the CC BY 4.0 license apply to all parts
# of the software developed or licensed by NIST.
#
# ACCESS THE FULL CC BY 4.0 LICENSE HERE:
# https://creativecommons.org/licenses/by/4.0/legalcode
"""A task plugin module for MLFlow artifacts management.

This module contains a set of task plugins for managing artifacts generated during an
entry point run.
"""
from __future__ import annotations

from pathlib import Path

import mlflow
import structlog
from structlog.stdlib import BoundLogger

from dioptra import pyplugs

LOGGER: BoundLogger = structlog.stdlib.get_logger()


@pyplugs.register
def upload_directory_as_artifact(
    directory: str | Path,
    artifact_prefix: str,
) -> None:
    directory = Path(directory)
    mlflow.log_artifacts(
        local_dir=str(directory),
        artifact_path=artifact_prefix,
    )
    LOGGER.info(
        "Artifact directory uploaded for current MLFlow run",
        directory=str(directory),
        artifact_prefix=artifact_prefix,
    )
