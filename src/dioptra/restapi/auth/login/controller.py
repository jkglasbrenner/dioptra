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
"""The module defining the auth/login endpoint."""
from __future__ import annotations

import uuid

import structlog
from flask import g, jsonify, request
from flask.wrappers import Response
from flask_accepts import accepts, responds
from flask_restx import Namespace, Resource
from injector import inject
from structlog.stdlib import BoundLogger

from dioptra.restapi.shared.password.service import PasswordService
from dioptra.restapi.utils import as_api_parser

LOGGER: BoundLogger = structlog.stdlib.get_logger()

api: Namespace = Namespace(
    "Auth/Login",
    description="Login",
)


@api.route("/")
class AuthLoginResource(Resource):
    @inject
    def __init__(self, password_service: PasswordService, *args, **kwargs) -> None:
        self._password_service = password_service
        super().__init__(*args, **kwargs)

    def post(self):
        log: BoundLogger = LOGGER.new(
            request_id=g.request_id, resource="auth/login", request_type="POST"
        )
