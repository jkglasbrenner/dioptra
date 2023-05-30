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
"""The server-side functions that perform auth/login endpoint operations."""
from __future__ import annotations

import structlog
from flask_jwt_extended import create_access_token, create_refresh_token
from injector import inject
from structlog.stdlib import BoundLogger

from dioptra.restapi.shared.key_value_store.service import KeyValueStoreService
from dioptra.restapi.shared.password.service import PasswordService
from dioptra.restapi.user.model import User

LOGGER: BoundLogger = structlog.stdlib.get_logger()


class AuthLoginService(object):
    @inject
    def __init__(
        self,
        key_value_store_service: KeyValueStoreService,
        password_service: PasswordService,
    ) -> None:
        self._key_value_store_service = key_value_store_service
        self._password_service = password_service

    def check_password(self, user: User, password: str, **kwargs) -> bool:
        return self._password_service.verify(
            password=password,
            hashed_password=user.password,
        )

    def create_access_token(self, user: User, **kwargs) -> str:
        log: BoundLogger = kwargs.get("log", LOGGER.new())  # noqa: F841
        access_token: str = create_access_token(identity=user)

        return access_token

    def create_refresh_token(self, user: User, **kwargs) -> str:
        log: BoundLogger = kwargs.get("log", LOGGER.new())  # noqa: F841
        refresh_token: str = create_refresh_token(identity=user)

        return refresh_token

    def _start_tracking_new_refresh_token_family(self):
        self._key_value_store_service
        pass

    def _associate_refresh_token_family_with_user(self):
        pass
