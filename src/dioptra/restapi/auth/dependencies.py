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
"""Binding configurations to shared services using dependency injection."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Callable
from urllib.parse import urlparse

from injector import Binder, Module, provider

from dioptra.restapi.shared.key_value_store.service import (
    KeyValueStoreService,
    LocalKeyValueStoreService,
    RedisKeyValueStoreService,
)


@dataclass
class AuthenticationServiceConfiguration(object):
    key_value_store_service: KeyValueStoreService


class AuthenticationServiceModule(Module):
    @provider
    def provide_authentication_service_module(
        self, configuration: AuthenticationServiceConfiguration
    ) -> AuthenticationService:
        return AuthenticationService(
            key_value_store_service=configuration.key_value_store_service,
        )


def _bind_authentication_service_configuration(binder: Binder):
    jwt_storage_uri = urlparse(
        os.getenv("DIOPTRA_JWT_STORAGE_URI", "local://127.0.0.1:50000")
    )
    uri_scheme: str = jwt_storage_uri.scheme.lower()

    if uri_scheme == "redis":
        key_value_store_service = RedisKeyValueStoreService()

    elif uri_scheme == "local":
        key_value_store_service = LocalKeyValueStoreService()

    configuration: AuthenticationServiceConfiguration = (
        AuthenticationServiceConfiguration(
            key_value_store_service=key_value_store_service,
        )
    )

    binder.bind(AuthenticationServiceConfiguration, to=configuration)


def bind_dependencies(binder: Binder) -> None:
    """Binds interfaces to implementations within the main application.

    Args:
        binder: A :py:class:`~injector.Binder` object.
    """
    _bind_authentication_service_configuration(binder)


def register_providers(modules: list[Callable[..., Any]]) -> None:
    """Registers type providers within the main application.

    Args:
        modules: A list of callables used for configuring the dependency injection
            environment.
    """
    modules.append(AuthenticationServiceModule)
