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
from __future__ import annotations

from abc import ABCMeta, abstractmethod
from datetime import timedelta as TimeDelta
from typing import Any, Callable, Iterator


class KeyValueStoreService(metaclass=ABCMeta):
    @abstractmethod
    def delete(self, keys: str | list[str]) -> int:
        raise NotImplementedError

    @abstractmethod
    def get(self, key: str) -> Any | None:
        raise NotImplementedError

    @abstractmethod
    def set(
        self,
        key: str,
        value: bytes | str | int | float,
        time_to_live: TimeDelta | None,
    ) -> bool | None:
        raise NotImplementedError

    @abstractmethod
    def expire(
        self,
        key: str,
        time: float | TimeDelta,
    ) -> bool:
        raise NotImplementedError

    @abstractmethod
    def lpop(
        self,
        key: str,
        count: int | None,
    ) -> Any:
        raise NotImplementedError

    @abstractmethod
    def lpush(
        self,
        key: str,
        *values: bytes | str | int | float,
    ) -> int:
        raise NotImplementedError

    @abstractmethod
    def rpop(
        self,
        key: str,
        count: int | None,
    ) -> Any:
        raise NotImplementedError

    @abstractmethod
    def rpush(
        self,
        key: str,
        *values: bytes | str | int | float,
    ) -> int:
        raise NotImplementedError

    @abstractmethod
    def zadd(
        self,
        key: str,
        mapping: dict[str, bytes | str | int | float],
    ) -> int:
        raise NotImplementedError

    @abstractmethod
    def zremrangebyscore(
        self,
        key: str,
        min: int | float,
        max: int | float,
    ) -> int:
        raise NotImplementedError

    @abstractmethod
    def zscan_iter(
        self,
        key: str,
        score_cast_func: type | Callable[[Any], Any],
    ) -> Iterator[tuple[Any, Any]]:
        raise NotImplementedError
