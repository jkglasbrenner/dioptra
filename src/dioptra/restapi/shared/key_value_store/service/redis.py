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

from datetime import datetime as DateTime
from datetime import timedelta as TimeDelta
from typing import Any, Callable, Iterator

from redis import Redis

from .base import KeyValueStoreService

try:
    from typing import TypedDict

except ImportError:
    from typing_extensions import TypedDict


class ScalarValue(TypedDict):
    value: bytes | str
    expires_on: DateTime


class RedisKeyValueStoreService(KeyValueStoreService):
    def __init__(self, redis: Redis) -> None:
        self._redis = redis

    def delete(self, keys: str | list[str]) -> int:
        if not isinstance(keys, str) and not isinstance(keys, list):
            raise RuntimeError

        if isinstance(keys, str):
            return self._redis.delete(keys)

        return self._redis.delete(*keys)

    def get(self, key: str) -> Any | None:
        return self._redis.get(name=key)

    def set(
        self,
        key: str,
        value: bytes | str | int | float,
        time_to_live: TimeDelta | None = None,
    ) -> bool | None:
        if not isinstance(value, (bytes, str, int, float)):
            raise RuntimeError

        return self._redis.set(name=key, value=value, ex=time_to_live)

    def expire(
        self,
        key: str,
        time: float | TimeDelta,
    ) -> bool:
        return self._redis.expire(name=key, time=time)

    def lpop(
        self,
        key: str,
        count: int | None = None,
    ) -> Any:
        return self._redis.lpop(name=key, count=count)

    def lpush(
        self,
        key: str,
        *values: bytes | str | int | float,
    ) -> int:
        return self._redis.lpush(key, *values)

    def rpop(
        self,
        key: str,
        count: int | None,
    ) -> Any:
        return self._redis.rpop(name=key, count=count)

    def rpush(
        self,
        key: str,
        *values: bytes | str | int | float,
    ) -> int:
        return self._redis.rpush(key, *values)

    def zadd(
        self,
        key: str,
        mapping: dict[str, bytes | str | int | float],
    ) -> int:
        return self._redis.zadd(name=key, mapping=mapping)

    def zremrangebyscore(
        self,
        key: str,
        min: int | float,
        max: int | float,
    ) -> int:
        return self._redis.zremrangebyscore(name=key, min=min, max=max)

    def zscan_iter(
        self,
        key: str,
        score_cast_func: type | Callable[[Any], Any] | None = None,
    ) -> Iterator[tuple[Any, Any]]:
        return self._redis.zscan_iter(name=key, score_cast_func=score_cast_func)
