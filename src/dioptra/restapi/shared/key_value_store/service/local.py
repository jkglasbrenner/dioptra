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

import atexit
import time
from datetime import datetime as DateTime
from datetime import timedelta as TimeDelta
from multiprocessing import Lock, Process
from multiprocessing.managers import BaseManager
from typing import Any, Callable, Iterator

from .base import KeyValueStoreService

try:
    from typing import TypedDict

except ImportError:
    from typing_extensions import TypedDict


class ScalarValue(TypedDict):
    value: bytes | str
    expires_on: DateTime


class LocalKeyValueStoreService(KeyValueStoreService):
    def __init__(
        self,
        authkey: bytes = b"dioptra-local-store",
        address: tuple[str, int] = ("127.0.0.1", 50000),
    ) -> None:
        self._address: tuple[str, int] = address
        self._authkey: bytes = authkey
        self._server: Process = _init_server(authkey=authkey, address=address)
        self._client: BaseManager = _init_client(authkey=authkey, address=address)

        atexit.register(self._shutdown)

    def delete(self, keys: str | list[str]) -> int:
        try:
            response: int = self._client.delete(keys=keys)._getvalue()
            return response

        except ConnectionRefusedError:
            return None

    def get(self, key: str) -> Any | None:
        try:
            response: Any | None = self._client.get(key=key)._getvalue()
            return response

        except ConnectionRefusedError:
            return None

    def set(
        self,
        key: str,
        value: bytes | str | int | float,
        time_to_live: TimeDelta | None = None,
    ) -> bool | None:
        try:
            response: bool = self._client.set(
                key=key, value=value, time_to_live=time_to_live
            )
            return response

        except ConnectionRefusedError:
            return None

    def expire(
        self,
        key: str,
        time: float | TimeDelta,
    ) -> bool:
        try:
            response: bool = self._client.expire(key=key, time=time)
            return response

        except ConnectionRefusedError:
            return None

    def lpop(
        self,
        key: str,
        count: int | None,
    ) -> Any:
        try:
            response: Any = self._client.lpop(key=key, count=count)
            return response

        except ConnectionRefusedError:
            return None

    def lpush(
        self,
        key: str,
        *values: bytes | str | int | float,
    ) -> int:
        try:
            response: Any = self._client.lpush(key=key, values=values)
            return response

        except ConnectionRefusedError:
            return None

    def rpop(
        self,
        key: str,
        count: int | None,
    ) -> Any:
        try:
            response: Any = self._client.rpop(key=key, count=count)
            return response

        except ConnectionRefusedError:
            return None

    def rpush(
        self,
        key: str,
        *values: bytes | str | int | float,
    ) -> int:
        try:
            response: Any = self._client.rpush(key=key, values=values)
            return response

        except ConnectionRefusedError:
            return None

    def zadd(
        self,
        key: str,
        mapping: dict[str, bytes | str | int | float],
    ) -> int:
        try:
            response: Any = self._client.rpush(key=key, mapping=mapping)
            return response

        except ConnectionRefusedError:
            return None

    def zremrangebyscore(
        self,
        key: str,
        min: int | float,
        max: int | float,
    ) -> int:
        try:
            response: Any = self._client.zremrangebyscore(key=key, min=min, max=max)
            return response

        except ConnectionRefusedError:
            return None

    def zscan_iter(
        self,
        key: str,
        score_cast_func: type | Callable[[Any], Any] | None = None,
    ) -> Iterator[tuple[Any, Any]]:
        cursor: int = 0
        response: tuple[Any, Any]

        while True:
            try:
                response, cursor = self._client.zscan_iter(key=key, cursor=cursor)

            except ConnectionRefusedError:
                break

            if score_cast_func is not None:
                response[1] = score_cast_func(response[1])

            yield response

            if cursor == 0:
                break

    def _shutdown(self):
        if not self._server.is_alive():
            return None

        try:
            self._client.stop_server()

        except ConnectionRefusedError:
            return None

        finally:
            while self._server.is_alive():
                time.sleep(0.5)

            self._server.close()


def _init_server(authkey: bytes, address: tuple[str, int]) -> Process:
    server_process = Process(
        target=_start_server,
        kwargs={"authkey": authkey, "address": address},
        daemon=True,
    )
    server_process.start()

    return server_process


def _init_client(authkey: bytes, address: tuple[str, int]) -> BaseManager:
    class LocalKeyValueStoreClient(BaseManager):
        pass

    LocalKeyValueStoreClient.register("delete")
    LocalKeyValueStoreClient.register("get")
    LocalKeyValueStoreClient.register("set")
    LocalKeyValueStoreClient.register("stop_server")
    client = LocalKeyValueStoreClient(address=address, authkey=authkey)

    num_retries: int = 0

    while True:
        time.sleep(1)

        try:
            client.connect()
            break

        except ConnectionRefusedError as e:
            num_retries += 1

            if num_retries > 3:
                raise e

    return client


def _start_server(authkey: bytes, address: tuple[str, int]) -> None:
    class LocalKeyValueStoreServer(BaseManager):
        pass

    data_store: dict[str, ScalarValue] = {}
    lock: Lock = Lock()

    def delete_data(keys: str | list[str]) -> int:
        return _delete(data_store=data_store, lock=lock, keys=keys)

    def get_data(key: str) -> bytes | str | None:
        return _get(data_store=data_store, lock=lock, key=key)

    def set_data(
        key: str, value: bytes | str | int | float, time_to_live: TimeDelta | None
    ) -> bytes | str | None:
        return _set(
            data_store=data_store,
            lock=lock,
            key=key,
            value=value,
            time_to_live=time_to_live,
        )

    def set_expire(key: str, time: int | float | TimeDelta) -> bool:
        return _expire(data_store=data_store, lock=lock, key=key, time=time)

    def lpop_data(key: str, count: int | None) -> Any:
        return _lpop(data_store=data_store, lock=lock, key=key, count=count)

    def lpush_data(key: str, *values: bytes | str | int | float) -> int:
        return _lpush(data_store=data_store, lock=lock, key=key, values=values)

    def rpop_data(key: str, count: int | None) -> Any:
        return _rpop(data_store=data_store, lock=lock, key=key, count=count)

    def rpush_data(key: str, *values: bytes | str | int | float) -> int:
        return _rpush(data_store=data_store, lock=lock, key=key, values=values)

    def zadd_data(key: str, mapping: dict[str, bytes | str | int | float]) -> int:
        return _zadd(data_store=data_store, lock=lock, key=key, mapping=mapping)

    def zremrangebyscore_data(key: str, min: int | float, max: int | float) -> int:
        return _zremrangebyscore(
            data_store=data_store, lock=lock, key=key, min=min, max=max
        )

    def zscan_iter_data(key: str, cursor: int) -> tuple[tuple[Any, Any], int]:
        return _zscan_iter(data_store=data_store, key=key, cursor=cursor)

    def stop_server():
        server.stop_event.set()

    LocalKeyValueStoreServer.register("delete", callable=delete_data)
    LocalKeyValueStoreServer.register("get", callable=get_data)
    LocalKeyValueStoreServer.register("set", callable=set_data)
    LocalKeyValueStoreServer.register("expire", callable=set_expire)
    LocalKeyValueStoreServer.register("lpop", callable=lpop_data)
    LocalKeyValueStoreServer.register("lpush", callable=lpush_data)
    LocalKeyValueStoreServer.register("rpop", callable=rpop_data)
    LocalKeyValueStoreServer.register("rpush", callable=rpush_data)
    LocalKeyValueStoreServer.register("zadd", callable=zadd_data)
    LocalKeyValueStoreServer.register(
        "zremrangebyscore", callable=zremrangebyscore_data
    )
    LocalKeyValueStoreServer.register("zscan_iter", callable=zscan_iter_data)
    LocalKeyValueStoreServer.register("stop_server", callable=stop_server)
    manager = LocalKeyValueStoreServer(address=address, authkey=authkey)

    try:
        server = manager.get_server()

    except OSError:
        return None

    try:
        server.serve_forever()

    finally:
        time.sleep(1)


def _delete(
    data_store: dict[str, ScalarValue], lock: Lock, keys: str | list[str]
) -> int:
    if not isinstance(keys, str) and not isinstance(keys, list):
        raise RuntimeError

    with lock:
        if isinstance(keys, str) and data_store.get(keys) is None:
            return 0

        if isinstance(keys, str):
            del data_store[keys]
            return 1

        existing_keys: list[str] = [key for key in keys if key in data_store]

        for key in existing_keys:
            del data_store[key]

        return len(existing_keys)


def _get(
    data_store: dict[str, ScalarValue], lock: Lock, key: str
) -> bytes | str | None:
    with lock:
        scalar_value: ScalarValue | None = data_store.get(key)

        if scalar_value is None:
            return None

        if (
            scalar_value["expires_on"] is not None
            and DateTime.utcnow() > scalar_value["expires_on"]
        ):
            del data_store[key]
            return None

        return scalar_value["value"]


def _set(
    data_store: dict[str, ScalarValue],
    lock: Lock,
    key: str,
    value: bytes | str | int | float,
    time_to_live: TimeDelta | None = None,
) -> bool:
    if not isinstance(value, (bytes, str, int, float)):
        raise RuntimeError

    if isinstance(value, (int, float)):
        value = str(value)

    expires_on: DateTime = (
        DateTime.utcnow() + time_to_live if time_to_live is not None else None
    )

    with lock:
        data_store[key] = ScalarValue(value=value, expires_on=expires_on)

    return True


def _expire(
    data_store: dict[str, ScalarValue],
    lock: Lock,
    key: str,
    time: int | float | TimeDelta,
) -> bool:
    if not isinstance(time, (int, float, TimeDelta)):
        raise RuntimeError

    if isinstance(time, (int, float)):
        time = TimeDelta(seconds=time)

    expires_on: DateTime = DateTime.utcnow() + time

    with lock:
        if data_store.get(key) is None:
            return False

        data_store[key]["expires_on"] = expires_on

    return True


def _lpop(data_store, lock: Lock, key: str, count: int | None) -> Any:
    pass


def _lpush(
    data_store, lock: Lock, key: str, values: tuple[bytes | str | int | float, ...]
) -> int:
    pass


def _rpop(data_store, lock: Lock, key: str, count: int | None) -> Any:
    pass


def _rpush(
    data_store, lock: Lock, key: str, values: tuple[bytes | str | int | float, ...]
) -> int:
    pass


def _zadd(
    data_store, lock: Lock, key: str, mapping: dict[str, bytes | str | int | float]
) -> int:
    pass


def _zremrangebyscore(
    data_store, lock: Lock, key: str, min: int | float, max: int | float
) -> int:
    pass


def _zscan_iter(
    data_store, lock: Lock, key: str, cursor: int
) -> tuple[tuple[Any, Any], int]:
    pass
