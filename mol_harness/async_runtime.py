"""Single-process asynchronous production runtime for the MoL proxy.

The protocol conversion and state models remain in the existing modules. This
module owns production HTTP serving, Gateway I/O, admission control, response
state publication, and routing-key lifecycle management.
"""
from __future__ import annotations

import asyncio
import hmac
import json
import os
import time
import uuid
from collections import Counter, OrderedDict, deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Awaitable, Callable

import aiohttp
from prometheus_client import CollectorRegistry, Counter as PromCounter
from prometheus_client import Gauge, Histogram, start_http_server

from . import proxy as core
from . import responses as responses_api
from .session import ConvoState, Task, extract_tool_results, split_system_and_turn


MAX_CONNECTIONS = max(1, int(os.environ.get("MOL_MAX_CONNECTIONS", "4096")))
if "MOL_MAX_INFLIGHT_REQUESTS" in os.environ:
    MAX_INFLIGHT = max(1, int(os.environ["MOL_MAX_INFLIGHT_REQUESTS"]))
else:
    MAX_INFLIGHT = max(
        1, int(os.environ.get("MOL_MAX_CONCURRENT_REQUESTS", "256")))
MAX_QUEUED = max(0, int(os.environ.get("MOL_MAX_QUEUED_REQUESTS", "256")))
QUEUE_TIMEOUT_S = max(0.001, float(os.environ.get("MOL_QUEUE_TIMEOUT_S", "1")))
KEEPALIVE_TIMEOUT_S = max(
    1, int(float(os.environ.get("MOL_KEEPALIVE_TIMEOUT_S", "15"))))
SSE_KEEPALIVE_INTERVAL_S = max(
    0.0, float(os.environ.get("MOL_SSE_KEEPALIVE_INTERVAL_S", "10")))
SSE_KEEPALIVE_FRAME = b": keepalive\n\n"
UPSTREAM_MAX_CONNECTIONS = max(
    1, int(os.environ.get("MOL_UPSTREAM_MAX_CONNECTIONS", "512")))
UPSTREAM_MAX_KEEPALIVE = max(
    1, int(os.environ.get("MOL_UPSTREAM_MAX_KEEPALIVE", "256")))
STATE_EXECUTOR_WORKERS = max(
    1, int(os.environ.get("MOL_STATE_EXECUTOR_WORKERS", "4")))
RELEASE_WORKERS = max(1, int(os.environ.get("MOL_RELEASE_WORKERS", "8")))
RELEASE_QUEUE_SIZE = max(
    RELEASE_WORKERS, int(os.environ.get("MOL_RELEASE_QUEUE_SIZE", "4096")))
RELEASE_BATCH_SIZE = max(
    1, int(os.environ.get("MOL_RELEASE_BATCH_SIZE", "256")))
RELEASE_TIMEOUT_S = max(
    1.0, float(os.environ.get("MOL_RELEASE_TIMEOUT_S", "10")))
METRICS_HOST = os.environ.get("MOL_METRICS_HOST", "127.0.0.1")
METRICS_PORT = max(0, int(os.environ.get("MOL_METRICS_PORT", "8201")))
DRAIN_TIMEOUT_S = max(1.0, float(os.environ.get("MOL_DRAIN_TIMEOUT_S", "600")))
STORE_CLEANUP_INTERVAL_S = max(
    1.0, float(os.environ.get("MOL_STORE_CLEANUP_INTERVAL_S", "5")))

if "MOL_MAX_CONCURRENT_REQUESTS" in os.environ \
        and "MOL_MAX_INFLIGHT_REQUESTS" not in os.environ:
    # Keep deployments from silently changing their active limit while they
    # migrate to the explicit async setting.
    core._log(
        "MOL_MAX_CONCURRENT_REQUESTS is deprecated; use "
        "MOL_MAX_INFLIGHT_REQUESTS")


class ClientDisconnected(ConnectionError):
    pass


class QueueOverloaded(RuntimeError):
    pass


class ResponseStoreCapacityError(RuntimeError):
    pass


class UpstreamHTTPError(RuntimeError):
    def __init__(self, status: int, body: dict):
        super().__init__(f"upstream returned HTTP {status}")
        self.status = status
        self.body = body


class RuntimeMetrics:
    def __init__(self) -> None:
        self.registry = CollectorRegistry()
        self.active = Gauge(
            "mol_proxy_active_requests", "Active orchestrations",
            registry=self.registry)
        self.queued = Gauge(
            "mol_proxy_queued_requests", "Queued orchestrations",
            registry=self.registry)
        self.queue_wait = Histogram(
            "mol_proxy_queue_wait_seconds", "Admission queue wait",
            buckets=(.001, .005, .01, .025, .05, .1, .25, .5, 1, 2),
            registry=self.registry)
        self.requests = PromCounter(
            "mol_proxy_requests_total", "Public requests", ("path", "status"),
            registry=self.registry)
        self.request_latency = Histogram(
            "mol_proxy_request_seconds", "Public request latency", ("path",),
            buckets=(.01, .025, .05, .1, .2, .3, .4, .5, 1, 2, 5, 10, 30, 60),
            registry=self.registry)
        self.hop_latency = Histogram(
            "mol_proxy_hop_seconds", "Gateway hop latency", ("hop", "mode"),
            buckets=(.005, .01, .025, .05, .1, .2, .5, 1, 2, 5, 10, 30, 60),
            registry=self.registry)
        self.upstream_errors = PromCounter(
            "mol_proxy_upstream_errors_total", "Gateway request errors",
            ("hop", "kind"), registry=self.registry)
        self.state_entries = Gauge(
            "mol_proxy_response_state_entries", "Committed Responses states",
            registry=self.registry)
        self.state_bytes = Gauge(
            "mol_proxy_response_state_bytes", "Approximate Responses state bytes",
            registry=self.registry)
        self.pending_states = Gauge(
            "mol_proxy_pending_response_states", "Pending Responses publications",
            registry=self.registry)
        self.routing_roots = Gauge(
            "mol_proxy_routing_roots", "Locally referenced Gateway routing roots",
            registry=self.registry)
        self.release_queue = Gauge(
            "mol_proxy_release_queue", "Queued Gateway routing-key releases",
            registry=self.registry)
        self.release_failures = PromCounter(
            "mol_proxy_release_failures_total", "Exhausted routing-key releases",
            registry=self.registry)
        self.event_loop_lag = Histogram(
            "mol_proxy_event_loop_lag_seconds", "Observed event-loop scheduling lag",
            buckets=(.001, .0025, .005, .01, .025, .05, .1, .25, .5, 1),
            registry=self.registry)
        self._server: Any = None

    def start(self) -> None:
        if METRICS_PORT and self._server is None:
            self._server = start_http_server(
                METRICS_PORT, addr=METRICS_HOST, registry=self.registry)

    def close(self) -> None:
        if self._server is None:
            return
        server = self._server[0] if isinstance(self._server, tuple) else self._server
        thread = (self._server[1]
                  if isinstance(self._server, tuple) and len(self._server) > 1
                  else None)
        with suppress(Exception):
            server.shutdown()
        with suppress(Exception):
            server.server_close()
        if thread is not None:
            with suppress(Exception):
                thread.join(timeout=2)
        self._server = None


class AdmissionLease:
    def __init__(self, controller: "AdmissionController") -> None:
        self.controller = controller
        self.released = False

    async def __aenter__(self) -> "AdmissionLease":
        return self

    async def __aexit__(self, *_args: object) -> None:
        if not self.released:
            self.released = True
            await self.controller.release()


class AdmissionController:
    """FIFO admission by active orchestration, independent of TCP connections."""

    def __init__(self, metrics: RuntimeMetrics) -> None:
        self.metrics = metrics
        self.max_active = MAX_INFLIGHT
        self.max_queued = MAX_QUEUED
        self.timeout = QUEUE_TIMEOUT_S
        self.active = 0
        self.waiters: deque[asyncio.Future[None]] = deque()
        self.lock = asyncio.Lock()

    def _update_metrics(self) -> None:
        self.metrics.active.set(self.active)
        self.metrics.queued.set(len(self.waiters))

    async def acquire(self) -> AdmissionLease:
        started = time.monotonic()
        async with self.lock:
            if self.active < self.max_active:
                self.active += 1
                self._update_metrics()
                self.metrics.queue_wait.observe(0)
                return AdmissionLease(self)
            if len(self.waiters) >= self.max_queued:
                raise QueueOverloaded("orchestration queue is full")
            waiter = asyncio.get_running_loop().create_future()
            self.waiters.append(waiter)
            self._update_metrics()
        try:
            await asyncio.wait_for(asyncio.shield(waiter), self.timeout)
        except BaseException:
            owns_slot = False
            async with self.lock:
                try:
                    self.waiters.remove(waiter)
                except ValueError:
                    owns_slot = waiter.done() and not waiter.cancelled()
                self._update_metrics()
            if owns_slot:
                await self.release()
            raise QueueOverloaded("orchestration queue wait timed out")
        self.metrics.queue_wait.observe(time.monotonic() - started)
        return AdmissionLease(self)

    async def release(self) -> None:
        async with self.lock:
            while self.waiters:
                waiter = self.waiters.popleft()
                if waiter.done():
                    continue
                waiter.set_result(None)
                self._update_metrics()
                return
            self.active = max(0, self.active - 1)
            self._update_metrics()


class AsyncGatewayClient:
    def __init__(self, metrics: RuntimeMetrics) -> None:
        self.metrics = metrics
        self.client: aiohttp.ClientSession | None = None
        self.release_client: aiohttp.ClientSession | None = None

    async def start(self) -> None:
        if self.client is not None:
            return
        # aiohttp keeps socket parsing and connection bookkeeping out of the
        # Python hot path at high fan-out while the public ASGI behavior stays
        # unchanged.
        connector = aiohttp.TCPConnector(
            limit=UPSTREAM_MAX_CONNECTIONS,
            limit_per_host=UPSTREAM_MAX_KEEPALIVE,
            keepalive_timeout=KEEPALIVE_TIMEOUT_S,
            force_close=False,
        )
        timeout = aiohttp.ClientTimeout(
            total=None,
            connect=2.0,
            sock_connect=2.0,
            sock_read=core.HOP_TIMEOUT,
        )
        self.client = aiohttp.ClientSession(
            base_url=core.UPSTREAM.rstrip("/"),
            connector=connector,
            timeout=timeout,
            trust_env=False,
        )
        self.release_client = aiohttp.ClientSession(
            base_url=core.UPSTREAM.rstrip("/"),
            connector=aiohttp.TCPConnector(
                limit=RELEASE_WORKERS,
                limit_per_host=RELEASE_WORKERS,
                keepalive_timeout=KEEPALIVE_TIMEOUT_S,
                force_close=False,
            ),
            timeout=aiohttp.ClientTimeout(
                total=None, connect=2.0, sock_connect=2.0,
                sock_read=RELEASE_TIMEOUT_S),
            trust_env=False,
        )

    async def close(self) -> None:
        if self.release_client is not None:
            await self.release_client.close()
            self.release_client = None
        if self.client is not None:
            await self.client.close()
            self.client = None

    def _client(self) -> aiohttp.ClientSession:
        if self.client is None:
            raise RuntimeError("Gateway client is not started")
        return self.client

    def _release_client(self) -> aiohttp.ClientSession:
        if self.release_client is None:
            raise RuntimeError("Gateway release client is not started")
        return self.release_client

    async def is_ready(self) -> bool:
        """Return whether Gateway reports at least one healthy Engine worker."""
        try:
            async with self._client().get(
                    "/readiness", timeout=2.0) as response:
                await response.read()
                return response.status == 200
        except (aiohttp.ClientError, asyncio.TimeoutError):
            return False

    async def post_json(
        self, path: str, payload: dict, *, hop: str,
    ) -> tuple[int, dict]:
        started = time.monotonic()
        try:
            async with self._client().post(
                    path, json=payload,
                    headers=core._routing_headers(
                        hop if hop == "route" else None)) as response:
                try:
                    body = await response.json(content_type=None)
                except (ValueError, json.JSONDecodeError):
                    body = {"error": await response.text()}
                return response.status, body
        except asyncio.CancelledError:
            raise
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            self.metrics.upstream_errors.labels(hop, type(exc).__name__).inc()
            return -1, {"error": f"{type(exc).__name__}: {exc}"}
        finally:
            self.metrics.hop_latency.labels(hop, "nonstream").observe(
                time.monotonic() - started)

    async def stream_chat(self, payload: dict) -> AsyncIterator[dict]:
        started = time.monotonic()
        saw_done = False
        try:
            async with self._client().post(
                    "/v1/chat/completions", json=payload,
                    headers=core._routing_headers()) as response:
                if response.status != 200:
                    raw = await response.read()
                    try:
                        body = json.loads(raw.decode("utf-8", "replace"))
                    except (ValueError, json.JSONDecodeError):
                        body = {"error": raw.decode("utf-8", "replace")}
                    raise UpstreamHTTPError(response.status, body)
                data_lines: list[str] = []
                async for raw_line in response.content:
                    line = raw_line.decode("utf-8", "replace").rstrip("\r\n")
                    if not line:
                        if not data_lines:
                            continue
                        payload_text = "\n".join(data_lines)
                        data_lines.clear()
                        if payload_text == "[DONE]":
                            saw_done = True
                            break
                        try:
                            yield json.loads(payload_text)
                        except (ValueError, json.JSONDecodeError):
                            continue
                    elif line.startswith("data:"):
                        data_lines.append(line[5:].lstrip())
                if data_lines and not saw_done:
                    payload_text = "\n".join(data_lines)
                    if payload_text == "[DONE]":
                        saw_done = True
                    else:
                        with suppress(ValueError, json.JSONDecodeError):
                            yield json.loads(payload_text)
                if not saw_done:
                    raise RuntimeError("upstream SSE ended before [DONE]")
        except asyncio.CancelledError:
            raise
        except UpstreamHTTPError:
            raise
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            self.metrics.upstream_errors.labels(
                "answer", type(exc).__name__).inc()
            raise RuntimeError(f"{type(exc).__name__}: {exc}") from exc
        finally:
            self.metrics.hop_latency.labels("answer", "stream").observe(
                time.monotonic() - started)

    async def delete_routing_keys(self, routing_keys: list[str]) -> None:
        if not routing_keys:
            return
        async with self._release_client().delete(
                "/_internal/routing-keys",
                json={"routing_keys": routing_keys}) as response:
            if response.status not in (200, 204, 404):
                body = await response.text()
                raise UpstreamHTTPError(
                    response.status, {"error": body[:200]})
            if response.status != 404:
                return

        # Rolling upgrades may briefly pair the new Proxy with an older
        # Gateway. Preserve cleanup through the legacy single-key endpoint.
        async def delete_one(routing_key: str) -> None:
            async with self._release_client().delete(
                    "/_internal/routing-key",
                    headers={"X-SMG-Routing-Key": routing_key}) as response:
                if response.status not in (200, 204, 404):
                    body = await response.text()
                    raise UpstreamHTTPError(
                        response.status, {"error": body[:200]})

        await asyncio.gather(*(delete_one(key) for key in routing_keys))


class RoutingReleaseManager:
    def __init__(
        self, gateway: AsyncGatewayClient, metrics: RuntimeMetrics,
    ) -> None:
        self.gateway = gateway
        self.metrics = metrics
        self.queue: asyncio.Queue[str] = asyncio.Queue(RELEASE_QUEUE_SIZE)
        self.overflow: OrderedDict[str, None] = OrderedDict()
        self.max_overflow = max(RELEASE_QUEUE_SIZE, 8192)
        self.workers: list[asyncio.Task[None]] = []

    async def start(self) -> None:
        if not self.workers:
            self.workers = [
                asyncio.create_task(self._worker(), name=f"mol-release-{index}")
                for index in range(RELEASE_WORKERS)
            ]

    async def schedule(self, routing_key: str | None) -> None:
        if not routing_key:
            return
        # A request must never wait for a Gateway DELETE. The queue is a
        # bounded durability buffer; an overflow is retried by workers and the
        # Gateway's idle TTL remains the final crash-recovery backstop.
        try:
            self.queue.put_nowait(routing_key)
        except asyncio.QueueFull:
            if len(self.overflow) < self.max_overflow:
                self.overflow.setdefault(routing_key, None)
            else:
                self.metrics.release_failures.inc()
                core._log("routing release overflow exhausted")
        self._update_queue_metric()

    def _update_queue_metric(self) -> None:
        self.metrics.release_queue.set(
            self.queue.qsize() + len(self.overflow))

    def _refill_queue(self) -> None:
        while self.overflow and not self.queue.full():
            routing_key, _ = self.overflow.popitem(last=False)
            self.queue.put_nowait(routing_key)
        self._update_queue_metric()

    async def _worker(self) -> None:
        while True:
            routing_keys = [await self.queue.get()]
            while len(routing_keys) < RELEASE_BATCH_SIZE:
                try:
                    routing_keys.append(self.queue.get_nowait())
                except asyncio.QueueEmpty:
                    break
            self._update_queue_metric()
            try:
                for attempt in range(3):
                    try:
                        await asyncio.wait_for(
                            self.gateway.delete_routing_keys(routing_keys),
                            RELEASE_TIMEOUT_S)
                        break
                    except asyncio.CancelledError:
                        raise
                    except Exception:
                        if attempt == 2:
                            self.metrics.release_failures.inc(
                                len(routing_keys))
                            core._log(
                                "routing key release batch exhausted "
                                f"count={len(routing_keys)} "
                                f"first={routing_keys[0][:16]}")
                        else:
                            await asyncio.sleep(0.1 * (2 ** attempt))
            finally:
                for _ in routing_keys:
                    self.queue.task_done()
                self._refill_queue()

    async def drain(self) -> None:
        deadline = time.monotonic() + DRAIN_TIMEOUT_S
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise asyncio.TimeoutError
            await asyncio.wait_for(self.queue.join(), remaining)
            if not self.overflow:
                return
            self._refill_queue()

    async def close(self) -> None:
        with suppress(asyncio.TimeoutError):
            await self.drain()
        for worker in self.workers:
            worker.cancel()
        for worker in self.workers:
            with suppress(asyncio.CancelledError):
                await worker
        self.workers.clear()


@dataclass
class PendingResponse:
    event: asyncio.Event = field(default_factory=asyncio.Event)
    approx_bytes: int = 0


@dataclass
class AsyncStoredResponse:
    resp_id: str
    state: ConvoState
    parent_id: str | None
    routing_root_id: str
    approx_bytes: int
    touched_at: float = field(default_factory=time.monotonic)


class AsyncResponseStore:
    """Memory-only immutable response snapshots with O(1) capacity counters."""

    def __init__(
        self, releases: RoutingReleaseManager, metrics: RuntimeMetrics,
    ) -> None:
        self.releases = releases
        self.metrics = metrics
        self.entries: OrderedDict[str, AsyncStoredResponse] = OrderedDict()
        self.pending: dict[str, PendingResponse] = {}
        self.total_bytes = 0
        self.pending_bytes = 0
        self.root_refs: Counter[str] = Counter()
        self.lock = asyncio.Lock()
        self.executor = ThreadPoolExecutor(
            max_workers=STATE_EXECUTOR_WORKERS,
            thread_name_prefix="mol-state")
        self.max_entries = max(
            1, int(os.environ.get("MOL_MAX_RESPONSES", "5000")))
        self.max_bytes = max(1, int(os.environ.get(
            "MOL_MAX_RESPONSE_STATE_BYTES", str(512 * 1024 * 1024))))
        self.max_pending = max(
            1, int(os.environ.get("MOL_MAX_PENDING_RESPONSES", "256")))
        self.max_pending_bytes = max(self.max_bytes, int(os.environ.get(
            "MOL_MAX_PENDING_RESPONSE_STATE_BYTES", str(self.max_bytes))))
        self.pending_wait = max(
            1.0, float(os.environ.get("MOL_PENDING_WAIT_S", "60")))

    async def state_size(self, state: ConvoState) -> int:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self.executor, responses_api._state_approx_bytes, state)

    async def fork(self, state: ConvoState) -> ConvoState:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self.executor, responses_api._fork_state, state)

    def _update_metrics_locked(self) -> None:
        self.metrics.state_entries.set(len(self.entries))
        self.metrics.state_bytes.set(self.total_bytes)
        self.metrics.pending_states.set(len(self.pending))
        self.metrics.routing_roots.set(len(self.root_refs))

    def _dec_root_locked(self, root: str | None) -> str | None:
        if not root:
            return None
        remaining = self.root_refs.get(root, 0) - 1
        if remaining <= 0:
            self.root_refs.pop(root, None)
            return root
        self.root_refs[root] = remaining
        return None

    async def resolve(
        self, previous_response_id: str | None,
    ) -> tuple[ConvoState | None, str | None, str | None, int]:
        if not previous_response_id:
            return ConvoState(convo_id="resp_root"), None, None, 200
        deadline = time.monotonic() + self.pending_wait
        while True:
            async with self.lock:
                pending = self.pending.get(previous_response_id)
                if pending is None:
                    stored = self.entries.get(previous_response_id)
                    if stored is None:
                        return None, None, None, 404
                    stored.touched_at = time.monotonic()
                    self.entries.move_to_end(previous_response_id)
                    root = stored.routing_root_id
                    self.root_refs[root] += 1  # in-flight lease
                    source = stored.state
                    self._update_metrics_locked()
                    break
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None, previous_response_id, None, 409
            try:
                await asyncio.wait_for(pending.event.wait(), remaining)
            except asyncio.TimeoutError:
                return None, previous_response_id, None, 409
        try:
            state = await self.fork(source)
        except BaseException:
            await self.release_lease(root)
            raise
        return state, previous_response_id, root, 200

    async def reserve(self, resp_id: str, approx_bytes: int) -> None:
        if approx_bytes > self.max_bytes:
            raise ResponseStoreCapacityError(
                "response state exceeds the configured byte budget")
        async with self.lock:
            if (len(self.pending) >= self.max_pending
                    or self.pending_bytes + approx_bytes > self.max_pending_bytes):
                raise ResponseStoreCapacityError(
                    "response publication capacity is temporarily exhausted")
            self.pending[resp_id] = PendingResponse(approx_bytes=approx_bytes)
            self.pending_bytes += approx_bytes
            self._update_metrics_locked()

    async def resize_pending(self, resp_id: str, approx_bytes: int) -> None:
        if approx_bytes > self.max_bytes:
            raise ResponseStoreCapacityError(
                "response state exceeds the configured byte budget")
        async with self.lock:
            pending = self.pending.get(resp_id)
            if pending is None:
                raise RuntimeError("response publication reservation was lost")
            projected = self.pending_bytes - pending.approx_bytes + approx_bytes
            if projected > self.max_pending_bytes:
                raise ResponseStoreCapacityError(
                    "response publication capacity is temporarily exhausted")
            self.pending_bytes = projected
            pending.approx_bytes = approx_bytes
            self._update_metrics_locked()

    def _evict_limits_locked(self) -> list[str]:
        release: list[str] = []
        while self.entries and (
            len(self.entries) > self.max_entries
            or self.total_bytes > self.max_bytes
        ):
            _, victim = self.entries.popitem(last=False)
            self.total_bytes -= victim.approx_bytes
            root = self._dec_root_locked(victim.routing_root_id)
            if root:
                release.append(root)
        return release

    async def commit(
        self, resp_id: str, state: ConvoState, parent_id: str | None,
        routing_root_id: str, approx_bytes: int,
    ) -> None:
        releases: list[str]
        async with self.lock:
            pending = self.pending.pop(resp_id, None)
            if pending is None:
                raise RuntimeError("response publication reservation was lost")
            self.pending_bytes -= pending.approx_bytes
            self.entries[resp_id] = AsyncStoredResponse(
                resp_id, state, parent_id, routing_root_id, approx_bytes)
            self.entries.move_to_end(resp_id)
            self.total_bytes += approx_bytes
            self.root_refs[routing_root_id] += 1
            releases = self._evict_limits_locked()
            pending.event.set()
            self._update_metrics_locked()
        for root in releases:
            await self.releases.schedule(root)

    async def abort(self, resp_id: str) -> None:
        async with self.lock:
            pending = self.pending.pop(resp_id, None)
            if pending is not None:
                self.pending_bytes -= pending.approx_bytes
                pending.event.set()
            self._update_metrics_locked()

    async def release_lease(self, routing_root_id: str | None) -> None:
        release = None
        async with self.lock:
            release = self._dec_root_locked(routing_root_id)
            self._update_metrics_locked()
        if release:
            await self.releases.schedule(release)

    async def cleanup(self) -> None:
        now = time.monotonic()
        releases: list[str] = []
        async with self.lock:
            for resp_id, stored in list(self.entries.items()):
                if now - stored.touched_at <= core.CONVO_TTL_S:
                    continue
                self.entries.pop(resp_id, None)
                self.total_bytes -= stored.approx_bytes
                root = self._dec_root_locked(stored.routing_root_id)
                if root:
                    releases.append(root)
            self._update_metrics_locked()
        for root in releases:
            await self.releases.schedule(root)

    async def close(self) -> None:
        releases: list[str] = []
        async with self.lock:
            for stored in self.entries.values():
                root = self._dec_root_locked(stored.routing_root_id)
                if root:
                    releases.append(root)
            self.entries.clear()
            self.total_bytes = 0
            releases.extend(root for root in self.root_refs if root not in releases)
            self.root_refs.clear()
            for pending in self.pending.values():
                pending.event.set()
            self.pending.clear()
            self.pending_bytes = 0
            self._update_metrics_locked()
        for root in releases:
            await self.releases.schedule(root)
        self.executor.shutdown(wait=True, cancel_futures=True)


StreamCallback = Callable[[tuple[str, Any]], Awaitable[None]]


class AsyncOrchestrator:
    """Async implementation of the shared route -> answer -> summary core."""

    def __init__(self, gateway: AsyncGatewayClient) -> None:
        self.gateway = gateway
        self.router_instruction = core._ROUTER._router_instruction()

    async def route_chat(
        self, router_prompt: str, model: str, max_tokens: int,
    ) -> str:
        prefill = "model_id="
        stripped = router_prompt.rstrip()
        user_content = (
            stripped[:-len(prefill)].rstrip()
            if stripped.endswith(prefill) else router_prompt)
        payload = {
            "model": model,
            "messages": [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": prefill},
            ],
            "max_tokens": max_tokens,
            "max_completion_tokens": max_tokens,
            "temperature": 0.0,
            "stream": False,
            "continue_final_message": True,
            "add_generation_prompt": False,
            "stop": None,
            **core._NO_THINK_OPTIONS,
            "structured_outputs": {"choice": list(core.CANONICAL_ROUTES)},
        }
        status, body = await self.gateway.post_json(
            "/v1/chat/completions", payload, hop="route")
        if status != 200:
            core._log(f"router chat status={status} body={str(body)[:200]}")
            raise core.RoutingUpstreamError(status, body)
        try:
            message = core._choice_message(body)
            text = core._msg_text(message)
            if not text.strip():
                text = (
                    message.get("reasoning")
                    or message.get("reasoning_content") or "")
            text = text.strip()
            if not text:
                raise core.RoutingError(
                    "upstream router returned an empty label")
            return text
        except core.RoutingError:
            raise
        except Exception as exc:
            raise core.RoutingError(
                f"invalid upstream router response: {type(exc).__name__}"
            ) from exc

    async def chat(
        self, messages: list[dict], model: str, max_tokens: int | None,
        tools: list | None = None, answer_options: dict | None = None,
        *, hop: str = "answer",
    ) -> tuple[int, dict]:
        payload = core._chat_payload(
            messages, model, max_tokens, tools, stream=False,
            answer_options=answer_options)
        return await self.gateway.post_json(
            "/v1/chat/completions", payload, hop=hop)

    async def stream_chat_accum(
        self, messages: list[dict], model: str, max_tokens: int | None,
        tools: list | None, on_chunk: StreamCallback | None,
        on_open: Callable[[], Awaitable[None]] | None = None,
        answer_options: dict | None = None,
    ) -> tuple[int, dict]:
        payload = core._chat_payload(
            messages, model, max_tokens, tools, stream=True,
            answer_options=answer_options)
        iterator = self.gateway.stream_chat(payload).__aiter__()
        acc = core._StreamAccum()
        buffered_tool_chunks: list[dict] = []

        async def next_chunk() -> dict | None:
            try:
                return await anext(iterator)
            except StopAsyncIteration:
                return None
            except UpstreamHTTPError as exc:
                raise exc
            except (asyncio.CancelledError, ClientDisconnected):
                raise
            except Exception as exc:
                core._log(
                    "stream chat error: "
                    f"{type(exc).__name__}: {str(exc)[:120]}")
                raise RuntimeError("answer_stream_error") from exc

        try:
            try:
                first = await next_chunk()
            except UpstreamHTTPError as exc:
                return exc.status, exc.body
            except RuntimeError:
                return -1, {"error": "answer_stream_error"}
            if first is not None and on_open is not None:
                await on_open()
            chunk = first
            while chunk is not None:
                try:
                    acc.feed(chunk)
                except Exception as exc:
                    core._log(
                        "stream chunk error: "
                        f"{type(exc).__name__}: {str(exc)[:120]}")
                    return 502, core._invalid_upstream_response()
                public_chunk, tool_chunk = core._split_tool_delta_chunk(chunk)
                if tool_chunk is not None:
                    buffered_tool_chunks.append(tool_chunk)
                if on_chunk is not None and public_chunk is not None:
                    await on_chunk(("chunk", public_chunk))
                try:
                    chunk = await next_chunk()
                except UpstreamHTTPError as exc:
                    return exc.status, exc.body
                except RuntimeError:
                    return -1, {"error": "answer_stream_error"}
        finally:
            with suppress(Exception):
                await iterator.aclose()
        if acc.finish is None:
            core._log(
                "stream chat error: upstream stream ended without finish_reason")
            return -1, {"error": "answer_stream_error"}
        if acc.finish == "tool_calls":
            signature = core._tool_call_signature(acc.tool_calls)
            call_ids = [entry[0] for entry in signature]
            if not signature or len(call_ids) != len(set(call_ids)):
                return 502, core._invalid_upstream_response()
        body = acc.response(model)
        if acc.finish in ("length", "content_filter"):
            body["choices"][0]["message"].pop("tool_calls", None)
        else:
            body["_mol_buffered_tool_chunks"] = (
                core._canonical_tool_delta_chunks(buffered_tool_chunks, acc))
        return 200, body

    async def call_answer(
        self, messages: list[dict], adapter: str, max_tokens: int | None,
        tools: list | None, stream_cb: StreamCallback | None,
        stream_start_cb: Callable[[], Awaitable[None]] | None = None,
        answer_options: dict | None = None,
    ) -> tuple[int, dict]:
        if stream_cb is None:
            return await self.chat(
                messages, adapter, max_tokens, tools, answer_options)
        return await self.stream_chat_accum(
            messages, adapter, max_tokens, tools, stream_cb,
            stream_start_cb, answer_options)

    async def route(
        self, state: Any, user_text: str,
    ) -> tuple[str, str, dict]:
        del state
        model_route = None
        if core.USE_MODEL_ROUTER and core.ENTRY_ADAPTER:
            prompt = f"{user_text.strip()}\n\n{self.router_instruction}"
            try:
                raw = await self.route_chat(
                    prompt, core.ENTRY_ADAPTER, core.ROUTER_MAX_TOKENS)
            except core.RoutingError:
                if core.PURE_MODEL_ROUTE:
                    raise
                raw = ""
            if core.PURE_MODEL_ROUTE:
                model_route = core._ROUTER.parse_canonical_output(raw)
            else:
                model_route = (
                    core._ROUTER.parse_router_output(f"model_id={raw}")
                    or core._ROUTER.parse_router_output(raw))
            core._log(
                f"route: L0 raw={raw!r} parsed_model_route={model_route}")
        if core.PURE_MODEL_ROUTE:
            if model_route not in core._TASKS:
                raise core.RoutingError(
                    "L0 emitted a non-canonical route label")
            return model_route, "pure_model_route", {
                "model_route": model_route, "route": model_route}
        decision = core._ROUTER.apply_guardrail(model_route, user_text)
        return decision.route_id, decision.decision, (
            decision.diagnostics or {})

    async def answer_hop(
        self, state: Any, route: str, user_text: str, max_tokens: int | None,
        tools: list | None = None, stream_cb: StreamCallback | None = None,
        stream_start_cb: Callable[[str], Awaitable[None]] | None = None,
        answer_options: dict | None = None,
    ) -> tuple[int, dict, str]:
        adapter = core.ROUTE_TO_ADAPTER.get(route, core.ENTRY_ADAPTER)
        state.begin_task(route, user_text)
        messages = state.own_view_messages(route)

        async def on_open() -> None:
            if stream_start_cb is not None:
                await stream_start_cb(route)

        status, body = await self.call_answer(
            messages, adapter, max_tokens, tools, stream_cb,
            on_open if stream_start_cb is not None else None, answer_options)
        if status == 400 and core._is_pool_miss(body):
            fallback = core._pool_miss_fallback()
            if fallback != route:
                core._log(
                    f"answer: pool-miss on {route}({adapter}) -> retry {fallback}")
                state.discard_open_task()
                state.begin_task(fallback, user_text)
                adapter = core.ROUTE_TO_ADAPTER.get(
                    fallback, core.ENTRY_ADAPTER)
                messages = state.own_view_messages(fallback)

                async def fallback_open() -> None:
                    if stream_start_cb is not None:
                        await stream_start_cb(fallback)

                status, body = await self.call_answer(
                    messages, adapter, max_tokens, tools, stream_cb,
                    fallback_open if stream_start_cb is not None else None,
                    answer_options)
                route = fallback
        if status == 200:
            if core._chat_response_is_malformed(
                    body, tools, answer_options):
                state.discard_open_task()
                return 502, core._invalid_upstream_response(), route
            state.append_assistant(core._message_for_state(body))
        return status, body, route

    async def summary_hop(self, state: Any, task: Task) -> str:
        adapter = core.ROUTE_TO_ADAPTER.get(
            task.owner, core.ENTRY_ADAPTER)
        messages = state.summary_context_messages(task)
        status, body = await self.chat(
            messages, adapter, core.SUMMARY_MAX_OUT,
            answer_options=core._NO_THINK_OPTIONS, hop="summary")
        summary = ""
        if status == 200:
            message = core._choice_message(body)
            content = message.get("content") if isinstance(message, dict) else None
            summary = content.strip() if isinstance(content, str) else ""
            if not summary and isinstance(message, dict):
                reasoning = (
                    message.get("reasoning")
                    or message.get("reasoning_content"))
                if isinstance(reasoning, str) and reasoning.strip():
                    summary = reasoning.strip()[:160]
        core._log(
            f"summary@{task.owner} status={status} ({len(summary)} chars)")
        return summary

    async def publish_buffered(
        self, body: dict, stream_cb: StreamCallback | None,
    ) -> None:
        if stream_cb is None:
            body.pop("_mol_buffered_tool_chunks", None)
            return
        for chunk in core._buffered_tool_chunks(body):
            await stream_cb(("chunk", chunk))

    async def orchestrate(
        self, state: Any, messages: list[dict], last_role: str,
        last_text: str, max_tokens: int | None, convo_id: str,
        tools: list | None = None, stream_cb: StreamCallback | None = None,
        answer_options: dict | None = None,
    ) -> tuple[int, dict, dict]:
        checkpoint = state.stream_checkpoint()
        try:
            result = await self._orchestrate_impl(
                state, messages, last_role, last_text, max_tokens, convo_id,
                tools, stream_cb, answer_options)
        except BaseException:
            state.restore_stream_checkpoint(checkpoint)
            raise
        if result[0] != 200:
            state.restore_stream_checkpoint(checkpoint)
        return result

    async def _orchestrate_impl(
        self, state: Any, messages: list[dict], last_role: str,
        last_text: str, max_tokens: int | None, convo_id: str,
        tools: list | None, stream_cb: StreamCallback | None,
        answer_options: dict | None,
    ) -> tuple[int, dict, dict]:
        diag = {
            "convo_id": convo_id,
            "n_completed": state.completed_count(),
        }
        if last_role == "tool":
            validation_error = core._validate_pending_tool_results(
                state, messages)
            replay = None
            if validation_error is not None:
                replay, replay_error = (
                    core._validate_self_contained_tool_results(messages))
                if (not isinstance(state, core.StatelessSideContext)
                        or replay_error is not None):
                    error = replay_error or validation_error
                    code = error["error"]["code"]
                    if stream_cb:
                        await stream_cb(("meta_post", {
                            **diag, "error": code}))
                    return 400, error, {**diag, "error": code}
                _, _, replay_user_text = split_system_and_turn(
                    messages[:replay["user_index"] + 1])
                try:
                    route, _, _ = await self.route(state, replay_user_text)
                except core.RoutingError as exc:
                    core._log(f"tool replay route failed: {exc}")
                    status, error, code = core._routing_error_response(exc)
                    diag["error"] = code
                    if stream_cb:
                        await stream_cb((
                            "meta_post", core._routing_error_event(diag, error)))
                    return status, error, diag
                state.begin_task(route, replay_user_text)
                state.set_pending_tool_route(route, replay["expected"])
                diag["decision"] = "self_contained_tool_replay"
            else:
                route = state.pending_tool_route
                diag["decision"] = "sticky_tool_continuation"
            adapter = core.ROUTE_TO_ADAPTER.get(route, core.ENTRY_ADAPTER)
            diag["route"] = route
            if stream_cb:
                await stream_cb(("meta_pre", diag))
            for tool_message in extract_tool_results(messages):
                state.append_tool_result(tool_message)
            answer_messages = state.own_view_messages(route)
            status, body = await self.call_answer(
                answer_messages, adapter, max_tokens, tools, stream_cb,
                answer_options=answer_options)
            if status == 200:
                if core._chat_response_is_malformed(
                        body, tools, answer_options):
                    if stream_cb:
                        await stream_cb(("meta_post", {
                            **diag, "error": "invalid_upstream_response"}))
                    return 502, core._invalid_upstream_response(), {
                        **diag, "error": "invalid_upstream_response"}
                state.append_assistant(core._message_for_state(body))
            if status != 200:
                context_error = core._context_length_error_response(
                    status, body)
                if context_error is not None:
                    diag["error"] = "context_length_exceeded"
                    if stream_cb:
                        await stream_cb(("meta_post", {
                            **diag, "error_body": context_error}))
                    return 400, context_error, diag
                if stream_cb:
                    await stream_cb(("meta_post", {
                        **diag, "error": "answer_hop_failed"}))
                return status, body, {**diag, "error": "answer_hop_failed"}
            if core._has_tool_call(body):
                state.set_pending_tool_route(
                    route, core._tool_call_ids_from_messages([
                        core._choice_message(body)]))
                await self.publish_buffered(body, stream_cb)
                diag["pending_tool_route"] = route
                if stream_cb:
                    await stream_cb(("meta_post", {
                        **diag, "engine_resp": body,
                        "finish_reason": core._finish_reason(body),
                        "tool_call": True}))
                    return 200, body, diag
                return core._shape_response(body, route, diag)
            await self.publish_buffered(body, stream_cb)
            task = state.close_open_task(summarize=False)
            if task is not None and state.should_summarize(route):
                task.summary = await self.summary_hop(state, task)
                state.record_summary(task)
            diag["summary"] = bool(task and task.summary)
            if stream_cb:
                await stream_cb(("meta_post", {
                    **diag, "engine_resp": body,
                    "finish_reason": core._finish_reason(body),
                    "tool_call": False}))
                return 200, body, diag
            return core._shape_response(body, route, diag)

        if last_role != "user":
            last_text = last_text or ""
        try:
            route, decision, _ = await self.route(state, last_text)
        except core.RoutingError as exc:
            core._log(f"route failed: {exc}")
            status, error, code = core._routing_error_response(exc)
            diag["error"] = code
            if stream_cb:
                await stream_cb((
                    "meta_post", core._routing_error_event(diag, error)))
            return status, error, diag
        diag["route"] = route
        diag["decision"] = decision
        initial_route = route

        async def stream_started(actual_route: str) -> None:
            diag["route"] = actual_route
            if actual_route != initial_route:
                diag["decision"] = "pool_miss_fallback"
            if stream_cb:
                await stream_cb(("meta_pre", diag))

        status, body, route = await self.answer_hop(
            state, route, last_text, max_tokens, tools, stream_cb,
            stream_started if stream_cb else None, answer_options)
        diag["route"] = route
        if status != 200:
            context_error = core._context_length_error_response(status, body)
            if context_error is not None:
                diag["error"] = "context_length_exceeded"
                if stream_cb:
                    await stream_cb(("meta_post", {
                        **diag, "error_body": context_error}))
                return 400, context_error, diag
            diag["error"] = "answer_hop_failed"
            if stream_cb:
                await stream_cb(("meta_post", {
                    **diag, "error": "answer_hop_failed"}))
            return status, body, diag
        if core._has_tool_call(body):
            state.set_pending_tool_route(
                route, core._tool_call_ids_from_messages([
                    core._choice_message(body)]))
            await self.publish_buffered(body, stream_cb)
            diag["pending_tool_route"] = route
            if stream_cb:
                await stream_cb(("meta_post", {
                    **diag, "engine_resp": body,
                    "finish_reason": core._finish_reason(body),
                    "tool_call": True}))
                return 200, body, diag
            return core._shape_response(body, route, diag)
        await self.publish_buffered(body, stream_cb)
        task = state.close_open_task(summarize=False)
        if task is not None and state.should_summarize(route):
            task.summary = await self.summary_hop(state, task)
            state.record_summary(task)
        diag["summary"] = bool(task and task.summary)
        if stream_cb:
            await stream_cb(("meta_post", {
                **diag, "engine_resp": body,
                "finish_reason": core._finish_reason(body),
                "tool_call": False}))
            return 200, body, diag
        return core._shape_response(body, route, diag)


class ASGIWriter:
    def __init__(
        self, send: Callable[[dict], Awaitable[None]],
        disconnected: asyncio.Event,
        response_complete: asyncio.Event,
    ) -> None:
        self._send = send
        self.disconnected = disconnected
        self.response_complete = response_complete
        self.started = False
        self.ended = False
        self._send_lock = asyncio.Lock()
        self._last_frame_at = time.monotonic()
        self._keepalive_task: asyncio.Task[None] | None = None
        self._keepalive_owner: asyncio.Task[Any] | None = None

    async def send(self, message: dict) -> None:
        async with self._send_lock:
            await self._send_unlocked(message)

    async def _send_unlocked(self, message: dict) -> None:
        if self.disconnected.is_set():
            raise ClientDisconnected("client disconnected")
        try:
            await self._send(message)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.disconnected.set()
            raise ClientDisconnected("client disconnected") from exc

    async def json(self, status: int, body: dict,
                   headers: dict[str, str] | None = None) -> None:
        payload = json.dumps(body, ensure_ascii=False,
                             separators=(",", ":")).encode("utf-8")
        response_headers = [
            (b"content-type", b"application/json; charset=utf-8"),
            (b"content-length", str(len(payload)).encode("ascii")),
        ]
        for key, value in (headers or {}).items():
            response_headers.append((key.lower().encode("ascii"),
                                     str(value).encode("latin-1")))
        async with self._send_lock:
            if self.started:
                return
            await self._send_unlocked({
                "type": "http.response.start", "status": status,
                "headers": response_headers,
            })
            self.started = True
        await self.send({
            "type": "http.response.body", "body": payload, "more_body": False,
        })
        self.response_complete.set()

    async def start_sse(
        self, status: int = 200,
        headers: dict[str, str] | None = None,
    ) -> None:
        response_headers = [
            (b"content-type", b"text/event-stream; charset=utf-8"),
            (b"cache-control", b"no-cache"),
            (b"connection", b"keep-alive"),
            (b"x-accel-buffering", b"no"),
        ]
        for key, value in (headers or {}).items():
            response_headers.append((key.lower().encode("ascii"),
                                     str(value).encode("latin-1")))
        async with self._send_lock:
            if self.started:
                return
            await self._send_unlocked({
                "type": "http.response.start", "status": status,
                "headers": response_headers,
            })
            self.started = True

    async def frame(self, payload: bytes) -> None:
        if not self.started:
            await self.start_sse()
        async with self._send_lock:
            await self._send_unlocked({
                "type": "http.response.body", "body": payload,
                "more_body": True,
            })
            self._last_frame_at = time.monotonic()

    async def start_keepalive(self, interval_s: float | None = None) -> None:
        interval = (SSE_KEEPALIVE_INTERVAL_S
                    if interval_s is None else max(0.0, interval_s))
        if interval == 0 or self.ended or self.response_complete.is_set():
            return
        if self._keepalive_task is not None:
            return
        if not self.started:
            await self.start_sse()
        self._keepalive_owner = asyncio.current_task()
        await self.frame(SSE_KEEPALIVE_FRAME)
        self._keepalive_task = asyncio.create_task(
            self._keepalive_loop(interval), name="mol-sse-keepalive")

    async def _keepalive_loop(self, interval_s: float) -> None:
        try:
            while (not self.ended and not self.response_complete.is_set()
                   and not self.disconnected.is_set()):
                delay = max(
                    0.0, self._last_frame_at + interval_s - time.monotonic())
                await asyncio.sleep(delay)
                if (self.ended or self.response_complete.is_set()
                        or self.disconnected.is_set()):
                    return
                await self._keepalive_if_silent(interval_s)
        except ClientDisconnected:
            owner = self._keepalive_owner
            if owner is not None and not owner.done():
                owner.cancel()

    async def _keepalive_if_silent(self, interval_s: float) -> None:
        async with self._send_lock:
            if time.monotonic() - self._last_frame_at < interval_s:
                return
            await self._send_unlocked({
                "type": "http.response.body", "body": SSE_KEEPALIVE_FRAME,
                "more_body": True,
            })
            self._last_frame_at = time.monotonic()

    async def stop_keepalive(self) -> None:
        task = self._keepalive_task
        self._keepalive_task = None
        self._keepalive_owner = None
        if task is None:
            return
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task

    async def end(self) -> None:
        if self.started and not self.ended:
            self.ended = True
            await self.send({
                "type": "http.response.body", "body": b"", "more_body": False,
            })
            self.response_complete.set()

    def mark_terminal(self) -> None:
        """The client-visible transaction is committed after this SSE frame."""
        self.response_complete.set()


class ProxyASGI:
    def __init__(self) -> None:
        self.metrics = RuntimeMetrics()
        self.admission = AdmissionController(self.metrics)
        self.gateway = AsyncGatewayClient(self.metrics)
        self.releases = RoutingReleaseManager(self.gateway, self.metrics)
        self.store = AsyncResponseStore(self.releases, self.metrics)
        self.orchestrator = AsyncOrchestrator(self.gateway)
        self._startup_lock = asyncio.Lock()
        self._started = False
        self._stopping = False
        self._cleanup_task: asyncio.Task[None] | None = None
        self._lag_task: asyncio.Task[None] | None = None

    async def startup(self) -> None:
        if self._started:
            return
        async with self._startup_lock:
            if self._started:
                return
            await self.gateway.start()
            await self.releases.start()
            self.metrics.start()
            self._cleanup_task = asyncio.create_task(
                self._cleanup_loop(), name="mol-state-cleanup")
            self._lag_task = asyncio.create_task(
                self._lag_loop(), name="mol-event-loop-lag")
            self._started = True
            core._log(
                "async runtime started "
                f"active={MAX_INFLIGHT} queue={MAX_QUEUED} "
                f"connections={MAX_CONNECTIONS} "
                f"sse_keepalive={SSE_KEEPALIVE_INTERVAL_S:g}s")

    async def shutdown(self) -> None:
        if not self._started or self._stopping:
            return
        self._stopping = True
        for task in (self._cleanup_task, self._lag_task):
            if task is not None:
                task.cancel()
        for task in (self._cleanup_task, self._lag_task):
            if task is not None:
                with suppress(asyncio.CancelledError):
                    await task
        await self.store.close()
        await self.releases.close()
        await self.gateway.close()
        await asyncio.to_thread(self.metrics.close)
        self._started = False

    async def _cleanup_loop(self) -> None:
        while True:
            await asyncio.sleep(STORE_CLEANUP_INTERVAL_S)
            await self.store.cleanup()
            self._cleanup_chat_registries()

    @staticmethod
    def _cleanup_chat_registries() -> None:
        """TTL cleanup is background-only; request paths no longer scan LRU maps."""
        now = time.monotonic()
        with core._SIDE_CTX_LOCK:
            for key, touched in list(core._SIDE_CTX_TOUCHED.items()):
                if now - touched <= core.CONVO_TTL_S:
                    continue
                state = core._SIDE_CTX.pop(key, None)
                core._SIDE_CTX_TOUCHED.pop(key, None)
                if state is not None:
                    core._drop_tool_context_state_locked(state)
        with core._CONVOS_LOCK:
            for key, touched in list(core._CONVOS_TOUCHED.items()):
                if now - touched > core.CONVO_TTL_S:
                    core._CONVOS.pop(key, None)
                    core._CONVOS_TOUCHED.pop(key, None)

    async def _lag_loop(self) -> None:
        interval = 0.025
        expected = time.monotonic() + interval
        while True:
            await asyncio.sleep(interval)
            now = time.monotonic()
            self.metrics.event_loop_lag.observe(max(0.0, now - expected))
            expected = now + interval

    async def _receive_body(
        self, receive: Callable[[], Awaitable[dict]],
        disconnected: asyncio.Event, response_complete: asyncio.Event,
    ) -> tuple[bytes, asyncio.Task[None]]:
        chunks: list[bytes] = []
        size = 0
        while True:
            event = await receive()
            if event["type"] == "http.disconnect":
                disconnected.set()
                raise ClientDisconnected("client disconnected while reading body")
            if event["type"] != "http.request":
                continue
            chunk = event.get("body", b"")
            size += len(chunk)
            if size > core.MAX_REQUEST_BYTES:
                raise ValueError("request_too_large")
            chunks.append(chunk)
            if not event.get("more_body", False):
                break

        request_task = asyncio.current_task()

        async def monitor() -> None:
            try:
                while True:
                    event = await receive()
                    if event["type"] == "http.disconnect":
                        disconnected.set()
                        if (not response_complete.is_set()
                                and request_task is not None):
                            request_task.cancel()
                        return
            except asyncio.CancelledError:
                raise

        monitor_task = asyncio.create_task(monitor(), name="mol-client-disconnect")
        return b"".join(chunks), monitor_task

    @staticmethod
    def _headers(scope: dict) -> dict[str, str]:
        values: dict[str, str] = {}
        for key, value in scope.get("headers", []):
            name = key.decode("latin-1").lower()
            text = value.decode("latin-1")
            if name in values:
                values[name] = values[name] + "," + text
            else:
                values[name] = text
        return values

    @staticmethod
    def _model_error(model: object) -> dict:
        return {"error": {
            "message": f"The model `{model}` does not exist",
            "type": "invalid_request_error", "code": "model_not_found",
            "param": "model",
        }}

    async def __call__(self, scope: dict, receive, send) -> None:
        if scope["type"] == "lifespan":
            await self._lifespan(receive, send)
            return
        if scope["type"] != "http":
            return
        await self.startup()
        path = scope.get("path", "/").rstrip("/") or "/"
        method = scope.get("method", "GET").upper()
        headers = self._headers(scope)
        started = time.monotonic()
        disconnected = asyncio.Event()
        response_complete = asyncio.Event()
        monitor_task: asyncio.Task[None] | None = None
        writer = ASGIWriter(send, disconnected, response_complete)
        status = 500
        try:
            if path == "/health" and method == "GET":
                health_status = "ok" if await self.gateway.is_ready() else "not_ready"
                await writer.json(200, {"status": health_status})
                status = 200
                return
            if path != "/health" and not self._authorized(headers):
                await writer.json(401, core._oai_error(
                    "Incorrect API key provided",
                    error_type="authentication_error", code="invalid_api_key"),
                    {"WWW-Authenticate": "Bearer"})
                status = 401
                return
            if path == "/v1/models" and method == "GET":
                await writer.json(200, {
                    "object": "list",
                    "data": [{"id": core.SERVED_MODEL_NAME,
                              "object": "model", "created": 0,
                              "owned_by": "mol"}],
                })
                status = 200
                return
            if path not in ("/v1/chat/completions", "/v1/responses") \
                    or method != "POST":
                await writer.json(404, core._oai_error(
                    f"Endpoint {path} does not exist",
                    code="endpoint_not_found", param="path"))
                status = 404
                return
            try:
                raw, monitor_task = await self._receive_body(
                    receive, disconnected, response_complete)
            except ValueError:
                await writer.json(413, core._oai_error(
                    f"Request body exceeds the {core.MAX_REQUEST_BYTES}-byte limit",
                    code="request_too_large", param="body"))
                status = 413
                return
            try:
                payload = json.loads(raw) if raw else {}
            except Exception as exc:
                await writer.json(400, core._oai_error(
                    f"bad json: {exc}", code="invalid_json", param="body"))
                status = 400
                return
            if not isinstance(payload, dict):
                await writer.json(400, core._oai_error(
                    "JSON request body must be an object",
                    code="invalid_json_body", param="body"))
                status = 400
                return
            try:
                lease = await self.admission.acquire()
            except QueueOverloaded:
                await writer.json(503, core._oai_error(
                    "Proxy is at capacity; retry the request",
                    error_type="server_error", code="overloaded"),
                    {"Retry-After": "1"})
                status = 503
                return
            async with lease:
                if path == "/v1/chat/completions":
                    status = await self._handle_chat(
                        payload, headers, writer)
                else:
                    status = await self._handle_responses(
                        payload, writer)
        except ClientDisconnected:
            status = 499
        except asyncio.CancelledError:
            if disconnected.is_set():
                status = 499
                return
            raise
        except Exception as exc:
            core._log(f"async request error path={path}: {type(exc).__name__}: {exc}")
            if not writer.started and not disconnected.is_set():
                with suppress(Exception):
                    await writer.json(500, core._oai_error(
                        "Internal proxy error", error_type="server_error",
                        code="proxy_error"))
            status = 500
        finally:
            if monitor_task is not None:
                monitor_task.cancel()
                with suppress(asyncio.CancelledError):
                    await monitor_task
            self.metrics.requests.labels(path, str(status)).inc()
            self.metrics.request_latency.labels(path).observe(
                time.monotonic() - started)

    async def _lifespan(self, receive, send) -> None:
        while True:
            event = await receive()
            if event["type"] == "lifespan.startup":
                try:
                    await self.startup()
                    await send({"type": "lifespan.startup.complete"})
                except Exception as exc:
                    await send({
                        "type": "lifespan.startup.failed", "message": str(exc)})
                    return
            elif event["type"] == "lifespan.shutdown":
                try:
                    await self.shutdown()
                    await send({"type": "lifespan.shutdown.complete"})
                except Exception as exc:
                    await send({
                        "type": "lifespan.shutdown.failed", "message": str(exc)})
                return

    @staticmethod
    def _authorized(headers: dict[str, str]) -> bool:
        if not core.API_KEY:
            return True
        return hmac.compare_digest(
            headers.get("authorization", ""), f"Bearer {core.API_KEY}")

    async def _handle_chat(
        self, payload: dict, headers: dict[str, str], writer: ASGIWriter,
    ) -> int:
        model = payload.get("model")
        if model != core.SERVED_MODEL_NAME:
            await writer.json(404, self._model_error(model))
            return 404
        messages = payload.get("messages")
        if (not isinstance(messages, list) or not messages
                or not all(isinstance(message, dict) for message in messages)):
            await writer.json(400, core._oai_error(
                "messages must be a non-empty array of objects",
                code="invalid_messages", param="messages"))
            return 400
        try:
            core._validate_chat_request_capabilities(payload)
        except ValueError as exc:
            await writer.json(400, core._oai_error(
                str(exc), code="invalid_request"))
            return 400

        system_msgs, last_role, last_text = split_system_and_turn(messages)
        max_tokens = core._clamp_max_tokens(payload)
        tools = payload.get("tools")
        answer_options = core._chat_answer_options(payload)
        stream = bool(payload.get("stream"))
        explicit_convo_id = headers.get("x-conversation-id")
        authorization = headers.get("authorization")
        sticky_context = (
            core._take_tool_context(messages, authorization)
            if explicit_convo_id is None and last_role == "tool" else None)
        try:
            client_convo_id = core._chat_conversation_token(explicit_convo_id)
        except ValueError as exc:
            await writer.json(400, core._oai_error(
                str(exc), code="invalid_conversation_id",
                param="X-Conversation-Id"))
            return 400
        if sticky_context is not None:
            state, client_convo_id, tool_binding = sticky_context
            convo_id = state.convo_key_id
        else:
            tool_binding = None
            convo_id = core._chat_convo_key(
                messages, client_convo_id, authorization)
            state = core._get_side_context(convo_id, system_msgs)
        if last_role == "tool":
            validation_error = core._validate_chat_tool_results(
                state, messages, tool_binding)
            if validation_error is not None:
                await writer.json(400, validation_error)
                return 400

        routing_key = core._new_chat_routing_key()
        if not stream:
            checkpoint = None
            registry_checkpoint = None
            diag: dict = {}
            try:
                async with state.async_lock:
                    checkpoint = state.stream_checkpoint()
                    registry_checkpoint = core._snapshot_tool_context(
                        state, authorization)
                    with core._routing_scope(routing_key):
                        state.set_request_messages(messages)
                        status, body, diag = await self.orchestrator.orchestrate(
                            state, messages, last_role, last_text, max_tokens,
                            convo_id, tools, answer_options=answer_options)
                    if (status != 200 and diag.get("error") not in (
                            "orphan_tool_turn", "routing_failed",
                            "context_length_exceeded")):
                        status = core._public_upstream_status(status)
                        body = core._oai_error(
                            "The upstream model request failed",
                            error_type="server_error", code="answer_hop_failed")
                    response_headers = None
                    if status == 200:
                        body.setdefault("metadata", {})[
                            "mol_conversation_id"] = client_convo_id
                        response_headers = {"X-Conversation-Id": client_convo_id}
                    try:
                        await writer.json(status, body, response_headers)
                    except BaseException:
                        if checkpoint is not None:
                            state.restore_stream_checkpoint(checkpoint)
                        if registry_checkpoint is not None:
                            core._restore_tool_context(
                                state, authorization, registry_checkpoint)
                        raise
                    if status == 200:
                        if core._has_tool_call(body):
                            core._register_tool_context(
                                body, state, client_convo_id, authorization)
                        else:
                            core._clear_tool_context(state, authorization)
                    return status
            except ClientDisconnected:
                raise
            except BaseException:
                if checkpoint is not None:
                    state.restore_stream_checkpoint(checkpoint)
                if registry_checkpoint is not None:
                    core._restore_tool_context(
                        state, authorization, registry_checkpoint)
                raise
            finally:
                await asyncio.shield(self.releases.schedule(routing_key))

        await writer.start_sse(
            headers={"X-Conversation-Id": client_convo_id})
        await writer.start_keepalive()
        checkpoint = None
        registry_checkpoint = None
        deferred_tool_frames: list[bytes] = []
        captured_engine: dict | None = None
        stream_id = "chatcmol-" + uuid.uuid4().hex[:24]
        stream_created = int(time.time())

        async def stream_cb(event: tuple[str, Any]) -> None:
            nonlocal captured_engine
            kind, data = event
            if kind == "meta_pre":
                from .responses import _mol_meta
                metadata = _mol_meta(data)
                metadata["mol_conversation_id"] = client_convo_id
                await writer.frame(core._sse_frame(core._chat_chunk(
                    stream_id,
                    core.SERVED_MODEL_NAME,
                    {"role": "assistant", "content": ""}, None,
                    meta=metadata, created=stream_created)))
                return
            if kind == "chunk":
                chunk = core._sanitize_chat_chunk(data)
                chunk.pop("usage", None)
                chunk["model"] = core.SERVED_MODEL_NAME
                chunk["id"] = stream_id
                chunk["created"] = stream_created
                chunk["object"] = "chat.completion.chunk"
                meaningful = False
                has_tool_delta = False
                for choice in chunk.get("choices") or []:
                    choice["finish_reason"] = None
                    delta = choice.get("delta")
                    if isinstance(delta, dict) and delta.get("tool_calls"):
                        has_tool_delta = True
                    if (isinstance(delta, dict) and bool(delta)) \
                            or choice.get("logprobs") is not None:
                        meaningful = True
                if meaningful:
                    frame = core._sse_frame(chunk)
                    if has_tool_delta:
                        deferred_tool_frames.append(frame)
                    else:
                        await writer.frame(frame)
                return
            if kind == "meta_post":
                if isinstance(data, dict) and data.get("error"):
                    error_body = data.get("error_body")
                    if not (isinstance(error_body, dict)
                            and isinstance(error_body.get("error"), dict)):
                        error_code = str(data.get("error"))
                        error_body = core._oai_error(
                            error_code, error_type="server_error",
                            code=error_code)
                    await writer.frame(core._sse_frame(error_body))
                    await writer.frame(b"data: [DONE]\n\n")
                    writer.mark_terminal()
                    return
                if isinstance(data, dict):
                    captured_engine = data.get("engine_resp")
                    from .responses import _mol_meta
                    metadata = _mol_meta({
                        key: value for key, value in data.items()
                        if key not in ("engine_resp", "finish_reason", "tool_call")
                    })
                    metadata["mol_conversation_id"] = client_convo_id
                    cached, prompt_tokens = core._cache_stats(
                        captured_engine if isinstance(captured_engine, dict) else {})
                    metadata["mol_cached_tokens"] = cached
                    metadata["mol_prompt_tokens"] = prompt_tokens
                    for frame in deferred_tool_frames:
                        await writer.frame(frame)
                    deferred_tool_frames.clear()
                    finish = data.get("finish_reason") or "stop"
                    await writer.frame(core._sse_frame(core._chat_chunk(
                        stream_id, core.SERVED_MODEL_NAME, {}, finish,
                        meta=metadata, created=stream_created)))
                    # The engine always returns usage because the upstream
                    # stream payload requests it. Forward it for billing even
                    # when the public client did not explicitly opt in.
                    if (isinstance(captured_engine, dict)
                            and isinstance(captured_engine.get("usage"), dict)):
                        usage = core._sanitize_usage(captured_engine["usage"])
                        if core._USAGE_COUNT_KEYS.issubset(usage):
                            usage_chunk = core._chat_chunk(
                                stream_id, core.SERVED_MODEL_NAME, {}, None,
                                created=stream_created)
                            usage_chunk["choices"] = []
                            usage_chunk["usage"] = usage
                            await writer.frame(core._sse_frame(usage_chunk))
                    await writer.frame(b"data: [DONE]\n\n")
                    writer.mark_terminal()

        try:
            async with state.async_lock:
                checkpoint = state.stream_checkpoint()
                registry_checkpoint = core._snapshot_tool_context(
                    state, authorization)
                with core._routing_scope(routing_key):
                    state.set_request_messages(messages)
                    status, body, diag = await self.orchestrator.orchestrate(
                        state, messages, last_role, last_text, max_tokens,
                        convo_id, tools, stream_cb=stream_cb,
                        answer_options=answer_options)
                if status != 200:
                    state.restore_stream_checkpoint(checkpoint)
                    core._restore_tool_context(
                        state, authorization, registry_checkpoint)
                if status == 200:
                    if core._has_tool_call(body):
                        core._register_tool_context(
                            body, state, client_convo_id, authorization)
                    else:
                        core._clear_tool_context(state, authorization)
                if (state.open_task() is not None
                        and state.pending_tool_route is None):
                    state.discard_open_task()
            return status
        except BaseException:
            if checkpoint is not None:
                state.restore_stream_checkpoint(checkpoint)
            if registry_checkpoint is not None:
                core._restore_tool_context(
                    state, authorization, registry_checkpoint)
            raise
        finally:
            await asyncio.shield(self.releases.schedule(routing_key))
            await writer.stop_keepalive()
            with suppress(Exception):
                await writer.end()

    async def _handle_responses(self, payload: dict, writer: ASGIWriter) -> int:
        if bool(payload.get("stream")):
            return await self._handle_responses_stream(payload, writer)
        model = payload.get("model")
        if model is not None and model != core.SERVED_MODEL_NAME:
            await writer.json(404, self._model_error(model))
            return 404
        try:
            responses_api._validate_request_capabilities(payload)
            messages = responses_api.responses_input_to_oai_messages(
                payload.get("input"), payload.get("instructions"))
            tools = responses_api.responses_tools_to_oai_tools(
                payload.get("tools"))
        except ValueError as exc:
            await writer.json(400, core._oai_error(
                str(exc), code="unsupported_parameter"))
            return 400
        if not messages:
            await writer.json(400, core._oai_error(
                "Responses `input` is empty", code="invalid_input", param="input"))
            return 400
        prev_id = payload.get("previous_response_id")
        state, parent_id, parent_root, resolve_status = await self.store.resolve(prev_id)
        if resolve_status != 200 or state is None:
            if resolve_status == 409:
                await writer.json(409, core._oai_error(
                    f"previous_response_id {prev_id!r} is still being committed",
                    error_type="conflict", code="response_not_ready",
                    param="previous_response_id"))
                return 409
            await writer.json(404, core._oai_error(
                f"previous_response_id {prev_id!r} not found",
                error_type="not_found", code="previous_response_not_found",
                param="previous_response_id"))
            return 404
        system_msgs, last_role, last_text = split_system_and_turn(messages)
        core._adopt_system(state, system_msgs, clear_empty=True)
        if last_role == "tool":
            validation_error = core._validate_pending_tool_results(
                state, messages, param="input")
            if validation_error is not None:
                await writer.json(400, validation_error)
                if parent_root:
                    await self.store.release_lease(parent_root)
                return 400
        max_tokens = core._clamp_max_tokens({
            "max_tokens": payload.get("max_output_tokens")})
        answer_options = core._responses_answer_options(payload)
        resp_id = "resp_" + uuid.uuid4().hex[:24]
        routing_root = parent_root or resp_id
        should_store = payload.get("store", True) is not False
        checkpoint = None
        reserved = False
        stored = False
        try:
            with core._routing_scope(routing_root):
                async with state.async_lock:
                    checkpoint = state.stream_checkpoint()
                    if last_role != "tool":
                        state.stage_external_history(
                            messages[len(system_msgs):-1])
                    status, chat_body, diag = await self.orchestrator.orchestrate(
                        state, messages, last_role, last_text, max_tokens,
                        resp_id, tools, answer_options=answer_options)
            if status != 200:
                if diag.get("error") == "context_length_exceeded":
                    body = chat_body
                else:
                    body = core._oai_error(
                        "answer hop failed", error_type="server_error",
                        code="answer_hop_failed")
                await writer.json(core._public_upstream_status(status), body)
                return core._public_upstream_status(status)
            response_obj = responses_api.chat_to_responses(
                chat_body, resp_id, prev_id, core.SERVED_MODEL_NAME,
                diag, request_payload=payload)
            if should_store:
                size = await self.store.state_size(state)
                await self.store.reserve(resp_id, size)
                reserved = True
            try:
                await writer.json(200, response_obj)
            except BaseException:
                if checkpoint is not None:
                    state.restore_stream_checkpoint(checkpoint)
                raise
            if should_store:
                await asyncio.shield(self.store.commit(
                    resp_id, state, parent_id, routing_root, size))
                stored = True
            return 200
        except ResponseStoreCapacityError:
            if not writer.started:
                await writer.json(503, core._oai_error(
                    "The response state store is at capacity",
                    error_type="server_error", code="response_store_capacity"))
            return 503
        finally:
            if reserved and not stored:
                await self.store.abort(resp_id)
            if parent_root:
                await self.store.release_lease(parent_root)
            elif not parent_root and not stored:
                await asyncio.shield(self.releases.schedule(routing_root))

    async def _handle_responses_stream(
        self, payload: dict, writer: ASGIWriter,
    ) -> int:
        """Responses SSE path with terminal publication before snapshot commit."""
        model = payload.get("model") or core.SERVED_MODEL_NAME
        prev_id = payload.get("previous_response_id")
        try:
            responses_api._validate_request_capabilities(payload)
            messages = responses_api.responses_input_to_oai_messages(
                payload.get("input"), payload.get("instructions"))
            tools = responses_api.responses_tools_to_oai_tools(
                payload.get("tools"))
        except ValueError as exc:
            await writer.json(400, core._oai_error(
                str(exc), code="unsupported_parameter"))
            return 400
        if not messages:
            await writer.json(400, core._oai_error(
                "Responses `input` is empty", code="invalid_input", param="input"))
            return 400
        state, parent_id, parent_root, resolve_status = await self.store.resolve(prev_id)
        if resolve_status != 200 or state is None:
            await writer.json(
                409 if resolve_status == 409 else 404,
                core._oai_error(
                    f"previous_response_id {prev_id!r} "
                    + ("is still being committed" if resolve_status == 409
                       else "not found"),
                    error_type="conflict" if resolve_status == 409 else "not_found",
                    code="response_not_ready" if resolve_status == 409
                    else "previous_response_not_found",
                    param="previous_response_id"))
            return 409 if resolve_status == 409 else 404
        system_msgs, last_role, last_text = split_system_and_turn(messages)
        core._adopt_system(state, system_msgs, clear_empty=True)
        if last_role == "tool":
            validation_error = core._validate_pending_tool_results(
                state, messages, param="input")
            if validation_error is not None:
                await writer.json(400, validation_error)
                if parent_root:
                    await self.store.release_lease(parent_root)
                return 400
        max_tokens = core._clamp_max_tokens({
            "max_tokens": payload.get("max_output_tokens")})
        answer_options = core._responses_answer_options(payload)
        resp_id = "resp_" + uuid.uuid4().hex[:24]
        routing_root = parent_root or resp_id
        should_store = payload.get("store", True) is not False
        reserved = False
        stored = False
        if should_store:
            try:
                initial_size = await self.store.state_size(state)
                await self.store.reserve(resp_id, initial_size)
                reserved = True
            except ResponseStoreCapacityError:
                if parent_root:
                    await self.store.release_lease(parent_root)
                await writer.json(503, core._oai_error(
                    "The response state store is at capacity",
                    error_type="server_error", code="response_store_capacity"))
                return 503

        await writer.start_sse()
        await writer.start_keepalive()
        frames: list[bytes] = []
        callback = responses_api._make_responses_cb(
            frames.append, resp_id, model, prev_id, payload)

        async def drain_frames() -> None:
            while frames:
                await writer.frame(frames.pop(0))

        async def stream_cb(event: tuple[str, Any]) -> None:
            callback(event)
            await drain_frames()

        async def emit_completed(response_obj: dict) -> None:
            callback.emit_completed(response_obj)
            await drain_frames()
            writer.mark_terminal()

        async def emit_failed(message: str, code: str,
                              response_obj: dict | None = None) -> None:
            callback.emit_failed(message, code, response_obj)
            await drain_frames()
            writer.mark_terminal()

        checkpoint = None
        try:
            with core._routing_scope(routing_root):
                async with state.async_lock:
                    checkpoint = state.stream_checkpoint()
                    if last_role != "tool":
                        state.stage_external_history(
                            messages[len(system_msgs):-1])
                    status, chat_body, diag = await self.orchestrator.orchestrate(
                        state, messages, last_role, last_text, max_tokens,
                        resp_id, tools, stream_cb=stream_cb,
                        answer_options=answer_options)
                    if status != 200:
                        state.restore_stream_checkpoint(checkpoint)
                    if (state.open_task() is not None
                            and state.pending_tool_route is None):
                        state.discard_open_task()
            if status == 200:
                response_obj = responses_api.chat_to_responses(
                    chat_body, resp_id, prev_id, model, diag,
                    request_payload=payload,
                    item_layout=callback.item_layout(),
                    created_at=callback.created_at)
                if should_store:
                    size = await self.store.state_size(state)
                    await self.store.resize_pending(resp_id, size)
                await emit_completed(response_obj)
                if should_store:
                    size = await self.store.state_size(state)
                    await asyncio.shield(self.store.commit(
                        resp_id, state, parent_id, routing_root, size))
                    stored = True
            else:
                # The shared core emits response.failed through meta_post before
                # returning a non-200 status. Do not publish a duplicate terminal
                # event here.
                writer.mark_terminal()
            return 200
        except ResponseStoreCapacityError:
            await emit_failed(
                "The response state store is at capacity",
                "response_store_capacity")
            return 200
        finally:
            if reserved and not stored:
                await self.store.abort(resp_id)
            if parent_root:
                await self.store.release_lease(parent_root)
            elif not parent_root and not stored:
                await asyncio.shield(self.releases.schedule(routing_root))
            await writer.stop_keepalive()
            with suppress(Exception):
                await writer.end()
