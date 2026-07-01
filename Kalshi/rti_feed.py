"""Background CF Benchmarks RTI index feed.

The BTC Real-Time Index (BRTI) that Kalshi uses to settle the KXBTC15M markets
is NOT available on the REST market object -- it is only published on the
authenticated ``cfbenchmarks_value`` WebSocket channel. This module runs that
WebSocket subscription in a daemon thread and exposes the latest value to the
synchronous REST polling loop via a thread-safe :class:`RTIFeed`.

Usage::

    feed = RTIFeed(api_key_id, key_file_path, index_id="BRTI")
    feed.start()
    ...
    price = feed.price          # latest value, or 0.0 until first tick
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time

from kalshi.auth import KalshiAuth
from kalshi.config import KalshiConfig
from kalshi.ws import KalshiWebSocket
from kalshi.ws.models.cfbenchmarks import CFBenchmarksValueMessage

logger = logging.getLogger("rti_feed")

# Keys the raw CF Benchmarks frame may carry the instantaneous value under.
# The exact upstream schema isn't pinned by the SDK, so we probe a few and
# fall back to the typed 60-second average, which is always present.
_RAW_VALUE_KEYS = ("value", "price", "level", "last", "index_value")


def _extract_price(msg: CFBenchmarksValueMessage) -> float | None:
    """Pull the freshest usable price out of a value message.

    Prefers the instantaneous value from the raw upstream frame (best for
    lead/lag analysis); falls back to the typed trailing 60s average.
    """
    payload = msg.msg

    # 1. Try the raw upstream frame for the instantaneous ("NOW") value.
    if payload.data:
        try:
            raw = json.loads(payload.data)
            if isinstance(raw, dict):
                for key in _RAW_VALUE_KEYS:
                    if key in raw:
                        return float(raw[key])
        except (json.JSONDecodeError, TypeError, ValueError):
            pass  # fall through to the typed average

    # 2. Fall back to the always-present trailing 60s average.
    try:
        return float(payload.avg_60s_data.value)
    except (AttributeError, TypeError, ValueError):
        return None


class RTIFeed:
    """Thread-safe holder for the latest RTI index value."""

    def __init__(self, key_id: str, key_path: str, index_id: str = "BRTI") -> None:
        self._auth = KalshiAuth.from_key_path(key_id, key_path)
        self._config = KalshiConfig()  # production defaults
        self._index_id = index_id

        self._lock = threading.Lock()
        self._price = 0.0
        self._updates = 0
        self._last_update = 0.0  # time.monotonic() of last tick
        self._logged_first = False

        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    @property
    def price(self) -> float:
        with self._lock:
            return self._price

    @property
    def updates(self) -> int:
        """Number of ticks received so far (0 means feed hasn't warmed up)."""
        with self._lock:
            return self._updates

    @property
    def age(self) -> float:
        """Seconds since the last tick (inf if none received yet).

        Use to detect a stalled feed during long unattended runs.
        """
        with self._lock:
            if self._updates == 0:
                return float("inf")
            return time.monotonic() - self._last_update

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._run, name="rti-feed", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    # --- internals ----------------------------------------------------------
    def _run(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._consume_forever())
        finally:
            loop.close()

    async def _consume_forever(self) -> None:
        """Subscribe and stream, reconnecting on any failure until stopped."""
        while not self._stop.is_set():
            try:
                ws = KalshiWebSocket(auth=self._auth, config=self._config)
                async with ws.connect() as session:
                    stream = await session.subscribe_cfbenchmarks_value(
                        index_ids=[self._index_id]
                    )
                    async for msg in stream:
                        if self._stop.is_set():
                            break
                        if not isinstance(msg, CFBenchmarksValueMessage):
                            continue  # skip indexlist control frames
                        price = _extract_price(msg)
                        if price is None:
                            continue
                        with self._lock:
                            self._price = price
                            self._updates += 1
                            self._last_update = time.monotonic()
                        if not self._logged_first:
                            self._logged_first = True
                            logger.info(
                                "RTI feed live: %s = %s", self._index_id, price
                            )
            except Exception as e:  # noqa: BLE001 - keep the feed alive
                logger.warning("RTI feed error: %r; reconnecting in 3s", e)
                await asyncio.sleep(3)
