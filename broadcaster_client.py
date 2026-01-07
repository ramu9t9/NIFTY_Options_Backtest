import json
import logging
import random
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

try:
    import websocket  # type: ignore
except Exception as e:  # pragma: no cover
    websocket = None  # type: ignore
    _import_error = e
else:
    _import_error = None


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BroadcasterClientConfig:
    ws_url: str = "ws://127.0.0.1:8765"
    # Heartbeat: server supports {"type":"ping"} -> {"type":"pong"}
    # NOTE: Some broadcaster deployments close the socket when receiving unknown messages.
    # So we default to NOT sending pings; we only monitor incoming activity and reconnect if stale.
    # Set ping_interval_s > 0 to enable sending ping messages.
    ping_interval_s: float = 0.0
    ping_timeout_s: float = 20.0
    # Reconnect backoff
    reconnect_min_s: float = 1.0
    reconnect_max_s: float = 30.0
    reconnect_jitter_s: float = 0.5


class BroadcasterWebSocketClient:
    """
    Robust WebSocket client for the Centralize Data Centre broadcaster.

    Key guarantees (prevents "ghost" connections):
    - Only ONE active websocket connection per client instance.
    - Reconnect will ALWAYS close the old connection first (best-effort).
    - Close() stops threads and closes socket.
    """

    def __init__(
        self,
        config: BroadcasterClientConfig = BroadcasterClientConfig(),
        on_data: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> None:
        if websocket is None:  # pragma: no cover
            raise ImportError(
                "Missing dependency 'websocket-client'. Install with: pip install websocket-client"
            ) from _import_error

        self._cfg = config
        self._on_data = on_data

        self._lock = threading.RLock()
        self._stop = threading.Event()

        self._ws_app: Optional["websocket.WebSocketApp"] = None
        self._ws_thread: Optional[threading.Thread] = None
        self._heartbeat_thread: Optional[threading.Thread] = None

        self._connected = threading.Event()
        # Treat ANY incoming message as activity; don't require pong support.
        self._last_activity_monotonic = 0.0

        self._latest_by_symbol: Dict[str, Dict[str, Any]] = {}
        self._last_message_monotonic = 0.0
        self._reconnect_attempt = 0

    def connect(self) -> None:
        with self._lock:
            if self._ws_thread and self._ws_thread.is_alive():
                return
            self._stop.clear()
            self._start_threads()

    def close(self) -> None:
        with self._lock:
            self._stop.set()
            self._connected.clear()
            self._close_ws_locked()
        self._join_threads()

    def is_connected(self) -> bool:
        return self._connected.is_set()

    def get_latest(self, symbol: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            return self._latest_by_symbol.get(symbol)

    def get_symbols(self) -> list[str]:
        with self._lock:
            return list(self._latest_by_symbol.keys())

    def last_message_age_s(self) -> Optional[float]:
        ts = self._last_message_monotonic
        if ts <= 0:
            return None
        return max(0.0, time.monotonic() - ts)

    # -----------------------
    # Internal implementation
    # -----------------------

    def _start_threads(self) -> None:
        self._ws_thread = threading.Thread(target=self._run_loop, name="broadcaster-ws", daemon=True)
        self._ws_thread.start()

        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop, name="broadcaster-heartbeat", daemon=True
        )
        self._heartbeat_thread.start()

    def _join_threads(self) -> None:
        for t in (self._ws_thread, self._heartbeat_thread):
            if t and t.is_alive():
                t.join(timeout=5)

    def _run_loop(self) -> None:
        """
        Outer loop that (re)creates a WebSocketApp. This ensures a reconnect does NOT stack
        multiple run_forever threads/sockets.
        """
        while not self._stop.is_set():
            try:
                self._reconnect_attempt += 1
                self._connected.clear()
                with self._lock:
                    # Always close any prior socket before creating a new one
                    self._close_ws_locked()
                    self._ws_app = websocket.WebSocketApp(
                        self._cfg.ws_url,
                        on_open=self._on_open,
                        on_message=self._on_message,
                        on_error=self._on_error,
                        on_close=self._on_close,
                    )
                    ws_app = self._ws_app

                # run_forever blocks until closed
                ws_app.run_forever(ping_interval=None, ping_timeout=None)
            except Exception as e:
                logger.warning("WebSocket loop error: %s", e)

            if self._stop.is_set():
                break

            # Backoff before reconnect
            delay = self._next_backoff_s()
            logger.info("Reconnecting in %.2fs ...", delay)
            time.sleep(delay)

        # Final cleanup
        with self._lock:
            self._close_ws_locked()
        self._connected.clear()

    def _heartbeat_loop(self) -> None:
        """
        Best-effort keepalive. If no activity is seen for ping_timeout_s,
        forces a close to trigger reconnect.
        """
        while not self._stop.is_set():
            # If ping disabled, just periodically check activity.
            if self._cfg.ping_interval_s and self._cfg.ping_interval_s > 0:
                time.sleep(max(0.5, self._cfg.ping_interval_s))
            else:
                time.sleep(1.0)
            if self._stop.is_set():
                break

            if not self.is_connected():
                continue

            now = time.monotonic()
            with self._lock:
                ws = self._ws_app

            # Optional best-effort ping
            if self._cfg.ping_interval_s and self._cfg.ping_interval_s > 0:
                try:
                    if ws and ws.sock and ws.sock.connected:
                        ws.send(json.dumps({"type": "ping"}))
                except Exception:
                    # Force reconnect
                    with self._lock:
                        self._close_ws_locked()
                    continue

            # Detect stale feed (no activity)
            last = self._last_activity_monotonic
            if last > 0 and (now - last) > self._cfg.ping_timeout_s:
                logger.warning(
                    "Feed timeout (no activity for %.2fs). Forcing reconnect.",
                    now - last,
                )
                with self._lock:
                    self._close_ws_locked()

    def _next_backoff_s(self) -> float:
        base = min(self._cfg.reconnect_max_s, self._cfg.reconnect_min_s * (2 ** min(self._reconnect_attempt, 8)))
        jitter = random.uniform(0, self._cfg.reconnect_jitter_s)
        return min(self._cfg.reconnect_max_s, base + jitter)

    def _close_ws_locked(self) -> None:
        ws = self._ws_app
        self._ws_app = None
        if not ws:
            return
        try:
            # Close with status to encourage server to drop quickly
            ws.close(status=1000, reason="client_close")
        except Exception:
            try:
                ws.close()
            except Exception:
                pass

    # WebSocketApp callbacks
    def _on_open(self, ws) -> None:
        self._reconnect_attempt = 0
        self._connected.set()
        self._last_activity_monotonic = time.monotonic()
        logger.info("Connected to broadcaster: %s", self._cfg.ws_url)

    def _on_close(self, ws, close_status_code, close_msg) -> None:
        self._connected.clear()
        logger.warning("Broadcaster connection closed (%s): %s", close_status_code, close_msg)

    def _on_error(self, ws, error) -> None:
        # Keep it light: websocket-client emits noisy errors on normal shutdown.
        self._connected.clear()
        logger.warning("Broadcaster websocket error: %s", error)

    def _on_message(self, ws, message: str) -> None:
        self._last_message_monotonic = time.monotonic()
        self._last_activity_monotonic = self._last_message_monotonic
        try:
            data = json.loads(message)
        except Exception:
            return

        msg_type = data.get("type")
        if msg_type == "welcome":
            # welcome contains subscribers count, timestamp, etc.
            return
        if msg_type == "pong":
            self._last_activity_monotonic = time.monotonic()
            return

        symbol = data.get("symbol")
        if symbol:
            with self._lock:
                self._latest_by_symbol[str(symbol)] = data

        if self._on_data:
            try:
                self._on_data(data)
            except Exception as e:
                logger.exception("on_data callback failed: %s", e)


