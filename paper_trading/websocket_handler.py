"""
Angel One WebSocket Handler (SmartApi SmartWebSocketV2)

Used ONLY for active trade tracking (Target/Stop) so exits are not delayed by 5-second DB polling.

Important:
- The older `SmartWebSocket` points to `wsfeeds.angelbroking.com` which is NXDOMAIN now and
  also breaks with modern `websocket-client` on_close signature.
- We use `SmartWebSocketV2` (endpoint `wss://smartapisocket.angelone.in/smart-stream`) which is current.

Design goals:
- Single websocket instance per process (avoid ghost connections)
- Subscribe to exactly ONE active token at a time (active option)
- Clean unsubscribe on exit
- Robust reconnect with backoff (via SmartWebSocketV2 retry)

Credentials (supports both naming schemes):
  - preferred: ANGEL_API_KEY, ANGEL_CLIENT_ID, ANGEL_PASSWORD, ANGEL_TOTP_SECRET
  - legacy:    API_KEY, CLIENT_ID, PASSWORD, TOTP_SECRET
"""

from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Optional

import pandas as pd

logger = logging.getLogger(__name__)

TickFn = Callable[[Dict[str, Any]], None]
StatusFn = Callable[[str], None]


def _parse_ts_utc(ts: Any) -> Optional[datetime]:
    if ts is None:
        return None
    try:
        # Angel payload can be 'timestamp' string; normalize to UTC
        dt = pd.to_datetime(ts, errors="coerce", utc=True)
        if pd.isna(dt):
            return None
        out = dt.to_pydatetime()
        if out.tzinfo is None:
            out = out.replace(tzinfo=timezone.utc)
        return out
    except Exception:
        return None


@dataclass
class AngelCredentials:
    api_key: str
    client_id: str
    password: str
    totp_secret: str

    @staticmethod
    def from_env() -> "AngelCredentials":
        # Support both naming schemes:
        # - preferred: ANGEL_API_KEY / ANGEL_CLIENT_ID / ANGEL_PASSWORD / ANGEL_TOTP_SECRET
        # - legacy (.env from VPS handoff): API_KEY / CLIENT_ID / PASSWORD / TOTP_SECRET
        api_key = (os.getenv("ANGEL_API_KEY") or os.getenv("API_KEY") or "").strip()
        client_id = (os.getenv("ANGEL_CLIENT_ID") or os.getenv("CLIENT_ID") or "").strip()
        password = (os.getenv("ANGEL_PASSWORD") or os.getenv("PASSWORD") or "").strip()
        totp_secret = (os.getenv("ANGEL_TOTP_SECRET") or os.getenv("TOTP_SECRET") or "").strip()
        missing = [k for k, v in {
            "ANGEL_API_KEY (or API_KEY)": api_key,
            "ANGEL_CLIENT_ID (or CLIENT_ID)": client_id,
            "ANGEL_PASSWORD (or PASSWORD)": password,
            "ANGEL_TOTP_SECRET (or TOTP_SECRET)": totp_secret,
        }.items() if not v]
        if missing:
            raise RuntimeError(f"Missing env vars for Angel login: {', '.join(missing)}")
        return AngelCredentials(api_key=api_key, client_id=client_id, password=password, totp_secret=totp_secret)


class AngelWebSocketHandler:
    """
    Minimal wrapper around SmartApi websocket.

    Public API:
      - start()
      - stop()
      - subscribe(token, exchange_type=2, mode=1)   # NFO=2 for options
      - unsubscribe()
      - set_tick_callback(fn)
    """

    def __init__(
        self,
        creds: Optional[AngelCredentials] = None,
        *,
        on_tick: Optional[TickFn] = None,
        on_status: Optional[StatusFn] = None,
        reconnect: bool = True,
        max_backoff_s: float = 30.0,
    ) -> None:
        self.creds = creds or AngelCredentials.from_env()
        self._on_tick = on_tick
        self._on_status = on_status
        self._reconnect = bool(reconnect)
        self._max_backoff_s = float(max_backoff_s)

        self._lock = threading.RLock()
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()

        self._active_token: Optional[str] = None
        self._active_exchange_type: int = 2  # NFO for NIFTY options
        self._active_mode: int = 1  # LTP mode
        self._pending_subscribe: bool = False

        self._obj = None
        self._sws = None
        self._feed_token: Optional[str] = None
        self._jwt_token: Optional[str] = None
        self._connected = False

    def set_tick_callback(self, fn: Optional[TickFn]) -> None:
        with self._lock:
            self._on_tick = fn

    def _emit_status(self, msg: str) -> None:
        logger.info("[WS] %s", msg)
        fn = None
        with self._lock:
            fn = self._on_status
        if fn:
            try:
                fn(msg)
            except Exception:
                pass

    def start(self) -> None:
        with self._lock:
            if self._thread and self._thread.is_alive():
                return
            self._stop.clear()
            self._thread = threading.Thread(target=self._run_forever, name="AngelWS", daemon=True)
            self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        with self._lock:
            sws = self._sws
        try:
            if sws is not None:
                sws.close_connection()  # type: ignore[attr-defined]
        except Exception:
            pass
        with self._lock:
            self._sws = None
            self._connected = False

    def subscribe(self, token: str, *, exchange_type: int = 2, mode: int = 1) -> None:
        """
        Subscribe to a token. If websocket isn't connected yet, it will subscribe on connect.
        exchange_type: 1=NSE, 2=NFO (options), per Angel examples.
        mode: 1=LTP (lightweight).
        """
        token = str(token)
        with self._lock:
            self._active_token = token
            self._active_exchange_type = int(exchange_type)
            self._active_mode = int(mode)
            self._pending_subscribe = True

        self._emit_status(f"subscribe requested token={token} exch={exchange_type} mode={mode}")
        # If already connected, try immediate subscribe
        with self._lock:
            sws = self._sws
            connected = self._connected
        if connected and sws is not None:
            self._try_subscribe_now()

    def unsubscribe(self) -> None:
        with self._lock:
            token = self._active_token
            exch = self._active_exchange_type
            sws = self._sws
            self._active_token = None
            self._pending_subscribe = False

        if not token or sws is None:
            return

        try:
            token_list = [{"exchangeType": int(exch), "tokens": [str(token)]}]
            sws.unsubscribe(correlation_id="papertrade-unsub", mode=int(self._active_mode), token_list=token_list)  # type: ignore[arg-type]
            self._emit_status(f"unsubscribed token={token}")
        except Exception as e:
            self._emit_status(f"unsubscribe failed token={token}: {e}")

    # ----- internal -----

    def _login(self) -> None:
        try:
            from SmartApi import SmartConnect  # type: ignore
            from SmartApi.smartWebSocketV2 import SmartWebSocketV2  # type: ignore
            import pyotp  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "SmartApi/pyotp not installed. Install VPS collector requirements first "
                "(see vps_data_collector/requirements.txt)."
            ) from e

        obj = SmartConnect(api_key=self.creds.api_key)
        totp = pyotp.TOTP(self.creds.totp_secret).now()
        data = obj.generateSession(self.creds.client_id, self.creds.password, totp)
        if not isinstance(data, dict) or not data.get("status"):
            raise RuntimeError(f"Angel login failed: {data}")
        jwt = (data.get("data") or {}).get("jwtToken")
        if not jwt:
            raise RuntimeError("Angel login missing jwtToken")
        feed_token = obj.getfeedToken()
        if not feed_token:
            raise RuntimeError("Angel feed token missing after login")
        self._obj = obj
        self._feed_token = str(feed_token)
        self._jwt_token = str(jwt)
        self._emit_status("login ok")

    def _build_ws(self) -> Any:
        try:
            from SmartApi.smartWebSocketV2 import SmartWebSocketV2  # type: ignore
        except Exception as e:
            raise RuntimeError(f"SmartWebSocketV2 not available in installed SmartApi: {e}") from e

        # SmartWebSocketV2 requires jwt auth token + api key + client code + feed token
        return SmartWebSocketV2(
            self._jwt_token,
            self.creds.api_key,
            self.creds.client_id,
            self._feed_token,
            max_retry_attempt=5,
            retry_strategy=1,  # exponential
            retry_delay=2,
            retry_multiplier=2,
            retry_duration=60,
        )

    def _try_subscribe_now(self) -> None:
        with self._lock:
            token = self._active_token
            exch = self._active_exchange_type
            mode = self._active_mode
            sws = self._sws
        if not token or sws is None:
            return
        try:
            token_list = [{"exchangeType": int(exch), "tokens": [str(token)]}]
            sws.subscribe(correlation_id="papertrade-sub", mode=int(mode), token_list=token_list)  # type: ignore[arg-type]
            with self._lock:
                self._pending_subscribe = False
            self._emit_status(f"subscribed token={token}")
        except Exception as e:
            self._emit_status(f"subscribe failed token={token}: {e}")

    def _handle_tick(self, raw: Any) -> None:
        # Normalize minimal fields
        if not isinstance(raw, dict):
            return
        token = raw.get("token") or raw.get("symbolToken") or raw.get("tk")
        ltp_raw = raw.get("ltp") or raw.get("last_traded_price") or raw.get("lastTradedPrice")
        ts = raw.get("timestamp") or raw.get("exchange_timestamp") or raw.get("ts")

        # SmartWebSocketV2 emits binary-parsed ints; LTP is typically in paise (x100).
        ltp = None
        try:
            if ltp_raw is not None:
                v = float(ltp_raw)
                # Heuristic: values are commonly scaled by 100 in V2.
                ltp = v / 100.0 if v > 1000 else v
        except Exception:
            ltp = None

        # exchange_timestamp in V2 is epoch millis
        ts_utc = None
        try:
            if isinstance(ts, (int, float)) and ts > 10_000_000_000:
                ts_utc = datetime.fromtimestamp(float(ts) / 1000.0, tz=timezone.utc)
            else:
                ts_utc = _parse_ts_utc(ts)
        except Exception:
            ts_utc = None
        out = {
            "token": str(token) if token is not None else None,
            "ltp": float(ltp) if ltp is not None else None,
            "volume": raw.get("volume") or raw.get("totaltradedvolume") or raw.get("tradeVolume"),
            "oi": raw.get("oi") or raw.get("openInterest"),
            "ts_utc": ts_utc or datetime.now(timezone.utc),
            "raw": raw,
        }
        fn = None
        with self._lock:
            fn = self._on_tick
        if fn:
            try:
                fn(out)
            except Exception:
                pass

    def _run_forever(self) -> None:
        backoff = 1.0
        while not self._stop.is_set():
            try:
                self._emit_status("connecting...")
                self._login()
                sws = self._build_ws()

                def _on_open(*args: Any, **kwargs: Any) -> None:
                    with self._lock:
                        self._connected = True
                        self._sws = sws
                    self._emit_status("connected")
                    # Subscribe if requested
                    with self._lock:
                        pending = self._pending_subscribe
                    if pending:
                        self._try_subscribe_now()

                def _on_close(*args: Any, **kwargs: Any) -> None:
                    with self._lock:
                        self._connected = False
                    self._emit_status("disconnected")

                def _on_error(*args: Any, **kwargs: Any) -> None:
                    self._emit_status(f"error args={args} kwargs={kwargs}")

                def _on_data(wsapp: Any, data: Any) -> None:
                    self._handle_tick(data)

                # SmartWebSocketV2 expects these public callbacks:
                # - on_open(wsapp)
                # - on_data(wsapp, parsed_dict)
                # - on_error(...)
                # - on_close(wsapp)
                sws.on_open = _on_open  # type: ignore[attr-defined]
                sws.on_data = _on_data  # type: ignore[attr-defined]
                sws.on_close = _on_close  # type: ignore[attr-defined]
                if hasattr(sws, "on_error"):
                    sws.on_error = _on_error  # type: ignore[attr-defined]

                with self._lock:
                    self._sws = sws

                backoff = 1.0
                # Blocking connect
                sws.connect()  # type: ignore[attr-defined]
            except Exception as e:
                self._emit_status(f"ws loop exception: {e}")
                with self._lock:
                    self._connected = False
                    self._sws = None
                if not self._reconnect:
                    break
                time.sleep(min(self._max_backoff_s, backoff))
                backoff = min(self._max_backoff_s, backoff * 2.0)


