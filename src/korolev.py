"""Королёв ИИ: протокол веб-клиента, НЕ публичный OpenAI API и НЕ API-key auth.

Официальные frontend-источники (проверены 2026-11-13):
https://assistant.korolev-ai.bmstu.ru/_app/immutable/chunks/CbpPkxpb.js
  POST /api/chat/completions; Authorization: Bearer <session token>; JSON body.
https://assistant.korolev-ai.bmstu.ru/_app/immutable/chunks/Ki5iZDfP.js
  model/messages/stream/params/features; временный чат имеет chat_id="local".
https://assistant.korolev-ai.bmstu.ru/_app/immutable/nodes/41.i5nMeham.js
  штатный вход /oauth/oidc/login. Форма/CSRF читаются из фактического HTML SSO.

Никаких файлов с cookies/токенами, API-ключей, CRUD чатов или профиля.
Секреты: KOROLEV_LOGIN/KOROLEV_PASSWORD только в окружении текущего процесса.
Для интерактивного запуска без shell history: prompt_credentials(), затем runner
в том же Python-процессе через runpy.run_module(..., run_name="__main__").
"""

import hashlib
import math
import os
import threading
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from getpass import getpass
from html.parser import HTMLParser
from http.cookiejar import CookieJar
from urllib.parse import parse_qs, urlencode, urljoin, urlparse

import httpx

ORIGIN = "https://assistant.korolev-ai.bmstu.ru"
COMPLETION_PATH = "/api/chat/completions"
MODELS_PATH = "/api/models"
AUTH_HOSTS = {
    "assistant.korolev-ai.bmstu.ru",
    "frontend.korolev-ai.bmstu.ru",
    "sso.bmstu.ru",
}
SOURCES = [
    ORIGIN + "/_app/immutable/chunks/CbpPkxpb.js",
    ORIGIN + "/_app/immutable/chunks/Ki5iZDfP.js",
]
_secrets: set[str] = set()
_secret_lock = threading.RLock()
_call_lock = (
    threading.RLock()
)  # Даже при ошибочной настройке runner — строго по одному.
_client = None
_fingerprint = None


def _remember(value):
    # Короткие технические cookie-флаги вроде "1" не являются секретом:
    # их замена повредила бы числа в сериализованном JSON. LOGIN/PASSWORD
    # любой длины отдельно очищаются llm._scrub из runtime env.
    if isinstance(value, str) and len(value) >= 8:
        with _secret_lock:
            _secrets.add(value)


def scrub(text: str) -> str:
    """Последняя страховка журнала; реальные session/cookie/CSRF — только в RAM."""
    with _secret_lock:
        for value in sorted(_secrets, key=len, reverse=True):
            text = text.replace(value, "***")
    return text


def prompt_credentials():
    """Никаких .env/файлов: оба значения вводятся скрыто и живут в процессе."""
    for name, label in (
        ("KOROLEV_LOGIN", "Логин SSO МГТУ (ввод скрыт): "),
        ("KOROLEV_PASSWORD", "Пароль SSO МГТУ (ввод скрыт): "),
    ):
        if not os.getenv(name):
            os.environ[name] = getpass(label)
        _remember(os.environ[name])


class KorolevError(Exception):
    """Только контролируемое сообщение: без response body, заголовков или URL."""

    def __init__(self, message, *, status_code=None, retry_after_s=None):
        super().__init__(message)
        self.status_code = status_code
        self.retry_after_s = retry_after_s
        # Не повторяем скрыто генерацию, auth и исчерпанные квоты.
        self.retryable = False


class _Forms(HTMLParser):
    def __init__(self):
        super().__init__()
        self.forms = []
        self.current = None

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "form":
            self.current = {
                "action": attrs.get("action", ""),
                "method": attrs.get("method", "get").lower(),
                "names": [],
                "hidden": {},
            }
            self.forms.append(self.current)
        if tag == "input" and self.current is not None and attrs.get("name"):
            self.current["names"].append(attrs["name"])
            if attrs.get("type", "").lower() == "hidden":
                self.current["hidden"][attrs["name"]] = attrs.get("value", "")

    def handle_endtag(self, tag):
        if tag == "form":
            self.current = None


def _retry_after(value):
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        try:
            return max(
                0.0,
                (
                    parsedate_to_datetime(value) - datetime.now(timezone.utc)
                ).total_seconds(),
            )
        except (ValueError, TypeError, OverflowError):
            return None


def _model_metadata(model):
    """Только сведения модели, не owner/user_id и не полный model_item из UI."""
    info = model.get("info") or {}
    params = info.get("params") or {}
    system = params.get("system")
    return {
        "id": model.get("id"),
        "name": model.get("name"),
        "model_defaults": {
            k: v
            for k, v in params.items()
            if k
            in {
                "temperature",
                "max_tokens",
                "num_ctx",
                "num_predict",
                "seed",
                "top_p",
                "top_k",
                "stream_response",
            }
            and isinstance(v, (str, int, float, bool, type(None)))
        },
        "configured_system_prompt_present": bool(system),
        "configured_system_prompt_sha256": (
            hashlib.sha256(system.encode()).hexdigest()
            if isinstance(system, str) and system
            else None
        ),
        "configured_system_prompt_chars": len(system)
        if isinstance(system, str)
        else None,
        "declared_limits": {
            k: model[k]
            for k in (
                "context_length",
                "max_model_len",
                "max_input_tokens",
                "max_output_tokens",
            )
            if k in model and isinstance(model[k], (int, float, str, type(None)))
        },
        "top_provider_limits": {
            k: (model.get("top_provider") or {})[k]
            for k in ("context_length", "max_completion_tokens")
            if k in (model.get("top_provider") or {})
        },
        "server_prompt_injection": "unknown; web backend may apply model defaults/filters",
        "context_limit_verified": False,
    }


class KorolevSession:
    def __init__(self, timeout_s=300, *, transport=None):
        self.http = httpx.Client(
            cookies=CookieJar(),
            timeout=timeout_s,
            follow_redirects=False,
            transport=transport,
            headers={
                "User-Agent": "Mozilla/5.0",
                "Accept": "application/json,text/html",
            },
        )
        self._token = None
        self._authenticated = False
        self._models = None
        self._paused_until = 0.0

    def close(self):
        self.http.close()
        self.http.cookies.clear()
        self._token = None
        self._authenticated = False

    def _capture(self, url):
        parsed = urlparse(str(url))
        for query in (parsed.query, parsed.fragment):
            values = parse_qs(query)
            for name in ("token", "state", "code", "nonce", "session_state"):
                for value in values.get(name, []):
                    _remember(value)
            if parsed.hostname == "assistant.korolev-ai.bmstu.ru" and values.get(
                "token"
            ):
                self._token = values["token"][0]
        for cookie in self.http.cookies.jar:
            _remember(cookie.value)
            if (
                cookie.name == "token"
                and cookie.domain.lstrip(".") == "assistant.korolev-ai.bmstu.ru"
            ):
                self._token = cookie.value

    def _request(
        self, method, url, *, data=None, json=None, authenticated=False, headers=None
    ):
        # Cookies автоматически ограничены доменами CookieJar. Bearer не уходит на SSO.
        headers = dict(headers or {})
        if authenticated and self._token:
            if urlparse(url).hostname != "assistant.korolev-ai.bmstu.ru":
                raise KorolevError(
                    "Session authorization outside assistant is forbidden"
                )
            headers["Authorization"] = "Bearer " + self._token
        for _ in range(12):
            parsed = urlparse(url)
            if parsed.scheme != "https" or parsed.hostname not in AUTH_HOSTS:
                raise KorolevError("SSO redirect outside approved HTTPS hosts; stopped")
            try:
                response = self.http.request(
                    method, url, data=data, json=json, headers=headers
                )
            except httpx.HTTPError as exc:
                raise KorolevError(
                    "HTTP transport failed (" + type(exc).__name__ + ")"
                ) from None
            self._capture(response.url)
            if response.status_code not in (301, 302, 303, 307, 308):
                return response
            if authenticated:
                raise KorolevError(
                    "Authenticated request redirected; session may have expired"
                )
            location = response.headers.get("location")
            if not location:
                raise KorolevError("SSO redirect missing Location")
            url = urljoin(str(response.url), location)
            self._capture(url)
            if response.status_code == 303 or (
                response.status_code in (301, 302) and method == "POST"
            ):
                method, data, json = "GET", None, None
                headers = {}
            elif method == "POST":
                # Нельзя переслать login/password при 307/308 на другой адрес.
                raise KorolevError("SSO requested credential POST redirect; stopped")
        raise KorolevError("SSO redirect limit reached")

    def _check(self, response, stage):
        status = response.status_code
        if status == 429:
            delay = _retry_after(response.headers.get("retry-after"))
            self._paused_until = time.monotonic() + max(delay or 60.0, 1.0)
            raise KorolevError(
                stage + ": rate/quota limit; no automatic retry",
                status_code=429,
                retry_after_s=delay,
            )
        if status in (401, 403):
            self._authenticated = False
            self._token = None
        if status >= 400:
            raise KorolevError(stage + ": HTTP " + str(status), status_code=status)

    def login(self):
        login, password = os.getenv("KOROLEV_LOGIN"), os.getenv("KOROLEV_PASSWORD")
        if not login or not password:
            raise KorolevError(
                "Set KOROLEV_LOGIN and KOROLEV_PASSWORD in the runtime process"
            )
        _remember(login)
        _remember(password)
        response = self._request("GET", ORIGIN + "/oauth/oidc/login")
        self._check(response, "SSO form")
        forms = _Forms()
        forms.feed(response.text)
        form = next(
            (f for f in forms.forms if {"login", "password"} <= set(f["names"])), None
        )
        if form is None:
            raise KorolevError(
                "SSO login form unavailable; interactive MFA/CAPTCHA may be required"
            )
        action = urljoin(str(response.url), form["action"])
        parsed = urlparse(action)
        if (parsed.scheme, parsed.hostname, parsed.path, form["method"]) != (
            "https",
            "sso.bmstu.ru",
            "/portal4/local",
            "post",
        ):
            raise KorolevError("SSO form action/method changed; stopped")
        fields = dict(form["hidden"])
        for value in fields.values():
            _remember(value)
        fields.update(login=login, password=password)
        response = self._request(
            "POST",
            action,
            data=fields,
            headers={"Referer": str(response.url), "Origin": "https://sso.bmstu.ru"},
        )
        self._check(response, "SSO submit")
        forms = _Forms()
        forms.feed(response.text)
        if any("password" in f["names"] for f in forms.forms):
            raise KorolevError(
                "SSO returned login form; no retry (credentials/MFA/CAPTCHA)"
            )
        if urlparse(str(response.url)).hostname != "assistant.korolev-ai.bmstu.ru":
            raise KorolevError(
                "SSO did not return to assistant; interactive action required"
            )
        self._authenticated = True

    def models(self):
        if not self._authenticated:
            self.login()
        if self._models is None:
            response = self._request("GET", ORIGIN + MODELS_PATH, authenticated=True)
            self._check(response, "Models")
            try:
                models = response.json()["data"]
                if not isinstance(models, list):
                    raise ValueError()
                self._models = {m["id"]: m for m in models}
            except (ValueError, TypeError, KeyError):
                raise KorolevError("Models response schema changed") from None
        return self._models

    def model_metadata(self, model_id):
        model = self.models().get(model_id)
        if model is None:
            raise KorolevError("Requested model is not in the authorized model list")
        return _model_metadata(model)

    def generate(self, model_id, messages, temperature, max_tokens, params):
        if self._paused_until > time.monotonic():
            raise KorolevError(
                "Quota cooldown active; no request sent", status_code=429
            )
        model_meta = self.model_metadata(model_id)
        if params:
            raise KorolevError(
                "Extra web parameters are not verified; params must be empty"
            )
        if not math.isfinite(temperature) or temperature < 0:
            raise KorolevError("Invalid temperature")
        if max_tokens is not None and (
            not isinstance(max_tokens, int)
            or isinstance(max_tokens, bool)
            or max_tokens < 1
        ):
            raise KorolevError("max_tokens must be a positive integer")
        request_params = {"temperature": temperature}
        if max_tokens is not None:
            request_params["max_tokens"] = max_tokens
        body = {
            "model": model_id,
            "messages": messages,
            "stream": False,
            "params": request_params,
            "chat_id": "local",
            "features": {
                "web_search": False,
                "image_generation": False,
                "code_interpreter": False,
            },
        }
        started = time.perf_counter()
        response = self._request(
            "POST", ORIGIN + COMPLETION_PATH, json=body, authenticated=True
        )
        latency = time.perf_counter() - started
        self._check(response, "Generation")
        try:
            data = response.json()
            choice = data["choices"][0]
            message = choice["message"]
            text = message.get("content")
            if text is not None and not isinstance(text, str):
                raise ValueError()
            if data.get("error"):
                raise ValueError()
        except (ValueError, TypeError, KeyError, IndexError):
            raise KorolevError(
                "Generation JSON schema changed; no automatic retry"
            ) from None
        usage = data.get("usage") or {}
        details = usage.get("prompt_tokens_details") or {}
        raw = {
            k: data[k]
            for k in (
                "id",
                "object",
                "created",
                "model",
                "choices",
                "usage",
                "system_fingerprint",
            )
            if k in data
        }
        return {
            "text": text,
            "tokens_in": usage.get("prompt_tokens"),
            "tokens_out": usage.get("completion_tokens"),
            "tokens_cached": details.get("cached_tokens"),
            "model_reported": data.get("model"),
            "finish_reason": choice.get("finish_reason"),
            "reasoning_text": message.get("reasoning_content"),
            "latency_s": latency,
            "raw_generation_response": raw,
            "provider_metadata": {
                "protocol": "korolev_web_frontend",
                "auth": "SSO session Bearer + in-memory CookieJar; NOT API key",
                "request_endpoint": ORIGIN + COMPLETION_PATH,
                "request_body": body,
                "frontend_sources": SOURCES,
                "chat_persistence": "temporary chat_id=local; no chat CRUD/background tasks",
                "stream": False,
                "temperature_sent": temperature,
                "temperature_effective_server": "not reported",
                "max_tokens_sent": max_tokens,
                "effective_limits": {
                    "declared_by_model_list": model_meta["declared_limits"],
                    "top_provider": model_meta["top_provider_limits"],
                    "server_enforcement": "not tested; no quota stress/probing",
                },
                "seed_sent": None,
                "model_metadata": model_meta,
                "usage": usage,
                "http_status": response.status_code,
            },
        }


def get_session(timeout_s=300):
    """Одна переиспользуемая сессия на процесс; fingerprint никогда не журналируется."""
    global _client, _fingerprint
    fingerprint = hashlib.sha256(
        (
            os.getenv("KOROLEV_LOGIN", "") + "\0" + os.getenv("KOROLEV_PASSWORD", "")
        ).encode()
    ).digest()
    with _call_lock:
        if _client is None or fingerprint != _fingerprint:
            if _client is not None:
                _client.close()
            _client = KorolevSession(timeout_s)
            _fingerprint = fingerprint
        return _client


def close_session():
    global _client, _fingerprint
    with _call_lock:
        if _client is not None:
            _client.close()
        _client, _fingerprint = None, None


def call(cfg, model_id, messages, temperature, max_tokens, repeat):
    # repeat различает стандартный кэш; недокументированный seed не выдумывается.
    with _call_lock:
        return get_session(cfg.get("timeout_s", 300)).generate(
            model_id, messages, temperature, max_tokens, cfg.get("params") or {}
        )
