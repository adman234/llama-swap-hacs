"""Thin async client for the llama-swap HTTP API."""

from __future__ import annotations

import logging
from typing import Any
from urllib.parse import quote

import aiohttp
from yarl import URL

from .const import LOAD_TIMEOUT

_LOGGER = logging.getLogger(__name__)


class LlamaSwapError(Exception):
    """Base error for llama-swap API failures."""


class LlamaSwapConnectionError(LlamaSwapError):
    """The server could not be reached."""


class LlamaSwapAuthError(LlamaSwapError):
    """The API key was rejected."""


class LlamaSwapNotFoundError(LlamaSwapError):
    """The server answered 404."""


class LlamaSwapRouteMissingError(LlamaSwapNotFoundError):
    """The server has no such endpoint, so this build predates it.

    Kept distinct from a 404 about the *target*: llama-swap answers 404 both
    for a route it does not have and for a model it declines to act on, and
    those must not be treated the same way.
    """


# Go's default mux writes exactly this when nothing is registered for a path.
# llama-swap's own 404s go through its error envelope instead, so the body is
# what tells the two apart.
_GO_NOT_FOUND = "404 page not found"


class LlamaSwapClient:
    """Talks to a single llama-swap instance.

    Endpoints that only exist on newer llama-swap builds raise
    LlamaSwapNotFoundError on older servers; callers use that to feature-detect
    rather than to fail the whole update.
    """

    def __init__(
        self,
        session: aiohttp.ClientSession,
        host: str,
        port: int,
        *,
        use_ssl: bool = False,
        verify_ssl: bool = True,
        api_key: str | None = None,
    ) -> None:
        """Initialise the client."""
        self._session = session
        self._base = URL.build(
            scheme="https" if use_ssl else "http", host=host, port=port
        )
        self._verify_ssl = verify_ssl
        self._api_key = api_key or None

    @property
    def base_url(self) -> str:
        """Return the server's base URL."""
        return str(self._base)

    @property
    def _headers(self) -> dict[str, str]:
        if not self._api_key:
            return {}
        return {"Authorization": f"Bearer {self._api_key}"}

    async def _request(
        self,
        method: str,
        path: str,
        *,
        timeout: int = 30,
        expect_json: bool = True,
    ) -> Any:
        """Perform one request, translating transport errors into ours."""
        url = self._base.join(URL(path, encoded=True))
        try:
            async with self._session.request(
                method,
                url,
                headers=self._headers,
                ssl=self._verify_ssl,
                timeout=aiohttp.ClientTimeout(total=timeout),
                allow_redirects=True,
            ) as response:
                if response.status in (401, 403):
                    raise LlamaSwapAuthError(
                        f"llama-swap rejected the API key ({response.status})"
                    )
                if response.status == 404:
                    body = (await response.text())[:200]
                    if _GO_NOT_FOUND in body:
                        raise LlamaSwapRouteMissingError(f"{path} is not served")
                    raise LlamaSwapNotFoundError(f"{path} returned 404: {body}")
                if response.status >= 400:
                    body = (await response.text())[:200]
                    raise LlamaSwapError(
                        f"{method} {path} failed with {response.status}: {body}"
                    )
                if not expect_json:
                    return await response.text()
                return await response.json(content_type=None)
        except TimeoutError as err:
            raise LlamaSwapConnectionError(f"Timeout calling {path}") from err
        except aiohttp.ClientError as err:
            raise LlamaSwapConnectionError(f"Error calling {path}: {err}") from err

    async def async_health(self) -> bool:
        """Return True when the server answers its health check."""
        await self._request("GET", "/health", timeout=10, expect_json=False)
        return True

    async def async_get_models(self) -> list[dict[str, Any]]:
        """Return the OpenAI-compatible model listing."""
        payload = await self._request("GET", "/v1/models")
        if not isinstance(payload, dict):
            raise LlamaSwapError("/v1/models returned an unexpected payload")
        data = payload.get("data")
        return data if isinstance(data, list) else []

    async def async_get_running(self) -> list[dict[str, Any]]:
        """Return the models that currently have a process."""
        payload = await self._request("GET", "/running")
        if not isinstance(payload, dict):
            raise LlamaSwapError("/running returned an unexpected payload")
        running = payload.get("running")
        return running if isinstance(running, list) else []

    async def async_get_version(self) -> dict[str, Any]:
        """Return build metadata. Requires a recent llama-swap."""
        payload = await self._request("GET", "/api/version")
        return payload if isinstance(payload, dict) else {}

    async def async_get_performance(self) -> dict[str, Any]:
        """Return buffered system and GPU stats."""
        payload = await self._request("GET", "/api/performance")
        return payload if isinstance(payload, dict) else {}

    async def async_get_profiles(self) -> dict[str, Any]:
        """Return the configured profiles and the active one."""
        payload = await self._request("GET", "/api/profiles")
        return payload if isinstance(payload, dict) else {}

    async def async_set_profile(self, profile: str | None) -> None:
        """Activate a profile, or deactivate the current one with None."""
        url = self._base.join(URL("/api/profiles/active"))
        try:
            async with self._session.put(
                url,
                headers=self._headers,
                json={"name": profile},
                ssl=self._verify_ssl,
                timeout=aiohttp.ClientTimeout(total=30),
            ) as response:
                if response.status in (401, 403):
                    raise LlamaSwapAuthError("llama-swap rejected the API key")
                if response.status == 404:
                    raise LlamaSwapNotFoundError(f"Unknown profile: {profile}")
                if response.status >= 400:
                    body = (await response.text())[:200]
                    raise LlamaSwapError(
                        f"Setting profile failed with {response.status}: {body}"
                    )
        except TimeoutError as err:
            raise LlamaSwapConnectionError("Timeout setting profile") from err
        except aiohttp.ClientError as err:
            raise LlamaSwapConnectionError(f"Error setting profile: {err}") from err

    async def async_unload_all(self) -> None:
        """Stop every running model."""
        await self._request("GET", "/unload", expect_json=False)

    async def async_unload_model(self, model: str) -> None:
        """Stop one model by ID.

        Only a genuinely absent endpoint falls back to unloading everything.
        A current llama-swap also answers 404 here for a model it does not
        know and for any peer model ("no local server found"), and unloading
        the whole server because one model was refused would be a disaster.
        """
        path = f"/api/models/unload/{quote(model, safe='')}"
        try:
            await self._request("POST", path, expect_json=False)
        except LlamaSwapRouteMissingError:
            _LOGGER.debug(
                "No per-model unload endpoint on this llama-swap build; "
                "unloading every model instead"
            )
            await self.async_unload_all()

    async def async_load_model(self, model: str) -> None:
        """Load a model by sending a request through the upstream proxy.

        llama-swap has no explicit load endpoint: any request routed to a model
        swaps it in first. The upstream's own response is irrelevant here, so a
        404 raised by the upstream server still means the model got loaded. A
        404 from llama-swap itself means the model ID is unknown, which is a
        real error, so the two are told apart by the response body.
        """
        url = self._base.join(
            URL(f"/upstream/{quote(model, safe='')}/health", encoded=True)
        )
        try:
            async with self._session.get(
                url,
                headers=self._headers,
                ssl=self._verify_ssl,
                timeout=aiohttp.ClientTimeout(total=LOAD_TIMEOUT),
                allow_redirects=True,
            ) as response:
                if response.status in (401, 403):
                    raise LlamaSwapAuthError("llama-swap rejected the API key")
                body = (await response.text())[:200]
                if response.status == 404 and "model not found" in body.lower():
                    raise LlamaSwapNotFoundError(f"Unknown model: {model}")
                if response.status >= 400 and response.status != 404:
                    raise LlamaSwapError(
                        f"Loading {model} failed with {response.status}: {body}"
                    )
        except TimeoutError as err:
            raise LlamaSwapConnectionError(
                f"Timed out waiting for {model} to load"
            ) from err
        except aiohttp.ClientError as err:
            raise LlamaSwapConnectionError(f"Error loading {model}: {err}") from err
