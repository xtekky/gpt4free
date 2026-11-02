"""HTTP routes for browsing APIs, live feeds, triggers and notifications."""

from __future__ import annotations

from typing import Dict, List, Optional

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.status import (
    HTTP_200_OK,
    HTTP_400_BAD_REQUEST,
    HTTP_404_NOT_FOUND,
    HTTP_500_INTERNAL_SERVER_ERROR,
)

from ..browsing import (
    create_trigger,
    delete_trigger,
    execute_api,
    fast_browse,
    fetch_feed_conversation,
    get_trigger,
    list_apis,
    list_triggers,
    notifications,
    parse_trigger_prompt,
    run_due_triggers,
    run_trigger,
    search,
    update_trigger,
)


def _error(message: str, status: int = HTTP_400_BAD_REQUEST) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": {"message": message}})


async def _json_body(request: Request) -> Dict:
    try:
        body = await request.json()
    except Exception:
        return {}
    return body if isinstance(body, dict) else {}


def register_browsing_routes(app: FastAPI) -> None:
    """Register the ``/v1/browsing`` API surface on *app*."""

    @app.get("/v1/browsing/apis", tags=["browsing"])
    async def browsing_apis(
        category: Optional[str] = None,
        capability: Optional[str] = None,
        include_unavailable: bool = True,
    ):
        apis = list_apis(
            category=category,
            capability=capability,
            include_unavailable=include_unavailable,
        )
        return {
            "object": "list",
            "data": apis,
            "count": len(apis),
            "available": sum(1 for api in apis if api["available"]),
        }

    @app.api_route("/v1/search", methods=["GET", "POST"], tags=["browsing"])
    async def search_endpoint(request: Request):
        """Top-level search endpoint (GET query params or POST JSON)."""
        if request.method == "GET":
            params = request.query_params
            apis_param = params.get("apis") or ""
            body = {
                "query": params.get("q") or params.get("query") or "",
                "apis": [item.strip() for item in apis_param.split(",") if item.strip()],
                "limit": params.get("limit") or 10,
                "timeout": params.get("timeout") or 20.0,
            }
        else:
            body = await _json_body(request)
        query = body.get("query") or body.get("q") or ""
        if not str(query).strip():
            return _error("A query is required")
        try:
            return await search(
                str(query),
                apis=body.get("apis") or [],
                limit=int(body.get("limit") or 10),
                timeout=float(body.get("timeout") or 20.0),
            )
        except Exception as exc:
            return _error(f"{type(exc).__name__}: {exc}", HTTP_500_INTERNAL_SERVER_ERROR)

    @app.post("/v1/browsing/search", tags=["browsing"])
    async def browsing_search(request: Request):
        body = await _json_body(request)
        query = body.get("query") or body.get("q") or ""
        if not str(query).strip():
            return _error("A query is required")
        try:
            return await search(
                str(query),
                apis=body.get("apis") or [],
                limit=int(body.get("limit") or 10),
                timeout=float(body.get("timeout") or 20.0),
            )
        except Exception as exc:
            return _error(f"{type(exc).__name__}: {exc}", HTTP_500_INTERNAL_SERVER_ERROR)

    @app.post("/v1/browsing/browse", tags=["browsing"])
    async def browsing_browse(request: Request):
        body = await _json_body(request)
        urls = body.get("urls")
        if isinstance(urls, str):
            urls = [urls]
        if not urls:
            return _error("One or more urls are required")
        try:
            results = await fast_browse(
                urls,
                max_words=int(body.get("max_words") or 600),
                timeout=float(body.get("timeout") or 15.0),
                concurrency=int(body.get("concurrency") or 6),
                allow_local=bool(body.get("allow_local") or False),
            )
            return {"data": results, "count": len(results)}
        except Exception as exc:
            return _error(f"{type(exc).__name__}: {exc}", HTTP_500_INTERNAL_SERVER_ERROR)

    @app.post("/v1/browsing/execute", tags=["browsing"])
    async def browsing_execute(request: Request):
        body = await _json_body(request)
        api_id = body.get("api") or body.get("id")
        if not api_id:
            return _error("An api id is required")
        try:
            result = await execute_api(
                str(api_id),
                params=body.get("params") or body,
                timeout=float(body.get("timeout") or 20.0),
            )
            return {"api": api_id, "result": result}
        except ValueError as exc:
            return _error(str(exc))
        except Exception as exc:
            return _error(f"{type(exc).__name__}: {exc}", HTTP_500_INTERNAL_SERVER_ERROR)

    @app.get("/v1/browsing/feeds", tags=["browsing"])
    async def browsing_feed(
        url: str,
        limit: int = 50,
        conversation_id: Optional[str] = None,
    ):
        if not url:
            return _error("A feed url is required")
        try:
            conversation = await fetch_feed_conversation(
                url,
                conversation_id=conversation_id,
                limit=max(1, min(limit, 200)),
            )
            return conversation
        except ValueError as exc:
            return _error(str(exc))
        except Exception as exc:
            return _error(f"{type(exc).__name__}: {exc}", HTTP_500_INTERNAL_SERVER_ERROR)

    @app.post("/v1/browsing/feeds/conversation", tags=["browsing"])
    async def browsing_feed_conversation(request: Request):
        body = await _json_body(request)
        url = body.get("url") or body.get("feed_url")
        if not url:
            return _error("A feed url is required")
        try:
            return await fetch_feed_conversation(
                str(url),
                conversation_id=body.get("conversation_id"),
                limit=int(body.get("limit") or 50),
            )
        except ValueError as exc:
            return _error(str(exc))
        except Exception as exc:
            return _error(f"{type(exc).__name__}: {exc}", HTTP_500_INTERNAL_SERVER_ERROR)

    @app.post("/v1/browsing/feeds/stream", tags=["browsing"])
    async def browsing_feed_stream(request: Request):
        """Server-sent events for a feed: initial snapshot then polls.

        Clients that can keep a request open get live conversations without a
        client-side timer.  Polling clients can use ``GET /v1/browsing/feeds``.
        """
        import asyncio
        import json

        body = await _json_body(request)
        url = body.get("url") or body.get("feed_url")
        if not url:
            return _error("A feed url is required")
        interval = max(15, min(int(body.get("interval") or 60), 3600))

        async def events():
            seen = set()
            while True:
                try:
                    conversation = await fetch_feed_conversation(str(url), limit=50)
                    new_items = [
                        item
                        for item in conversation.get("items", [])
                        if item.get("id") not in seen
                    ]
                    seen.update(item.get("id") for item in conversation.get("items", []) if item.get("id"))
                    payload = {
                        "conversation": {
                            key: value
                            for key, value in conversation.items()
                            if key != "items"
                        },
                        "items": new_items if seen else conversation.get("items", []),
                    }
                    yield f"data: {json.dumps(payload)}\n\n"
                except Exception as exc:
                    yield f"data: {json.dumps({'error': str(exc)})}\n\n"
                await asyncio.sleep(interval)

        return StreamingResponse(
            events(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
        )

    # Triggers -------------------------------------------------------------
    @app.get("/v1/browsing/triggers", tags=["browsing"])
    async def browsing_triggers():
        triggers = list_triggers()
        return {"object": "list", "data": triggers, "count": len(triggers)}

    @app.post("/v1/browsing/triggers", tags=["browsing"])
    async def browsing_create_trigger(request: Request):
        body = await _json_body(request)
        try:
            return create_trigger(body)
        except ValueError as exc:
            return _error(str(exc))

    @app.post("/v1/browsing/triggers/from-prompt", tags=["browsing"])
    async def browsing_trigger_from_prompt(request: Request):
        body = await _json_body(request)
        prompt = body.get("prompt") or body.get("name") or ""
        try:
            draft = parse_trigger_prompt(str(prompt), body.get("source"))
        except ValueError as exc:
            return _error(str(exc))
        if body.get("save") or body.get("create"):
            try:
                return create_trigger(draft)
            except ValueError as exc:
                return _error(str(exc))
        return {"trigger": draft, "saved": False}

    @app.post("/v1/browsing/triggers/run-due", tags=["browsing"])
    async def browsing_run_due_triggers():
        return {"data": await run_due_triggers()}

    @app.get("/v1/browsing/triggers/{trigger_id}", tags=["browsing"])
    async def browsing_get_trigger(trigger_id: str):
        trigger = get_trigger(trigger_id)
        if trigger is None:
            return _error("Trigger not found", HTTP_404_NOT_FOUND)
        return trigger

    @app.patch("/v1/browsing/triggers/{trigger_id}", tags=["browsing"])
    async def browsing_update_trigger(trigger_id: str, request: Request):
        body = await _json_body(request)
        try:
            trigger = update_trigger(trigger_id, body)
        except ValueError as exc:
            return _error(str(exc))
        if trigger is None:
            return _error("Trigger not found", HTTP_404_NOT_FOUND)
        return trigger

    @app.delete("/v1/browsing/triggers/{trigger_id}", tags=["browsing"])
    async def browsing_delete_trigger(trigger_id: str):
        if not delete_trigger(trigger_id):
            return _error("Trigger not found", HTTP_404_NOT_FOUND)
        return {"deleted": True, "id": trigger_id}

    @app.post("/v1/browsing/triggers/{trigger_id}/run", tags=["browsing"])
    async def browsing_run_trigger(trigger_id: str, request: Request):
        body = await _json_body(request)
        try:
            result = await run_trigger(
                trigger_id, timeout=float(body.get("timeout") or 20.0)
            )
        except ValueError as exc:
            return _error(str(exc), HTTP_404_NOT_FOUND)
        except Exception as exc:
            return _error(f"{type(exc).__name__}: {exc}", HTTP_500_INTERNAL_SERVER_ERROR)
        return result

    # Notifications --------------------------------------------------------
    @app.get("/v1/browsing/notifications", tags=["browsing"])
    async def browsing_notifications(unread_only: bool = False, limit: int = 50):
        return {
            "object": "list",
            "data": notifications.list(unread_only=unread_only, limit=limit),
            "unread": notifications.unread_count(),
        }

    @app.post("/v1/browsing/notifications", tags=["browsing"])
    async def browsing_add_notification(request: Request):
        """Push a notification, e.g. when a chat or job finished."""
        body = await _json_body(request)
        title = body.get("title") or body.get("body") or "Notification"
        return notifications.add(
            title=str(title),
            body=str(body.get("body") or ""),
            kind=str(body.get("kind") or "info"),
            source=str(body.get("source") or ""),
            url=str(body.get("url") or ""),
            trigger_id=str(body.get("trigger_id") or ""),
            metadata=body.get("metadata") if isinstance(body.get("metadata"), dict) else {},
        )

    @app.post("/v1/browsing/notifications/read", tags=["browsing"])
    async def browsing_read_notifications(request: Request):
        body = await _json_body(request)
        notification_id = body.get("id")
        changed = notifications.mark_read(notification_id)
        return {"updated": changed, "unread": notifications.unread_count()}

    @app.delete("/v1/browsing/notifications", tags=["browsing"])
    async def browsing_clear_notifications():
        notifications.clear()
        return {"cleared": True}
