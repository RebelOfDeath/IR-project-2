from dataclasses import dataclass
from uuid import uuid4

from starlette.datastructures import MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from server.context import correlation_id
from server.headers import X_REQUEST_ID


@dataclass
class CorrelationIdMiddleware:
    app: ASGIApp

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return

        headers = MutableHeaders(scope=scope)
        id_value = str(uuid4().hex)
        headers[X_REQUEST_ID] = id_value
        correlation_id.set(id_value)

        async def handle_outgoing_request(message: Message) -> None:
            if message["type"] == "http.response.start" and correlation_id.get():
                headers = MutableHeaders(scope=message)
                headers.append(X_REQUEST_ID, correlation_id.get())
            await send(message)

        await self.app(scope, receive, handle_outgoing_request)
        return
