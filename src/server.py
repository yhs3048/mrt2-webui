"""FastAPI app: index page, WAV export, and the streaming WebSocket endpoint."""

from __future__ import annotations

import asyncio
import json
import threading
import time

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse
from starlette.websockets import WebSocketState

from .config import FAVICON_PATH, FRAMES_PER_SECOND, INDEX_HTML_PATH, SAMPLE_RATE
from .engine import MRT2StreamEngine, samples_to_pcm_b64


def create_app(engine: MRT2StreamEngine) -> FastAPI:
    app = FastAPI(title="MRT2 WebUI")
    index_html = INDEX_HTML_PATH.read_text(encoding="utf-8")

    @app.get("/", response_class=HTMLResponse)
    async def index() -> str:
        return index_html

    @app.get("/favicon.ico", include_in_schema=False)
    async def favicon() -> FileResponse:
        return FileResponse(FAVICON_PATH, media_type="image/x-icon")

    @app.get("/api/export")
    async def export_wav():
        path = engine.export_session_wav()
        if path is None:
            return HTMLResponse("No audio yet. Play first.", status_code=404)
        return FileResponse(
            path,
            media_type="audio/wav",
            filename="session_export.wav",
        )

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket) -> None:
        await websocket.accept()
        stop_event = threading.Event()
        stream_task: asyncio.Task | None = None

        async def send_json(payload: dict) -> None:
            if websocket.client_state == WebSocketState.CONNECTED:
                await websocket.send_text(json.dumps(payload))

        async def run_stream(msg: dict) -> None:
            stop_event.clear()
            prompt = (msg.get("prompt") or "disco funk").strip() or "disco funk"
            temperature = float(msg.get("temperature", engine.temperature))
            top_k = int(msg.get("top_k", engine.top_k))
            cfg_musiccoca = float(msg.get("cfg_musiccoca", engine.cfg_musiccoca))
            cfg_notes = float(msg.get("cfg_notes", engine.cfg_notes))
            if msg.get("frames"):
                engine.set_yield_frames(int(msg["frames"]))
            loop = asyncio.get_running_loop()

            chunk_idx = 0
            line = ""
            try:
                await loop.run_in_executor(
                    None,
                    lambda: engine.update_sampling(
                        temperature, top_k, cfg_musiccoca, cfg_notes
                    ),
                )
                init_msg = await loop.run_in_executor(
                    None, lambda: engine.set_prompt(prompt, reset_state=True)
                )
                await send_json({"type": "model_status", **engine.model_status()})
                await send_json({"type": "reset"})
                await send_json(
                    {
                        "type": "status",
                        "text": f"Starting stream...\n{init_msg}",
                    }
                )

                while not stop_event.is_set():
                    t0 = time.perf_counter()
                    chunk_duration_sec = engine.yield_frames / FRAMES_PER_SECOND
                    samples, metrics, current_prompt = await loop.run_in_executor(
                        None, engine.stream_yield_batch
                    )
                    chunk_idx += 1
                    line = metrics.status_line(current_prompt)
                    if chunk_idx == 1:
                        line = f"Starting stream...\n{init_msg}\n{line}"

                    await send_json(
                        {
                            "type": "audio",
                            "sr": SAMPLE_RATE,
                            "pcm_b64": samples_to_pcm_b64(samples),
                            "total_sec": metrics.total_frames / FRAMES_PER_SECOND,
                        }
                    )
                    await send_json({"type": "status", "text": line})

                    wait_sec = chunk_duration_sec - (time.perf_counter() - t0)
                    if wait_sec > 0:
                        await asyncio.sleep(wait_sec)
                        if stop_event.is_set():
                            break
            except asyncio.CancelledError:
                pass
            finally:
                suffix = "\nStream stopped." if stop_event.is_set() else "\nStream ended."
                await send_json({"type": "status", "text": line + suffix if chunk_idx else suffix.strip()})
                await send_json({"type": "stream_end"})

        try:
            while True:
                raw = await websocket.receive_text()
                msg = json.loads(raw)
                cmd = msg.get("cmd")

                if cmd == "play":
                    stop_event.set()
                    if stream_task and not stream_task.done():
                        stream_task.cancel()
                        try:
                            await stream_task
                        except asyncio.CancelledError:
                            pass
                    stop_event.clear()
                    stream_task = asyncio.create_task(run_stream(msg))

                elif cmd == "stop":
                    stop_event.set()
                    if stream_task and not stream_task.done():
                        stream_task.cancel()
                        try:
                            await stream_task
                        except asyncio.CancelledError:
                            pass
                    await send_json({"type": "status", "text": "Stop requested."})

                elif cmd == "apply_prompt":
                    prompt = (msg.get("prompt") or "").strip()
                    result = await asyncio.get_running_loop().run_in_executor(
                        None, lambda: engine.set_prompt(prompt, reset_state=False)
                    )
                    delay = engine.yield_frames / FRAMES_PER_SECOND
                    await send_json(
                        {
                            "type": "status",
                            "text": f"{result}\n→ next chunk (~{delay:.1f}s) uses new prompt",
                        }
                    )

                elif cmd == "apply_image":
                    image_b64 = msg.get("image_b64") or ""
                    filename = (msg.get("filename") or "image").strip() or "image"
                    recurring = bool(msg.get("recurring", False))
                    if not image_b64:
                        await send_json({"type": "status", "text": "No image data received."})
                    else:
                        loop = asyncio.get_running_loop()
                        try:
                            # CLIP embed time is reported in the streaming status_line
                            # (clip embed: N ms), so don't overwrite status here.
                            await loop.run_in_executor(
                                None,
                                lambda: engine.apply_image_b64(
                                    image_b64,
                                    filename,
                                    reset_state=False,
                                    recurring=recurring,
                                ),
                            )
                        except Exception as e:
                            await send_json(
                                {"type": "status", "text": f"Image apply failed: {e}"}
                            )

                elif cmd == "reset":
                    result = await asyncio.get_running_loop().run_in_executor(
                        None, engine.reset_state
                    )
                    await send_json({"type": "status", "text": result})

                elif cmd == "set_frames":
                    frames = int(msg.get("frames", engine.yield_frames))
                    await asyncio.get_running_loop().run_in_executor(
                        None, lambda: engine.set_yield_frames(frames)
                    )
                    await send_json(
                        {
                            "type": "status",
                            "text": f"frames/chunk = {engine.yield_frames}",
                        }
                    )

                elif cmd == "set_model":
                    model_name = (msg.get("model") or "").strip()
                    loop = asyncio.get_running_loop()
                    await send_json(
                        {"type": "status", "text": f"Loading model: {model_name} ..."}
                    )
                    try:
                        result = await loop.run_in_executor(
                            None, lambda: engine.set_model(model_name)
                        )
                        await send_json({"type": "status", "text": result})
                    except Exception as e:
                        await send_json(
                            {"type": "status", "text": f"Model load failed: {e}"}
                        )
                    await send_json(
                        {"type": "model_status", **engine.model_status()}
                    )

                elif cmd == "set_preallocate":
                    enabled = bool(msg.get("enabled", True))
                    result = engine.set_preallocate(enabled)
                    await send_json({"type": "status", "text": result})
                    await send_json(
                        {"type": "model_status", **engine.model_status()}
                    )

                elif cmd == "get_model":
                    await send_json(
                        {"type": "model_status", **engine.model_status()}
                    )

                elif cmd == "update_sampling":
                    await asyncio.get_running_loop().run_in_executor(
                        None,
                        lambda: engine.update_sampling(
                            float(msg.get("temperature", engine.temperature)),
                            int(msg.get("top_k", engine.top_k)),
                            float(msg.get("cfg_musiccoca", engine.cfg_musiccoca)),
                            float(msg.get("cfg_notes", engine.cfg_notes)),
                        ),
                    )

        except WebSocketDisconnect:
            stop_event.set()
            if stream_task and not stream_task.done():
                stream_task.cancel()

    return app
