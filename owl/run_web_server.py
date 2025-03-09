import uvicorn
import asyncio
import logging
import queue
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from loguru import logger as loguru_logger

from run import construct_society
from utils import run_society

app = FastAPI()
templates = Jinja2Templates(directory="owl/templates")

# Create two queues for log handling
log_queue = asyncio.Queue()       # Async queue for frontend SSE
sync_log_queue = queue.Queue()    # Thread-safe sync queue for logging

# Intercept standard logging and forward to Loguru
class InterceptHandler(logging.Handler):
    def emit(self, record):
        try:
            level = loguru_logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        frame, depth = logging.currentframe(), 2
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        loguru_logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())

# Critical step: fully reset logging configuration
logging.root.handlers = [InterceptHandler()]
logging.root.setLevel(logging.INFO)

# Ensure all loggers propagate logs to the root logger
for logger_name in logging.root.manager.loggerDict:
    logger = logging.getLogger(logger_name)
    logger.handlers = []
    logger.propagate = True  # Force propagation to root logger

# Add log handler to push formatted logs into synchronous queue
loguru_logger.add(lambda msg: sync_log_queue.put(msg), format="{level} | {message}")

# On startup, create a log transfer task
@app.on_event("startup")
async def startup_event():
    asyncio.create_task(transfer_logs())

async def transfer_logs():
    """Transfer logs from the synchronous queue to the asynchronous queue."""
    while True:
        try:
            message = sync_log_queue.get_nowait()
            await log_queue.put(message)
        except queue.Empty:
            await asyncio.sleep(0.1)
        except Exception as e:
            loguru_logger.error(f"Log transfer error: {e}")

# SSE log stream
async def log_stream():
    while True:
        message = await log_queue.get()
        yield f"data: {message}\n\n"

@app.get("/logs")
async def stream_logs():
    """Endpoint for frontend to access real-time log stream."""
    return StreamingResponse(log_stream(), media_type="text/event-stream")

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("web.html", {"request": request})

class PromptRequest(BaseModel):
    prompt: str

@app.post("/run")
async def run(request: PromptRequest):
    loguru_logger.info("Constructing society and processing request...")

    def blocking_task(prompt: str):
        society = construct_society(prompt)
        answer, chat_history, token_count = run_society(society)
        logging.info("Internal system log test")  # Testing standard logging integration
        loguru_logger.success(f"Task completed successfully. Token count: {token_count}")
        return answer

    answer = await asyncio.to_thread(blocking_task, request.prompt)
    return {"result": answer}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
