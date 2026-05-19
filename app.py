#!/usr/bin/env python3
"""
深度学习训练任务队列监控工具
单文件应用 — FastAPI + 嵌入式前端
"""

import asyncio
import json
import os
import signal
import sys
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------
CONDA_SEARCH_PATHS = [
    "~/miniconda3/envs",
    "~/anaconda3/envs",
    "/opt/conda/envs",
    "/home/*/miniconda3/envs",
    "/home/*/anaconda3/envs",
]

MAX_LOG_LINES = 2000  # 每个任务保留的最大日志行数

# ---------------------------------------------------------------------------
# 数据模型
# ---------------------------------------------------------------------------
class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class Task(BaseModel):
    id: str
    name: str
    conda_env: str = ""
    script_path: str = ""
    args: str = ""
    status: TaskStatus = TaskStatus.PENDING
    exit_code: Optional[int] = None
    created_at: float = 0.0
    started_at: Optional[float] = None
    finished_at: Optional[float] = None

    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "conda_env": self.conda_env,
            "script_path": self.script_path,
            "args": self.args,
            "status": self.status.value,
            "exit_code": self.exit_code,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
        }


class AddTaskRequest(BaseModel):
    name: str
    conda_env: str = ""
    script_path: str = ""
    args: str = ""


class ReorderRequest(BaseModel):
    task_ids: list[str]


# ---------------------------------------------------------------------------
# 任务队列管理器
# ---------------------------------------------------------------------------
class QueueManager:
    def __init__(self):
        self.tasks: list[Task] = []
        self._log_buffers: dict[str, list[dict]] = {}  # task_id → [{ts, stream, text}]
        self._current_process: Optional[asyncio.subprocess.Process] = None
        self._running_task_id: Optional[str] = None
        self._cancel_event: Optional[asyncio.Event] = None
        self._ws_clients: set[WebSocket] = set()
        self._lock = asyncio.Lock()

    # ---- WebSocket 广播 ----
    async def broadcast(self, message: dict):
        data = json.dumps(message)
        dead: list[WebSocket] = []
        for ws in self._ws_clients:
            try:
                await ws.send_text(data)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self._ws_clients.discard(ws)

    async def broadcast_state(self):
        await self.broadcast({
            "type": "queue_state",
            "tasks": [t.to_dict() for t in self.tasks],
            "running_task_id": self._running_task_id,
        })

    async def broadcast_log(self, task_id: str, line: dict):
        await self.broadcast({
            "type": "log",
            "task_id": task_id,
            "line": line,
        })

    # ---- 日志缓冲区 ----
    def append_log(self, task_id: str, stream: str, text: str):
        if task_id not in self._log_buffers:
            self._log_buffers[task_id] = []
        entry = {"ts": time.time(), "stream": stream, "text": text}
        buf = self._log_buffers[task_id]
        buf.append(entry)
        if len(buf) > MAX_LOG_LINES:
            self._log_buffers[task_id] = buf[-MAX_LOG_LINES:]

    def get_logs(self, task_id: str) -> list[dict]:
        return self._log_buffers.get(task_id, [])

    # ---- 任务管理 ----
    def _find_index(self, task_id: str) -> int:
        for i, t in enumerate(self.tasks):
            if t.id == task_id:
                return i
        return -1

    async def add_task(self, req: AddTaskRequest) -> Task:
        task = Task(
            id=uuid.uuid4().hex[:8],
            name=req.name,
            conda_env=req.conda_env,
            script_path=req.script_path,
            args=req.args,
            created_at=time.time(),
        )
        async with self._lock:
            self.tasks.append(task)
        await self.broadcast_state()
        return task

    async def remove_task(self, task_id: str) -> bool:
        async with self._lock:
            idx = self._find_index(task_id)
            if idx < 0:
                return False
            task = self.tasks[idx]
            if task.status == TaskStatus.RUNNING:
                await self._cancel_running()
                # 等待进程退出
                if self._current_process:
                    try:
                        await asyncio.wait_for(self._current_process.wait(), timeout=5)
                    except asyncio.TimeoutError:
                        self._current_process.kill()
            del self.tasks[idx]
        await self.broadcast_state()
        if task.status == TaskStatus.RUNNING:
            await self._try_start_next()
        return True

    async def reorder_tasks(self, task_ids: list[str]) -> bool:
        async with self._lock:
            if set(task_ids) != {t.id for t in self.tasks}:
                return False
            id_to_task = {t.id: t for t in self.tasks}
            self.tasks = [id_to_task[tid] for tid in task_ids]
        await self.broadcast_state()
        return True

    async def rerun_task(self, task_id: str) -> bool:
        async with self._lock:
            idx = self._find_index(task_id)
            if idx < 0:
                return False
            task = self.tasks[idx]
            if task.status not in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED):
                return False
            task.status = TaskStatus.PENDING
            task.exit_code = None
            task.started_at = None
            task.finished_at = None
            self._log_buffers.pop(task_id, None)
        await self.broadcast_state()
        return True

    async def cancel_task(self, task_id: str) -> bool:
        async with self._lock:
            idx = self._find_index(task_id)
            if idx < 0:
                return False
            task = self.tasks[idx]
            if task.status != TaskStatus.RUNNING:
                return False
            await self._cancel_running()
        return True

    # ---- 执行引擎 ----
    async def _cancel_running(self):
        """终止当前运行的进程"""
        if self._cancel_event:
            self._cancel_event.set()
        proc = self._current_process
        if proc and proc.returncode is None:
            try:
                if sys.platform == "win32":
                    proc.terminate()
                else:
                    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except Exception:
                try:
                    proc.terminate()
                except Exception:
                    pass
            # 等待一秒后强制 kill
            try:
                await asyncio.wait_for(proc.wait(), timeout=3)
            except asyncio.TimeoutError:
                try:
                    proc.kill()
                except Exception:
                    pass

    def _build_command(self, task: Task) -> str:
        """构建完整的执行命令"""
        if task.conda_env:
            python = f"{task.conda_env}/bin/python"
        else:
            python = sys.executable
        cmd = f"{python} {task.script_path}"
        if task.args:
            cmd += f" {task.args}"
        return cmd

    async def _run_task(self, task: Task):
        """执行单个任务（由 _try_start_next 调用）"""
        cmd = self._build_command(task)
        task.status = TaskStatus.RUNNING
        task.started_at = time.time()
        self._running_task_id = task.id
        cancel_event = asyncio.Event()
        self._cancel_event = cancel_event
        await self.broadcast_state()

        self.append_log(task.id, "system", f"[INFO] 开始执行: {cmd}\n")
        await self.broadcast_log(task.id, {"ts": time.time(), "stream": "system", "text": f"开始执行: {cmd}\n"})

        try:
            # 设置环境变量，让 tqdm 等进度条库输出更适合日志查看
            child_env = {
                **os.environ,
                "PYTHONUNBUFFERED": "1",
                "TQDM_MININTERVAL": "2",  # 最少 2 秒刷新一次，减少日志刷屏
            }
            if sys.platform == "win32":
                child_env["PYTHONIOENCODING"] = "utf-8"
                proc = await asyncio.create_subprocess_exec(
                    sys.executable, task.script_path, *task.args.split(),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    env=child_env,
                )
            else:
                proc = await asyncio.create_subprocess_exec(
                    "/bin/bash", "-c", cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    preexec_fn=os.setsid,
                    env=child_env,
                )
            self._current_process = proc

            def _decode(data: bytes) -> str:
                try:
                    return data.decode("utf-8")
                except UnicodeDecodeError:
                    return data.decode(sys.getdefaultencoding(), errors="replace")

            async def read_stream(stream, stream_name):
                """按 chunk 读取，正确处理 tqdm 的 \\r 进度条输出"""
                buf = ""
                while True:
                    chunk = await stream.read(8192)
                    if not chunk:
                        break
                    buf += _decode(chunk)
                    # 按行分割，同时处理 \\r（tqdm）和 \\n 作为行分隔符
                    while True:
                        cr = buf.find("\r")
                        nl = buf.find("\n")
                        term = -1
                        term_char = ""
                        if cr != -1 and nl != -1:
                            term = min(cr, nl)
                            term_char = buf[term]
                        elif cr != -1:
                            term = cr
                            term_char = "\r"
                        elif nl != -1:
                            term = nl
                            term_char = "\n"
                        else:
                            break

                        line = buf[:term]
                        # 跳过跟在 \r 后面的 \n（处理 \r\n）
                        skip = 1
                        if term_char == "\r" and term + 1 < len(buf) and buf[term + 1] == "\n":
                            skip = 2
                        buf = buf[term + skip :]

                        if line:
                            self.append_log(task.id, stream_name, line + "\n")
                            await self.broadcast_log(task.id, {
                                "ts": time.time(),
                                "stream": stream_name,
                                "text": line + "\n",
                            })
                # 输出剩余缓冲区
                if buf:
                    self.append_log(task.id, stream_name, buf + "\n")
                    await self.broadcast_log(task.id, {
                        "ts": time.time(),
                        "stream": stream_name,
                        "text": buf + "\n",
                    })

            stdout_task = asyncio.create_task(read_stream(proc.stdout, "stdout"))
            stderr_task = asyncio.create_task(read_stream(proc.stderr, "stderr"))

            # 等待进程结束或被取消
            done, pending = await asyncio.wait(
                [asyncio.create_task(proc.wait()), asyncio.create_task(cancel_event.wait())],
                return_when=asyncio.FIRST_COMPLETED,
            )

            if cancel_event.is_set():
                # 用户取消了任务
                try:
                    if sys.platform == "win32":
                        proc.terminate()
                    else:
                        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                except Exception:
                    pass
                try:
                    await asyncio.wait_for(proc.wait(), timeout=3)
                except asyncio.TimeoutError:
                    try:
                        proc.kill()
                    except Exception:
                        pass
                    await proc.wait()
                task.status = TaskStatus.CANCELLED
                self.append_log(task.id, "system", "[INFO] 任务已被取消\n")
                await self.broadcast_log(task.id, {"ts": time.time(), "stream": "system", "text": "任务已被取消\n"})
            else:
                exit_code = proc.returncode
                task.exit_code = exit_code
                if exit_code == 0:
                    task.status = TaskStatus.COMPLETED
                    self.append_log(task.id, "system", f"[INFO] 任务完成 (exit_code={exit_code})\n")
                    await self.broadcast_log(task.id, {"ts": time.time(), "stream": "system", "text": f"任务完成 (exit_code={exit_code})\n"})
                else:
                    task.status = TaskStatus.FAILED
                    self.append_log(task.id, "system", f"[ERROR] 任务失败 (exit_code={exit_code})\n")
                    await self.broadcast_log(task.id, {"ts": time.time(), "stream": "system", "text": f"任务失败 (exit_code={exit_code})\n"})

            # 等待流读取完成
            for t in pending:
                t.cancel()
            await stdout_task
            await stderr_task

        except Exception as e:
            task.status = TaskStatus.FAILED
            task.exit_code = -1
            self.append_log(task.id, "system", f"[ERROR] 执行异常: {e}\n")
            await self.broadcast_log(task.id, {"ts": time.time(), "stream": "system", "text": f"执行异常: {e}\n"})

        finally:
            task.finished_at = time.time()
            self._current_process = None
            self._running_task_id = None
            self._cancel_event = None
            await self.broadcast_state()
            # 自动启动下一个任务
            await self._try_start_next()

    async def _try_start_next(self) -> bool:
        """启动队列中下一个待执行的任务。返回 True 表示找到了待执行任务。"""
        async with self._lock:
            if self._running_task_id is not None:
                return False
            for task in self.tasks:
                if task.status == TaskStatus.PENDING:
                    # 预占位，避免 start 端点返回后状态仍为未启动
                    self._running_task_id = task.id
                    break
            else:
                return False  # 没有待执行的任务

        asyncio.create_task(self._run_task(task))
        return True


# ---------------------------------------------------------------------------
# 全局实例
# ---------------------------------------------------------------------------
queue = QueueManager()

# ---------------------------------------------------------------------------
# FastAPI 应用
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(_app: FastAPI):
    yield
    if queue._current_process and queue._current_process.returncode is None:
        try:
            if sys.platform == "win32":
                queue._current_process.terminate()
            else:
                os.killpg(os.getpgid(queue._current_process.pid), signal.SIGTERM)
        except Exception:
            pass

app = FastAPI(title="DL Training Monitor", version="1.0.0", lifespan=lifespan)


@app.get("/", response_class=HTMLResponse)
async def index():
    return HTML_TEMPLATE


# ---- 任务 API ----
@app.get("/api/tasks")
async def list_tasks():
    return [t.to_dict() for t in queue.tasks]


@app.post("/api/tasks")
async def add_task(req: AddTaskRequest):
    task = await queue.add_task(req)
    return task.to_dict()


@app.delete("/api/tasks/{task_id}")
async def delete_task(task_id: str):
    ok = await queue.remove_task(task_id)
    if not ok:
        raise HTTPException(404, "任务不存在")
    return {"ok": True}


@app.put("/api/tasks/reorder")
async def reorder_tasks(req: ReorderRequest):
    ok = await queue.reorder_tasks(req.task_ids)
    if not ok:
        raise HTTPException(400, "任务ID列表不匹配")
    await queue.broadcast_state()
    return {"ok": True}


@app.post("/api/tasks/{task_id}/rerun")
async def rerun_task(task_id: str):
    ok = await queue.rerun_task(task_id)
    if not ok:
        raise HTTPException(400, "仅已完成/失败/已取消的任务可重跑")
    return {"ok": True}


@app.post("/api/tasks/{task_id}/cancel")
async def cancel_task(task_id: str):
    ok = await queue.cancel_task(task_id)
    if not ok:
        raise HTTPException(400, "任务未在运行")
    return {"ok": True}


@app.get("/api/tasks/{task_id}/logs")
async def get_task_logs(task_id: str):
    return queue.get_logs(task_id)


@app.post("/api/queue/start")
async def start_queue():
    """手动启动队列中第一个等待中的任务"""
    if queue._running_task_id is not None:
        raise HTTPException(400, "已有任务正在运行")
    started = await queue._try_start_next()
    return {"ok": True, "started": started}


# ---- Conda 环境 ----
@app.get("/api/conda-envs")
async def list_conda_envs():
    envs = []
    home = Path.home()

    for search_path in CONDA_SEARCH_PATHS:
        expanded = str(search_path).replace("~", str(home))
        p = Path(expanded)
        if p.is_dir():
            for env_dir in sorted(p.iterdir()):
                if env_dir.is_dir():
                    python_bin = env_dir / "bin" / "python"
                    if python_bin.is_file():
                        envs.append({
                            "name": env_dir.name,
                            "path": str(env_dir),
                            "python_path": str(python_bin),
                        })

    # 当前Python环境
    envs.insert(0, {
        "name": "系统默认 (当前Python)",
        "path": "",
        "python_path": sys.executable,
    })

    return envs


# ---- 文件浏览器 ----
@app.get("/api/browse")
async def browse_filesystem(path: str = Query(default="")):
    if not path:
        path = str(Path.home())
    elif path.startswith("~"):
        path = str(Path(path).expanduser())

    p = Path(path).resolve()
    if not p.exists():
        return {"error": "路径不存在", "path": path}
    if not p.is_dir():
        return {"error": "不是目录", "path": str(p)}

    # 安全限制：不允许浏览系统敏感目录
    if sys.platform != "win32" and str(p) in ("/", "/root", "/etc", "/sys", "/proc", "/dev"):
        pass  # 允许浏览，但前端会提示

    dirs = []
    files = []

    try:
        for entry in sorted(p.iterdir()):
            if entry.name.startswith("."):
                continue
            if entry.is_dir():
                dirs.append({"name": entry.name, "path": str(entry)})
            elif entry.suffix == ".py":
                files.append({"name": entry.name, "path": str(entry)})
    except PermissionError:
        return {"error": "无权限访问", "path": str(p)}

    parent = str(p.parent) if str(p) != str(Path.home().anchor) else None

    return {
        "current": str(p),
        "parent": parent,
        "dirs": dirs,
        "files": files,
    }


# ---- WebSocket ----
@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    queue._ws_clients.add(ws)
    try:
        # 发送当前状态
        await ws.send_text(json.dumps({
            "type": "queue_state",
            "tasks": [t.to_dict() for t in queue.tasks],
            "running_task_id": queue._running_task_id,
        }))
        # 保持连接
        while True:
            data = await ws.receive_text()
            msg = json.loads(data)
            if msg.get("type") == "get_state":
                await ws.send_text(json.dumps({
                    "type": "queue_state",
                    "tasks": [t.to_dict() for t in queue.tasks],
                    "running_task_id": queue._running_task_id,
                }))
            elif msg.get("type") == "get_logs":
                tid = msg.get("task_id")
                if tid:
                    logs = queue.get_logs(tid)
                    await ws.send_text(json.dumps({
                        "type": "log_history",
                        "task_id": tid,
                        "logs": logs,
                    }))
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        queue._ws_clients.discard(ws)


# ---------------------------------------------------------------------------
# 前端 UI（嵌入式 HTML）
# ---------------------------------------------------------------------------
HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>深度学习训练任务队列</title>
<style>
:root {
  --radius: 8px;
  --font: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  --mono: "JetBrains Mono", "Fira Code", "Cascadia Code", Consolas, monospace;
}

/* ---- Dark theme (default) ---- */
[data-theme="dark"] {
  --bg: #1a1b26;
  --card: #24283b;
  --card2: #2f3348;
  --border: #3b4261;
  --text: #c0caf5;
  --text2: #787c99;
  --accent: #7aa2f7;
  --accent-hover: #89b4fa;
  --accent-bg: rgba(122,162,247,0.12);
  --green: #9ece6a;
  --red: #f7768e;
  --yellow: #e0af68;
  --orange: #ff9e64;
  --log-bg: #1a1b26;
  --log-stdout: #c0caf5;
  --toast-bg: #24283b;
  --modal-overlay: rgba(0,0,0,0.55);
  --scrollbar-thumb: #3b4261;
  --scrollbar-thumb-hover: #565f89;
  --shadow: 0 1px 3px rgba(0,0,0,0.3);
}

/* ---- Light theme ---- */
[data-theme="light"] {
  --bg: #f4f5f9;
  --card: #ffffff;
  --card2: #f0f1f5;
  --border: #d9dbe3;
  --text: #2c2e3a;
  --text2: #6c7080;
  --accent: #4a6cf7;
  --accent-hover: #3b5de7;
  --accent-bg: rgba(74,108,247,0.08);
  --green: #2e8b57;
  --red: #d14343;
  --yellow: #b8860b;
  --orange: #d4731a;
  --log-bg: #fafafa;
  --log-stdout: #2c2e3a;
  --toast-bg: #ffffff;
  --modal-overlay: rgba(0,0,0,0.25);
  --scrollbar-thumb: #ccced6;
  --scrollbar-thumb-hover: #a8abb8;
  --shadow: 0 1px 3px rgba(0,0,0,0.08);
}

* { margin:0; padding:0; box-sizing:border-box; }
body { background:var(--bg); color:var(--text); font-family:var(--font); height:100vh; display:flex; flex-direction:column; overflow:hidden; }

/* Header */
header { background:var(--card); border-bottom:1px solid var(--border); padding:10px 24px; display:flex; align-items:center; justify-content:space-between; flex-shrink:0; box-shadow:var(--shadow); }
header h1 { font-size:17px; font-weight:600; }
.header-right { display:flex; align-items:center; gap:16px; }
header .stats { font-size:12px; color:var(--text2); }
header .stats span { margin-left:14px; }
.theme-toggle { background:none; border:1px solid var(--border); color:var(--text); width:32px; height:32px; border-radius:50%; cursor:pointer; font-size:15px; display:flex; align-items:center; justify-content:center; transition:background 0.2s; }
.theme-toggle:hover { background:var(--card2); }

main { display:flex; flex:1; overflow:hidden; }
.panel { display:flex; flex-direction:column; }
.panel-left { width:480px; flex-shrink:0; border-right:1px solid var(--border); }
.panel-right { flex:1; }
.panel-header { padding:10px 16px; background:var(--card); border-bottom:1px solid var(--border); font-size:12px; font-weight:600; color:var(--text2); text-transform:uppercase; letter-spacing:0.5px; flex-shrink:0; display:flex; align-items:center; justify-content:space-between; }
.panel-header button { background:var(--accent); color:#fff; border:none; padding:5px 14px; border-radius:6px; cursor:pointer; font-size:12px; font-weight:500; }
.panel-header button:hover { background:var(--accent-hover); }

/* Task cards */
.task-list { flex:1; overflow-y:auto; padding:8px; }
.task-card { background:var(--card); border:1px solid var(--border); border-radius:var(--radius); padding:12px; margin-bottom:8px; cursor:pointer; transition:border-color 0.2s, box-shadow 0.2s; box-shadow:var(--shadow); }
.task-card:hover { border-color:var(--accent); }
.task-card.active { border-color:var(--accent); background:var(--accent-bg); }
.task-card-header { display:flex; align-items:center; gap:8px; margin-bottom:6px; }
.task-index { font-size:11px; color:var(--text2); min-width:24px; }
.task-name { font-size:14px; font-weight:500; flex:1; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.task-status { font-size:11px; padding:2px 8px; border-radius:10px; font-weight:500; white-space:nowrap; }
.status-pending { background:var(--card2); color:var(--text2); }
.status-running { background:var(--accent-bg); color:var(--accent); animation:pulse 1.5s infinite; }
.status-completed { background:rgba(46,139,87,0.12); color:var(--green); }
.status-failed { background:rgba(209,67,67,0.12); color:var(--red); }
.status-cancelled { background:rgba(255,158,100,0.12); color:var(--orange); }
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.6} }
.task-card-body { font-size:12px; color:var(--text2); }
.task-card-body .task-time { font-size:11px; color:var(--text2); margin-bottom:4px; }
.task-card-body .path { font-family:var(--mono); font-size:11px; word-break:break-all; }
.task-card-actions { display:flex; gap:6px; margin-top:8px; }
.task-card-actions button { background:var(--card2); border:1px solid var(--border); color:var(--text2); padding:4px 10px; border-radius:4px; cursor:pointer; font-size:11px; }
.task-card-actions button:hover { color:var(--text); border-color:var(--accent); }
.task-card-actions button.danger:hover { border-color:var(--red); color:var(--red); }

/* Log viewer */
.log-viewer { flex:1; overflow-y:auto; padding:12px; background:var(--log-bg); font-family:var(--mono); font-size:12px; line-height:1.6; }
.log-viewer .log-line { white-space:pre-wrap; word-break:break-all; }
.log-viewer .log-stdout { color:var(--log-stdout); }
.log-viewer .log-stderr { color:var(--red); }
.log-viewer .log-system { color:var(--text2); font-style:italic; }
.log-empty { color:var(--text2); text-align:center; padding-top:80px; font-size:14px; }

/* Add form */
.add-form-wrap { border-top:1px solid var(--border); background:var(--card); flex-shrink:0; }
.add-form-toggle { padding:10px 16px; cursor:pointer; font-size:13px; color:var(--accent); user-select:none; display:flex; align-items:center; gap:6px; }
.add-form-toggle:hover { color:var(--accent-hover); }
.add-form { padding:0 16px 14px; display:none; }
.add-form.open { display:block; }
.form-row { display:flex; gap:8px; margin-bottom:8px; align-items:center; }
.form-row label { font-size:12px; color:var(--text2); min-width:40px; }
.form-row input, .form-row select { flex:1; background:var(--bg); border:1px solid var(--border); color:var(--text); padding:6px 10px; border-radius:4px; font-size:13px; font-family:var(--font); }
.form-row input:focus, .form-row select:focus { outline:none; border-color:var(--accent); }
.form-row .browse-btn { background:var(--card2); border:1px solid var(--border); color:var(--text); padding:6px 12px; border-radius:4px; cursor:pointer; font-size:12px; white-space:nowrap; }
.form-row .browse-btn:hover { border-color:var(--accent); }
.form-row .btn-add { background:var(--accent); color:#fff; border:none; padding:8px 20px; border-radius:6px; cursor:pointer; font-size:13px; font-weight:500; }
.form-row .btn-add:hover { background:var(--accent-hover); }

/* File browser modal */
.modal-overlay { display:none; position:fixed; inset:0; background:var(--modal-overlay); z-index:100; align-items:center; justify-content:center; }
.modal-overlay.open { display:flex; }
.modal { background:var(--card); border:1px solid var(--border); border-radius:12px; width:600px; max-height:70vh; display:flex; flex-direction:column; box-shadow:0 8px 32px rgba(0,0,0,0.2); }
.modal-header { padding:14px 18px; border-bottom:1px solid var(--border); display:flex; align-items:center; justify-content:space-between; }
.modal-header h3 { font-size:15px; }
.modal-header button { background:none; border:none; color:var(--text2); cursor:pointer; font-size:18px; }
.modal-header button:hover { color:var(--text); }
.modal-breadcrumbs { padding:8px 18px; font-size:12px; color:var(--text2); font-family:var(--mono); word-break:break-all; }
.modal-breadcrumbs span { cursor:pointer; color:var(--accent); }
.modal-breadcrumbs span:hover { text-decoration:underline; }
.modal-body { flex:1; overflow-y:auto; padding:8px 12px; }
.modal-body .entry { display:flex; align-items:center; gap:8px; padding:6px 8px; border-radius:4px; cursor:pointer; font-size:13px; }
.modal-body .entry:hover { background:var(--card2); }
.modal-body .entry.dir { color:var(--accent); }
.modal-body .entry.file { color:var(--green); }
.modal-body .entry .icon { width:20px; text-align:center; }

/* Toast */
.toast-container { position:fixed; top:20px; right:20px; z-index:200; }
.toast { background:var(--toast-bg); border:1px solid var(--border); color:var(--text); padding:10px 16px; border-radius:8px; margin-bottom:8px; font-size:13px; animation:slideIn 0.3s ease; box-shadow:0 4px 12px rgba(0,0,0,0.15); }
@keyframes slideIn { from{transform:translateX(100%)} to{transform:translateX(0)} }

/* Scrollbar */
::-webkit-scrollbar { width:6px; }
::-webkit-scrollbar-track { background:transparent; }
::-webkit-scrollbar-thumb { background:var(--scrollbar-thumb); border-radius:3px; }
::-webkit-scrollbar-thumb:hover { background:var(--scrollbar-thumb-hover); }

.empty-state { text-align:center; color:var(--text2); padding:60px 20px; font-size:14px; }
.empty-state .icon { font-size:40px; margin-bottom:12px; }
</style>
</head>
<body>

<header>
  <h1>深度学习训练任务队列</h1>
  <div class="header-right">
    <div class="stats">
      <span id="stat-pending">等待: 0</span>
      <span id="stat-running">运行中: 0</span>
      <span id="stat-completed">完成: 0</span>
      <span id="stat-failed">失败: 0</span>
    </div>
    <button class="theme-toggle" onclick="toggleTheme()" title="切换亮色/暗色模式" id="theme-btn">🌙</button>
  </div>
</header>

<main>
  <div class="panel panel-left">
    <div class="panel-header">
      任务列表
      <button onclick="startFirstPending()" title="手动启动第一个等待中的任务">▶ 启动</button>
    </div>
    <div class="task-list" id="task-list">
      <div class="empty-state">
        <div class="icon">📋</div>
        暂无任务，点击下方添加
      </div>
    </div>
    <div class="add-form-wrap">
      <div class="add-form-toggle" onclick="toggleAddForm()">＋ 添加任务</div>
      <div class="add-form" id="add-form">
        <div class="form-row">
          <label>名称</label>
          <input id="input-name" placeholder="例如: train_vggt_exp1" onkeydown="if(event.key==='Enter')addTask()">
        </div>
        <div class="form-row">
          <label>环境</label>
          <select id="input-env"></select>
        </div>
        <div class="form-row">
          <label>脚本</label>
          <input id="input-script" placeholder="选择Python脚本..." readonly style="cursor:pointer" onclick="openFileBrowser()">
          <button class="browse-btn" onclick="openFileBrowser()">📂 浏览</button>
        </div>
        <div class="form-row">
          <label>参数</label>
          <input id="input-args" placeholder="额外命令行参数" onkeydown="if(event.key==='Enter')addTask()">
        </div>
        <div class="form-row" style="justify-content:flex-end">
          <button class="btn-add" onclick="addTask()">添加任务</button>
        </div>
      </div>
    </div>
  </div>
  <div class="panel panel-right">
    <div class="panel-header" id="log-panel-title">运行日志</div>
    <div class="log-viewer" id="log-viewer">
      <div class="log-empty">选择一个任务查看日志</div>
    </div>
  </div>
</main>

<!-- File browser modal -->
<div class="modal-overlay" id="file-modal">
  <div class="modal">
    <div class="modal-header">
      <h3>选择 Python 脚本</h3>
      <button onclick="closeFileBrowser()">&times;</button>
    </div>
    <div class="modal-breadcrumbs" id="file-breadcrumbs"></div>
    <div class="modal-body" id="file-browser-body"></div>
  </div>
</div>

<!-- Toast container -->
<div class="toast-container" id="toast-container"></div>

<script>
// ---- State ----
let tasks = [];
let activeTaskId = null;
let ws = null;
let wsReconnectTimer = null;
let condaEnvs = [];
let fileBrowserPath = '';
let selectedScript = '';
let logAutoScroll = true;

// ---- WebSocket ----
function connectWS() {
  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  ws = new WebSocket(`${proto}://${location.host}/ws`);
  ws.onopen = () => {
    if (wsReconnectTimer) { clearInterval(wsReconnectTimer); wsReconnectTimer = null; }
  };
  ws.onmessage = (e) => {
    const msg = JSON.parse(e.data);
    handleWSMessage(msg);
  };
  ws.onclose = () => {
    if (!wsReconnectTimer) wsReconnectTimer = setInterval(connectWS, 3000);
  };
}

let lastRunningTaskId = null;

function handleWSMessage(msg) {
  switch (msg.type) {
    case 'queue_state': {
      const prevRunning = lastRunningTaskId;
      lastRunningTaskId = msg.running_task_id;
      tasks = msg.tasks;
      renderTaskList();
      updateStats();

      // 检测任务切换：新的任务开始运行
      if (msg.running_task_id && msg.running_task_id !== prevRunning) {
        // 如果用户正在跟踪旧任务（或没选中任何任务），自动切换到新任务
        if (!activeTaskId || activeTaskId === prevRunning || !tasks.find(t => t.id === activeTaskId)) {
          selectTask(msg.running_task_id);
        }
      }
      // 任务结束且用户正在跟踪它：保持当前日志视图，让用户看到最终日志
      break;
    }
    case 'log':
      if (msg.task_id === activeTaskId) {
        appendLogLine(msg.line);
      }
      break;
    case 'log_history':
      // 防止旧请求的响应覆盖当前选中任务的日志
      if (msg.task_id === activeTaskId) {
        renderLogHistory(msg.logs);
      }
      break;
  }
}

// ---- Render ----
function renderTaskList() {
  const el = document.getElementById('task-list');
  if (tasks.length === 0) {
    el.innerHTML = '<div class="empty-state"><div class="icon">📋</div>暂无任务，点击下方添加</div>';
    return;
  }
  el.innerHTML = tasks.map((t, i) => {
    const statusText = {pending:'等待',running:'运行中',completed:'完成',failed:'失败',cancelled:'已取消'}[t.status]||t.status;
    const isActive = t.id === activeTaskId;
    return `
    <div class="task-card ${isActive?'active':''}" onclick="selectTask('${t.id}')">
      <div class="task-card-header">
        <span class="task-index">#${i+1}</span>
        <span class="task-name">${esc(t.name)}</span>
        <span class="task-status status-${t.status}">${statusText}</span>
      </div>
      <div class="task-card-body">
        <div class="task-time">${taskTimeInfo(t)}</div>
        ${t.conda_env ? '<div>🐍 ' + esc(t.conda_env.split('/').pop()||t.conda_env) + '</div>' : ''}
        <div class="path">${esc(t.script_path)||'(无脚本)'}</div>
        ${t.args ? '<div>⚙ ' + esc(t.args) + '</div>' : ''}
        ${t.exit_code !== null && t.exit_code !== undefined ? '<div>退出码: ' + t.exit_code + '</div>' : ''}
      </div>
      <div class="task-card-actions" onclick="event.stopPropagation()">
        ${t.status === 'running' ? '<button class="danger" onclick="cancelTask(\'' + t.id + '\')">⏹ 停止</button>' : ''}
        ${t.status === 'completed' || t.status === 'failed' || t.status === 'cancelled' ? '<button onclick="rerunTask(\'' + t.id + '\')">🔄 重跑</button>' : ''}
        <button onclick="moveUp('${t.id}')" ${i===0?'disabled':''}>↑</button>
        <button onclick="moveDown('${t.id}')" ${i===tasks.length-1?'disabled':''}>↓</button>
        <button class="danger" onclick="deleteTask('${t.id}')">✕ 删除</button>
      </div>
    </div>`;
  }).join('');
}

function updateStats() {
  const counts = {pending:0,running:0,completed:0,failed:0,cancelled:0};
  tasks.forEach(t => { counts[t.status] = (counts[t.status]||0) + 1; });
  document.getElementById('stat-pending').textContent = '等待: ' + counts.pending;
  document.getElementById('stat-running').textContent = '运行中: ' + counts.running;
  document.getElementById('stat-completed').textContent = '完成: ' + counts.completed;
  document.getElementById('stat-failed').textContent = '失败: ' + counts.failed;
}

function selectTask(taskId) {
  activeTaskId = taskId;
  document.getElementById('log-panel-title').textContent = '运行日志 — ' + (tasks.find(t=>t.id===taskId)?.name||taskId);
  document.getElementById('log-viewer').innerHTML = '<div class="log-empty">加载中...</div>';
  renderTaskList();
  // 请求日志历史
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({type:'get_logs',task_id:taskId}));
  }
}

function renderLogHistory(logs) {
  const el = document.getElementById('log-viewer');
  if (!logs || logs.length === 0) {
    el.innerHTML = '<div class="log-empty">暂无日志</div>';
    return;
  }
  el.innerHTML = logs.map(l => {
    const cls = 'log-' + (l.stream||'stdout');
    return `<div class="log-line ${cls}">${esc(l.text)}</div>`;
  }).join('');
  scrollLogToBottom();
}

function appendLogLine(line) {
  const el = document.getElementById('log-viewer');
  // 移除空状态
  const empty = el.querySelector('.log-empty');
  if (empty) empty.remove();
  const cls = 'log-' + (line.stream||'stdout');
  const div = document.createElement('div');
  div.className = 'log-line ' + cls;
  div.textContent = line.text;
  el.appendChild(div);
  scrollLogToBottom();
}

function scrollLogToBottom() {
  if (!logAutoScroll) return;
  const el = document.getElementById('log-viewer');
  el.scrollTop = el.scrollHeight;
}

// Auto-scroll toggle: pause when user scrolls up
document.getElementById('log-viewer').addEventListener('scroll', function() {
  const el = this;
  logAutoScroll = (el.scrollTop + el.clientHeight + 20 >= el.scrollHeight);
});

// ---- Actions ----
async function addTask() {
  const name = document.getElementById('input-name').value.trim();
  const script = selectedScript || document.getElementById('input-script').value.trim();
  if (!name) { toast('请输入任务名称'); return; }
  const body = {
    name: name,
    conda_env: document.getElementById('input-env').value,
    script_path: script,
    args: document.getElementById('input-args').value.trim(),
  };
  const resp = await fetch('/api/tasks', {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
  if (resp.ok) {
    document.getElementById('input-name').value = '';
    document.getElementById('input-script').value = '';
    document.getElementById('input-args').value = '';
    selectedScript = '';
    toast('任务已添加');
  } else {
    toast('添加失败');
  }
}

async function deleteTask(taskId) {
  if (!confirm('确认删除该任务？')) return;
  const resp = await fetch('/api/tasks/' + taskId, {method:'DELETE'});
  if (resp.ok) {
    if (activeTaskId === taskId) { activeTaskId = null; document.getElementById('log-viewer').innerHTML = '<div class="log-empty">选择一个任务查看日志</div>'; document.getElementById('log-panel-title').textContent = '运行日志'; }
    toast('任务已删除');
  } else {
    toast('删除失败');
  }
}

async function cancelTask(taskId) {
  if (!confirm('确认停止该任务？')) return;
  await fetch('/api/tasks/' + taskId + '/cancel', {method:'POST'});
  toast('已发送停止信号');
}

async function rerunTask(taskId) {
  await fetch('/api/tasks/' + taskId + '/rerun', {method:'POST'});
  toast('已加入队列');
}

async function moveUp(taskId) {
  const idx = tasks.findIndex(t => t.id === taskId);
  if (idx <= 0) return;
  const ids = tasks.map(t => t.id);
  [ids[idx-1], ids[idx]] = [ids[idx], ids[idx-1]];
  await fetch('/api/tasks/reorder', {method:'PUT',headers:{'Content-Type':'application/json'},body:JSON.stringify({task_ids:ids})});
}

async function moveDown(taskId) {
  const idx = tasks.findIndex(t => t.id === taskId);
  if (idx < 0 || idx >= tasks.length-1) return;
  const ids = tasks.map(t => t.id);
  [ids[idx], ids[idx+1]] = [ids[idx+1], ids[idx]];
  await fetch('/api/tasks/reorder', {method:'PUT',headers:{'Content-Type':'application/json'},body:JSON.stringify({task_ids:ids})});
}

async function startFirstPending() {
  const resp = await fetch('/api/queue/start', {method:'POST'});
  const data = await resp.json();
  if (data.started) {
    toast('队列已启动');
  } else if (data.detail) {
    toast(data.detail);
  } else {
    toast('没有等待中的任务');
  }
}

// ---- Add form ----
function toggleAddForm() {
  document.getElementById('add-form').classList.toggle('open');
}

// ---- Conda envs ----
async function loadCondaEnvs() {
  try {
    const resp = await fetch('/api/conda-envs');
    condaEnvs = await resp.json();
    const sel = document.getElementById('input-env');
    sel.innerHTML = condaEnvs.map(e => `<option value="${escAttr(e.path)}">${esc(e.name)}</option>`).join('');
  } catch(e) {}
}

// ---- File browser ----
async function openFileBrowser() {
  document.getElementById('file-modal').classList.add('open');
  if (!fileBrowserPath) fileBrowserPath = '';
  await navigateFileBrowser(fileBrowserPath);
}

function closeFileBrowser() {
  document.getElementById('file-modal').classList.remove('open');
}

async function navigateFileBrowser(path) {
  const resp = await fetch('/api/browse?path=' + encodeURIComponent(path));
  const data = await resp.json();
  if (data.error) {
    toast(data.error);
    return;
  }
  fileBrowserPath = data.current;
  // Breadcrumbs
  const parts = data.current.split('/').filter(Boolean);
  if (data.current.startsWith('\\\\') || data.current.includes(':\\')) {
    // Windows
    const bcHtml = parts.map((p,i) => {
      const subpath = parts.slice(0,i+1).join('\\') + '\\';
      return `<span onclick="navigateFileBrowser('${escAttr(subpath)}')">${esc(p)}</span> / `;
    }).join('');
    document.getElementById('file-breadcrumbs').innerHTML = bcHtml || data.current;
  } else {
    let cum = '';
    const bcHtml = parts.map((p,i) => {
      cum += '/' + p;
      if (i === parts.length - 1) return esc(p);
      return `<span onclick="navigateFileBrowser('${escAttr(cum)}')">${esc(p)}</span> / `;
    }).join('');
    document.getElementById('file-breadcrumbs').innerHTML = '<span onclick="navigateFileBrowser(\'/\')">/</span>' + (bcHtml?'/'+bcHtml:'');
  }
  // Entries
  let html = '';
  if (data.parent) {
    html += `<div class="entry dir" onclick="navigateFileBrowser('${escAttr(data.parent)}')"><span class="icon">📁</span>..</div>`;
  }
  data.dirs.forEach(d => {
    html += `<div class="entry dir" onclick="navigateFileBrowser('${escAttr(d.path)}')"><span class="icon">📁</span>${esc(d.name)}</div>`;
  });
  data.files.forEach(f => {
    html += `<div class="entry file" onclick="selectFile('${escAttr(f.path)}')"><span class="icon">🐍</span>${esc(f.name)}</div>`;
  });
  if (data.dirs.length === 0 && data.files.length === 0 && data.parent) {
    html += '<div style="color:var(--text2);padding:12px;text-align:center">空目录</div>';
  }
  document.getElementById('file-browser-body').innerHTML = html;
}

function selectFile(path) {
  selectedScript = path;
  document.getElementById('input-script').value = path;
  closeFileBrowser();
}

// ---- Toast ----
function toast(msg) {
  const container = document.getElementById('toast-container');
  const t = document.createElement('div');
  t.className = 'toast';
  t.textContent = msg;
  container.appendChild(t);
  setTimeout(() => { t.remove(); }, 3000);
}

// ---- Helpers ----
function esc(s) {
  if (!s) return '';
  return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}
function escAttr(s) {
  if (!s) return '';
  return s.replace(/&/g,'&amp;').replace(/"/g,'&quot;').replace(/\\/g,'\\\\');
}
function fmtTime(ts) {
  if (!ts) return '';
  const d = new Date(ts * 1000);
  return d.toLocaleString('zh-CN', {month:'2-digit',day:'2-digit',hour:'2-digit',minute:'2-digit',second:'2-digit'});
}
function fmtDuration(seconds) {
  if (seconds == null || isNaN(seconds)) return '';
  const s = Math.round(seconds);
  if (s < 60) return s + '秒';
  if (s < 3600) return Math.floor(s/60) + '分' + (s%60) + '秒';
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  return h + '时' + m + '分';
}
function taskTimeInfo(t) {
  const now = Date.now() / 1000;
  const lines = [];
  lines.push('创建: ' + fmtTime(t.created_at));
  if (t.started_at) {
    if (t.status === 'running') {
      lines.push('运行中: ' + fmtDuration(now - t.started_at));
    } else if (t.finished_at) {
      lines.push('耗时: ' + fmtDuration(t.finished_at - t.started_at));
      lines.push('结束: ' + fmtTime(t.finished_at));
    }
  }
  return lines.join(' · ');
}

// ---- Theme ----
function initTheme() {
  const saved = localStorage.getItem('dl-monitor-theme') || 'dark';
  applyTheme(saved);
}
function toggleTheme() {
  const current = document.documentElement.getAttribute('data-theme') || 'dark';
  const next = current === 'dark' ? 'light' : 'dark';
  applyTheme(next);
  localStorage.setItem('dl-monitor-theme', next);
}
function applyTheme(theme) {
  document.documentElement.setAttribute('data-theme', theme);
  document.getElementById('theme-btn').textContent = theme === 'dark' ? '☀️' : '🌙';
}

// ---- Init ----
initTheme();
loadCondaEnvs();
connectWS();
</script>
</body>
</html>"""

# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    print("=" * 50)
    print("  深度学习训练任务队列监控")
    print("  打开浏览器访问: http://localhost:8000")
    print("=" * 50)
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
