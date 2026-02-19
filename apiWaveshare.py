# Run with: uvicorn apiWaveshare:app --host 0.0.0.0 --port 8000
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from threading import Lock, Event
from pathlib import Path
import json, time, threading

from PCA9685 import PCA9685

# --------------- hardware init ---------------
pwm = PCA9685(0x40, debug=False)
pwm.setPWMFreq(50)


def _set_dutycycle(channel: int, percent: int):
    """Set PWM duty cycle. percent: 0-100."""
    percent = max(0, min(100, percent))
    pwm.setDutycycle(channel, percent)


def _set_level(channel: int, level: int):
    """Set a channel fully HIGH (1) or fully LOW (0)."""
    pwm.setLevel(channel, 1 if level else 0)


class MotorDriver:
    def __init__(self):
        self.PWMA = 0
        self.AIN1 = 1
        self.AIN2 = 2
        self.PWMB = 5
        self.BIN1 = 3
        self.BIN2 = 4
        self._lock = Lock()          # protects all I2C / PWM calls

    def MotorRun(self, motor: int, direction: str, speed: int):
        speed = max(0, min(100, int(speed)))
        if motor == 0:
            _set_dutycycle(self.PWMA, speed)
            if direction == "forward":
                _set_level(self.AIN1, 0); _set_level(self.AIN2, 1)
            else:
                _set_level(self.AIN1, 1); _set_level(self.AIN2, 0)
        else:
            _set_dutycycle(self.PWMB, speed)
            if direction == "forward":
                _set_level(self.BIN1, 0); _set_level(self.BIN2, 1)
            else:
                _set_level(self.BIN1, 1); _set_level(self.BIN2, 0)

    def MotorStop(self, motor: int):
        if motor == 0:
            _set_dutycycle(self.PWMA, 0)
        else:
            _set_dutycycle(self.PWMB, 0)

    def Tank(self, left: float, right: float):
        """Thread-safe tank drive. Values in -1.0 .. 1.0 range."""
        with self._lock:
            self._tank_unlocked(left, right)

    def _tank_unlocked(self, left: float, right: float):
        def side(motor, val):
            if abs(val) < 1e-3:
                self.MotorStop(motor)
                return
            sp = int(abs(val) * 100)
            direction = 'forward' if val > 0 else 'backward'
            self.MotorRun(motor, direction, sp)

        side(0, left)
        side(1, right)


MOTOR = MotorDriver()

# --------------- drive task with cancellation ---------------
_drive_cancel = Event()      # set this to cancel the current drive_for
_drive_lock = Lock()         # serialises drive_for launches


def drive_for(left: float, right: float, seconds: float):
    """Drive motors, then stop. Cancellable by setting _drive_cancel."""
    MOTOR.Tank(left, right)
    # sleep in small steps so we can be cancelled quickly
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        if _drive_cancel.is_set():
            break
        time.sleep(0.05)
    MOTOR.Tank(0.0, 0.0)


def launch_drive(left: float, right: float, seconds: float):
    """Cancel any running drive, then start a new one in a background thread."""
    _drive_cancel.set()                # tell any running drive to stop
    with _drive_lock:
        _drive_cancel.clear()          # reset for the new drive
        t = threading.Thread(target=drive_for, args=(left, right, seconds),
                             daemon=True)
        t.start()


# --------------- FastAPI app ---------------
@asynccontextmanager
async def lifespan(_app: FastAPI):
    yield
    # shutdown: stop motors
    _drive_cancel.set()
    MOTOR.Tank(0.0, 0.0)


app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

STATE_PATH = Path(__file__).with_name("robotState.json")
stateLock = Lock()


class ControlData(BaseModel):
    up: bool = False
    down: bool = False
    left: bool = False
    right: bool = False
    command: str | None = None
    speed: float = Field(0.6, ge=0.0, le=1.0)
    duration: float = Field(0.8, ge=0.05, le=5.0)


robotState = {
    "up": False, "down": False, "left": False, "right": False,
    "command": "stop", "command_id": 0, "timestamp": int(time.time()),
    "speed": 0.6, "duration": 0.8,
}


def write_state_to_disk(state: dict):
    tmp = STATE_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state))
    tmp.replace(STATE_PATH)


def cmd_to_tank(cmd: str, sp: float) -> tuple[float, float]:
    c = (cmd or "").lower()
    if c in ("forward", "start", "move"): return sp, sp
    if c in ("back", "backward"):         return -sp, -sp
    if c == "left":                        return -sp, sp
    if c == "right":                       return sp, -sp
    return 0.0, 0.0


@app.get("/")
def root():
    return {"ok": True, "driver": "Waveshare PCA9685 + MotorDriver", "pwm_freq": 50}


@app.get("/control/status")
def status():
    with stateLock:
        return dict(robotState)          # return a snapshot copy


@app.post("/control/stop")
def stop():
    _drive_cancel.set()                  # cancel any timed drive
    MOTOR.Tank(0.0, 0.0)
    with stateLock:
        robotState.update({"command": "stop"})
        robotState["command_id"] += 1
        robotState["timestamp"] = int(time.time())
        snap = dict(robotState)
        write_state_to_disk(robotState)
    return {"message": "stopped", "state": snap}


@app.post("/control/set")
def update_controls(data: ControlData):
    sp = max(0.0, min(1.0, float(data.speed)))
    dur = float(data.duration)

    cmd = data.command
    if not cmd:
        if   data.up:    cmd = "forward"
        elif data.down:  cmd = "back"
        elif data.left:  cmd = "left"
        elif data.right: cmd = "right"
        else:            cmd = "stop"

    L, R = cmd_to_tank(cmd, sp)

    if (L != 0 or R != 0) and dur > 0:
        launch_drive(L, R, dur)          # cancels previous, starts new thread
    else:
        _drive_cancel.set()              # cancel any running timed drive
        MOTOR.Tank(L, R)

    with stateLock:
        robotState.update({
            "up": data.up, "down": data.down, "left": data.left, "right": data.right,
            "command": cmd, "speed": sp, "duration": dur
        })
        robotState["command_id"] += 1
        robotState["timestamp"] = int(time.time())
        snap = dict(robotState)
        write_state_to_disk(robotState)

    return {"message": "Updated & driving", "state": snap}
