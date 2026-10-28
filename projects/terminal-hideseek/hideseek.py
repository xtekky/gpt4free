#!/usr/bin/env python3
"""HIDE & SEEK 3D - a terminal raycasting game.

You are the SEEKER. The HIDER is a glowing orb somewhere in the maze.
Find it before the clock runs out. The orb runs away when it sees you.

Pure standard library: no dependencies, no external assets.
"""

from __future__ import annotations

import argparse
import math
import os
import random
import select
import shutil
import signal
import sys
import termios
import time
import tty
from dataclasses import dataclass, field

# --------------------------------------------------------------------------
# terminal helpers
# --------------------------------------------------------------------------

RESET = "\x1b[0m"
HIDE_CURSOR = "\x1b[?25l"
SHOW_CURSOR = "\x1b[?25h"
ALT_SCREEN = "\x1b[?1049h"
MAIN_SCREEN = "\x1b[?1049l"
CLEAR = "\x1b[2J"
HOME = "\x1b[H"


def fg(r: int, g: int, b: int) -> str:
    return f"\x1b[38;2;{r};{g};{b}m"


def bg(r: int, g: int, b: int) -> str:
    return f"\x1b[48;2;{r};{g};{b}m"


class Terminal:
    """Raw-mode terminal with a double-buffered character grid."""

    def __init__(self) -> None:
        self.fd = sys.stdin.fileno()
        self.saved = None
        self.w = 80
        self.h = 24
        self.buf: list[list[tuple[str, str, str]]] = []
        self._prev: list[str] = []
        self.truecolor = True
        self.measure()

    # -- lifecycle ---------------------------------------------------------
    def __enter__(self) -> "Terminal":
        self.saved = termios.tcgetattr(self.fd)
        tty.setraw(self.fd)
        sys.stdout.write(ALT_SCREEN + HIDE_CURSOR + CLEAR)
        sys.stdout.flush()
        return self

    def __exit__(self, *exc) -> None:
        sys.stdout.write(RESET + SHOW_CURSOR + MAIN_SCREEN)
        sys.stdout.flush()
        if self.saved is not None:
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self.saved)

    def measure(self) -> None:
        size = shutil.get_terminal_size((80, 24))
        self.w = max(40, size.columns)
        self.h = max(16, size.lines)
        self.buf = [[(" ", "", "") for _ in range(self.w)] for _ in range(self.h)]
        self._prev = [""] * self.h

    # -- drawing -----------------------------------------------------------
    def clear(self, top: str = "", bottom: str = "") -> None:
        for row in self.buf:
            for i in range(self.w):
                row[i] = (" ", top, bottom)

    def put(self, x: int, y: int, text: str, f: str = "", b: str = "") -> None:
        if not (0 <= y < self.h):
            return
        row = self.buf[y]
        for ch in text:
            if 0 <= x < self.w:
                row[x] = (ch, f, b)
            x += 1

    def put_center(self, y: int, text: str, f: str = "", b: str = "") -> None:
        self.put(max(0, (self.w - len(text)) // 2), y, text, f, b)

    def flush(self) -> None:
        out = [HOME]
        for y, row in enumerate(self.buf):
            line = "".join(ch + f + b for ch, f, b in row)
            if line != self._prev[y]:
                out.append(f"\x1b[{y + 1};1H" + line + RESET)
                self._prev[y] = line
        sys.stdout.write("".join(out))
        sys.stdout.flush()

    # -- input -------------------------------------------------------------
    def read_keys(self) -> list[str]:
        keys: list[str] = []
        while select.select([sys.stdin], [], [], 0)[0]:
            data = os.read(self.fd, 64)
            if not data:
                break
            keys.extend(decode_keys(data.decode("utf-8", "ignore")))
        return keys


def decode_keys(data: str) -> list[str]:
    keys: list[str] = []
    i = 0
    while i < len(data):
        ch = data[i]
        if ch == "\x1b":
            seq = data[i : i + 3]
            if seq.startswith("\x1b[A"):
                keys.append("up")
                i += 3
                continue
            if seq.startswith("\x1b[B"):
                keys.append("down")
                i += 3
                continue
            if seq.startswith("\x1b[C"):
                keys.append("right")
                i += 3
                continue
            if seq.startswith("\x1b[D"):
                keys.append("left")
                i += 3
                continue
            keys.append("esc")
            i += 1
            continue
        keys.append(ch.lower())
        i += 1
    return keys


# --------------------------------------------------------------------------
# world
# --------------------------------------------------------------------------

WALL = "#"
MAP_W = 31
MAP_H = 27


def generate_maze(rng: random.Random, loop_chance: float = 0.14) -> list[str]:
    """Recursive-backtracker maze with extra loops knocked out.

    Every open cell is guaranteed reachable from every other open cell.
    """
    grid = [[WALL] * MAP_W for _ in range(MAP_H)]
    start = (1, 1)
    grid[start[1]][start[0]] = "."
    stack = [start]
    while stack:
        x, y = stack[-1]
        neighbours = []
        for dx, dy in ((2, 0), (-2, 0), (0, 2), (0, -2)):
            nx, ny = x + dx, y + dy
            if 0 < nx < MAP_W - 1 and 0 < ny < MAP_H - 1 and grid[ny][nx] == WALL:
                neighbours.append((nx, ny, dx, dy))
        if not neighbours:
            stack.pop()
            continue
        nx, ny, dx, dy = rng.choice(neighbours)
        grid[y + dy // 2][x + dx // 2] = "."
        grid[ny][nx] = "."
        stack.append((nx, ny))

    # knock out walls to create loops and open rooms
    for y in range(1, MAP_H - 1):
        for x in range(1, MAP_W - 1):
            if grid[y][x] != WALL:
                continue
            horizontal = grid[y][x - 1] == "." and grid[y][x + 1] == "."
            vertical = grid[y - 1][x] == "." and grid[y + 1][x] == "."
            if (horizontal or vertical) and rng.random() < loop_chance:
                grid[y][x] = "."

    return ["".join(row) for row in grid]


MAZE: list[str] = generate_maze(random.Random(0))

# wall id -> (top rgb, bottom rgb)
WALL_STYLES = {
    "1": ((196, 92, 60), (92, 40, 26)),      # brick
    "2": ((96, 150, 210), (40, 66, 100)),    # slate
    "3": ((150, 200, 120), (58, 84, 46)),    # moss
    "4": ((210, 190, 120), (96, 84, 50)),    # sand
}


def is_wall(x: float, y: float) -> bool:
    ix, iy = int(x), int(y)
    if ix < 0 or iy < 0 or ix >= MAP_W or iy >= MAP_H:
        return True
    return MAZE[iy][ix] == WALL


def wall_style(ix: int, iy: int) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    return WALL_STYLES[str((ix * 7 + iy * 13) % 4 + 1)]


def open_cells() -> list[tuple[int, int]]:
    return [
        (x, y)
        for y in range(MAP_H)
        for x in range(MAP_W)
        if MAZE[y][x] != WALL
    ]


# --------------------------------------------------------------------------
# entities
# --------------------------------------------------------------------------


@dataclass
class Player:
    x: float
    y: float
    angle: float = 0.0
    bob: float = 0.0
    stamina: float = 1.0
    sprinting: bool = False


@dataclass
class Hider:
    x: float
    y: float
    vx: float = 0.0
    vy: float = 0.0
    panic: float = 0.0
    phase: float = 0.0
    trail: list[tuple[float, float]] = field(default_factory=list)


# --------------------------------------------------------------------------
# game
# --------------------------------------------------------------------------

FOV = math.radians(72)
MAX_DEPTH = 26.0
SPRITE_CHARS = " .:-=+*#%@"
FOG = (6, 8, 16)


class Game:
    def __init__(self, seed: int | None = None, time_limit: float = 120.0) -> None:
        self.rng = random.Random(seed)
        self.time_limit = time_limit
        self.state = "menu"
        self.message = ""
        self.message_until = 0.0
        self.frame = 0
        self.fps = 0.0
        self._fps_t = time.time()
        self._fps_n = 0
        self.new_round()

    # -- setup -------------------------------------------------------------
    def new_round(self) -> None:
        global MAZE
        MAZE = generate_maze(self.rng)
        cells = open_cells()
        px, py = self.rng.choice(cells)
        self.player = Player(px + 0.5, py + 0.5, self.rng.uniform(0, math.tau))
        far = sorted(
            cells,
            key=lambda c: (c[0] - px) ** 2 + (c[1] - py) ** 2,
            reverse=True,
        )
        hx, hy = self.rng.choice(far[: max(1, len(far) // 4)])
        self.hider = Hider(hx + 0.5, hy + 0.5)
        self.time_left = self.time_limit
        self.found = False
        self.distance = math.dist((self.player.x, self.player.y), (self.hider.x, self.hider.y))
        self.best_distance = self.distance
        self.steps = 0
        self.visibility = 0.0
        self.state = "play"
        self.notify("Find the glowing orb. It flees when it sees you.", 4.0)

    def notify(self, text: str, seconds: float = 2.0) -> None:
        self.message = text
        self.message_until = time.time() + seconds

    # -- input -------------------------------------------------------------
    def handle(self, keys: list[str]) -> None:
        for k in keys:
            if k in ("q", "esc"):
                if self.state == "play":
                    self.state = "paused"
                else:
                    self.state = "quit"
            elif k == "r":
                self.new_round()
            elif k == "n" and self.state in ("won", "lost"):
                self.new_round()
            elif k == "p" and self.state == "play":
                self.state = "paused"
            elif k == " " and self.state == "paused":
                self.state = "play"

    # -- simulation --------------------------------------------------------
    def update(self, dt: float, keys: list[str]) -> None:
        if self.state != "play":
            return
        self.frame += 1
        self.time_left -= dt
        if self.time_left <= 0:
            self.time_left = 0
            self.state = "lost"
            return

        p = self.player
        held = set(keys)
        forward = 1.0 if ("w" in held or "up" in held) else 0.0
        back = 1.0 if ("s" in held or "down" in held) else 0.0
        strafe = (1.0 if ("d" in held or "right" in held) else 0.0) - (
            1.0 if ("a" in held or "left" in held) else 0.0
        )
        turn = (1.0 if "l" in held else 0.0) - (1.0 if "j" in held else 0.0)

        p.sprinting = ("shift" in held or "k" in held) and p.stamina > 0.02
        speed = 3.4 if p.sprinting else 2.1
        if p.sprinting:
            p.stamina = max(0.0, p.stamina - dt * 0.45)
        else:
            p.stamina = min(1.0, p.stamina + dt * 0.28)

        p.angle = (p.angle + turn * dt * 2.6) % math.tau

        move = forward - back
        if move or strafe:
            dx = math.cos(p.angle) * move + math.cos(p.angle + math.pi / 2) * strafe
            dy = math.sin(p.angle) * move + math.sin(p.angle + math.pi / 2) * strafe
            length = math.hypot(dx, dy) or 1.0
            dx, dy = dx / length * speed * dt, dy / length * speed * dt
            self._try_move(p, dx, dy)
            p.bob += dt * (11.0 if p.sprinting else 7.0)
            self.steps += 1

        self._update_hider(dt)

        self.distance = math.dist((p.x, p.y), (self.hider.x, self.hider.y))
        self.best_distance = min(self.best_distance, self.distance)
        if self.distance < 0.75:
            self.found = True
            self.state = "won"

    def _try_move(self, p: Player, dx: float, dy: float) -> None:
        radius = 0.22
        if not is_wall(p.x + dx + math.copysign(radius, dx), p.y):
            p.x += dx
        if not is_wall(p.x, p.y + dy + math.copysign(radius, dy)):
            p.y += dy

    def _update_hider(self, dt: float) -> None:
        h = self.hider
        p = self.player
        h.phase += dt
        dx, dy = h.x - p.x, h.y - p.y
        dist = math.hypot(dx, dy) or 1e-6
        nx, ny = dx / dist, dy / dist

        # line of sight check
        seen = self._line_of_sight(p.x, p.y, h.x, h.y)
        self.visibility = 1.0 if seen else 0.0
        if seen and dist < 14.0:
            h.panic = min(1.0, h.panic + dt * 1.6)
        else:
            h.panic = max(0.0, h.panic - dt * 0.5)

        if h.panic > 0.05:
            # flee: away from player, with wall sliding
            flee_speed = 1.5 + 1.9 * h.panic
            fx, fy = -nx, -ny
            # add a little wobble so it does not get stuck in corners
            wob = math.sin(h.phase * 3.1) * 0.5
            fx += -ny * wob
            fy += nx * wob
            fl = math.hypot(fx, fy) or 1.0
            h.vx += (fx / fl * flee_speed - h.vx) * min(1.0, dt * 4.0)
            h.vy += (fy / fl * flee_speed - h.vy) * min(1.0, dt * 4.0)
        else:
            # idle drift
            h.vx *= 0.9
            h.vy *= 0.9

        step_x, step_y = h.vx * dt, h.vy * dt
        if not is_wall(h.x + step_x + math.copysign(0.2, step_x), h.y):
            h.x += step_x
        else:
            h.vx = 0.0
        if not is_wall(h.x, h.y + step_y + math.copysign(0.2, step_y)):
            h.y += step_y
        else:
            h.vy = 0.0

        h.trail.append((h.x, h.y))
        if len(h.trail) > 26:
            h.trail.pop(0)

    def _line_of_sight(self, x0: float, y0: float, x1: float, y1: float) -> bool:
        dist = math.hypot(x1 - x0, y1 - y0)
        steps = int(dist * 8) + 1
        for i in range(1, steps):
            t = i / steps
            if is_wall(x0 + (x1 - x0) * t, y0 + (y1 - y0) * t):
                return False
        return True

    # -- rendering ---------------------------------------------------------
    def render(self, term: Terminal) -> None:
        term.measure()
        w, h = term.w, term.h
        view_h = h - 3
        horizon = view_h // 2
        zbuf = [MAX_DEPTH] * w

        self._render_walls(term, w, view_h, horizon, zbuf)
        self._render_hider(term, w, view_h, horizon, zbuf)
        self._render_hud(term, w, h, view_h)

    def _render_walls(
        self, term: Terminal, w: int, view_h: int, horizon: int, zbuf: list[float]
    ) -> None:
        p = self.player
        bob = math.sin(p.bob) * 0.35
        for col in range(w):
            cam = (col / w - 0.5) * FOV
            ray = p.angle + cam
            rx, ry = math.cos(ray), math.sin(ray)
            dist, side, hit = self._cast(p.x, p.y, rx, ry)
            zbuf[col] = dist
            if hit is None:
                continue
            corrected = max(0.05, dist * math.cos(cam))
            wall_h = min(view_h, int(view_h / corrected))
            top = int(horizon - wall_h / 2 + bob)
            bottom = top + wall_h
            top_rgb, bottom_rgb = wall_style(*hit)
            shade = 1.0 / (1.0 + corrected * 0.16)
            if side == 1:
                shade *= 0.72
            # subtle vertical texture banding
            band = 0.86 + 0.14 * ((int(hit[0] * 3 + hit[1] * 5) % 3) / 2.0)
            shade *= band
            for y in range(max(0, top), min(view_h, bottom)):
                t = (y - top) / max(1, wall_h - 1)
                r = int((top_rgb[0] * (1 - t) + bottom_rgb[0] * t) * shade)
                g = int((top_rgb[1] * (1 - t) + bottom_rgb[1] * t) * shade)
                b = int((top_rgb[2] * (1 - t) + bottom_rgb[2] * t) * shade)
                term.put(col, y, " ", "", bg(r, g, b))
            # ceiling / floor
            for y in range(0, max(0, top)):
                t = y / max(1, horizon)
                term.put(col, y, " ", "", bg(int(10 + 26 * t), int(12 + 30 * t), int(30 + 60 * t)))
            for y in range(min(view_h, bottom), view_h):
                t = (y - bottom) / max(1, view_h - bottom)
                term.put(col, y, " ", "", bg(int(30 - 18 * t), int(26 - 14 * t), int(22 - 12 * t)))

    def _cast(self, x: float, y: float, rx: float, ry: float):
        """DDA raycast. Returns (distance, side, cell)."""
        map_x, map_y = int(x), int(y)
        delta_x = abs(1.0 / rx) if rx else 1e9
        delta_y = abs(1.0 / ry) if ry else 1e9
        if rx < 0:
            step_x, side_x = -1, (x - map_x) * delta_x
        else:
            step_x, side_x = 1, (map_x + 1.0 - x) * delta_x
        if ry < 0:
            step_y, side_y = -1, (y - map_y) * delta_y
        else:
            step_y, side_y = 1, (map_y + 1.0 - y) * delta_y

        side = 0
        for _ in range(256):
            if side_x < side_y:
                side_x += delta_x
                map_x += step_x
                side = 0
            else:
                side_y += delta_y
                map_y += step_y
                side = 1
            if map_x < 0 or map_y < 0 or map_x >= MAP_W or map_y >= MAP_H:
                return MAX_DEPTH, side, None
            if MAZE[map_y][map_x] == WALL:
                dist = side_x - delta_x if side == 0 else side_y - delta_y
                return max(0.05, dist), side, (map_x, map_y)
        return MAX_DEPTH, side, None

    def _render_hider(
        self, term: Terminal, w: int, view_h: int, horizon: int, zbuf: list[float]
    ) -> None:
        p, hd = self.player, self.hider
        dx, dy = hd.x - p.x, hd.y - p.y
        dist = math.hypot(dx, dy)
        if dist < 0.05:
            return
        angle = (math.atan2(dy, dx) - p.angle + math.pi) % math.tau - math.pi
        if abs(angle) > FOV * 0.75:
            return
        corrected = dist * math.cos(angle)
        if corrected <= 0.1:
            return
        screen_x = int((0.5 + angle / FOV) * w)
        size = max(1.0, min(view_h * 0.9, view_h / corrected * 0.55))
        bob = math.sin(p.bob) * 0.35
        center_y = horizon + bob + (view_h / corrected) * 0.12
        pulse = 0.75 + 0.25 * math.sin(hd.phase * 6.0)
        panic = hd.panic

        # trail of fading echoes
        for i, (tx, ty) in enumerate(hd.trail[:-1]):
            tdx, tdy = tx - p.x, ty - p.y
            tdist = math.hypot(tdx, tdy)
            if tdist < 0.2:
                continue
            tangle = (math.atan2(tdy, tdx) - p.angle + math.pi) % math.tau - math.pi
            if abs(tangle) > FOV * 0.75:
                continue
            tcorr = tdist * math.cos(tangle)
            if tcorr <= 0.1:
                continue
            tsx = int((0.5 + tangle / FOV) * w)
            tsize = max(1.0, min(view_h * 0.5, view_h / tcorr * 0.22))
            fade = (i + 1) / len(hd.trail)
            tcy = horizon + bob + (view_h / tcorr) * 0.12
            self._blit_orb(term, tsx, int(tcy), tsize, zbuf, tcorr, fade * 0.45, 0.0, w, view_h)

        self._blit_orb(term, screen_x, int(center_y), size, zbuf, corrected, pulse, panic, w, view_h)

    def _blit_orb(
        self,
        term: Terminal,
        cx: int,
        cy: int,
        size: float,
        zbuf: list[float],
        dist: float,
        intensity: float,
        panic: float,
        w: int,
        view_h: int,
    ) -> None:
        radius = size / 2
        r0, r1 = int(cy - radius), int(cy + radius)
        c0, c1 = int(cx - radius), int(cx + radius)
        for y in range(max(0, r0), min(view_h, r1 + 1)):
            for x in range(max(0, c0), min(w, c1 + 1)):
                if dist >= zbuf[x]:
                    continue
                nx = (x - cx) / radius
                ny = (y - cy) / radius
                d = math.hypot(nx, ny)
                if d > 1.0:
                    continue
                # fake sphere shading
                nz = math.sqrt(max(0.0, 1.0 - d * d))
                light = 0.35 + 0.65 * nz
                glow = (1.0 - d) ** 2
                base = 255 * intensity * light
                r = int(min(255, base * (1.0 + panic * 0.4)))
                g = int(min(255, base * (0.55 - panic * 0.35)))
                b = int(min(255, base * (0.25 - panic * 0.2) + glow * 60))
                ch = SPRITE_CHARS[min(len(SPRITE_CHARS) - 1, int(glow * 9))]
                term.put(x, y, ch, fg(max(0, r), max(0, g), max(0, b)), "")

    def _render_hud(self, term: Terminal, w: int, h: int, view_h: int) -> None:
        p = self.player
        bar_y = view_h
        # crosshair
        term.put(w // 2, view_h // 2, "+", fg(230, 230, 230))
        # top bar
        term.put(1, bar_y, " " * (w - 2), fg(220, 220, 220), bg(18, 20, 30))
        time_txt = f"TIME {self.time_left:5.1f}s"
        dist_txt = f"DIST {self.distance:5.1f}m"
        seen = "SPOTTED" if self.visibility else "HIDDEN "
        seen_col = fg(255, 90, 90) if self.visibility else fg(120, 220, 140)
        term.put(2, bar_y, time_txt, fg(255, 220, 120), bg(18, 20, 30))
        term.put(2 + len(time_txt) + 2, bar_y, dist_txt, fg(160, 200, 255), bg(18, 20, 30))
        term.put(w - 12, bar_y, seen, seen_col, bg(18, 20, 30))
        # stamina bar
        filled = int(p.stamina * 12)
        term.put(2, bar_y + 1, "STAMINA ", fg(200, 200, 200))
        term.put(10, bar_y + 1, "=" * filled + "-" * (12 - filled), fg(120, 230, 160))
        term.put(26, bar_y + 1, f"FPS {self.fps:4.1f}", fg(140, 140, 160))
        term.put(40, bar_y + 1, "WASD move  J/L turn  SHIFT sprint  R restart  Q quit", fg(120, 120, 140))

        if self.message and time.time() < self.message_until:
            term.put_center(view_h // 2 + 3, self.message, fg(255, 240, 180))

        if self.state == "paused":
            self._overlay(term, w, view_h, "PAUSED", "SPACE resume   R restart   Q quit")
        elif self.state == "won":
            self._overlay(
                term,
                w,
                view_h,
                "FOUND IT!",
                f"time left {self.time_left:.1f}s   N new round   Q quit",
                fg(140, 255, 170),
            )
        elif self.state == "lost":
            self._overlay(
                term,
                w,
                view_h,
                "TIME UP",
                f"closest approach {self.best_distance:.1f}m   N new round   Q quit",
                fg(255, 130, 130),
            )

    def _overlay(self, term: Terminal, w: int, h: int, title: str, sub: str, color: str = "") -> None:
        color = color or fg(255, 255, 255)
        box_w = max(len(title), len(sub)) + 6
        x0 = (w - box_w) // 2
        y0 = h // 2 - 2
        for y in range(y0, y0 + 5):
            term.put(x0, y, " " * box_w, "", bg(10, 12, 22))
        term.put_center(y0 + 1, title, color, bg(10, 12, 22))
        term.put_center(y0 + 3, sub, fg(190, 190, 200), bg(10, 12, 22))

    # -- main loop ---------------------------------------------------------
    def run(self) -> None:
        with Terminal() as term:
            last = time.time()
            while self.state != "quit":
                now = time.time()
                dt = min(0.1, now - last)
                last = now
                keys = term.read_keys()
                self.handle(keys)
                self.update(dt, keys)
                self.render(term)
                term.flush()
                self._fps_n += 1
                if now - self._fps_t >= 0.5:
                    self.fps = self._fps_n / (now - self._fps_t)
                    self._fps_n = 0
                    self._fps_t = now
                time.sleep(0.012)


def main() -> int:
    parser = argparse.ArgumentParser(description="HIDE & SEEK 3D - terminal raycasting game")
    parser.add_argument("--seed", type=int, default=None, help="deterministic maze seed")
    parser.add_argument("--time", type=float, default=120.0, help="round time limit in seconds")
    args = parser.parse_args()

    if not sys.stdin.isatty():
        print("This game needs an interactive terminal (TTY).", file=sys.stderr)
        return 1

    signal.signal(signal.SIGINT, lambda *_: sys.exit(0))
    try:
        Game(seed=args.seed, time_limit=args.time).run()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())