// Pushwise: a deterministic push-level Sokoban solver.
//
// The levels made for this competition are a hallway joining independent
// rooms. Pushwise first separates those rooms, then solves each room with an
// A* search over pushes. Its estimate is the cheapest assignment of distinct
// goals to boxes, using exact one-box push distances; that assignment is a
// lower bound, so the first complete A* result is push-optimal. All limits
// below count expanded search states. The program reads no clock and uses no
// randomness, so equal inputs produce equal output bytes.
import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';
const DIRS = [
  { dx: 0, dy: -1, move: 'u', push: 'U' },
  { dx: 0, dy: 1, move: 'd', push: 'D' },
  { dx: -1, dy: 0, move: 'l', push: 'L' },
  { dx: 1, dy: 0, move: 'r', push: 'R' },
];
const ROOM_ASTAR = 400_000;
const ROOM_BFS = 400_000;
const ROOM_GREEDY = 400_000;
const WHOLE_ASTAR = 600_000;
const WHOLE_GREEDY = 600_000;
const INF = Number.POSITIVE_INFINITY;

function parseLevel(text) {
  const rows = text.split('\n').filter((row) => row.length > 0);
  if (rows.length === 0) return null;
  const w = Math.max(...rows.map((row) => row.length));
  const h = rows.length;
  const wall = new Uint8Array(w * h).fill(1);
  const goals = [];
  const boxes = [];
  let player = -1;
  rows.forEach((row, y) => {
    for (let x = 0; x < row.length; x++) {
      const ch = row[x];
      const i = y * w + x;
      if (ch === '#') continue;
      wall[i] = 0;
      if (ch === '.' || ch === '*' || ch === '+') goals.push(i);
      if (ch === '$' || ch === '*') boxes.push(i);
      if (ch === '@' || ch === '+') player = i;
    }
  });
  for (let x = 0; x < w; x++) {
    wall[x] = 1;
    wall[(h - 1) * w + x] = 1;
  }
  for (let y = 0; y < h; y++) {
    wall[y * w] = 1;
    wall[y * w + w - 1] = 1;
  }
  boxes.sort((a, b) => a - b);
  goals.sort((a, b) => a - b);
  return { w, h, wall, goals, boxes, player };
}

function inside(level, x, y) {
  return x >= 0 && y >= 0 && x < level.w && y < level.h;
}

// For every goal, the exact number of pushes needed to bring one box from
// each square to that goal when other boxes are ignored. A reverse search
// from the goal follows pushes backwards: a box now at `here` was pushed
// there from `here - step`, with the player standing at `here - 2 * step`.
function pushTables(level) {
  const size = level.w * level.h;
  const live = new Uint8Array(size);
  const tables = [];
  for (const goal of level.goals) {
    const dist = new Int32Array(size).fill(-1);
    dist[goal] = 0;
    const queue = [goal];
    for (let head = 0; head < queue.length; head++) {
      const here = queue[head];
      const hx = here % level.w;
      const hy = Math.floor(here / level.w);
      for (const { dx, dy } of DIRS) {
        const bx = hx - dx;
        const by = hy - dy;
        const sx = bx - dx;
        const sy = by - dy;
        if (!inside(level, bx, by) || !inside(level, sx, sy)) continue;
        const boxAt = by * level.w + bx;
        const standAt = sy * level.w + sx;
        if (level.wall[boxAt] || level.wall[standAt] || dist[boxAt] !== -1) continue;
        dist[boxAt] = dist[here] + 1;
        queue.push(boxAt);
      }
    }
    for (let i = 0; i < size; i++) if (dist[i] >= 0) live[i] = 1;
    tables.push(dist);
  }
  return { tables, live };
}

// The minimum-cost matching of boxes to distinct goals. With at most eight
// boxes this is an exact dynamic program over goal subsets; larger sets use
// the sum of individual minima, which is also a lower bound.
function lowerBound(boxes, tables, cache) {
  const key = boxes.join(',');
  const known = cache.get(key);
  if (known !== undefined) return known;
  let answer = INF;
  if (boxes.length === 0) {
    answer = 0;
  } else if (boxes.length > 8) {
    answer = 0;
    for (const box of boxes) {
      let best = INF;
      for (const dist of tables) if (dist[box] >= 0 && dist[box] < best) best = dist[box];
      if (best === INF) {
        answer = INF;
        break;
      }
      answer += best;
    }
  } else {
    const n = boxes.length;
    const dp = new Float64Array(1 << n).fill(INF);
    dp[0] = 0;
    for (let mask = 0; mask < dp.length; mask++) {
      if (dp[mask] === INF) continue;
      let boxIndex = 0;
      for (let m = mask; m !== 0; m &= m - 1) boxIndex++;
      if (boxIndex >= n) continue;
      for (let goalIndex = 0; goalIndex < n; goalIndex++) {
        if (mask & (1 << goalIndex)) continue;
        const cost = tables[goalIndex][boxes[boxIndex]];
        if (cost < 0) continue;
        const next = mask | (1 << goalIndex);
        if (dp[mask] + cost < dp[next]) dp[next] = dp[mask] + cost;
      }
    }
    answer = dp[dp.length - 1];
  }
  cache.set(key, answer);
  return answer;
}

// The squares the player can walk to from `start` without crossing a wall or
// a box, with the step that first reached each square (for rebuilding the
// walking part of a solution) and the region's lowest square (the canonical
// player mark of a search state).
function reachable(level, occupied, start) {
  const size = level.w * level.h;
  const seen = new Uint8Array(size);
  const prev = new Int32Array(size).fill(-1);
  const how = new Int8Array(size).fill(-1);
  const queue = [start];
  seen[start] = 1;
  prev[start] = start;
  let lowest = start;
  for (let head = 0; head < queue.length; head++) {
    const p = queue[head];
    if (p < lowest) lowest = p;
    const px = p % level.w;
    const py = Math.floor(p / level.w);
    for (let d = 0; d < 4; d++) {
      const qx = px + DIRS[d].dx;
      const qy = py + DIRS[d].dy;
      if (!inside(level, qx, qy)) continue;
      const q = qy * level.w + qx;
      if (level.wall[q] || occupied[q] || seen[q]) continue;
      seen[q] = 1;
      prev[q] = p;
      how[q] = d;
      queue.push(q);
    }
  }
  return { seen, prev, how, lowest };
}

function regionLowest(level, occupied, start) {
  const seen = new Uint8Array(level.w * level.h);
  const stack = [start];
  seen[start] = 1;
  let lowest = start;
  while (stack.length > 0) {
    const p = stack.pop();
    if (p < lowest) lowest = p;
    const px = p % level.w;
    const py = Math.floor(p / level.w);
    for (let d = 0; d < 4; d++) {
      const qx = px + DIRS[d].dx;
      const qy = py + DIRS[d].dy;
      if (!inside(level, qx, qy)) continue;
      const q = qy * level.w + qx;
      if (level.wall[q] || occupied[q] || seen[q]) continue;
      seen[q] = 1;
      stack.push(q);
    }
  }
  return lowest;
}

function walkMoves(reach, start, target) {
  if (target === start) return '';
  if (!reach.seen[target]) return null;
  let out = '';
  for (let p = target; p !== start; p = reach.prev[p]) {
    out = DIRS[reach.how[p]].move + out;
  }
  return out;
}

// A 2x2 block of walls and boxes can never change; if one of its boxes is
// off a goal the position is dead. Only blocks touching the pushed box can
// be newly formed, plus a full scan for the starting position.
function frozenAt(level, occupied, goalSet, focus) {
  const spots = [];
  if (focus === undefined) {
    for (let y = 0; y + 1 < level.h; y++) for (let x = 0; x + 1 < level.w; x++) spots.push([x, y]);
  } else {
    const fx = focus % level.w;
    const fy = Math.floor(focus / level.w);
    for (const [x, y] of [[fx - 1, fy - 1], [fx, fy - 1], [fx - 1, fy], [fx, fy]]) {
      if (x >= 0 && y >= 0 && x + 1 < level.w && y + 1 < level.h) spots.push([x, y]);
    }
  }
  for (const [x, y] of spots) {
    const a = y * level.w + x;
    const cells = [a, a + 1, a + level.w, a + level.w + 1];
    let blocked = true;
    let badBox = false;
    for (const c of cells) {
      if (!level.wall[c] && !occupied[c]) {
        blocked = false;
        break;
      }
      if (occupied[c] && !goalSet[c]) badBox = true;
    }
    if (blocked && badBox) return true;
  }
  return false;
}

// One push-level search. Modes: 'astar' (pushes + estimate; push-optimal),
// 'bfs' (pushes only; push-optimal), 'greedy' (estimate only; any solution).
// Returns { pushes, moves } on success, { capped: true } when the expansion
// budget ran out, and null when the reachable space is exhausted.
function pushSearch(level, mode, limit) {
  const size = level.w * level.h;
  const goalSet = new Uint8Array(size);
  for (const g of level.goals) goalSet[g] = 1;
  const { tables, live } = pushTables(level);
  const cache = new Map();
  const startBoxes = level.boxes.slice().sort((a, b) => a - b);
  if (startBoxes.some((b) => !live[b])) return null;
  const isGoal = (boxes) => boxes.every((b) => goalSet[b] === 1);
  if (isGoal(startBoxes)) return { pushes: 0, moves: '' };
  const startOcc = new Uint8Array(size);
  for (const b of startBoxes) startOcc[b] = 1;
  if (frozenAt(level, startOcc, goalSet)) return null;
  const h0 = lowerBound(startBoxes, tables, cache);
  if (h0 === INF) return null;

  const nodeBoxes = [startBoxes];
  const nodePlayer = [level.player];
  const nodeParent = [-1];
  const nodeFrom = [-1];
  const nodeTo = [-1];
  const nodeDir = [-1];
  const nodeG = [0];
  const best = new Map();
  best.set(startBoxes.join(',') + '|' + regionLowest(level, startOcc, level.player), 0);

  const heap = [];
  const before = (a, b) =>
    a[0] !== b[0] ? a[0] < b[0] : a[1] !== b[1] ? a[1] < b[1] : a[2] < b[2];
  const heapPush = (entry) => {
    heap.push(entry);
    let i = heap.length - 1;
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (!before(heap[i], heap[p])) break;
      const t = heap[i];
      heap[i] = heap[p];
      heap[p] = t;
      i = p;
    }
  };
  const heapPop = () => {
    const top = heap[0];
    const last = heap.pop();
    if (heap.length > 0) {
      heap[0] = last;
      for (let i = 0; ; ) {
        let c = 2 * i + 1;
        if (c >= heap.length) break;
        if (c + 1 < heap.length && before(heap[c + 1], heap[c])) c++;
        if (!before(heap[c], heap[i])) break;
        const t = heap[i];
        heap[i] = heap[c];
        heap[c] = t;
        i = c;
      }
    }
    return top;
  };
  const priority = (g, h) => (mode === 'bfs' ? g : mode === 'greedy' ? h : g + h);
  heapPush([priority(0, h0), h0, 0]);

  let sequence = 0;
  let expanded = 0;
  let found = -1;
  while (heap.length > 0) {
    const [, , nodeIndex] = heapPop();
    const boxes = nodeBoxes[nodeIndex];
    const g = nodeG[nodeIndex];
    if (expanded >= limit) return { capped: true };
    expanded++;
    const occupied = new Uint8Array(size);
    for (const b of boxes) occupied[b] = 1;
    const reach = reachable(level, occupied, nodePlayer[nodeIndex]);
    for (let bi = 0; bi < boxes.length && found < 0; bi++) {
      const b = boxes[bi];
      const bx = b % level.w;
      const by = Math.floor(b / level.w);
      for (let d = 0; d < 4; d++) {
        const sx = bx - DIRS[d].dx;
        const sy = by - DIRS[d].dy;
        const tx = bx + DIRS[d].dx;
        const ty = by + DIRS[d].dy;
        if (!inside(level, sx, sy) || !inside(level, tx, ty)) continue;
        const stance = sy * level.w + sx;
        const dest = ty * level.w + tx;
        if (!reach.seen[stance] || level.wall[dest] || occupied[dest] || !live[dest]) continue;
        const nextBoxes = boxes.slice();
        nextBoxes[bi] = dest;
        nextBoxes.sort((a, b2) => a - b2);
        const nextOcc = occupied.slice();
        nextOcc[b] = 0;
        nextOcc[dest] = 1;
        if (frozenAt(level, nextOcc, goalSet, dest)) continue;
        const h = lowerBound(nextBoxes, tables, cache);
        if (h === INF) continue;
        const key = nextBoxes.join(',') + '|' + regionLowest(level, nextOcc, b);
        const known = best.get(key);
        if (known !== undefined && known <= g + 1) continue;
        best.set(key, g + 1);
        const idx = nodeBoxes.length;
        nodeBoxes.push(nextBoxes);
        nodePlayer.push(b);
        nodeParent.push(nodeIndex);
        nodeFrom.push(b);
        nodeTo.push(dest);
        nodeDir.push(d);
        nodeG.push(g + 1);
        if (isGoal(nextBoxes)) {
          found = idx;
          break;
        }
        heapPush([priority(g + 1, h), h, ++sequence, idx]);
      }
    }
  }
  if (found < 0) return null;

  const chain = [];
  for (let s = found; s > 0; s = nodeParent[s]) chain.push(s);
  chain.reverse();
  let moves = '';
  let player = level.player;
  let boxesNow = startBoxes;
  for (const s of chain) {
    const occ = new Uint8Array(size);
    for (const b of boxesNow) occ[b] = 1;
    const reach = reachable(level, occ, player);
    const d = nodeDir[s];
    const stance = nodeFrom[s] - (DIRS[d].dy * level.w + DIRS[d].dx);
    const walk = walkMoves(reach, player, stance);
    if (walk === null) return null;
    moves += walk + DIRS[d].push;
    player = nodeFrom[s];
    boxesNow = nodeBoxes[s];
  }
  return { pushes: chain.length, moves };
}

function solveSub(level, caps) {
  const first = pushSearch(level, 'astar', caps[0]);
  if (first !== null && first.moves !== undefined) return first;
  if (first === null || !first.capped) return null;
  if (caps.length > 2) {
    const second = pushSearch(level, 'bfs', caps[1]);
    if (second !== null && second.moves !== undefined) return second;
    if (second === null || !second.capped) return null;
    const third = pushSearch(level, 'greedy', caps[2]);
    return third !== null && third.moves !== undefined ? third : null;
  }
  const second = pushSearch(level, 'greedy', caps[1]);
  return second !== null && second.moves !== undefined ? second : null;
}

// Split a level into rooms: cut the dead squares of the player's row (the
// hallway) out of the floor. A box can never cross the cut, so each part
// holding boxes is an independent room whose optimum adds to the level's.
// Returns null when the level does not split cleanly.
function roomsOf(level) {
  const { live } = pushTables(level);
  const row = Math.floor(level.player / level.w);
  const cut = new Uint8Array(level.w * level.h);
  for (let x = 0; x < level.w; x++) {
    const i = row * level.w + x;
    if (!level.wall[i] && !live[i]) cut[i] = 1;
  }
  if (level.boxes.some((b) => cut[b] === 1)) return null;
  const part = new Int32Array(level.w * level.h).fill(-1);
  const byPart = new Map();
  for (let s = 0; s < level.wall.length; s++) {
    if (level.wall[s] || cut[s] || part[s] >= 0) continue;
    const squares = [s];
    part[s] = s;
    for (let i = 0; i < squares.length; i++) {
      const p = squares[i];
      const px = p % level.w;
      const py = Math.floor(p / level.w);
      for (let d = 0; d < 4; d++) {
        const qx = px + DIRS[d].dx;
        const qy = py + DIRS[d].dy;
        if (!inside(level, qx, qy)) continue;
        const q = qy * level.w + qx;
        if (level.wall[q] || cut[q] || part[q] >= 0) continue;
        part[q] = s;
        squares.push(q);
      }
    }
    byPart.set(s, { boxes: [], goals: [] });
  }
  for (const b of level.boxes) byPart.get(part[b]).boxes.push(b);
  for (const g of level.goals) {
    if (part[g] < 0) return null;
    byPart.get(part[g]).goals.push(g);
  }
  const rooms = [...byPart.values()].filter((r) => r.boxes.length > 0);
  if (rooms.some((r) => r.boxes.length !== r.goals.length)) return null;
  if (rooms.reduce((n, r) => n + r.boxes.length, 0) !== level.boxes.length) return null;
  rooms.sort((a, b) => a.boxes[0] - b.boxes[0]);
  return rooms;
}

function endOf(level, from, moves) {
  let p = from;
  for (const ch of moves.toLowerCase()) {
    if (ch === 'u') p -= level.w;
    else if (ch === 'd') p += level.w;
    else if (ch === 'l') p -= 1;
    else if (ch === 'r') p += 1;
  }
  return p;
}

function solveLevel(text) {
  const level = parseLevel(text);
  if (level === null || level.player < 0 || level.boxes.length === 0) return '';
  const rooms = roomsOf(level);
  if (rooms !== null) {
    let player = level.player;
    let moves = '';
    let ok = true;
    for (const room of rooms) {
      const wall = level.wall.slice();
      for (const other of rooms) {
        if (other === room) continue;
        for (const b of other.done ? other.goals : other.boxes) wall[b] = 1;
      }
      const sub = {
        w: level.w,
        h: level.h,
        wall,
        goals: room.goals.slice().sort((a, b) => a - b),
        boxes: room.boxes.slice().sort((a, b) => a - b),
        player,
      };
      const hit = solveSub(sub, [ROOM_ASTAR, ROOM_BFS, ROOM_GREEDY]);
      if (hit === null) {
        ok = false;
        break;
      }
      moves += hit.moves;
      player = endOf(level, player, hit.moves);
      room.done = true;
    }
    if (ok) return moves;
  }
  const whole = solveSub(level, [WHOLE_ASTAR, WHOLE_GREEDY]);
  return whole === null ? '' : whole.moves;
}

for (const id of readdirSync(INPUT_DIR).sort()) {
  let moves = '';
  try {
    moves = solveLevel(readFileSync(join(INPUT_DIR, id, 'level.txt'), 'utf8'));
  } catch {
    moves = '';
  }
  mkdirSync(join(OUTPUT_DIR, id), { recursive: true });
  writeFileSync(join(OUTPUT_DIR, id, 'solution.txt'), `${moves}\n`);
}
