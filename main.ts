// Pushwise: a push-optimal Sokoban solver for hallway-and-rooms levels.
//
// A level is split into rooms by cutting the dead squares of the player's row (the hallway): a box standing
// there can be pushed to no goal, so no box ever crosses the cut and each room is an independent push problem
// whose optimum adds up to the level's. Each room is solved with an A* search over pushes whose estimate is a
// minimum-cost matching (Hungarian) between the boxes and the goals over per-goal push-distance tables built
// by reverse pull-BFS; the estimate never overstates the pushes still needed, so the first solution the A*
// search returns is push-optimal. Dead squares (a box there reaches no goal) and frozen 2x2 blocks prune the
// search. If the A* budget runs out, a plain push-BFS runs, then a greedy best-first search that settles for
// any solution; a level that does not split into rooms is searched whole with the same ladder. Every budget
// is a count of expanded states, never the clock, and every tie is broken by square order, so two runs of
// this program write byte-identical answers.
import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
const I = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const O = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

// Work budgets, in expanded states. A room of at most four boxes on a 6x6 slot is small; these caps sit far
// above what the searches below ever reach on such rooms and far inside the set's wall budget.
const ROOM_ASTAR = 600_000;
const ROOM_BFS = 600_000;
const ROOM_GREEDY = 600_000;
const WHOLE_ASTAR = 1_200_000;
const WHOLE_BFS = 800_000;
const WHOLE_GREEDY = 800_000;
// A cost no real assignment can reach: stands in for infinity inside the matching, kept finite so the
// arithmetic there can never turn into NaN.
const BIG = 1_000_000;

const DIRS: [number, number, string, string][] = [
  [0, -1, 'u', 'U'],
  [0, 1, 'd', 'D'],
  [-1, 0, 'l', 'L'],
  [1, 0, 'r', 'R'],
];

interface Level {
  w: number;
  h: number;
  wall: Uint8Array;
  goals: number[];
  boxes: number[];
  player: number;
}

function parseLevel(text: string): Level {
  const rows = text.split('\n').filter((r) => r.length > 0);
  const h = rows.length;
  const w = Math.max(...rows.map((r) => r.length));
  const wall = new Uint8Array(w * h).fill(1);
  const goals: number[] = [];
  const boxes: number[] = [];
  let player = -1;
  rows.forEach((row, y) => {
    for (let x = 0; x < row.length; x++) {
      const ch = row[x]!;
      const i = y * w + x;
      if (ch === '#') continue;
      wall[i] = 0;
      if (ch === '.' || ch === '*' || ch === '+') goals.push(i);
      if (ch === '$' || ch === '*') boxes.push(i);
      if (ch === '@' || ch === '+') player = i;
    }
  });
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      if (x === 0 || y === 0 || x === w - 1 || y === h - 1) wall[y * w + x] = 1;
    }
  }
  boxes.sort((a, b) => a - b);
  goals.sort((a, b) => a - b);
  return { w, h, wall, goals, boxes, player };
}

// Per-goal push distances: dist[g][p] is the fewest pushes that bring a box standing on p onto goal g when the
// other boxes are ignored, found by pulling a box backwards off the goal over open floor. A square absent from
// every table is dead: a box pushed there can never reach a goal.
function pushTables(level: Level): Int32Array[] {
  const { w, wall, goals } = level;
  const size = wall.length;
  return goals.map((g) => {
    const dist = new Int32Array(size).fill(-1);
    dist[g] = 0;
    const queue = [g];
    for (let qi = 0; qi < queue.length; qi++) {
      const c = queue[qi]!;
      const cx = c % w;
      for (const [dx, dy] of DIRS) {
        const d = dy * w + dx;
        const p = c - d;
        const stand = c - 2 * d;
        if (p < 0 || stand < 0 || p >= size || stand >= size) continue;
        if (dx !== 0 && (p % w === 0 || p % w === w - 1 || stand % w === 0 || stand % w === w - 1) && Math.floor(p / w) !== Math.floor(c / w)) continue;
        if (dx !== 0 && Math.floor(p / w) !== Math.floor(c / w)) continue;
        if (wall[p] || wall[stand] || dist[p] !== -1) continue;
        void cx;
        dist[p] = dist[c]! + 1;
        queue.push(p);
      }
    }
    return dist;
  });
}

// The Hungarian algorithm on a square cost matrix whose unreachable pairs cost BIG: the cheapest assignment's
// total, or BIG-or-more when some box cannot be assigned a reachable goal. Costs are push distances, so the
// cheapest box-to-goal assignment never overstates the pushes a position still needs.
function hungarian(cost: number[][]): number {
  const n = cost.length;
  if (n === 0) return 0;
  const u = new Array<number>(n + 1).fill(0);
  const v = new Array<number>(n + 1).fill(0);
  const p = new Array<number>(n + 1).fill(0);
  const way = new Array<number>(n + 1).fill(0);
  for (let i = 1; i <= n; i++) {
    p[0] = i;
    let j0 = 0;
    const minv = new Array<number>(n + 1).fill(BIG);
    const used = new Array<boolean>(n + 1).fill(false);
    do {
      used[j0] = true;
      const i0 = p[j0]!;
      let delta = BIG;
      let j1 = 0;
      for (let j = 1; j <= n; j++) {
        if (used[j]) continue;
        const cur = cost[i0 - 1]![j - 1]! - u[i0]! - v[j]!;
        if (cur < minv[j]!) {
          minv[j] = cur;
          way[j] = j0;
        }
        if (minv[j]! < delta) {
          delta = minv[j]!;
          j1 = j;
        }
      }
      for (let j = 0; j <= n; j++) {
        if (used[j]) {
          u[p[j]!] += delta;
          v[j] -= delta;
        } else {
          minv[j] -= delta;
        }
      }
      j0 = j1;
    } while (p[j0] !== 0);
    do {
      const j1 = way[j0]!;
      p[j0] = p[j1]!;
      j0 = j1;
    } while (j0 !== 0);
  }
  return -v[0]!;
}

// The squares the player reaches from `from` with the boxes of `occupied` in the way; returns the region's
// lowest square, the state's canonical player mark.
function regionLow(level: Level, occupied: Uint8Array, from: number, seen: Int32Array, stamp: number, stack: Int32Array): number {
  const { w, wall } = level;
  let low = from;
  let n = 0;
  stack[n++] = from;
  seen[from] = stamp;
  while (n > 0) {
    const p = stack[--n]!;
    if (p < low) low = p;
    const up = p - w;
    const down = p + w;
    const left = p - 1;
    const right = p + 1;
    if (!wall[up] && !occupied[up] && seen[up] !== stamp) { seen[up] = stamp; stack[n++] = up; }
    if (!wall[down] && !occupied[down] && seen[down] !== stamp) { seen[down] = stamp; stack[n++] = down; }
    if (!wall[left] && !occupied[left] && seen[left] !== stamp) { seen[left] = stamp; stack[n++] = left; }
    if (!wall[right] && !occupied[right] && seen[right] !== stamp) { seen[right] = stamp; stack[n++] = right; }
  }
  return low;
}

// Whether the box just pushed onto t completes a frozen 2x2 block of walls and boxes with a box off its goal.
function frozenBlock(level: Level, occupied: Uint8Array, goal: Uint8Array, t: number): boolean {
  const { w, wall } = level;
  for (const a of [t, t - 1, t - w, t - w - 1]) {
    const b = a + 1;
    const c = a + w;
    const d = c + 1;
    if (a < 0 || d >= wall.length) continue;
    if (!(wall[a] || occupied[a]) || !(wall[b] || occupied[b]) || !(wall[c] || occupied[c]) || !(wall[d] || occupied[d])) continue;
    if ((occupied[a] && !goal[a]) || (occupied[b] && !goal[b]) || (occupied[c] && !goal[c]) || (occupied[d] && !goal[d])) return true;
  }
  return false;
}

// The shortest walk from `from` to `to` around walls and boxes, lower-case LURD, first direction winning ties.
function walkTo(level: Level, occupied: Uint8Array, from: number, to: number): string {
  const { w, wall } = level;
  if (from === to) return '';
  const prev = new Int32Array(wall.length).fill(-1);
  const how = new Array<string>(wall.length).fill('');
  const queue = [from];
  prev[from] = from;
  for (let qi = 0; qi < queue.length; qi++) {
    const p = queue[qi]!;
    if (p === to) break;
    for (const [dx, dy, letter] of DIRS) {
      const q = p + dy * w + dx;
      if (q < 0 || q >= wall.length || wall[q] || occupied[q] || prev[q] !== -1) continue;
      if (dx !== 0 && Math.floor(q / w) !== Math.floor(p / w)) continue;
      prev[q] = p;
      how[q] = letter;
      queue.push(q);
    }
  }
  if (prev[to] === -1) return '';
  let s = '';
  for (let p = to; p !== from; p = prev[p]!) s = how[p]! + s;
  return s;
}

interface PushHit {
  pushes: number;
  moves: string;
}

// One push-level search over a (sub-)level. mode 'astar' orders by pushes + matching estimate and is
// push-optimal; 'bfs' orders by pushes alone and is push-optimal too; 'greedy' orders by the estimate alone
// and settles for the first solution it meets. The player component of a state is its reachable region's
// lowest square, so positions that differ only in where the player stands inside one region are one state.
function pushSearch(level: Level, mode: 'astar' | 'bfs' | 'greedy', cap: number): PushHit | null {
  const { w, wall, goals } = level;
  const size = wall.length;
  const goal = new Uint8Array(size);
  for (const g of goals) goal[g] = 1;
  const tables = pushTables(level);
  const live = new Uint8Array(size);
  for (const t of tables) for (let i = 0; i < size; i++) if (t[i]! >= 0) live[i] = 1;
  if (level.boxes.some((b) => !live[b])) return null;
  const useMatching = level.boxes.length <= 6;
  const estimate = (boxes: number[]): number => {
    if (useMatching) {
      const cost = boxes.map((b) => tables.map((t) => (t[b]! < 0 ? BIG : t[b]!)));
      const total = hungarian(cost);
      return total >= BIG ? Infinity : total;
    }
    let sum = 0;
    for (const b of boxes) {
      let best = Infinity;
      for (const t of tables) if (t[b]! >= 0 && t[b]! < best) best = t[b]!;
      if (best === Infinity) return Infinity;
      sum += best;
    }
    return sum;
  };
  if (estimate(level.boxes) === Infinity) return null;
  const solved = (boxes: number[]): boolean => boxes.every((b) => goal[b] === 1);

  const occupied = new Uint8Array(size);
  const seen = new Int32Array(size);
  const mark = new Int32Array(size);
  const stack = new Int32Array(size);
  let stamp = 0;
  const keyOf = (boxes: number[], low: number): string => String.fromCharCode(low, ...boxes);

  const statesBoxes: number[][] = [level.boxes.slice()];
  const statesPlayer: number[] = [level.player];
  const statesG: number[] = [0];
  const parent: number[] = [-1];
  const best = new Map<string, number>();
  for (const b of level.boxes) occupied[b] = 1;
  best.set(keyOf(level.boxes, regionLow(level, occupied, level.player, seen, ++stamp, stack)), 0);
  for (const b of level.boxes) occupied[b] = 0;

  // A binary heap on (priority, estimate, insertion order): fully deterministic.
  const heap: [number, number, number][] = [];
  const less = (a: [number, number, number], b: [number, number, number]): boolean =>
    a[0] < b[0] || (a[0] === b[0] && (a[1] < b[1] || (a[1] === b[1] && a[2] < b[2])));
  const hpush = (e: [number, number, number]): void => {
    let i = heap.length;
    heap.push(e);
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (!less(heap[i]!, heap[p]!)) break;
      const tmp = heap[i]!;
      heap[i] = heap[p]!;
      heap[p] = tmp;
      i = p;
    }
  };
  const hpop = (): [number, number, number] => {
    const top = heap[0]!;
    const last = heap.pop()!;
    if (heap.length > 0) {
      heap[0] = last;
      for (let i = 0; ; ) {
        let c = 2 * i + 1;
        if (c >= heap.length) break;
        if (c + 1 < heap.length && less(heap[c + 1]!, heap[c]!)) c++;
        if (!less(heap[c]!, heap[i]!)) break;
        const tmp = heap[i]!;
        heap[i] = heap[c]!;
        heap[c] = tmp;
        i = c;
      }
    }
    return top;
  };
  const priority = (g: number, h: number): number => (mode === 'bfs' ? g : mode === 'greedy' ? h : g + h);
  const h0 = estimate(level.boxes);
  hpush([priority(0, h0), h0, 0]);

  let found = solved(level.boxes) ? 0 : -1;
  let expanded = 0;
  while (heap.length > 0 && found < 0 && expanded < cap) {
    const [, , s] = hpop();
    const boxes = statesBoxes[s]!;
    const g = statesG[s]!;
    for (const b of boxes) occupied[b] = 1;
    regionLow(level, occupied, statesPlayer[s]!, seen, ++stamp, stack);
    const here = stamp;
    expanded++;
    for (let bi = 0; bi < boxes.length && found < 0; bi++) {
      const b = boxes[bi]!;
      for (const [dx, dy] of DIRS) {
        const d = dy * w + dx;
        const t = b + d;
        if (t < 0 || t >= size || b - d < 0 || b - d >= size) continue;
        if (dx !== 0 && Math.floor(t / w) !== Math.floor(b / w)) continue;
        if (seen[b - d] !== here || wall[t] || occupied[t] || !live[t]) continue;
        occupied[b] = 0;
        occupied[t] = 1;
        const frozen = frozenBlock(level, occupied, goal, t);
        let next: number[] | null = null;
        let key = '';
        let h = 0;
        if (!frozen) {
          next = boxes.slice();
          next[bi] = t;
          next.sort((a, b2) => a - b2);
          h = estimate(next);
          if (h !== Infinity) key = keyOf(next, regionLow(level, occupied, b, mark, ++stamp, stack));
        }
        occupied[t] = 0;
        occupied[b] = 1;
        if (frozen || h === Infinity || key === '') continue;
        const seenG = best.get(key);
        if (seenG !== undefined && seenG <= g + 1) continue;
        best.set(key, g + 1);
        const idx = statesBoxes.length;
        statesBoxes.push(next);
        statesPlayer.push(b);
        statesG.push(g + 1);
        parent.push(s);
        hpush([priority(g + 1, h), h, idx]);
        if (solved(next)) found = idx;
      }
    }
    for (const b of boxes) occupied[b] = 0;
  }
  if (found < 0) return null;
  const path: number[] = [];
  for (let s = found; s > 0; s = parent[s]!) path.push(s);
  path.reverse();
  let moves = '';
  let player = level.player;
  let boxes = level.boxes;
  for (const s of path) {
    const nextBoxes = statesBoxes[s]!;
    const from = boxes.find((b) => !nextBoxes.includes(b))!;
    const to = nextBoxes.find((b) => !boxes.includes(b))!;
    const d = to - from;
    const dir = DIRS.find(([dx, dy]) => dy * w + dx === d)!;
    for (const b of boxes) occupied[b] = 1;
    moves += walkTo(level, occupied, player, from - d);
    for (const b of boxes) occupied[b] = 0;
    moves += dir[3];
    player = from;
    boxes = nextBoxes;
  }
  return { pushes: path.length, moves };
}

// The ladder for one (sub-)level: optimal searches first, a greedy any-solution search last.
function solveSub(level: Level, caps: [number, number, number]): PushHit | null {
  return (
    pushSearch(level, 'astar', caps[0]) ??
    pushSearch(level, 'bfs', caps[1]) ??
    pushSearch(level, 'greedy', caps[2])
  );
}

interface Room {
  squares: number[];
  boxes: number[];
  goals: number[];
  done?: boolean;
}

// The level's rooms: cut the dead squares of the player's row out of the floor; each part left holding a box
// is a room. Null when the level does not split cleanly (a box or goal on the cut, a part whose boxes and
// goals differ in count) — the caller then searches the level whole.
function roomsOf(level: Level): Room[] | null {
  const { w, wall, goals, boxes, player } = level;
  const tables = pushTables(level);
  const live = new Uint8Array(wall.length);
  for (const t of tables) for (let i = 0; i < wall.length; i++) if (t[i]! >= 0) live[i] = 1;
  const row = Math.floor(player / w);
  const cut = new Uint8Array(wall.length);
  for (let i = row * w; i < (row + 1) * w; i++) if (!wall[i] && !live[i]) cut[i] = 1;
  if (boxes.some((b) => cut[b] === 1)) return null;
  const part = new Int32Array(wall.length).fill(-1);
  const rooms: Room[] = [];
  for (let s = 0; s < wall.length; s++) {
    if (wall[s] || cut[s] || part[s] >= 0) continue;
    const squares = [s];
    part[s] = rooms.length;
    for (let i = 0; i < squares.length; i++) {
      for (const [dx, dy] of DIRS) {
        const q = squares[i]! + dy * w + dx;
        if (q >= 0 && q < wall.length && !wall[q] && !cut[q] && part[q] < 0) {
          if (dx !== 0 && Math.floor(q / w) !== Math.floor(squares[i]! / w)) continue;
          part[q] = rooms.length;
          squares.push(q);
        }
      }
    }
    rooms.push({ squares, boxes: [], goals: [] });
  }
  for (const b of boxes) rooms[part[b]!]!.boxes.push(b);
  for (const g of goals) if (part[g]! >= 0) rooms[part[g]!]!.goals.push(g);
  if (goals.some((g) => part[g]! < 0) || rooms.some((r) => r.boxes.length !== r.goals.length)) return null;
  return rooms.filter((r) => r.boxes.length > 0);
}

// The square the player stands on after `moves` from `from` on a board `w` wide.
function endOf(w: number, from: number, moves: string): number {
  let p = from;
  for (const ch of moves.toLowerCase()) p += ch === 'u' ? -w : ch === 'd' ? w : ch === 'l' ? -1 : 1;
  return p;
}

function solveLevel(text: string): string {
  const level = parseLevel(text);
  if (level.boxes.length === 0 || level.player < 0) return '';
  const rooms = roomsOf(level);
  if (rooms === null) {
    const whole = solveSub(level, [WHOLE_ASTAR, WHOLE_BFS, WHOLE_GREEDY]);
    return whole === null ? '' : whole.moves;
  }
  let player = level.player;
  let moves = '';
  for (const room of rooms) {
    const wall = level.wall.slice();
    for (const other of rooms) {
      if (other === room) continue;
      for (const b of other.done ? other.goals : other.boxes) wall[b] = 1;
    }
    const sub: Level = {
      w: level.w,
      h: level.h,
      wall,
      goals: room.goals.slice().sort((a, b) => a - b),
      boxes: room.boxes.slice().sort((a, b) => a - b),
      player,
    };
    const hit = solveSub(sub, [ROOM_ASTAR, ROOM_BFS, ROOM_GREEDY]);
    if (hit === null) {
      const whole = solveSub(level, [WHOLE_ASTAR, WHOLE_BFS, WHOLE_GREEDY]);
      return whole === null ? '' : whole.moves;
    }
    moves += hit.moves;
    player = endOf(level.w, player, hit.moves);
    room.done = true;
  }
  return moves;
}

for (const id of readdirSync(I).sort()) {
  let moves = '';
  try {
    moves = solveLevel(readFileSync(join(I, id, 'level.txt'), 'utf8'));
  } catch {
    moves = '';
  }
  mkdirSync(join(O, id), { recursive: true });
  writeFileSync(join(O, id, 'solution.txt'), `${moves}\n`);
}
