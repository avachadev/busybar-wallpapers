import { device } from "@shared/device";
import manifest from "./appmeta/manifest.json";

/** The name the app draws under; the device clears elements by this id. */
const APP = manifest.id;

/**
 * Each theme is one .anim with two sections: `idle` loops forever, `action` plays once on Start and its tail
 * blends into idle, so idle resumes without a cut. The frames are rendered offline by tools/render.py.
 */
type Theme = { id: string; name: string; action: string };

const THEMES: Theme[] = [
  { id: "inferno", name: "Inferno", action: "stoke the fire" },
  { id: "hyperspace", name: "Hyperspace", action: "jump" },
  { id: "aurora", name: "Aurora", action: "solar storm" },
  { id: "rain", name: "Neon Rain", action: "lightning" },
  { id: "synthwave", name: "Synthwave", action: "overdrive" },
  { id: "lava", name: "Lava Lamp", action: "heat it up" },
  { id: "matrix", name: "Matrix", action: "decode" },
  { id: "fireworks", name: "Fireworks", action: "grand finale" },
  { id: "blackhole", name: "Black Hole", action: "feed it a star" },
  { id: "train", name: "Night Train", action: "tunnel" },
  { id: "orbit", name: "Low Orbit", action: "sunrise" },
  { id: "slime", name: "Slime Mold", action: "feed it" },
  { id: "plasma", name: "Plasma", action: "touch" },
];

const STORAGE_KEY = "theme";
/** The theme is remembered once the wheel has been still for this long: every write goes to flash. */
const SAVE_DELAY_MS = 3000;

/** Layers, bottom to top. */
const Z_SCENE = 0;
const Z_SHADE = 1;
const Z_CLOCK = 2;

/** Front: the clock sits on the right over a soft shade, so it reads on any scene. */
const SHADE_WIDTH = 34;

/** `afterAction`: idle queued behind an action waits for it to end instead of cutting it short. */
function scene(theme: Theme, section: "idle" | "action", afterAction = false): object {
  return {
    id: "scene",
    type: "animation",
    path: `animations/${theme.id}.anim`,
    section,
    loop: section === "idle",
    await_previous_end: afterAction,
    x: 0,
    y: 0,
    z_index: Z_SCENE,
  };
}

const SHADE = {
  id: "shade",
  type: "rectangle",
  x: 72 - SHADE_WIDTH,
  y: 0,
  width: SHADE_WIDTH,
  height: 16,
  fill: "gradient_h",
  fill_colors: ["#00000000", "#000000C0"],
  border_width: 0,
  z_index: Z_SHADE,
};

function clock(time: string): object {
  return {
    id: "clock",
    type: "text",
    text: time,
    font: "condensed",
    align: "mid_right",
    x: 70,
    y: 8,
    color: "#FFFFFFFF",
    z_index: Z_CLOCK,
  };
}

function backInfo(theme: Theme, index: number): object[] {
  return [
    { id: "back_name", type: "text", text: theme.name, font: "large", align: "center", x: 72, y: 30, display: "back" },
    {
      id: "back_count",
      type: "text",
      text: `${index + 1} / ${THEMES.length}`,
      font: "small",
      align: "center",
      x: 72,
      y: 12,
      display: "back",
    },
    {
      id: "back_hint",
      type: "text",
      text: `START: ${theme.action}`,
      font: "small",
      align: "center",
      x: 72,
      y: 56,
      display: "back",
    },
    { id: "back_wheel", type: "text", text: "< wheel >", font: "small", align: "center", x: 72, y: 68, display: "back" },
  ];
}

function loadThemeIndex(): number {
  try {
    const saved = localStorage.getItem(STORAGE_KEY);
    const index = THEMES.findIndex((theme) => theme.id === saved);
    return index >= 0 ? index : 0;
  } catch {
    return 0;
  }
}

/** Local "HH:MM" from the bar's clock; the ISO string it returns carries the local offset. */
async function localTime(): Promise<string> {
  const time = await device.TimeGet();
  return time.timestamp.slice(11, 16);
}

type Element = { id: string; section?: string };

function main(): void {
  let stopped = false;
  let index = loadThemeIndex();
  let clockTimer: ReturnType<typeof setTimeout> | undefined;
  let saveTimer: ReturnType<typeof setTimeout> | undefined;
  // One draw request in flight at a time. Queued requests absorb later updates of the same elements, so a fast
  // wheel sends only the latest theme; an action stays its own request so the idle queued behind it can wait for it.
  const queue: Element[][] = [];
  let drawing = false;

  const report = (err: unknown) => console.error(`${APP}: ${err instanceof Error ? err.message : String(err)}`);

  function send(elements: object[], ownRequest = false): void {
    const last = queue[queue.length - 1];
    if (ownRequest || last === undefined || last.some((e) => e.section === "action")) {
      queue.push(elements as Element[]);
    } else {
      for (const element of elements as Element[]) {
        const at = last.findIndex((e) => e.id === element.id);
        if (at >= 0) last[at] = element;
        else last.push(element);
      }
    }
    if (!drawing) void flush();
  }

  async function flush(): Promise<void> {
    drawing = true;
    while (queue.length > 0 && !stopped) {
      const elements = queue.shift()!;
      // busy-lib's element union is strict about optional fields; the shapes here follow the firmware schema.
      await device.DisplayDraw({ application_name: APP, priority: 50, elements } as never).catch(report);
    }
    drawing = false;
  }

  function showTheme(): void {
    const theme = THEMES[index]!;
    send([scene(theme, "idle"), ...backInfo(theme, index)]);
  }

  function play(): void {
    const theme = THEMES[index]!;
    send([scene(theme, "action")], true);
    send([scene(theme, "idle", true)], true);
  }

  function rememberTheme(): void {
    clearTimeout(saveTimer);
    saveTimer = setTimeout(() => {
      try {
        localStorage.setItem(STORAGE_KEY, THEMES[index]!.id);
      } catch (err) {
        report(err);
      }
    }, SAVE_DELAY_MS);
  }

  async function tickClock(): Promise<void> {
    try {
      send([clock(await localTime())]);
    } catch (err) {
      report(err);
    }
    if (stopped) return;
    // Wake just after the next minute starts.
    const now = new Date();
    clockTimer = setTimeout(() => void tickClock(), (60 - now.getSeconds()) * 1000 + 200);
  }

  send([SHADE]);
  showTheme();
  void tickClock();

  const unbind = listen("input", (event) => {
    if (stopped) return;

    if (event.key === "encoder") {
      index = (index + event.delta + THEMES.length) % THEMES.length;
      showTheme();
      rememberTheme();
    } else if (event.key === "start" && event.action === "short") {
      play();
    } else if (event.key === "back" && event.action === "short") {
      stopped = true;
      clearTimeout(clockTimer);
      if (saveTimer !== undefined) {
        clearTimeout(saveTimer);
        try {
          localStorage.setItem(STORAGE_KEY, THEMES[index]!.id);
        } catch (err) {
          report(err);
        }
      }
      // Not detached from inside the handler itself: with nothing else left the unbind stops the app on the spot.
      setTimeout(unbind, 0);
    }
  });
}

main();
