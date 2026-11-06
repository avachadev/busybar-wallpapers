// Works around a Windows bug in @busy-app/cli 0.6.0: the font-maps plugin writes absolute paths into
// `import ... from '<path>'` unescaped, so backslashes become escape sequences and the build fails.
// Runs as postinstall (`node tools/patch-busy-cli.mjs`); harmless on other systems.
import { existsSync, readFileSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";

const file = resolve("node_modules/@busy-app/cli/dist/lib/fontMapsPlugin.js");
const broken = "`import f${i} from '${resolve(maps, `${f}.json`)}';`";
const fixed = "`import f${i} from ${JSON.stringify(resolve(maps, `${f}.json`).replace(/\\\\/g, '/'))};`";

if (!existsSync(file)) {
  console.log("busy-cli font-maps: CLI not installed, nothing to patch");
} else {
  const source = readFileSync(file, "utf8");
  if (source.includes(fixed)) {
    console.log("busy-cli font-maps: already patched");
  } else if (source.includes(broken)) {
    writeFileSync(file, source.replace(broken, fixed));
    console.log("busy-cli font-maps: patched");
  } else {
    // Not fatal: a newer CLI may have fixed the bug upstream.
    console.warn("busy-cli font-maps: pattern not found, the CLI changed; check whether the Windows build still fails");
  }
}
