import { existsSync } from "node:fs";

const distInstallUrl = new URL("./dist/install-model.js", import.meta.url);

if (!existsSync(distInstallUrl)) {
  process.exitCode = 0;
} else {
  await import(distInstallUrl.href);
}
