// SPDX-License-Identifier: AGPL-3.0-or-later
/** Node filesystem convenience for the browser-compatible unified Wasm host. */

import {readdir, readFile} from 'node:fs/promises';
import {basename, dirname, join, resolve} from 'node:path';
import {pathToFileURL} from 'node:url';
import {UnifiedWasmFlyWorld} from '../../native/fly-world/runtime.mjs';

async function copyTreeToMemfs(module, sourceDirectory, destination = '/world') {
  module.FS.mkdirTree(destination);
  for (const entry of await readdir(sourceDirectory, {withFileTypes: true})) {
    const source = join(sourceDirectory, entry.name);
    const target = `${destination}/${entry.name}`;
    if (entry.isDirectory()) await copyTreeToMemfs(module, source, target);
    else if (entry.isFile()) module.FS.writeFile(target, await readFile(source));
    else throw new Error(`scene tree contains unsupported entry: ${source}`);
  }
}

export async function openUnifiedWasmFlyWorld({modulePath, scene, seed}) {
  const artifact = resolve(modulePath), sceneFile = resolve(scene);
  return UnifiedWasmFlyWorld.open({
    moduleURL: pathToFileURL(artifact),
    locateFile: name => resolve(dirname(artifact), name),
    scenePath: `/world/${basename(sceneFile)}`,
    seed,
    prepareFilesystem: module => copyTreeToMemfs(module, dirname(sceneFile)),
  });
}
