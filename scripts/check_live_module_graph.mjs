// SPDX-License-Identifier: AGPL-3.0-or-later
// Link the published entry's real static imports without a browser or DOM.
// This catches bare specifiers and missing exports before an inert Start button
// can be deployed. It does not execute WebGPU, render, or establish behavior.
import {readFile} from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL, fileURLToPath} from 'node:url';
import vm from 'node:vm';

if (!vm.SourceTextModule) throw new Error('Run with node --experimental-vm-modules');
const root = path.resolve(process.argv[2] ?? 'dist/site');
const pageURL = pathToFileURL(path.join(root, 'live.html'));
const html = await readFile(pageURL, 'utf8');
const mapText = html.match(/<script\s+type="importmap"\s*>([\s\S]*?)<\/script>/)?.[1];
const imports = mapText ? JSON.parse(mapText).imports : {};
const entry = html.match(/<script\s+type="module"\s+src="([^"]+)"/)?.[1];
if (!entry) throw new Error('Published live module entry is absent');
const context = vm.createContext({});
const modules = new Map();
async function load(url) {
  if (!modules.has(url.href)) {
    modules.set(url.href, (async () => {
      const filename = fileURLToPath(url);
      const relative = path.relative(root, filename);
      if (relative.startsWith('..' + path.sep) || path.isAbsolute(relative)) {
        throw new Error(`Module escapes publication: ${url.href}`);
      }
      return new vm.SourceTextModule(await readFile(url, 'utf8'), {
        context, identifier: url.href,
      });
    })());
  }
  return modules.get(url.href);
}
const module = await load(new URL(entry, pageURL));
await module.link((specifier, parent) => {
  if (Object.hasOwn(imports, specifier)) return load(new URL(imports[specifier], pageURL));
  if (!specifier.startsWith('./') && !specifier.startsWith('../') && !specifier.startsWith('/')) {
    throw new Error(`Unmapped browser module '${specifier}' imported by ${parent.identifier}`);
  }
  return load(new URL(specifier, parent.identifier));
});
process.stdout.write(JSON.stringify({entry, linkedModules: modules.size, evaluated: false}) + '\n');
