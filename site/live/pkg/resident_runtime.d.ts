/* tslint:disable */
/* eslint-disable */

export class ResidentRuntime {
    free(): void;
    [Symbol.dispose](): void;
    acknowledge(ticks: BigUint64Array, delivered_context: Float32Array): Uint8Array;
    diagnosticsJson(): string;
    expanded(additions: number, action_seed: bigint, suffix_seed: bigint): ResidentRuntime;
    /**
     * Restore validates all identities and state before committing any mutation.
     */
    loadBytes(bytes: Uint8Array): void;
    /**
     * Float32Array data is packed in the canonical training artifact order.
     * Artifact checksums identify the immutable model; the asset loader verifies
     * transport hashes before entering this host. No URLs or raw sensory ports.
     */
    constructor(config_json: string, core: Float32Array, predictor: Float32Array, sequence: Float32Array);
    saveBytes(): Uint8Array;
    /**
     * Inputs are resident-major B×512 and B×12; ticks are BigUint64Array,
     * resets Uint8Array containing exactly 0 or 1. The proposed context needs
     * a receipt after it actually enters CNS recurrence; until then no next
     * decision is allowed.
     */
    stepFlat(z: Float32Array, previous: Float32Array, ticks: BigUint64Array, reset: Uint8Array): ResidentStep;
    readonly batch: number;
}

/**
 * A copy of one completed decision, stable across subsequent engine calls.
 */
export class ResidentStep {
    private constructor();
    free(): void;
    [Symbol.dispose](): void;
    readonly diagnosticsJson: string;
    readonly proposedContext: Float32Array;
}

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
    readonly memory: WebAssembly.Memory;
    readonly __wbg_residentruntime_free: (a: number, b: number) => void;
    readonly __wbg_residentstep_free: (a: number, b: number) => void;
    readonly residentruntime_acknowledge: (a: number, b: number, c: number, d: number, e: number) => [number, number, number, number];
    readonly residentruntime_batch: (a: number) => number;
    readonly residentruntime_diagnosticsJson: (a: number) => [number, number, number, number];
    readonly residentruntime_expanded: (a: number, b: number, c: bigint, d: bigint) => [number, number, number];
    readonly residentruntime_loadBytes: (a: number, b: number, c: number) => [number, number];
    readonly residentruntime_new: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number) => [number, number, number];
    readonly residentruntime_saveBytes: (a: number) => [number, number, number, number];
    readonly residentruntime_stepFlat: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number) => [number, number, number];
    readonly residentstep_diagnosticsJson: (a: number) => [number, number];
    readonly residentstep_proposedContext: (a: number) => [number, number];
    readonly __wbindgen_externrefs: WebAssembly.Table;
    readonly __wbindgen_malloc: (a: number, b: number) => number;
    readonly __externref_table_dealloc: (a: number) => void;
    readonly __wbindgen_free: (a: number, b: number, c: number) => void;
    readonly __wbindgen_realloc: (a: number, b: number, c: number, d: number) => number;
    readonly __wbindgen_start: () => void;
}

export type SyncInitInput = BufferSource | WebAssembly.Module;

/**
 * Instantiates the given `module`, which can either be bytes or
 * a precompiled `WebAssembly.Module`.
 *
 * @param {{ module: SyncInitInput }} module - Passing `SyncInitInput` directly is deprecated.
 *
 * @returns {InitOutput}
 */
export function initSync(module: { module: SyncInitInput } | SyncInitInput): InitOutput;

/**
 * If `module_or_path` is {RequestInfo} or {URL}, makes a request and
 * for everything else, calls `WebAssembly.instantiate` directly.
 *
 * @param {{ module_or_path: InitInput | Promise<InitInput> }} module_or_path - Passing `InitInput` directly is deprecated.
 *
 * @returns {Promise<InitOutput>}
 */
export default function __wbg_init (module_or_path?: { module_or_path: InitInput | Promise<InitInput> } | InitInput | Promise<InitInput>): Promise<InitOutput>;
