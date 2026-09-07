#!/usr/bin/env python3
"""Pack an authenticated current CNS service + resident for headless/WebGPU delivery.

This is an explicit numerical export: only graph and rank projection weights
are rounded to IEEE binary16. Anatomy, row ordering and all edges are retained.
Large blobs belong in a release/Pages build artifact, not the source checkout.
"""
from __future__ import annotations
import argparse
import gzip
import hashlib
import json
import struct
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from chreatures.cns_adapter_contract import ARRAY_SPECS, MAGIC, FORMAT, DIMENSIONS, canonical, adapter_identity_payload, validate_arrays
from chreatures.sequence_control import CORE_ORDER, PREDICTOR_ORDER, EMBEDDED_ORDER, packed_sha256

HALF = {"graph.weight", "readout.projection.weight"}
def sha(data): return hashlib.sha256(data).hexdigest()
def file_sha(path):
    with Path(path).open('rb') as stream: return hashlib.file_digest(stream, 'sha256').hexdigest()

def read_service(path):
    with path.open('rb') as stream:
        if stream.read(8) != MAGIC: raise ValueError('Current CHCNS2 artifact required')
        size = struct.unpack('<I', stream.read(4))[0]
        if not 0 < size < 8 * 1024**2: raise ValueError('Invalid metadata length')
        meta = json.loads(stream.read(size))
    if meta['format'] != FORMAT or meta['dimensions'] != DIMENSIONS: raise ValueError('CNS format/dimensions differ')
    if sha(canonical(adapter_identity_payload(meta))) != meta['adapter_sha256']: raise ValueError('CNS internal identity differs')
    offset = 12 + size
    arrays = {}
    for name, dtype, shape in ARRAY_SPECS:
        a = np.memmap(path, mode='r', dtype=dtype, shape=shape, offset=offset)
        if sha(a.tobytes()) != meta['array_sha256'][name]: raise ValueError(f'CNS tensor checksum differs: {name}')
        arrays[name] = a
        offset += a.nbytes
    if path.stat().st_size != offset: raise ValueError('CNS file length differs')
    mask = validate_arrays(arrays)
    return meta, arrays, mask

def blob(directory, name, array, *, half=False):
    source = np.asarray(array)
    if half:
        rounded = source.astype('<f2').reshape(-1)
        if not np.isfinite(rounded).all(): raise ValueError(f'Half overflow: {name}')
        if rounded.size % 2: rounded = np.pad(rounded, (0, 1))
        raw = rounded.tobytes()
        dtype = 'packed-f16'
    else:
        raw = source.tobytes(order='C')
        dtype = {'f': 'f32', 'u': 'u32' if source.itemsize == 4 else 'u8'}[source.dtype.kind]
    compressed = gzip.compress(raw, compresslevel=6, mtime=0)
    filename = name.replace('.', '-') + '.bin.gz'
    (directory / filename).write_bytes(compressed)
    entry = dict(url=filename, byteLength=len(raw), sha256=sha(raw), dtype=dtype,
                 shape=list(source.shape), encoding='gzip', transportByteLength=len(compressed),
                 transportSha256=sha(compressed))
    if half:
        error = source.astype(np.float64) - source.astype('<f2').astype(np.float64)
        entry['quantization'] = dict(maxAbsoluteError=float(np.abs(error).max()),
                                     rmsError=float(np.sqrt(np.mean(error**2))), sourceSha256=sha(source.tobytes()))
    return entry

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--service', type=Path, required=True)
    p.add_argument('--resident', type=Path, required=True)
    p.add_argument('--soma-directory', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--revision', required=True)
    a=p.parse_args()
    if a.output.exists(): raise FileExistsError(a.output)
    if len(a.revision)!=40 or any(c not in '0123456789abcdef' for c in a.revision): raise ValueError('Full source revision required')
    meta, arrays, mask = read_service(a.service)
    service_hash = file_sha(a.service)
    # Use the production immutable loader, including component/ancestor checks.
    from chreatures.sensorimotor_worker_native import _load_resident
    resident_meta, resident_arrays, control = _load_resident(a.resident)
    if resident_meta['cns_service']['service_artifact_sha256'] != service_hash: raise ValueError('Resident belongs to a different CNS artifact')
    a.output.mkdir(parents=True)
    entries={name:blob(a.output,name,arrays[name],half=name in HALF) for name,_,_ in ARRAY_SPECS}
    entries['afferent.mask']=blob(a.output,'afferent.mask',mask.astype('<u4'))
    manifest=dict(format='chreatures-cns-webgpu-v2', version=2,
        counts=dict(neurons=165122,edges=25563197,types=11752,opticSites=1771,bodyChannels=43,afferents=15340,rank=64,latent=512,opticValues=5313,receptors=4107,receptorTypes=10,receptorSiteEdges=4669,bodyHidden=128,bodyTargets=11233),
        identity=dict(artifact=meta['adapter_sha256'],graph=meta['graph_sha256'],atlas=meta['atlas_sha256'],mask=meta['readout_mask_sha256']),
        serviceArtifactSha256=service_hash,sourceRevision=a.revision,buffers=entries,
        numericalExport='IEEE binary16 graph and rank projection weights; all edges retained; all other arrays float32/uint32',
        trainingStatus=meta['training_status'],trainingScope=meta.get('provenance',{}).get('scope','See source training receipt; no embodied competence inferred'))
    (a.output/'cns-manifest.json').write_bytes(canonical(manifest)+b'\n')
    components=resident_meta['controller_components']
    packs={}
    for name,order in [('core',CORE_ORDER),('predictor',PREDICTOR_ORDER),('sequence',EMBEDDED_ORDER)]:
        packs[name]=blob(a.output,'resident-'+name,np.concatenate([resident_arrays[k].reshape(-1) for k in order]).astype('<f4'))
    resident=dict(format='chreatures-browser-resident-v1',sourceRevision=a.revision,
        cnsServiceArtifactSha256=service_hash,artifactSha256=resident_meta['artifact_sha256'],
        config=dict(batch=3,action_mode='sample',action_seed=314159,suffix_seed=271828,
                    core_sha256=components['core_packed_sha256'],predictor_sha256=components['predictor_packed_sha256'],
                    sequence_control_version=control.version,sequence_control_sha256=control.sha256,research_training=False),
        trainingStatus=resident_meta['initialization']['training_status'],buffers=packs)
    (a.output/'resident-manifest.json').write_bytes(canonical(resident)+b'\n')
    soma=np.load(a.soma_directory/'soma_positions.npy',allow_pickle=False)
    valid=np.load(a.soma_directory/'soma_valid.npy',allow_pickle=False)
    if soma.shape!=(165122,3) or valid.shape!=(165122,) or not np.isfinite(soma[valid]).all(): raise ValueError('Soma atlas alignment differs')
    soma=np.where(valid[:,None],soma,0).astype('<f4')
    observer=dict(format='chreatures-browser-cns-observer-v1',neurons=165122,validSoma=int(valid.sum()),
                  graph=meta['graph_sha256'],coordinates='measured reconstruction coordinates; missing somas excluded',
                  buffers=dict(positions=blob(a.output,'observer-soma',soma),valid=blob(a.output,'observer-valid',valid.astype('u1'))))
    (a.output/'observer-manifest.json').write_bytes(canonical(observer)+b'\n')
    files={f.name:dict(bytes=f.stat().st_size,sha256=file_sha(f)) for f in sorted(a.output.iterdir())}
    receipt=dict(format='chreatures-browser-release-v1',sourceRevision=a.revision,serviceArtifactSha256=service_hash,
                 residentArtifactSha256=resident['artifactSha256'],files=files,totalBytes=sum(f['bytes'] for f in files.values()))
    (a.output/'release.json').write_bytes(canonical(receipt)+b'\n')
    print(json.dumps(dict(output=str(a.output),totalBytes=receipt['totalBytes'],serviceArtifactSha256=service_hash,files=len(files))))
if __name__=='__main__': main()
