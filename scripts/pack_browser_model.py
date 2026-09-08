#!/usr/bin/env python3
"""Pack an authenticated current CNS service + resident for headless/WebGPU delivery.

Canonical graph binary16 bits and all float32 parameters are copied exactly.
Anatomy, row ordering and all edges are retained; no backend rounding occurs.
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
from chreatures.cns_adapter_contract import ARRAY_SPECS, MAGIC, FORMAT, DIMENSIONS, canonical, adapter_identity_payload, validate_arrays, service_identity
from chreatures.sequence_control import CORE_ORDER, PREDICTOR_ORDER, EMBEDDED_ORDER, packed_sha256

def sha(data): return hashlib.sha256(data).hexdigest()
def file_sha(path):
    with Path(path).open('rb') as stream: return hashlib.file_digest(stream, 'sha256').hexdigest()

def read_service(path):
    with path.open('rb') as stream:
        if stream.read(8) != MAGIC: raise ValueError('Current CHCNS4 artifact required')
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
    if sha(mask.tobytes()) != meta['readout_mask_sha256']: raise ValueError('CNS injected readout mask identity differs')
    return meta, arrays, mask

def blob(directory, name, array):
    source = np.asarray(array)
    raw = source.tobytes(order='C')
    dtype = {('f', 4): 'f32', ('u', 4): 'u32', ('u', 2): 'u16', ('u', 1): 'u8'}[(source.dtype.kind, source.itemsize)]
    compressed = gzip.compress(raw, compresslevel=6, mtime=0)
    filename = name.replace('.', '-') + '.bin.gz'
    (directory / filename).write_bytes(compressed)
    return dict(url=filename, byteLength=len(raw), sha256=sha(raw), dtype=dtype,
                shape=list(source.shape), encoding='gzip', transportByteLength=len(compressed),
                transportSha256=sha(compressed))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--service', type=Path, required=True)
    p.add_argument('--resident', type=Path)
    p.add_argument('--soma-directory', type=Path)
    p.add_argument('--cns-only', action='store_true', help='pack only the CNS manifest and tensors for parity/integration')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--revision', required=True)
    a=p.parse_args()
    if a.output.exists(): raise FileExistsError(a.output)
    if len(a.revision)!=40 or any(c not in '0123456789abcdef' for c in a.revision): raise ValueError('Full source revision required')
    meta, arrays, mask = read_service(a.service)
    service_hash = file_sha(a.service)
    service_identity(meta, service_hash)
    a.output.mkdir(parents=True)
    entries={name:blob(a.output,name,arrays[name]) for name,_,_ in ARRAY_SPECS}
    entries['afferent.mask']=blob(a.output,'afferent.mask',mask.astype('<u4'))
    manifest=dict(format='chreatures-cns-webgpu-v4', version=4,
        counts=dict(neurons=165122,edges=25563197,types=11752,opticSites=1771,bodyChannels=807,
                    contextChannels=12,contextTargets=1314,motor=92,motorTargets=815,afferents=int(np.count_nonzero(mask == 0)),
                    rank=64,latent=512,opticValues=5313,receptors=4107,receptorTypes=10,
                    receptorSiteEdges=4669,bodyTargets=11798),
        identity=dict(artifact=meta['adapter_sha256'],graph=meta['graph_sha256'],atlas=meta['atlas_sha256'],
                      anatomy=meta['anatomy_sha256'],mask=meta['readout_mask_sha256'],
                      morphology=meta['morphology_sha256'],sensorySchema=meta['sensory_schema_sha256'],
                      actuatorSchema=meta['actuator_schema_sha256'],motorCalibration=meta['motor_calibration_sha256'],
                      graphSourceWeight=meta['graph_source_weight_sha256']),
        graphQuantization=meta['graph_quantization'], controlDt=0.01, substeps=2,
        serviceArtifactSha256=service_hash,sourceRevision=a.revision,buffers=entries,
        numericalExport='V4 canonical graph binary16 bits decoded once to float32; all other tensors remain float32; no backend rounding',
        trainingStatus=meta['training_status'],trainingScope=meta.get('provenance',{}).get('scope','See source training receipt; no embodied competence inferred'))
    (a.output/'cns-manifest.json').write_bytes(canonical(manifest)+b'\n')
    if a.cns_only:
        if a.resident or a.soma_directory: raise ValueError('--cns-only cannot include resident or soma inputs')
        files={f.name:dict(bytes=f.stat().st_size,sha256=file_sha(f)) for f in sorted(a.output.iterdir())}
        receipt=dict(format='chreatures-browser-cns-release-v4',sourceRevision=a.revision,
                     serviceArtifactSha256=service_hash,files=files,totalBytes=sum(f['bytes'] for f in files.values()))
        (a.output/'release.json').write_bytes(canonical(receipt)+b'\n')
        print(json.dumps(dict(output=str(a.output),totalBytes=receipt['totalBytes'],serviceArtifactSha256=service_hash,files=len(files))))
        return
    if not a.resident or not a.soma_directory: raise ValueError('--resident and --soma-directory are required unless --cns-only')
    # Use the production immutable loader, including component/ancestor checks.
    from chreatures.sensorimotor_worker_native import _load_resident
    resident_meta, resident_arrays, control = _load_resident(a.resident)
    if resident_meta['cns_service']['service_artifact_sha256'] != service_hash: raise ValueError('Resident belongs to a different CNS artifact')
    components=resident_meta['controller_components']
    packs={}
    for name,order in [('core',CORE_ORDER),('predictor',PREDICTOR_ORDER),('sequence',EMBEDDED_ORDER)]:
        packs[name]=blob(a.output,'resident-'+name,np.concatenate([resident_arrays[k].reshape(-1) for k in order]).astype('<f4'))
    resident=dict(format='chreatures-browser-resident-v1',sourceRevision=a.revision,
        cnsServiceArtifactSha256=service_hash,artifactSha256=resident_meta['artifact_sha256'],
        config=dict(batch=3,action_mode='sample',action_seed=314159,suffix_seed=271828,tick_seconds=0.01,
                    context_policy_version='signed-context12-v1',
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
