const FORMAT='chreatures-bad-apple-public-playback-v1';
const SHA=/^[0-9a-f]{64}$/;
const video=document.querySelector('#response-video');
const awaiting=document.querySelector('#awaiting-recording');
const status=document.querySelector('#playback-status');
const play=document.querySelector('#play-recording');
const facts=document.querySelector('#recording-facts');
const ledger=document.querySelector('#source-ledger');
const resultSummary=document.querySelector('#result-summary');
const MODEL_STATUS=new Set(['initialized, untrained rate model','trained rate model with cited training receipt']);
const CONTROLLER_STATUS=new Set(['initialized controller; motor and memory core untrained','briefly trained controller with cited receipt; no competence claim','trained controller with cited receipt; no competence claim']);
const NEURAL_DISPLAY=new Set(['maximum clipped absolute rate per soma display pixel; fixed global scale; not spikes or calcium','maximum positive and negative rate delta per soma display pixel from first recorded frame; fixed symmetric global scale; not spikes or calcium']);

function requireSha(value,label){if(typeof value!=='string'||!SHA.test(value))throw new Error(`${label} is invalid`);return value}
function validate(data){
  if(!data||data.format!==FORMAT||data.status!=='recorded synchronized stimulus, MaleCNS rate-model state, and physical response')throw new Error('Playback manifest is not a completed recording');
  if(!Number.isInteger(data.frames)||data.frames<2||!Number.isFinite(data.playback_fps)||data.playback_fps<1||data.playback_fps>120)throw new Error('Playback frame contract is invalid');
  if(!Array.isArray(data.source_time_seconds)||data.source_time_seconds.length!==2||data.source_time_seconds.some(value=>!Number.isFinite(value))||data.source_time_seconds[0]>=data.source_time_seconds[1])throw new Error('Playback time range is invalid');
  if(data.neurons!==165122||!Number.isInteger(data.valid_soma_positions)||data.valid_soma_positions<1||data.valid_soma_positions>data.neurons)throw new Error('MaleCNS extent is invalid');
  if(!MODEL_STATUS.has(data.model_status))throw new Error('Neural model status is invalid');
  if(!CONTROLLER_STATUS.has(data.controller_status))throw new Error('Controller status is invalid');
  const source=data.sources,sourceKeys=['stimulus_sha256','stimulus_receipt_sha256','malecns_graph_sha256','rate_capture_sha256','body_recording_sha256'];
  if(!source||sourceKeys.some(key=>!requireSha(source[key],key))||typeof source.world_source_revision!=='string'||!source.world_source_revision)throw new Error('Playback sources are incomplete');
  if(!data.video||typeof data.video.url!=='string'||!data.video.url.endsWith('.mp4')||new URL(data.video.url,location.href).origin!==location.origin||!Number.isInteger(data.video.bytes)||data.video.bytes<1)throw new Error('Playback video is invalid');
  requireSha(data.video.sha256,'video SHA-256');
  if(!data.receipt||typeof data.receipt.url!=='string'||!data.receipt.url.endsWith('.json')||new URL(data.receipt.url,location.href).origin!==location.origin)throw new Error('Playback receipt is invalid');
  requireSha(data.receipt.sha256,'receipt SHA-256');
  if(!NEURAL_DISPLAY.has(data.neural_display))throw new Error('Neural display semantics differ');
  return data;
}

export function validatePlaybackForTest(data){return validate(data)}

function short(value){return value.slice(0,12)}
function fact(label,value){const row=document.createElement('div'),name=document.createElement('span'),content=document.createElement('strong');name.textContent=label;content.textContent=value;row.append(name,content);return row}
function sourceLine(label,value){const row=document.createElement('p'),name=document.createElement('strong'),code=document.createElement('code');name.textContent=`${label}: `;code.textContent=value;row.append(name,code);return row}

async function load(){
  try{
    const response=await fetch('./assets/bad-apple-playback.json',{cache:'no-store'});
    if(!response.ok)throw new Error(`manifest HTTP ${response.status}`);
    const data=validate(await response.json());
    video.src=data.video.url;video.hidden=false;awaiting.hidden=true;play.disabled=false;play.textContent='Play the recorded response';
    const duration=data.frames/data.playback_fps;resultSummary.textContent=`This is the recorded ${duration.toFixed(2)}-second run. Neural state: ${data.model_status}. Controller: ${data.controller_status}.`;
    status.textContent=`Recorded manifest loaded · ${data.frames.toLocaleString()} synchronized frames · ${data.playback_fps} fps`;
    facts.replaceChildren(fact('Samples',data.frames.toLocaleString()),fact('Source time',`${data.source_time_seconds[0].toFixed(3)}–${data.source_time_seconds[1].toFixed(3)} s`),fact('Neural model',data.model_status),fact('Controller',data.controller_status),fact('Neural display',data.neural_display),fact('Video SHA-256',short(data.video.sha256)));facts.hidden=false;
    const receipt=document.createElement('a');receipt.href=data.receipt.url;receipt.textContent='Open the complete composition receipt ↗';ledger.replaceChildren(sourceLine('Stimulus',data.sources.stimulus_sha256),sourceLine('Stimulus receipt',data.sources.stimulus_receipt_sha256),sourceLine('MaleCNS graph',data.sources.malecns_graph_sha256),sourceLine('Rate capture',data.sources.rate_capture_sha256),sourceLine('Body recording',data.sources.body_recording_sha256),sourceLine('World revision',data.sources.world_source_revision),receipt);
    video.addEventListener('error',()=>{video.hidden=true;awaiting.hidden=false;play.disabled=true;play.textContent='Recording unavailable';status.textContent='The manifest loaded, but the hash-bound video could not be opened.'},{once:true});
  }catch(error){
    status.textContent='No verified public recording is present yet.';
    play.disabled=true;play.textContent='Recording not yet published';
    console.info('Bad Apple playback remains pending:',error.message);
  }
}

play.addEventListener('click',async()=>{if(play.disabled||video.hidden)return;await video.play();video.scrollIntoView({behavior:'smooth',block:'center'})});
load();
