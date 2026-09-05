const ALLOWED_OPS = new Set(['signal','light','hand','release']);
const MAX_DURATION = 120;
const MAX_EVENTS = 64;
const MODEL_TICK = 0.05;
const HAND_SAMPLE_SECONDS = 0.08;
const HAND_SAMPLE_DISTANCE = 0.05;

const finite = value => typeof value === 'number' && Number.isFinite(value);
const clamp = (value,min,max) => Math.max(min,Math.min(max,value));
const clone = value => JSON.parse(JSON.stringify(value));
const formatTime = value => `${Number(value||0).toFixed(2)} s`;
const modelTick = value => value<=0?0:Number((Math.ceil((value-1e-10)/MODEL_TICK)*MODEL_TICK).toFixed(2));
const node = (tag,className,text) => {const value=document.createElement(tag);if(className)value.className=className;if(text!==undefined)value.textContent=text;return value;};

function commandPosition(command){return [command.x,command.y,command.z].every(finite)?[command.x,command.y,command.z]:null;}
function normalizeCommand(value){
  if(!value||typeof value!=='object'||!ALLOWED_OPS.has(value.op))return null;
  if(value.op==='release')return {op:'release'};
  const position=commandPosition(value);if(!position)return null;
  if(value.op==='signal'){
    if(!Number.isInteger(value.tone)||value.tone<0||value.tone>2)return null;
    const result={op:'signal',x:position[0],y:position[1],z:position[2],tone:value.tone};if(finite(value.strength))result.strength=clamp(value.strength,.001,1);return result;
  }
  if(value.op==='light'){
    const result={op:'light',x:position[0],y:position[1],z:position[2]};if(finite(value.intensity))result.intensity=clamp(value.intensity,0,1);if(finite(value.duration))result.duration=clamp(value.duration,.01,30);if(Array.isArray(value.color)&&value.color.length===3&&value.color.every(finite))result.color=value.color.map(item=>clamp(item,0,1));return result;
  }
  if(typeof value.id!=='string'||!value.id)return null;
  const result={op:'hand',id:value.id,x:position[0],y:position[1],z:position[2]};if(finite(value.stiffness))result.stiffness=clamp(value.stiffness,.1,80);if(finite(value.damping))result.damping=clamp(value.damping,0,20);return result;
}

export class VisitorPattern {
  constructor(){this.events=[];this.recording=false;this.startedAt=null;this.lastModelTime=0;this.recordedDuration=0;}
  start(modelTime){if(!finite(modelTime))throw Error('Recording needs the current model time');this.events=[];this.recording=true;this.startedAt=modelTime;this.lastModelTime=modelTime;this.recordedDuration=0;return this.snapshot();}
  stop(modelTime=this.lastModelTime){if(this.recording&&finite(modelTime)){this.lastModelTime=modelTime;this.recordedDuration=modelTick(clamp(modelTime-this.startedAt,0,MAX_DURATION));}this.recording=false;return this.snapshot();}
  capture(command,modelTime){
    const clean=normalizeCommand(command);if(!this.recording||!clean||!finite(modelTime))return false;if(this.events.length>=MAX_EVENTS){this.stop(modelTime);return false;}
    const elapsed=modelTime-this.startedAt;if(elapsed>MAX_DURATION){this.stop(modelTime);return false;}const at=modelTick(clamp(elapsed,0,MAX_DURATION));
    if(clean.op==='hand'){
      const previous=[...this.events].reverse().find(event=>event.command.op==='hand'&&event.command.id===clean.id);
      if(previous){const elapsed=at-previous.at,here=commandPosition(clean),there=commandPosition(previous.command),distance=Math.hypot(...here.map((value,index)=>value-there[index]));if(elapsed<HAND_SAMPLE_SECONDS&&distance<HAND_SAMPLE_DISTANCE)return false;}
    }
    this.events.push({at,command:clean});this.lastModelTime=modelTime;this.recordedDuration=Math.max(this.recordedDuration,at);return true;
  }
  add(command,at){const clean=normalizeCommand(command);if(!clean||!finite(at)||this.events.length>=MAX_EVENTS)return false;const offset=modelTick(clamp(at,0,MAX_DURATION));this.events.push({at:offset,command:clean});this.events.sort((a,b)=>a.at-b.at);this.recordedDuration=Math.max(this.recordedDuration,offset);return true;}
  remove(index){if(index>=0&&index<this.events.length)this.events.splice(index,1);}
  clear(){this.events=[];this.recording=false;this.startedAt=null;this.recordedDuration=0;}
  get duration(){return this.events.length?Math.max(.05,this.recordedDuration,this.events[this.events.length-1].at):0;}
  snapshot(){return {duration:this.duration,event_count:this.events.length,events:clone(this.events),recording:this.recording};}
}

async function request(url,options={}){const response=await fetch(url,{headers:{Accept:'application/json',...(options.body?{'Content-Type':'application/json'}:{})},...options});if(!response.ok){let detail=`HTTP ${response.status}`;try{detail=(await response.json()).detail||detail;}catch{}throw Error(detail);}return response.status===204?null:response.json();}
export function createVisitorTransport(base='/api/visitor'){
  return {
    read:()=>request(base),
    save:motif=>request(`${base}/motifs`,{method:'POST',body:JSON.stringify(motif)}),
    queue:motif=>request(`${base}/schedules`,{method:'POST',body:JSON.stringify(motif)}),
    cancel:id=>request(`${base}/schedules/${encodeURIComponent(id)}`,{method:'DELETE'})
  };
}
export function createMemoryVisitorTransport(initialTime=0){
  const data={model_time:initialTime,paused:false,revision:0,motifs:[],queue:[]};let next=1;
  return {async read(){return clone(data);},async save(motif){const saved={...clone(motif),id:`motif-${next++}`,event_count:motif.events.length};data.motifs.push(saved);data.revision++;return {motif:clone(saved)};},async queue(motif){const source=motif.motif_id?data.motifs.find(item=>item.id===motif.motif_id):motif;if(!source)throw Error('Unknown motif');const schedule={...clone(source),id:`schedule-${next++}`,start_time:data.model_time+(motif.start_in??.25),status:'queued'};data.queue.push(schedule);data.revision++;return {schedule:clone(schedule)};},async cancel(id){const target=data.queue.find(item=>item.id===id);if(target)target.status='cancelled';data.revision++;return {cancelled:Boolean(target)};},setModelTime(value){data.model_time=value;}};
}

export function mountVisitorPanel(options={}){
  const transport=options.transport||createVisitorTransport(options.endpoint);
  const pattern=new VisitorPattern();let visitor={model_time:0,paused:false,motifs:[],queue:[]},selectedMotif=null,pollTimer=null,busy=false;
  const host=node('section','visitor-panel');host.setAttribute('aria-label','Visitor pattern recorder');host.innerHTML=`<button class="visitor-launch" type="button" aria-expanded="false"><span>♩</span> Pattern</button><div class="visitor-sheet" hidden><header><div><span class="visitor-kicker">PHYSICAL VISITOR PATTERN</span><h2>Make a motif in model time</h2></div><button class="visitor-close" type="button" aria-label="Close pattern panel">×</button></header><p class="visitor-explainer">Record tones, light, and hand motion already happening in the garden. Replays contain only those physical events; names stay in this visitor archive.</p><div class="visitor-now"><span>WORLD TIME</span><strong>—</strong><i></i></div><div class="visitor-actions"><button class="visitor-record" type="button">● Record</button><button class="visitor-stop" type="button" disabled>Stop</button><button class="visitor-clear" type="button">Clear</button></div><div class="visitor-add"><span>Add at current recording time</span><div><button type="button" data-tone="0">A</button><button type="button" data-tone="1">S</button><button type="button" data-tone="2">D</button><button type="button" data-light>☼ Light</button></div></div><div class="visitor-timeline"><div class="visitor-ruler"></div><div class="visitor-events"></div><p>Record a short performance with the garden tools.</p></div><div class="visitor-summary"><span><b class="visitor-count">0</b> events</span><span><b class="visitor-duration">0.00</b> s model time</span></div><label class="visitor-name-label">Motif name<input class="visitor-name" maxlength="60" placeholder="e.g. three notes by the seesaw"></label><div class="visitor-save-row"><button class="visitor-save" type="button">Bookmark motif</button><button class="visitor-queue" type="button">Queue replay</button></div><div class="visitor-library"><div class="visitor-section-title"><span>BOOKMARKED MOTIFS</span><small class="visitor-library-count">0</small></div><div class="visitor-motifs"></div></div><div class="visitor-queue-section"><div class="visitor-section-title"><span>MODEL-TIME QUEUE</span><small class="visitor-queue-count">0</small></div><div class="visitor-queue-list"></div></div><div class="visitor-message" role="status"></div></div>`;
  (options.mount||document.body).append(host);
  const find=selector=>host.querySelector(selector),all=selector=>[...host.querySelectorAll(selector)];
  const sheet=find('.visitor-sheet'),launch=find('.visitor-launch'),record=find('.visitor-record'),stop=find('.visitor-stop'),name=find('.visitor-name');
  function modelTime(){const current=options.getModelTime?.();return finite(current)?current:visitor.model_time;}
  function cursorCommand(op,tone){const position=options.getCursor?.()||{x:0,y:0,z:.2};return op==='signal'?{op,...position,tone}:{op,...position,intensity:.8,duration:2};}
  function message(value,error=false){const target=find('.visitor-message');target.textContent=value;target.classList.toggle('error',error);}
  function open(value=true){sheet.hidden=!value;launch.setAttribute('aria-expanded',String(value));if(value)refresh();}
  function renderPattern(){
    const snapshot=pattern.snapshot(),events=find('.visitor-events');events.replaceChildren();find('.visitor-count').textContent=snapshot.event_count;find('.visitor-duration').textContent=snapshot.duration.toFixed(2);find('.visitor-timeline>p').hidden=Boolean(snapshot.event_count);record.disabled=snapshot.recording;stop.disabled=!snapshot.recording;host.classList.toggle('is-recording',snapshot.recording);
    const width=Math.max(snapshot.duration,1);snapshot.events.forEach((event,index)=>{const button=node('button',`visitor-event visitor-${event.command.op}`,event.command.op==='signal'?['A','S','D'][event.command.tone]:event.command.op==='light'?'☼':event.command.op==='release'?'↑':'↗');button.type='button';button.style.left=`${clamp(event.at/width*100,1,97)}%`;button.title=`${event.command.op} at ${formatTime(event.at)} — click to remove`;button.onclick=()=>{pattern.remove(index);renderPattern();};events.append(button);});
  }
  function renderRemote(){
    const time=find('.visitor-now strong');time.textContent=formatTime(visitor.model_time);find('.visitor-now i').textContent=visitor.paused?'paused':'advancing';find('.visitor-now').classList.toggle('paused',visitor.paused);
    const motifs=find('.visitor-motifs');motifs.replaceChildren();find('.visitor-library-count').textContent=visitor.motifs.length;
    if(!visitor.motifs.length)motifs.append(node('p','visitor-empty','No bookmarked motifs yet.'));
    for(const motif of visitor.motifs){const button=node('button',`visitor-motif${selectedMotif===motif.id?' selected':''}`);button.type='button';const label=node('span','',motif.name),meta=node('small','',`${motif.event_count??motif.events?.length??0} events · ${formatTime(motif.duration)}`);button.append(label,meta);button.onclick=()=>{selectedMotif=motif.id;name.value=motif.name;renderRemote();};motifs.append(button);}
    const queue=find('.visitor-queue-list'),active=visitor.queue.filter(item=>['queued','playing'].includes(item.status));queue.replaceChildren();find('.visitor-queue-count').textContent=active.length;
    if(!active.length)queue.append(node('p','visitor-empty','Nothing is scheduled.'));
    for(const item of active){const row=node('div','visitor-queue-item');const label=node('div');const delivered=Number.isInteger(item.delivered)&&Array.isArray(item.events)?`${item.delivered}/${item.events.length} delivered · `:'';label.append(node('span','',item.name||'Untitled motif'),node('small','',`starts t ${Number(item.start_time).toFixed(2)} · ${formatTime(item.duration)} · ${delivered}${item.status}`));const cancel=node('button','','Cancel');cancel.type='button';cancel.onclick=()=>cancelSchedule(item.id);row.append(label,cancel);queue.append(row);}
  }
  async function refresh(){try{visitor=await transport.read();visitor.motifs=Array.isArray(visitor.motifs)?visitor.motifs:[];visitor.queue=Array.isArray(visitor.queue)?visitor.queue:[];renderRemote();message('');}catch(error){message(`Queue unavailable: ${error.message}`,true);}}
  async function perform(command){try{if(options.perform)await options.perform(command);else{selectedMotif=null;pattern.add(command,pattern.duration);renderPattern();}}catch(error){message(`Physical event was not accepted: ${error.message}`,true);}}
  function capture(command,time=modelTime()){const wasRecording=pattern.recording,wasFull=pattern.events.length>=MAX_EVENTS;if(wasRecording&&wasFull)pattern.stop(time);const changed=wasFull?false:pattern.capture(command,time);if(changed)selectedMotif=null;if(changed||wasRecording!==pattern.recording)renderPattern();if(wasRecording&&wasFull)message(`Recording stopped at the ${MAX_EVENTS}-event limit.`);else if(wasRecording&&!pattern.recording)message(`Recording stopped at the ${MAX_DURATION}-second limit.`);return changed;}
  async function save(){if(busy)return;const motif=payload();if(!motif)return;busy=true;try{const result=await transport.save(motif);selectedMotif=(result?.motif||result)?.id||null;message(`Bookmarked “${motif.name}”.`);await refresh();}catch(error){message(`Could not bookmark motif: ${error.message}`,true);}finally{busy=false;}}
  async function queue(){if(busy)return;const current=payload();if(!selectedMotif&&!current)return;busy=true;try{const body=selectedMotif?{motif_id:selectedMotif,start_in:.25}:{...current,start_in:.25};await transport.queue(body);message('Replay placed on the model-time queue.');await refresh();}catch(error){message(`Could not queue replay: ${error.message}`,true);}finally{busy=false;}}
  async function cancelSchedule(id){try{await transport.cancel(id);message('Scheduled replay cancelled.');await refresh();}catch(error){message(`Could not cancel replay: ${error.message}`,true);}}
  function payload(){const value=name.value.trim();if(!value){message('Name the motif before bookmarking or replaying it.',true);name.focus();return null;}if(!pattern.events.length&&!selectedMotif){message('Record at least one physical event.',true);return null;}return {name:value,duration:pattern.duration,events:clone(pattern.events)};}
  function update(value){if(value&&finite(value.model_time))visitor={...visitor,...value};else if(value&&finite(value.time))visitor={...visitor,model_time:value.time,paused:Boolean(value.paused)};renderRemote();}
  launch.onclick=()=>open(sheet.hidden);find('.visitor-close').onclick=()=>open(false);record.onclick=()=>{selectedMotif=null;pattern.start(modelTime());renderPattern();message('Recording model-time offsets. Use the garden tools.');};stop.onclick=()=>{pattern.stop(modelTime());renderPattern();message(`Captured ${pattern.events.length} physical events.`);};find('.visitor-clear').onclick=()=>{selectedMotif=null;pattern.clear();name.value='';renderPattern();message('');};find('.visitor-save').onclick=save;find('.visitor-queue').onclick=queue;
  all('[data-tone]').forEach(button=>button.onclick=()=>perform(cursorCommand('signal',Number(button.dataset.tone))));find('[data-light]').onclick=()=>perform(cursorCommand('light'));
  renderPattern();renderRemote();if(options.poll!==false)pollTimer=setInterval(()=>{if(!sheet.hidden)refresh();},2000);
  return {capture,update,open,refresh,get pattern(){return pattern.snapshot();},destroy(){clearInterval(pollTimer);host.remove();}};
}

export {normalizeCommand,modelTick,ALLOWED_OPS,MAX_DURATION,MAX_EVENTS,MODEL_TICK};
