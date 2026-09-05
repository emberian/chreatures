const $ = id => document.getElementById(id);
const svgNS = 'http://www.w3.org/2000/svg';
const state = {overview:null, graph:null, nodeById:new Map(), artifactByUri:new Map(), selected:null};

const number = (value, digits=2) => Number(value).toLocaleString(undefined,{maximumFractionDigits:digits});
const scientific = value => Number(value).toExponential(3);
const bytes = value => value >= 1e6 ? `${(value/1e6).toFixed(1)} MB` : value >= 1e3 ? `${(value/1e3).toFixed(1)} kB` : `${value} B`;
const shortHash = value => value ? `${value.slice(0,12)}…${value.slice(-8)}` : '—';
const nameForModel = name => ({energy_dynamics:'Bodily energy',movement_outcome_dynamics:'Movement outcome',prediction_error_dynamics:'Prediction error'})[name] || name.replaceAll('_',' ');
const purpose = name => {
  if(name.endsWith('.gam.gz')) return 'Native GAM model';
  if(name.includes('timeseries')) return name.startsWith('adult') ? 'Adult checkpoint history' : 'Development world telemetry';
  if(name==='observatory.json') return 'Verified observatory report';
  if(name==='dag-view.json') return 'Formatted evidence graph';
  if(name.includes('weave-request')) return 'Native Weave import request';
  if(name.includes('weave')) return 'Reloaded native Weave DAG';
  return 'Archived evidence';
};

function text(id,value){$(id).textContent=value;}
function el(tag,className,content){const node=document.createElement(tag);if(className)node.className=className;if(content!==undefined)node.textContent=content;return node;}
async function read(url){const response=await fetch(url,{headers:{Accept:'application/json'}});if(!response.ok){let detail=`HTTP ${response.status}`;try{detail=(await response.json()).detail||detail;}catch{}throw Error(detail);}return response.json();}

function renderSources(){
  const {manifest,world,development}=state.overview;
  const adult=world.summary, adultSource=world.source, run=development.summary, receipts=development.source.receipts;
  text('artifact-set',manifest.artifact_set_sha256);text('archive-count',`${Object.keys(manifest.artifacts).length} receipt-checked artifacts`);
  text('adult-name',adult.physics.habitat||'3D checkpoint');text('adult-verified',adultSource.checksum_verified?'✓ checksum verified':'checksum unavailable');$('adult-verified').classList.toggle('ok',adultSource.checksum_verified);
  text('adult-moment',`tick ${number(adult.tick,0)} · t ${number(adult.model_time,2)}`);text('adult-physics',`${adult.physics.engine.name} ${adult.physics.engine.version} · ${adult.physics.dimension}D`);
  text('adult-bodies',`${adult.articulation.resident_pose_3d_count}/${adult.physics.resident_count} pose + velocity`);text('adult-ecology',`${adult.ecology.kind} · ${Object.values(adult.ecology.component_counts).reduce((a,b)=>a+b,0)} components`);
  text('adult-memory',`${number(adult.cognition.memory_records,0)} records counted; contents excluded`);text('adult-hash',shortHash(adultSource.state_sha256));$('adult-hash').title=adultSource.state_sha256;
  text('adult-caveat',adult.articulation.body_mode_recorded?`Body mode: ${adult.articulation.body_mode}.`:`Body mode: ${adult.articulation.body_mode}. The snapshot still contains complete 3D poses and velocities.`);
  text('run-name',`${run.worlds} physical worlds`);text('run-verified',development.source.all_receipts_verified?'✓ all receipts verified':'receipt failure');$('run-verified').classList.toggle('ok',development.source.all_receipts_verified);
  text('run-population',`${run.residents} residents · ${run.worlds} worlds`);text('run-duration',`${number(run.steps,0)} steps · ${number(run.phases.simple_steps,0)} simple + ${number(run.phases.rich_steps,0)} rich`);
  text('run-nutrition',number(run.outcomes.nutrition_total,6));text('run-contact',number(run.outcomes.contact_positive_resident_steps,0));text('run-memory',`${number(run.cognition.memory_records.total,0)} records counted; contents excluded`);
  const trajectory=receipts['trajectory.npz'];text('run-hash',shortHash(trajectory.sha256));$('run-hash').title=trajectory.sha256;
  text('run-caveat',`${run.physics.world_family}. Articulation was not recorded in this run.`);
}

function renderModels(){
  const gam=state.overview.gamfit, split=gam.split, grid=$('model-grid');grid.replaceChildren();
  text('split-summary',`${split.training_worlds.length} complete worlds fit · ${split.held_out_worlds.join(' and ')} held out in full · ${number(split.held_out_rows,0)} unseen rows`);
  for(const [name,model] of Object.entries(gam.models)){
    const card=el('article','model-card');const header=el('header');const title=el('h3','',nameForModel(name));
    if(model.status!=='complete'){
      const pill=el('span','status-pill worse',model.status);header.append(title,pill);card.append(header,el('p','diagnostics',model.reason||'The native fit did not produce an artifact.'));grid.append(card);continue;
    }
    const ratio=model.rmse_vs_persistence;
    const better=ratio<1, delta=Math.abs(1-ratio)*100;const pill=el('span',`status-pill${better?'':' worse'}`,model.status);header.append(title,pill);
    const comparison=el('div','comparison-value',`${delta.toFixed(2)}%`);const label=el('div','comparison-label',`${better?'lower':'higher'} RMSE than persistence on held-out worlds`);
    const errors=[['GAM',model.held_out.rmse,'fit'],['Persistence',model.baselines.persistence.rmse,'baseline'],['Training mean',model.baselines.training_mean.rmse,'mean']];const max=Math.max(...errors.map(row=>row[1]));const bars=el('div','error-bars');
    for(const [labelText,value,kind] of errors){const row=el('div',`bar-row ${kind}`);const labelNode=el('span','',labelText);const track=el('span','bar-track');const bar=el('i');bar.style.width=`${Math.max(2,value/max*100)}%`;track.append(bar);const output=el('output','',scientific(value));row.append(labelNode,track,output);bars.append(row);}
    const meta=el('div','model-meta');meta.append(el('span','',`${number(state.overview.gamfit.split.held_out_rows,0)} held-out rows`),el('span','',`reload Δ ${model.artifact.reload_max_abs_prediction_delta}`));
    card.append(header,comparison,label,bars,meta);
    const specification=el('details','model-formula');const specificationLabel=el('summary','','Model specification');const formula=el('code','',model.formula);const modelLink=el('a','',`Native model · ${bytes(model.artifact.bytes)} ↓`);modelLink.href=`/api/observatory/artifacts/${encodeURIComponent(model.artifact.file)}`;modelLink.download=model.artifact.file;specification.append(specificationLabel,formula,modelLink);card.append(specification);
    const diagnostics=[...(model.warnings||[]),...(model.native_messages||[])];if(diagnostics.length){const details=el('details','diagnostics');const summary=el('summary','',`${diagnostics.length} retained fit diagnostic${diagnostics.length>1?'s':''}`);details.append(summary);for(const item of diagnostics)details.append(el('p','',item));card.append(details);}
    grid.append(card);
  }
}

function svgEl(tag,attributes={}){const node=document.createElementNS(svgNS,tag);for(const [name,value] of Object.entries(attributes))node.setAttribute(name,String(value));return node;}
function wrapLabel(label,max=27){const words=label.split(/\s+/),lines=[];let line='';for(const word of words){if((line+' '+word).trim().length>max&&line){lines.push(line);line=word;}else line=(line+' '+word).trim();if(lines.length===2)break;}if(line&&lines.length<3)lines.push(line);return lines.slice(0,3);}

function renderGraph(){
  const graph=state.graph, svg=$('evidence-graph');svg.replaceChildren();state.nodeById=new Map(graph.nodes.map(node=>[node.id,node]));
  const width=1040,height=460,nodeWidth=188,nodeHeight=76,padX=34,padY=34;svg.setAttribute('viewBox',`0 0 ${width} ${height}`);
  const levels=new Map();for(const node of graph.nodes){if(!levels.has(node.depth))levels.set(node.depth,[]);levels.get(node.depth).push(node);}
  const positions=new Map();for(const [depth,nodes] of levels){const x=padX+depth*((width-nodeWidth-padX*2)/Math.max(1,graph.levels-1));const gap=(height-padY*2-nodeHeight)/(Math.max(1,nodes.length));nodes.forEach((node,index)=>positions.set(node.id,{x,y:nodes.length===1?(height-nodeHeight)/2:padY+index*gap}));}
  const edgeLayer=svgEl('g');for(const edge of graph.edges){const a=positions.get(edge.source),b=positions.get(edge.target);if(!a||!b)continue;const path=svgEl('path',{class:'graph-edge',d:`M ${a.x+nodeWidth} ${a.y+nodeHeight/2} C ${a.x+nodeWidth+55} ${a.y+nodeHeight/2}, ${b.x-55} ${b.y+nodeHeight/2}, ${b.x} ${b.y+nodeHeight/2}`});edgeLayer.append(path);}svg.append(edgeLayer);
  const nodeLayer=svgEl('g');for(const node of graph.nodes){const p=positions.get(node.id),group=svgEl('g',{class:`graph-node ${node.lane}`,'data-id':node.id,tabindex:'0',role:'button','aria-label':node.title});group.setAttribute('transform',`translate(${p.x} ${p.y})`);group.append(svgEl('rect',{width:nodeWidth,height:nodeHeight,rx:8}));const kicker=svgEl('text',{x:12,y:17,class:'node-kicker'});kicker.textContent=node.kind.toUpperCase().replaceAll('_',' ');group.append(kicker);wrapLabel(node.title).forEach((line,index)=>{const label=svgEl('text',{x:12,y:35+index*13,class:'node-title'});label.textContent=line+(index===2&&node.title.length>line.length?'…':'');group.append(label);});const parents=svgEl('text',{x:nodeWidth-12,y:17,'text-anchor':'end',class:'node-parents'});parents.textContent=node.parent_count?`${node.parent_count} parent${node.parent_count>1?'s':''}`:'source';group.append(parents);group.addEventListener('click',()=>selectNode(node.id));group.addEventListener('keydown',event=>{if(event.key==='Enter'||event.key===' '){event.preventDefault();selectNode(node.id);}});nodeLayer.append(group);}svg.append(nodeLayer);
  const preferred=graph.nodes.find(node=>node.lane==='comparison')||graph.nodes[0];if(preferred)selectNode(preferred.id);
}

function selectNode(id){
  const node=state.nodeById.get(id);if(!node)return;state.selected=id;document.querySelectorAll('.graph-node').forEach(item=>item.classList.toggle('selected',item.dataset.id===id));
  const detail=$('node-detail');detail.replaceChildren();detail.append(el('p','eyebrow',node.lane==='comparison'?'MULTI-PARENT COMPARISON':node.kind.toUpperCase().replaceAll('_',' ')),el('h3','',node.title));
  detail.append(el('p','',`Recorded at ${typeof node.time==='number'?number(node.time,3):'an archived source time'}. This node has ${node.parent_count} direct parent${node.parent_count===1?'':'s'}.`));
  const parents=state.graph.edges.filter(edge=>edge.target===id).map(edge=>edge.source),children=state.graph.edges.filter(edge=>edge.source===id).map(edge=>edge.target);
  if(parents.length)detail.append(linkGroup('PARENTS',parents));if(children.length)detail.append(linkGroup('DESCENDANTS',children));
  if(node.artifact_uri){const group=el('div','detail-group');group.append(el('span','detail-label','CONTENT-ADDRESSED ARTIFACT'),el('div','artifact-ref',node.artifact_uri));const artifact=state.artifactByUri.get(node.artifact_uri);if(artifact){const link=el('a','detail-download','Download verified artifact ↗');link.href=`/api/observatory/artifacts/${encodeURIComponent(artifact.file)}`;link.download=artifact.file;group.append(link);}else group.append(el('p','','Source receipt is verified in the report; the original source is not copied into this archive.'));detail.append(group);}
}
function linkGroup(label,ids){const group=el('div','detail-group');group.append(el('span','detail-label',label));const links=el('div','node-links');for(const id of ids){const button=el('button','node-link',id);button.type='button';button.onclick=()=>selectNode(id);links.append(button);}group.append(links);return group;}

function renderArtifacts(){
  const manifest=state.overview.manifest,body=$('artifact-rows');body.replaceChildren();state.artifactByUri.clear();
  for(const artifact of Object.values(manifest.artifacts))state.artifactByUri.set(artifact.artifact_uri,artifact);
  for(const [name,artifact] of Object.entries(manifest.artifacts).sort(([a],[b])=>a.localeCompare(b))){const row=document.createElement('tr');const nameCell=el('td','',name),purposeCell=el('td','',purpose(name)),sizeCell=el('td','',bytes(artifact.bytes)),hashCell=el('td');const code=el('code','',artifact.sha256);code.title=artifact.sha256;hashCell.append(code);const action=el('td');const link=el('a','download-link','Download ↓');link.href=`/api/observatory/artifacts/${encodeURIComponent(name)}`;link.download=name;action.append(link);row.append(nameCell,purposeCell,sizeCell,hashCell,action);body.append(row);}
}

function renderLimitations(){const list=$('limitations-list');list.replaceChildren();for(const item of state.overview.limitations||[])list.append(el('li','',item));}

async function init(){
  try{
    const [overview,graph]=await Promise.all([read('/api/observatory'),read('/api/observatory/graph')]);state.overview=overview;state.graph=graph;renderSources();renderModels();renderArtifacts();renderGraph();renderLimitations();
    text('weave-status',`${graph.nodes.length} nodes · ${graph.edges.length} edges · native reload ${graph.native_roundtrip.reload_equal?'equal':'failed'}`);text('api-state','● archive verified');$('api-state').classList.add('ready');
  }catch(error){text('api-state','archive unavailable');const fatal=$('fatal');fatal.hidden=false;fatal.textContent=`The observatory could not verify its archive: ${error.message}`;}
}
init();
