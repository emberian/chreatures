const root=document.querySelector('#surfaces');
const labels={experience_tick:['experience','log₁₊tick'],goal_distance:['goal distance','code RMS'],history_rms:['action history','RMS'],motor_magnitude:['motor magnitude','mean |command|'],fatigue:['fatigue','fraction'],thrust:['thrust','command'],speed:['speed','m/s']};
const titles={body_law_residual:'Deployed body-law residual',effort:'Physical effort'};
const colors=[[24,49,40],[104,137,100],[231,190,112],[196,91,59]];
const fmt=value=>{const magnitude=Math.abs(value);return magnitude&&magnitude<.001?value.toExponential(2):value.toFixed(4)};
function nativeValue(data,name,z){const normal=data.feature_normalization[name];return z*normal.scale+normal.mean}
function colorAt(t){t=Math.max(0,Math.min(.9999,t));const scaled=t*(colors.length-1),index=Math.floor(scaled),mix=scaled-index,a=colors[index],b=colors[Math.min(colors.length-1,index+1)];return a.map((v,i)=>Math.round(v+(b[i]-v)*mix))}
function card(data,modelName,surface,index){
 const figure=document.createElement('figure');figure.className='surface-card';
 const [xName,yName]=surface.axes,[xLabel,xUnit]=labels[xName]||[xName,'native'],[yLabel,yUnit]=labels[yName]||[yName,'native'];
 figure.innerHTML=`<h3>${titles[modelName]} · ${index+1}</h3><p>${yLabel} × ${xLabel} · prediction: ${data.models[modelName].unit.replaceAll('_',' ')}</p><div class="plot-wrap"><canvas width="500" height="300" aria-label="${titles[modelName]} surface over ${xLabel} and ${yLabel}"></canvas><span class="plot-label y">${yLabel} · ${yUnit}</span><div class="plot-label"><span>${xLabel} · ${xUnit}</span><span class="range"></span></div><output class="plot-tooltip" hidden></output></div>`;
 const canvas=figure.querySelector('canvas'),context=canvas.getContext('2d'),values=surface.prediction.flat(),low=Math.min(...values),high=Math.max(...values),span=Math.max(1e-12,high-low),rows=surface.prediction.length,columns=surface.prediction[0].length,cellW=canvas.width/columns,cellH=canvas.height/rows;
 for(let y=0;y<rows;y++)for(let x=0;x<columns;x++){const rgb=colorAt((surface.prediction[y][x]-low)/span);context.fillStyle=`rgb(${rgb.join(',')})`;context.fillRect(x*cellW,y*cellH,Math.ceil(cellW),Math.ceil(cellH))}
 figure.querySelector('.range').textContent=`${fmt(low)} → ${fmt(high)}`;
 const tooltip=figure.querySelector('.plot-tooltip'),wrap=figure.querySelector('.plot-wrap');
 canvas.addEventListener('pointermove',event=>{const box=canvas.getBoundingClientRect(),x=Math.min(columns-1,Math.max(0,Math.floor((event.clientX-box.left)/box.width*columns))),y=Math.min(rows-1,Math.max(0,Math.floor((event.clientY-box.top)/box.height*rows))),xv=nativeValue(data,xName,surface.x_standardized[x]),yv=nativeValue(data,yName,surface.y_standardized[y]);tooltip.hidden=false;tooltip.innerHTML=`${xLabel}: ${fmt(xv)} ${xUnit}<br>${yLabel}: ${fmt(yv)} ${yUnit}<br>prediction: ${fmt(surface.prediction[y][x])}`;const parent=wrap.getBoundingClientRect();tooltip.style.left=`${Math.min(parent.width-190,event.clientX-parent.left+12)}px`;tooltip.style.top=`${Math.max(3,event.clientY-parent.top-58)}px`});
 canvas.addEventListener('pointerleave',()=>tooltip.hidden=true);return figure;
}
try{const response=await fetch('./assets/developmental-atlas.json');if(!response.ok)throw Error(`HTTP ${response.status}`);const data=await response.json();if(data.schema!=='chreatures-public-developmental-atlas-v1')throw Error('atlas schema differs');const cards=[];for(const modelName of ['body_law_residual','effort'])for(const [index,surface] of data.models[modelName].surfaces.entries())cards.push(card(data,modelName,surface,index));root.replaceChildren(...cards)}catch(error){root.textContent='The compact atlas could not be loaded.';console.error(error)}
