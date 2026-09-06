import * as THREE from 'three';
import {OrbitControls} from './vendor/three/OrbitControls.js';

const canvas=document.querySelector('#reef'), error=document.querySelector('#error');
try{
 const renderer=new THREE.WebGLRenderer({canvas,antialias:true});
 renderer.setPixelRatio(Math.min(devicePixelRatio,2));
 renderer.setClearColor('#d8dfcf');
 renderer.outputColorSpace=THREE.SRGBColorSpace;
 const scene=new THREE.Scene();scene.fog=new THREE.Fog('#d8dfcf',23,55);
 scene.add(new THREE.HemisphereLight('#fff9e8','#4d6651',2.7));
 const sun=new THREE.DirectionalLight('#fff3ce',3);sun.position.set(-1,-4,10);scene.add(sun);
 const camera=new THREE.PerspectiveCamera(36,1,.01,80);camera.up.set(0,0,1);
 const controls=new OrbitControls(camera,canvas);controls.enableDamping=true;controls.minDistance=.3;controls.maxDistance=32;controls.maxPolarAngle=Math.PI*.495;
 const geometries={
  sphere:new THREE.SphereGeometry(1,14,10),
  box:new THREE.BoxGeometry(2,2,2),
  cylinder:new THREE.CylinderGeometry(1,1,2,14).rotateX(Math.PI/2),
 };
 const material=new THREE.MeshStandardMaterial({roughness:.88,metalness:0});
 const root=new THREE.Group();scene.add(root);
 const dummy=new THREE.Object3D(), color=new THREE.Color();
 let frames=[],stage=1,close=true;
 function view(){
  if(close){camera.position.set(4.75,.1,2.25);controls.target.set(2.8,2.9,1.1);}
  else{camera.position.set(15,-12,12.5);controls.target.set(6,4,.55);}
  controls.update();
 }
 function display(index){
  stage=index;
  for(const child of [...root.children]){root.remove(child);child.dispose();}
  const groups={sphere:[],box:[],cylinder:[]};
  for(const entity of frames[index].entities)for(const shape of entity.shapes){
   const q=new THREE.Quaternion(shape.quaternion[1],shape.quaternion[2],shape.quaternion[3],shape.quaternion[0]);
   const position=new THREE.Vector3(...shape.position),s=shape.size;
   const add=(kind,p,scale)=>groups[kind].push({position:p,quaternion:q,scale,color:shape.color});
   if(shape.type==='box')add('box',position,s);
   else if(shape.type==='sphere')add('sphere',position,[s[0],s[0],s[0]]);
   else if(shape.type==='ellipsoid')add('sphere',position,s);
   else if(shape.type==='cylinder'||shape.type==='capsule'){
    add('cylinder',position,[s[0],s[0],s[1]]);
    if(shape.type==='capsule'){
     const offset=new THREE.Vector3(0,0,s[1]).applyQuaternion(q);
     add('sphere',position.clone().add(offset),[s[0],s[0],s[0]]);
     add('sphere',position.clone().sub(offset),[s[0],s[0],s[0]]);
    }
   }
  }
  for(const [kind,parts] of Object.entries(groups)){
   if(!parts.length)continue;
   const mesh=new THREE.InstancedMesh(geometries[kind],material,parts.length);
   parts.forEach((part,i)=>{
    dummy.position.copy(part.position);dummy.quaternion.copy(part.quaternion);dummy.scale.set(...part.scale);dummy.updateMatrix();
    mesh.setMatrixAt(i,dummy.matrix);mesh.setColorAt(i,color.set(part.color));
   });
   mesh.instanceMatrix.needsUpdate=true;mesh.instanceColor.needsUpdate=true;mesh.computeBoundingSphere();root.add(mesh);
  }
  document.querySelector('#time').textContent=`${Math.round(frames[index].time)} model seconds · ${index===0?'supplied terrain and colony mounts':'1,608 added physical parts'}`;
  for(const button of document.querySelectorAll('[data-frame]'))button.setAttribute('aria-pressed',String(Number(button.dataset.frame)===index));
 }
 for(const button of document.querySelectorAll('[data-frame]'))button.addEventListener('click',()=>{if(frames.length)display(Number(button.dataset.frame));});
 document.querySelector('#closer').addEventListener('click',()=>{close=!close;document.querySelector('#closer').setAttribute('aria-pressed',String(close));document.querySelector('#closer').textContent=close?'Whole habitat':'Colony view';view();});
 new ResizeObserver(()=>{const w=canvas.clientWidth,h=canvas.clientHeight;renderer.setSize(w,h,false);camera.aspect=w/h;camera.updateProjectionMatrix();}).observe(canvas);
 view();
 renderer.setAnimationLoop(()=>{controls.update();renderer.render(scene,camera);});
 const response=await fetch('assets/reef-recording.json');if(!response.ok)throw Error('recording unavailable');
 const data=await response.json();frames=data.frames;display(stage);
}catch(reason){error.hidden=false;document.querySelector('#time').textContent='Recording preview unavailable';console.error(reason);}
