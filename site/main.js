for(const button of document.querySelectorAll('[data-filter]')){
 button.addEventListener('click',()=>{
  const filter=button.dataset.filter;
  for(const b of document.querySelectorAll('[data-filter]'))b.setAttribute('aria-pressed',String(b===button));
  for(const entry of document.querySelectorAll('.entry[data-status]'))entry.hidden=filter!=='all'&&entry.dataset.status!==filter;
 });
}
fetch('build-info.json').then(response=>response.ok?response.json():null).then(info=>{
 if(!info)return;
 const revision=info.commit||info.revision||info.git_sha;
 if(revision)for(const node of document.querySelectorAll('[data-revision]')){
  node.textContent=revision.slice(0,8);
  node.href=`https://github.com/emberian/chreatures/commit/${revision}`;
 }
}).catch(()=>{});
