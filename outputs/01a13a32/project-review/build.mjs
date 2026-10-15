import fs from 'node:fs/promises';
import path from 'node:path';
import {createRequire} from 'node:module';
import {fileURLToPath, pathToFileURL} from 'node:url';
import {execFileSync} from 'node:child_process';
const here=path.dirname(fileURLToPath(import.meta.url)), root='F:/CMBL';
const req=createRequire('C:/Users/tonmo/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/package.json');
const {Workbook,SpreadsheetFile}=await import(pathToFileURL(req.resolve('@oai/artifact-tool')).href);
const tracked=new Set(execFileSync('git',['ls-files'],{cwd:root,encoding:'utf8'}).trim().split(/\r?\n/));
const bulk=new Map(), files=[];
function bucket(p){
 if(p.startsWith('outputs/'))return 'outputs/';
 if(p.startsWith('.claude/worktrees/'))return p.split('/').slice(0,3).join('/')+'/';
 if(p.startsWith('.git/'))return '.git/';
 if(p.startsWith('.venv/'))return '.venv/';
 if(p.includes('/__pycache__/'))return p.slice(0,p.indexOf('/__pycache__/'))+'/__pycache__/';
 if(p.startsWith('datasets/raw/UAV_VisLoc_dataset/')&&p.split('/').length>4)return p.split('/').slice(0,4).join('/')+'/';
 if(p.startsWith('datasets/sample/'))return p.split('/').slice(0,3).join('/')+'/';
 if(p.startsWith('datasets/cache/'))return p.split('/').slice(0,3).join('/')+'/';

 if(p.startsWith('build/'))return p.split('/').slice(0,3).join('/')+'/';
 if(p.startsWith('results/'))return path.posix.dirname(p)+'/';
 if(p.startsWith('figures/'))return path.posix.dirname(p)+'/';
 return null;
}
async function walk(dir,rel=''){
 for(const d of await fs.readdir(dir,{withFileTypes:true})){
  const p=rel?rel+'/'+d.name:d.name, full=path.join(dir,d.name);
  if(d.isSymbolicLink())continue;
  if(d.isDirectory()){await walk(full,p);continue;}
  const st=await fs.stat(full), b=bucket(p);
  if(b){const v=bulk.get(b)||{count:0,bytes:0};v.count++;v.bytes+=st.size;bulk.set(b,v);}
  else files.push({p,bytes:st.size});
 }
}
await walk(root);
const texts=[];
for(const {p} of files.filter(x=>/\.(py|yaml|json|md|txt)$/.test(x.p)&&!/sources\//.test(x.p))){
 if((await fs.stat(root+'/'+p)).size<200000)texts.push([p,await fs.readFile(root+'/'+p,'utf8')]);
}
function decision(p,body='',refs=0){
 if(p.startsWith('outputs/'))return ['KEEP','Current inventory workbook, previews and generator','Review deliverables; inventory sizes are a snapshot before this workbook export.'];
 if(p.startsWith('build/cloud4080'))return ['KEEP','Prepared cloud RTX 4080 code/cache bundle','Committed repair source and hash manifest for the pending cloud benchmark.'];
 if(p.startsWith('.claude/worktrees/'))return ['KEEP','Separate Claude Git worktree','May contain another chat’s work; do not delete as duplicate source.'];
 if(p.startsWith('.git/'))return ['KEEP','Git history and recovery','Never remove: contains commits, tags and worktree links.'];
 if(p==='.gitignore'||p==='.gitattributes')return ['KEEP','Git file handling rules','Preserves local-data exclusions and frozen artifact byte identities.'];
 if(p.startsWith('.venv'))return ['KEEP','Project Python environment','Keep until a replacement environment passes verification.'];
 if(p.includes('__pycache__'))return ['REMOVE CANDIDATE','Python bytecode cache','Automatically regenerated; safe after stopping Python jobs.'];
 if(/\.part$/.test(p))return ['REVIEW','Incomplete dataset download','May be resumable; compare with complete release before removal.'];
 if(p.startsWith('datasets/raw/'))return ['KEEP','Original UAV-VisLoc research inputs','Required for labels, provenance and reproduction; do not publish.'];
 if(p.startsWith('datasets/cache/'))return ['KEEP','Computed reference embeddings or terrain cache','Expensive to regenerate; needed for current pipeline.'];
 if(p.startsWith('datasets/sample/'))return ['REVIEW','Earlier sample release','Compare byte hashes against full release before archiving; not assumed duplicate.'];
 if(p.startsWith('datasets/manifests/'))return ['KEEP','Dataset identity, geometry, splits or quality metadata','Required scientific inputs; local paths and provenance must remain traceable.'];
 if(p.startsWith('build/kaggle/'))return ['KEEP','Kaggle bundles, job metadata or downloaded artifacts','Retain until all jobs and bundle provenance are reconciled.'];
 if(p.startsWith('build/verify/20261014'))return ['KEEP','Current verification evidence','Documents current environment and reproduction checks.'];
 if(p.startsWith('build/local_smoke')||p.startsWith('build/local_launcher')||p.startsWith('build/cleanup'))return ['KEEP','Current local execution and cleanup evidence','Retain current run logs, hash audit and validation evidence.'];
 if(p.startsWith('build/'))return ['ARCHIVE','Generated build or verification history','Regenerable, but keep failed-run evidence and a verified archive first.'];
 if(p.startsWith('results/'))return ['KEEP','Experiment outcomes, features or predictions','Scientific evidence and golden references; archive only with hashes.'];
 if(p.startsWith('figures/'))return ['ARCHIVE','Rendered experiment figures','Preserve reviewed evidence; regenerate only after inputs and commands are secured.'];
 if(p.startsWith('sources/'))return ['ARCHIVE','Captured literature or dataset source evidence','Move to a documented literature archive; copyright/source provenance retained.'];
 if(p.startsWith('A_help/'))return ['ARCHIVE','Internal research plan or earlier source notes','Historical rationale; preserve before moving out of the active workspace.'];
 if(p.endsWith('.docx'))return ['KEEP','Paper draft','User-authored and untracked; never delete as generated clutter.'];
 if(p.startsWith('configs/'))return ['KEEP','Experiment protocol or configuration','Older versions are referenced by provenance and reproduction checks.'];
 if(p.startsWith('reports/'))return ['KEEP','Research decision or measured evidence','Preserves the chronology of methods, failures and deviations.'];
 if(p.startsWith('literature/'))return ['KEEP','Literature comparison and source notes','Supports claims, citations and novelty assessment.'];
 if(/SUPERSEDED/.test(body))return ['KEEP','Superseded historical executable','Still referenced by imports/provenance; archive only after dependency isolation.'];
 if(p.endsWith('.py'))return ['KEEP',body.match(/"""([^\n]+)/)?.[1]?.slice(0,150)||'Research pipeline script',refs?'Referenced by other project files.':'Entry point or utility; absence of text references is not proof of disuse.'];
 return ['KEEP','Project documentation or settings','Retain unless an explicit replacement is identified.'];
}
const rows=[];
for(const f of files.sort((a,b)=>a.p.localeCompare(b.p))){
 const body=texts.find(t=>t[0]===f.p)?.[1]||'';
 const ref=texts.filter(t=>t[0]!==f.p&&t[1].includes(path.posix.basename(f.p))).map(t=>t[0]);
 const [dec,purpose,why]=decision(f.p,body,ref.length);
 rows.push([f.p,purpose,dec,why,1,f.bytes/1048576,tracked.has(f.p)?'Tracked':'Local / untracked',ref.slice(0,5).join('; ')||'No text reference found','Not removed']);
}
for(const [p,v]of [...bulk].sort()){
 const [dec,purpose,why]=decision(p);
 rows.push([p,purpose,dec,why,v.count,v.bytes/1048576,'Grouped collection','All regular files below this path; links excluded','Not removed']);
}
const issues=[
 ['CLEAN','P1','Review removal candidates','Generate inventory and sizes; preserve evidence; delete only verified disposable files.','Inventory complete; decisions pending','No irreplaceable file removed'],
 ['H1-H3','P1','Numeric hypothesis decisions absent','Write prospective thresholds and chronology disclosure; commit before additional outcome inspection.','Planned','Paired group contrasts and predeclared pass/fail rules'],
 ['S11','P1','Remote kernel state unknown','Query each private kernel; record missing, running, failed or complete separately from local files.','Blocked: network','All ten kernels timestamped'],
 ['R1','P0','Wrong attempt grid can pass','Require exact frame/radius/offset/variant keys, expected pilot keys, and row contracts.','Planned','Malformed grid and zero overlap fail'],
 ['R2','P0','Resume overwrites provenance','Freeze fingerprint; reject incompatible resumes; append events without replacing initial manifest.','Planned','Changed config, selection or code rejected'],
 ['R3','P0','Truncated rows skipped on resume','Validate complete schema and rows; fail safely on a damaged tail; preserve source bytes.','Planned','Partial row never marked complete'],
 ['R4','P0','Unknown outcomes become negative labels','Reject unknown/unlabelled outcomes and inconsistent acceptance or hashes before feature work.','Planned','Blank and misspelled outcomes fail'],
 ['R5','P0','Embedding identity not checked','Verify ordered tile IDs and geometry; bind raster, cache, model and preprocessing hashes.','Planned','Reordered or stale caches fail'],
 ['R6','P0','Features and embeddings can become misaligned','Store attempt keys in NPZ; verify order, dimensions, uniqueness, finite values and group membership.','Planned','Reordered CSV or wrong NPZ fails'],
 ['VERIFY','P0','Full-scale artifacts not revalidated','Run repaired acceptance on fetched 05/10 and full verifier in project environment.','Planned','Integrity reports saved for each fetched region'],
 ['S9','P1','Replay protocol not frozen','Resolve hybrid/live decision; calibrate drift on synthetic routes; declare exact seeds and approval record.','Planned','Committed drift calibration and protocol'],
 ['R7','P1','DINO code and weights unpinned','Identify cached upstream revision; pin and enforce code/weight identity; record actual environments.','Planned','Fresh/offline load rejects mismatches'],
 ['GUIDE','P1','Stage 11 guide absent','Document stage, push, status, fetch, acceptance and rewritten tag mapping.','Planned','Exact tag commit and legacy mapping recorded'],
 ['REPLAY','P1','Replay engine and live check absent','Implement paired replay, budget matching, truth boundary and false fixes; run 2,000 live comparisons.','Planned','Determinism tests and measured fidelity gate'],
 ['BASELINE','P1','Current baseline report absent','Use a small fixed PC batch; compare frozen methods and save dated measured report.','Planned','No invented results; exact batch and runtime'],
 ['LICENSE','P1','Code licence unspecified','Select code licence with author; add separate data rights statement citing original sources.','Needs licence choice','Licence covers original code only'],
 ['PAPER','P2','Paper has unfinished evidence','Preserve existing v0.2; update methods and prereg chronology; keep results placeholders.','Draft exists','No pilot findings represented as full results'],
 ['LAUNCH','P0','4080 equivalence absent','Run the same 180 attempts on 4080 and compare T4 outcomes/positions before launch.','Blocked: no 4080','Hardware recorded and agreement gate passes'],
 ['ENV','P1','Global Python has two OpenCV installs','Use and validate the project venv; avoid changing shared system packages.','Planned','Exactly one OpenCV distribution'],
 ['R8','P1','Unsafe archive replacement','Stage archive imports with prefix, type and size checks; validate before replacing results.','Planned','Unexpected entries rejected, old output retained'],
 ['R9','P1','Negative-path regression tests absent','Add synthetic fixtures for corruption, missing keys, cache mismatch and alignment.','Planned','Regression suite passes'],
 ['R10','P2','Prose and executable config can drift','Reject unsupported versions; validate frozen constants and explicit model settings.','Planned','Unknown protocol fails; golden predictions unchanged'],
 ['R11','P2','Repeated feature computation','Profile first; cache only query-independent features with strict size/fold boundaries.','Planned','Measured benefit and unchanged numerical results'],
 ['R12','P1','Manifest identity contains local paths','Add semantic dataset identity alongside original byte hash; preserve legacy provenance.','Planned','Relocation preserves semantic identity'],
 ['R13','P1','Replay requirement incomplete','Tracked under REPLAY and S9.','Planned','End-to-end route evidence'],
 ['R14','P1','Macro differences lack paired uncertainty','Resample same groups for paired macro differences and report eligible fold counts.','Planned','Fixed-seed paired contrasts and intervals'],
 ['R15','P2','Full reports retain pilot scope wording','Make report scope follow actual inputs; distinguish development and headline evidence.','Planned','Full run never labelled pilot automatically'],
 ['PROVENANCE','P1','Earlier audit reported false commit mismatch','25b6060 maps to 88631e3 through recorded history rewrite; retain both identifiers.','Corrected','reports/history_rewrite_2026-09-30.md']
];
try{const updates=JSON.parse(await fs.readFile(here+'/status.json','utf8'));for(const row of issues)if(updates[row[0]])row[4]=updates[row[0]];}catch{}
const wb=Workbook.create();
function sheet(name,title,headers,data,widths){
 const s=wb.worksheets.add(name);s.showGridLines=false;s.tabColor='#243B53';
 s.getRange('A2').values=[[title]];s.getRange('A2').format.font={name:'Arial',size:15,bold:true};
 s.getRange('A3').values=[['2026-10-14 | Verified duplicate sample removed (3.19 GiB). Remaining decisions are recommendations.']];
 const last=String.fromCharCode(64+headers.length),end=data.length+5;
 s.getRange(`A5:${last}${end}`).values=[headers,...data];
 const all=s.getRange(`A5:${last}${end}`);all.format.font={name:'Arial',size:10};all.format.rowHeight=52;all.format.verticalAlignment='top';all.format.wrapText=true;
 s.tables.add(`A5:${last}${end}`,true,name.replace(/ /g,'')+'Table');
 s.getRange(`A5:${last}5`).format.fill='#243B53';s.getRange(`A5:${last}5`).format.font={bold:true,color:'#FFFFFF'};
 widths.forEach((w,i)=>s.getRange(`${String.fromCharCode(65+i)}5:${String.fromCharCode(65+i)}${end}`).format.columnWidth=w);
 s.freezePanes.freezeRows(5);return s;
}
const inv=sheet('Inventory','CMBL — keep, archive or remove', ['Path / collection','What it does','Decision','Reason / prerequisite','Files','Size (MiB)','Tracking','References / evidence','Action'],rows,[53,52,23,65,9,14,23,60,18]);
inv.getRange(`F6:F${rows.length+5}`).setNumberFormat('#,##0.00');
inv.getRange(`C6:C${rows.length+5}`).dataValidation={rule:{type:'list',values:['KEEP','ARCHIVE','REVIEW','REMOVE CANDIDATE']}};
inv.getRange(`C6:C${rows.length+5}`).conditionalFormats.add('containsText',{text:'REMOVE CANDIDATE',format:{fill:'#FCE4D6'}});
sheet('Fix plan','CMBL — problems, implementation and acceptance', ['ID','Priority','Problem','Fix plan','Current state','Acceptance evidence'],issues,[19,12,44,80,42,65]);
const storage=[...bulk].map(([p,v])=>[p,v.count,v.bytes/1048576,...decision(p)]).sort((a,b)=>b[2]-a[2]);
const st=sheet('Storage review','CMBL — largest collections first',['Collection','Files','Size (MiB)','Decision','Purpose','Removal condition'],storage,[68,12,17,23,55,75]);st.getRange(`C6:C${storage.length+5}`).setNumberFormat('#,##0.00');
wb.recalculate();
console.log((await wb.inspect({kind:'table',range:'Inventory!A5:F9',include:'values',tableMaxRows:5,tableMaxCols:6,maxChars:2000})).ndjson);
console.log((await wb.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#NUM!',options:{useRegex:true,maxResults:10},maxChars:800})).ndjson);
for(const name of ['Inventory','Fix plan','Storage review']){
 const blob=await wb.render({sheetName:name,range:'A1:D11',scale:1,format:'png'});
 await fs.writeFile(here+'/'+name.replace(/ /g,'_')+'.png',new Uint8Array(await blob.arrayBuffer()));
}
await (await SpreadsheetFile.exportXlsx(wb)).save(here+'/CMBL_inventory_and_fix_plan.xlsx');
await fs.writeFile(here+'/inventory.json',JSON.stringify({date:'2026-10-14',rows,issues},null,2));
console.log(JSON.stringify({inventoryEntries:rows.length,totalFiles:rows.reduce((s,r)=>s+r[4],0),issues:issues.length,output:here+'/CMBL_inventory_and_fix_plan.xlsx'}));
