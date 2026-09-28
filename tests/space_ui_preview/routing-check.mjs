/* Routing check on a session's detail page (Agents > Sessions > a session).
   Run with the local fixture server from this directory. API responses below
   are synthetic and answers stay in memory; no real sessions are read. */
import assert from 'node:assert/strict';
import {pathToFileURL} from 'node:url';

const {chromium}=await import(process.env.PLAYWRIGHT_MODULE
  ?pathToFileURL(process.env.PLAYWRIGHT_MODULE).href:'playwright');
const origin=process.env.SPACE_PREVIEW_URL||'http://127.0.0.1:5100';
const browser=await chromium.launch({headless:true,
  executablePath:process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE||undefined});
const context=await browser.newContext({viewport:{width:1440,height:1000}});
const page=await context.newPage();
const errors=[];
page.on('pageerror',error=>errors.push(error.message));

const session=(id,i)=>({
  id,agent:'demo_alpha',project:'Aurora Console',project_path:'/demo/aurora-console',
  model:'model-atlas',started_at:new Date(Date.UTC(2026,8,28,0,i)).toISOString(),
  ended_at:new Date(Date.UTC(2026,8,28,0,i+1)).toISOString(),
  duration_sec:60,total_tokens:100,cost:0,cost_known:false,turns:1,subagents:[],tools:[],
});
/* routed-1 went through XO; manual-2 was started elsewhere */
const status={
  'routed-1':{routed:true,applied:{profile:'light',model:'model-comet',effort:'low',source:'sage'},answer:null},
  'manual-2':{routed:false,applied:null,answer:null},
};
const writes=[];
const json=(route,data,code=200)=>route.fulfill({status:code,contentType:'application/json',body:JSON.stringify(data)});
await context.route('**/xo/sessions.json',route=>json(route,{
  meta:{sources:[{id:'demo_alpha',label:'Runtime Alpha',available:true}]},
  totals:{sessions:2,sessions_by_agent:{demo_alpha:2}},
  sessions:[session('routed-1',2),session('manual-2',1)],daily_models:[],daily_sessions:[],daily_tools:[],
}));
await context.route('**/data/session_prompts.json?*',route=>json(route,{supported:false,prompts:[],total_prompts:0}));
await context.route('**/api/intelligence/sessions/**',async route=>{
  const request=route.request();
  const [agent,id,tail]=new URL(request.url()).pathname.split('/').slice(4);
  assert.equal(agent,'demo_alpha');
  if(request.method()==='POST'){
    assert.equal(tail,'feedback');
    const body=request.postDataJSON();
    writes.push({id,...body});
    status[id]={...status[id],answer:body.answer};
  }else assert.equal(request.method(),'GET');
  await json(route,{session_id:id,agent,...status[id]});
});

const card=page.locator('#sess-routing');
async function openSession(id){
  /* from a session's detail, back through the breadcrumb; otherwise load the list */
  if(await page.locator('#sess-back').count())await page.locator('#sess-back').click();
  else await page.goto(origin+'/space/#/agents/sessions',{waitUntil:'networkidle'});
  await page.locator('#sess-body tr[data-sid="demo_alpha:'+id+'"]').click();
  await card.locator('.sess-routing').waitFor();
}

try{
  await openSession('routed-1');
  assert.match(await card.textContent(),/ran on light · model-comet, low effort, chosen by Levanto Sage/);
  assert.match(await card.textContent(),/Did it still complete your task\?/);
  await card.locator('[data-routing-answer="no"]').click();
  await card.locator('.sess-routing-done').waitFor();
  assert.match(await card.textContent(),/Recorded: No/);
  assert.equal(await card.locator('[data-routing-answer="no"]').getAttribute('aria-pressed'),'true');
  assert.deepEqual(writes,[{id:'routed-1',answer:'no',model:'model-atlas'}]);

  await openSession('manual-2');
  assert.match(await card.textContent(),/ran on model-atlas \(not routed\)/);
  assert.match(await card.textContent(),/Did it complete your task\?/);
  await card.locator('[data-routing-answer="yes"]').click();
  await card.locator('.sess-routing-done').waitFor();
  assert.deepEqual(writes.at(-1),{id:'manual-2',answer:'yes',model:'model-atlas'});
  await page.screenshot({path:'/tmp/space-routing-check.png'});

  assert.deepEqual(errors,[]);
  console.log('Routing check: the layer a routed session ran on, "not routed" for others, and yes/no answers saved and shown.');
}finally{
  await browser.close();
}
