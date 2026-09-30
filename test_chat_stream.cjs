// Exercise the shipped inline event handlers with a minimal DOM/EventSource.
// No browser packages or network are needed: node --test test_chat_stream.cjs.
const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');

class Element {
  constructor() { this.children=[]; this.textContent=''; this.className=''; this.value=''; this.attributes={}; this.dataset={}; }
  append(...nodes) { this.children.push(...nodes); }
  appendChild(node) { this.append(node); }
  prepend(node) { this.children.unshift(node); }
  replaceChildren(...nodes) { this.children=[...nodes]; }
  cloneNode() { return new Element(); }
  querySelector(selector) { return selector==='p:last-child' ? (this.paragraph ||= new Element()) : null; }
  setAttribute(key,value) { this.attributes[key]=value; }
  focus() { this.focused=true; }
  get firstElementChild() { return this.children[0]; }
  get classList() {
    return {
      contains: value=>this.className.split(' ').includes(value),
      add: value=>{this.className+=' '+value;},
      remove: value=>{this.className=this.className.split(' ').filter(c=>c!==value).join(' ');},
    };
  }
}

function fixture() {
  const elements = new Map();
  const el = id=>{
    if (!elements.has(id)) elements.set(id,new Element());
    return elements.get(id);
  };
  el('messages').append(new Element());
  class EventSource {
    static CLOSED=2;
    static instances=[];
    constructor(url) { this.url=url; this.handlers={}; this.readyState=1; EventSource.instances.push(this); }
    addEventListener(name,handler) { (this.handlers[name] ||= []).push(handler); }
    close() { this.readyState=2; }
    dispatch(name,data,id) {
      for (const handler of this.handlers[name] || []) {
        handler({data: data===undefined ? undefined : JSON.stringify(data), lastEventId:String(id || '')});
      }
    }
  }
  const decisions=['approve','conditional','deny'].map(decision=>{
    const button=new Element(); button.dataset.inboxDecision=decision; return button;
  });
  const context=vm.createContext({
    document: {getElementById:el, createElement:()=>new Element(), querySelectorAll:selector=>selector==='[data-inbox-decision]' ? decisions : []},
    EventSource, URLSearchParams, setInterval:()=>0,
    // Startup catalog remains pending; only the real stream handlers are tested.
    fetch:()=>new Promise(()=>{}),
  });
  const html=fs.readFileSync(new URL('./index.html', `file://${__filename}`),'utf8');
  vm.runInContext(html.match(/<script>([\s\S]*?)<\/script>/)[1],context);
  el('messages').replaceChildren();
  context.connect('test-run');
  return {context,el,decisions,EventSource,source:EventSource.instances.at(-1)};
}

test('real deltas update one bubble and final/replayed events never duplicate text',()=>{
  const {el,source}=fixture();
  source.dispatch('response_start',{message_id:'m1'},1);
  source.dispatch('response_delta',{message_id:'m1',content:'Meridian '},2);
  assert.equal(el('messages').children.length,1);
  const bubble=el('messages').children[0];
  assert.equal(bubble.children[1].textContent,'Meridian ');
  assert.equal(bubble.attributes['aria-busy'],'true');
  source.dispatch('response_delta',{message_id:'m1',content:'Meridian '},2);
  source.dispatch('response_delta',{message_id:'m1',content:'Retail.'},3);
  source.dispatch('response_end',{message_id:'m1',intermediate:false},4);
  source.dispatch('agent_response',{message_id:'m1',content:'Meridian Retail.'},5);
  source.dispatch('done',{paused:false},6);
  assert.equal(el('messages').children.length,1);
  assert.equal(bubble.children[1].textContent,'Meridian Retail.');
  assert.equal(bubble.attributes['aria-busy'],'false');
  assert.equal(el('status').textContent,'Ready');
});

test('tool preamble and final answer are separate, and stale run events are ignored',()=>{
  const {context,el,source,EventSource}=fixture();
  source.dispatch('response_start',{message_id:'preamble'},1);
  source.dispatch('response_delta',{message_id:'preamble',content:'Checking.'},2);
  source.dispatch('response_end',{message_id:'preamble',intermediate:true},3);
  source.dispatch('response_start',{message_id:'answer'},4);
  source.dispatch('response_delta',{message_id:'answer',content:'Done.'},5);
  source.dispatch('agent_response',{message_id:'answer',content:'Done.'},6);
  assert.equal(el('messages').children.length,2);
  assert.equal(el('messages').children[0].children[0].textContent,'Assistant · tool update');
  context.connect('next-run');
  source.dispatch('response_delta',{message_id:'answer',content:'STALE'},7);
  EventSource.instances.at(-1).dispatch('agent_response',{message_id:'deterministic',content:'Handoff result.'},1);
  assert.deepEqual(el('messages').children.map(card=>card.children[1].textContent),['Checking.','Done.','Handoff result.']);
});

test('disconnect retains partial text for replay and provider errors mark it interrupted',()=>{
  const {el,source}=fixture();
  source.dispatch('response_start',{message_id:'m1'},1);
  source.dispatch('response_delta',{message_id:'m1',content:'Partial'},2);
  source.readyState=0;
  source.dispatch('error');
  assert.equal(el('status').textContent,'Reconnecting…');
  source.dispatch('response_delta',{message_id:'m1',content:'Partial'},2);
  assert.equal(el('messages').children[0].children[1].textContent,'Partial');
  source.dispatch('error',{message:'Provider failed'},3);
  assert.equal(el('messages').children[0].children[0].textContent,'Assistant · interrupted');
  assert.equal(el('messages').children.length,2);
});

test('no-tool denials show tenant and request decisions in the activity feed',()=>{
  const {el,source}=fixture();
  source.dispatch('tenant_verification_completed',{resource:'tenant:beacon',decision:'allow'},1);
  source.dispatch('authorization_completed',{resource:'Meridian Retail',decision:'deny',auth_scope:'request'},2);
  source.dispatch('request_denied',{resource:'Meridian Retail',reason_code:'resource_unavailable'},3);
  source.dispatch('agent_response',{message_id:'denial',content:'Access denied.'},4);
  source.dispatch('done',{paused:false},5);
  assert.equal(el('activity').children.length,3);
  assert.equal(el('activity').children[0].children[0].textContent,'request denied');
  assert.equal(el('activity').children[1].children[0].textContent,'authorization completed · deny');
  assert.equal(el('activity').children[2].children[0].textContent,'tenant verification completed · allow');
  assert.equal(el('messages').children[0].children[1].textContent,'Access denied.');
});

function inboxFixture() {
  const f=fixture();
  f.el('tenant').value='northstar'; f.el('persona').value='user:northstar/lead';
  vm.runInContext('canReview=true',f.context);
  f.item={id:'approval-1',status:'pending',account_name:'Meridian Retail',operation:'save_account_brief',
    requester:'Maya Chen',demo:true,expected_version:1,created_at:'2026-09-28T12:00:00Z',
    request_run_id:'request-run',conversation_id:'conversation-1',content:'<b>Proposed brief</b>',comment:''};
  f.data={items:[f.item],pending_count:1};
  return f;
}
const response=(data,status=200)=>({ok:status<400,status,json:async()=>data});
const deferred=()=>{let resolve;const promise=new Promise(done=>{resolve=done;});return {promise,resolve};};

test('inbox renders a proposal safely, preserves notes on refresh, and disables resolved actions',()=>{
  const {context,el,decisions,data,item}=inboxFixture();
  context.applyInbox(data);
  assert.equal(el('inbox-detail').hidden,false);
  assert.equal(el('inbox-proposal').textContent,'<b>Proposed brief</b>');
  assert.match(el('inbox-metadata').textContent,/Source run: request-run/);
  assert.equal(el('inbox-tab').textContent,'Approvals inbox (1)');
  assert.ok(decisions.every(button=>!button.disabled));
  el('inbox-comment').value='Confirm the fix date.';
  context.applyInbox(data);
  assert.equal(el('inbox-comment').value,'Confirm the fix date.');
  const second={...item,id:'approval-2',account_name:'Westhaven Energy',comment:'Existing note'};
  context.applyInbox({items:[item,second],pending_count:2});
  el('inbox-list').children[1].onclick();
  assert.match(el('inbox-title').textContent,/Westhaven/);
  assert.equal(el('inbox-comment').value,'Existing note');
  context.applyInbox({items:[{...second,status:'approved',outcome:'saved'}],pending_count:0});
  assert.ok(decisions.every(button=>button.disabled));
  assert.equal(el('inbox-comment').disabled,true);
  assert.match(el('inbox-outcome').textContent,/Approved and saved/);
});

test('context reset clears proposal data and discards an old tenant response',async()=>{
  const {context,el,data}=inboxFixture();
  context.applyInbox(data);
  const wait=deferred(); context.fetch=()=>wait.promise;
  const loading=context.loadInbox();
  context.resetInbox();
  el('tenant').value='beacon'; el('persona').value='user:beacon/lead';
  wait.resolve(response(data)); await loading;
  assert.equal(el('inbox-list').children.length,0);
  assert.equal(el('inbox-proposal').textContent,'');
  assert.equal(el('inbox-metadata').textContent,'');
  assert.equal(el('inbox-detail').hidden,true);
  assert.equal(el('inbox-tab').textContent,'Approvals inbox');
});

test('revoked inbox permission clears cached content and offers only lead guidance',async()=>{
  const {context,el,decisions,data}=inboxFixture();
  context.applyInbox(data); el('inbox-comment').value='Sensitive draft note';
  context.fetch=async()=>response({error:'Select an eligible lead'},403);
  await context.loadInbox();
  assert.equal(el('inbox-proposal').textContent,'');
  assert.equal(el('inbox-comment').value,'');
  assert.equal(el('inbox-access').hidden,false);
  assert.equal(el('populate-approvals').disabled,true);
  assert.ok(decisions.every(button=>button.disabled));
});

test('populate is single-flight and reports reused pending requests without claiming a write',async()=>{
  const {context,el,data}=inboxFixture();
  const wait=deferred(), calls=[];
  context.fetch=(path,options)=>{calls.push({path,body:JSON.parse(options.body)});return wait.promise;};
  const populating=context.populateInbox(); await context.populateInbox();
  assert.equal(calls.length,1);
  assert.equal(calls[0].path,'/api/approvals/demo');
  assert.deepEqual(calls[0].body,{tenant_id:'northstar',user_id:'user:northstar/lead'});
  assert.equal(el('populate-approvals').disabled,true);
  wait.resolve(response({created:2,reused:1,inbox:data})); await populating;
  assert.match(el('inbox-notice').textContent,/Started 2 demo requests; reused 1 pending requests. No records were changed/);
  assert.equal(el('populate-approvals').disabled,false);
});

test('review sends the exact selected proposal, prevents duplicate submits, and waits for real outcome',async()=>{
  const {context,el,decisions,data,item}=inboxFixture();
  context.applyInbox(data); el('inbox-comment').value='Ready for review';
  const wait=deferred(),calls=[];
  context.fetch=async(path,options)=>{
    calls.push({path,options});
    return options.method==='POST' ? wait.promise : response({items:[{...item,status:'resuming'}],pending_count:1});
  };
  const reviewing=context.reviewInbox('approve'); await context.reviewInbox('approve');
  assert.equal(calls.length,1);
  assert.ok(decisions.every(button=>button.disabled));
  assert.deepEqual(JSON.parse(calls[0].options.body),{tenant_id:'northstar',user_id:'user:northstar/lead',decision:'approve',comment:'Ready for review'});
  assert.equal(calls[0].path,'/api/approvals/approval-1/decision');
  wait.resolve(response({status:'resuming'},202)); await reviewing;
  assert.match(el('inbox-outcome').textContent,/Waiting for the original run/);
  assert.ok(decisions.every(button=>button.disabled));
});

test('waiting owner reconnects to decision stream and ignores stale owner responses',async()=>{
  const {context,EventSource}=inboxFixture();
  vm.runInContext("conversationId='conversation-1';pending={approval_id:'approval-1'}",context);
  context.fetch=async()=>response({resolution_run_id:'decision-run',status:'resuming'});
  await context.syncOwnerReview();
  assert.equal(EventSource.instances.at(-1).url,'/api/stream/decision-run');
  const count=EventSource.instances.length, wait=deferred();
  vm.runInContext("busy=false;pending={approval_id:'approval-2'}",context);
  context.fetch=()=>wait.promise;
  const syncing=context.syncOwnerReview();
  context.newChat();
  wait.resolve(response({resolution_run_id:'stale-run'})); await syncing;
  assert.equal(EventSource.instances.length,count);
});

test('new chat reloads lead eligibility and empty inbox stays actionable',async()=>{
  const {context,el}=inboxFixture();
  context.fetch=async()=>response({can_review_approvals:true,tenants:[],accounts:[],
    scenarios:[{prompt:'Look up Meridian Retail'}],counts:{tenants:3,accounts:9,relationship_tuples:171},reference_date:'2026-09-28'});
  await el('new-chat').onclick();
  context.applyInbox({items:[],pending_count:0});
  assert.equal(el('populate-approvals').disabled,false);
  assert.equal(el('inbox-detail').hidden,true);
  assert.match(el('inbox-list').children[0].textContent,/No approvals to review/);
});

function scenarioCatalog(tenant='northstar') {
  const name=tenant==='northstar' ? 'Northstar Cloud' : 'Beacon Data';
  const account=tenant==='northstar' ? 'Meridian Retail' : 'Juniper Manufacturing';
  return {tenants:[{id:tenant,name,users:[]}],accounts:[],can_review_approvals:false,
    scenarios:[{id:'renewal-risk',name:'Renewal meeting: open blockers',account,
      prompt:`Prepare me for ${account}'s renewal meeting. Find the open issues and cite the evidence.`,
      description:'Tests governed evidence retrieval.',expected:'A cited summary; no records change.'}],
    counts:{tenants:3,accounts:9,relationship_tuples:171},reference_date:'2026-09-28'};
}

test('tenant scenarios expose the test and expected result in hover and accessible descriptions',()=>{
  const {context,el}=fixture(); el('tenant').value='beacon';
  const data=scenarioCatalog('beacon'); context.renderScenarios(data);
  assert.equal(el('scenarios-heading').textContent,'Beacon Data scenarios');
  const [button,description]=el('scenarios').children;
  assert.equal(button.type,'button');
  assert.equal(button.children[0].textContent,'Renewal meeting: open blockers');
  assert.equal(button.children[1].textContent,'Juniper Manufacturing');
  assert.match(button.title,/Tests governed evidence retrieval/);
  assert.match(button.title,/Expected: A cited summary; no records change/);
  assert.equal(button.attributes['aria-describedby'],description.id);
  assert.equal(button.title,description.textContent);
});

test('scenario selection only fills and focuses the message without submitting or changing context',()=>{
  const {context,el,EventSource}=fixture(), data=scenarioCatalog();
  let calls=0; context.fetch=()=>{calls++;throw new Error('Scenario click must not submit');};
  const before=EventSource.instances.length;
  el('tenant').value='northstar'; el('persona').value='user:northstar/csm';
  el('profile').value='support'; el('mode').value='handoff';
  context.switchView('inbox'); context.renderScenarios(data);
  el('scenarios').children[0].onclick();
  assert.equal(el('prompt').value,data.scenarios[0].prompt);
  assert.equal(el('prompt').focused,true);
  assert.equal(el('chat-panel').hidden,false);
  assert.equal(el('persona').value,'user:northstar/csm');
  assert.equal(el('profile').value,'support');
  assert.equal(el('mode').value,'handoff');
  assert.equal(calls,0);
  assert.equal(EventSource.instances.length,before);
  assert.equal(el('messages').children.length,0);
});

test('scenario text remains literal, including markup in tooltip and prompt',()=>{
  const {context,el}=fixture(), data=scenarioCatalog();
  Object.assign(data.scenarios[0],{name:'<img src=x>',description:'<script>not executed</script>',
    expected:'<b>literal result</b>',prompt:'<script>literal prompt</script>'});
  context.renderScenarios(data);
  const [button,description]=el('scenarios').children;
  assert.equal(button.children[0].textContent,'<img src=x>');
  assert.equal(description.children.length,0);
  assert.match(button.title,/<b>literal result<\/b>/);
  button.onclick();
  assert.equal(el('prompt').value,'<script>literal prompt</script>');
});

test('context changes clear old scenarios and ignore a stale catalog response',async()=>{
  const {context,el}=fixture(), old=scenarioCatalog(), fresh=scenarioCatalog('beacon');
  el('tenant').value='northstar'; context.renderScenarios(old);
  el('prompt').value=old.scenarios[0].prompt;
  const wait=deferred(); context.fetch=()=>wait.promise;
  const loading=context.refresh(); context.newChat();
  assert.equal(el('scenarios').children.length,0);
  assert.equal(el('prompt').value,'');
  el('tenant').value='beacon'; el('persona').value='user:beacon/csm';
  context.fetch=async()=>response(fresh); await context.refresh();
  wait.resolve(response(old)); await loading;
  assert.equal(el('scenarios-heading').textContent,'Beacon Data scenarios');
  assert.equal(el('scenarios').children[0].children[1].textContent,'Juniper Manufacturing');
});

test('skills and SQL events appear in activity without leaking SQL into chat',()=>{
  const {el,source}=fixture();
  source.dispatch('skills_discovered',{skill_ids:['sql-analysis']},1);
  source.dispatch('skill_instructions_loaded',{resource:'skill:northstar/sql-analysis',file_path:'/skills/sql-analysis/SKILL.md'},2);
  source.dispatch('sql_query_completed',{resource:'account:northstar/AC-100',row_count:2},3);
  source.dispatch('sql_query_rejected',{reason_code:'unsafe_or_invalid_sql'},4);
  source.dispatch('tool_rejected',{tool_name:'query_customer_analytics',reason_code:'unsafe_or_invalid_sql'},5);
  assert.equal(el('activity').children.length,5);
  assert.equal(el('activity').children[0].className,'event failed');
  assert.equal(el('messages').children.length,0);
});

test('mock service failure activity is visible without fabricating an assistant answer',()=>{
  const {el,source}=fixture();
  source.dispatch('service_call_started',{mock_service:true},1);
  source.dispatch('service_call_failed',{reason_code:'service_timeout',mock_service:true},2);
  source.dispatch('tool_failed',{tool_name:'get_support_sla_report',reason_code:'service_timeout'},3);
  assert.equal(el('activity').children.length,3);
  assert.equal(el('activity').children[0].className,'event failed');
  assert.equal(el('messages').children.length,0);
});

test('default skills render literal text and clear on context change',async()=>{
  const {context,el}=fixture(), data=scenarioCatalog();
  data.default_skills=[{id:'sql-analysis',name:'Scoped SQL analysis'},{id:'test',name:'<b>literal</b>'}];
  data.counts.analytics_rows=297;
  context.fetch=async()=>response(data); await context.refresh();
  assert.deepEqual(el('default-skills').children.map(item=>item.textContent),['Scoped SQL analysis','<b>literal</b>']);
  assert.equal(el('default-skills').children[1].children.length,0);
  assert.match(el('counts').textContent,/297 SQL rows/);
  context.newChat();
  assert.equal(el('default-skills').children.length,0);
});
