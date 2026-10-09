// Exercise the actual privileged workflow guard without network access or tokens.
const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const workflow = fs.readFileSync('.github/workflows/dependabot-automerge.yml', 'utf8');
const script = workflow.split('          script: |\n')[1].split('\n      - name:')[0]
  .split('\n').map(line => line.replace(/^ {12}/, '')).join('\n');
const AsyncFunction = Object.getPrototypeOf(async function(){}).constructor;
const verify = new AsyncFunction('context', 'github', 'core', 'require', 'process', script);
const bot = {login: 'dependabot[bot]', id: 49699333};
const sha = 'a'.repeat(40);
const names = ['Lint EffektGuard', 'Validate EffektGuard', 'Validate Automation'];
async function scenario(change = () => {}) {
  const pr = {number: 42, user: bot, state: 'open', draft: false, commits: 1,
    base: {ref: 'main'}, head: {sha, repo: {full_name: 'owner/repo'}}, html_url: 'https://github.com/owner/repo/pull/42'};
  const run = {id: 1, actor: bot, triggering_actor: bot, event: 'pull_request',
    conclusion: 'success', head_sha: sha, head_repository: {full_name: 'owner/repo'}};
  const commits = [{sha, author: bot, commit: {verification: {verified: true, reason: 'valid'}}}];
  const jobs = names.map(name => ({name, conclusion: 'success'}));
  const data = {pr, run, commits, jobs, actor: bot.login};
  change(data);
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), 'approval-test-'));
  const output = {};
  const api = {
    repos: {get: async () => ({data: {default_branch: 'main', full_name: 'owner/repo'}}), listPullRequestsAssociatedWithCommit: 'prs'},
    pulls: {get: async () => ({data: pr}), listCommits: 'commits'},
    actions: {listJobsForWorkflowRun: 'jobs'}
  };
  try {
    await verify({payload: {workflow_run: run}, actor: data.actor, repo: {owner: 'owner', repo: 'repo'}},
      {rest: api, paginate: async key => ({prs: [pr], commits, jobs})[key]},
      {setOutput: (key, value) => {output[key] = value;}}, require, {env: {RUNNER_TEMP: directory}});
    assert.equal(output['head-sha'], sha);
    assert.equal(JSON.parse(fs.readFileSync(output['event-path'])).pull_request.number, 42);
  } finally {
    fs.rmSync(directory, {recursive: true, force: true});
  }
}
test('verified bot update with all validation jobs is eligible for metadata', () => scenario());
for (const [name, change] of [
  ['human triggering actor', d => {d.run.triggering_actor = {login: 'owner', id: 1};}],
  ['human workflow actor', d => {d.actor = 'owner';}],
  ['spoofed Dependabot identity', d => {d.pr.user = {...bot, id: 1};}],
  ['foreign repository', d => {d.run.head_repository.full_name = 'attacker/repo';}],
  ['ordinary push workflow', d => {d.run.event = 'push';}],
  ['merge-group workflow', d => {d.run.event = 'merge_group';}],
  ['invalid signature reason', d => {d.commits[0].commit.verification.reason = 'invalid';}],
  ['failed workflow', d => {d.run.conclusion = 'failure';}],
  ['failed lint', d => {d.jobs[0].conclusion = 'failure';}],
  ['skipped validation', d => {d.jobs[1].conclusion = 'skipped';}],
  ['missing required job', d => {d.jobs.pop();}],
  ['duplicate job name', d => {d.jobs.push(d.jobs[0]);}],
  ['stale validated SHA', d => {d.run.head_sha = 'b'.repeat(40);}],
  ['non-default target', d => {d.pr.base.ref = 'release';}],
  ['closed PR', d => {d.pr.state = 'closed';}],
  ['draft PR', d => {d.pr.draft = true;}],
  ['unverified commit', d => {d.commits[0].commit.verification.verified = false;}],
  ['extra human commit after signed first commit', d => {
    d.commits.push({...d.commits[0], author: {login: 'owner', id: 1}});
    d.commits[0].sha = 'b'.repeat(40); d.pr.commits = 2;
  }],
  ['truncated commit response', d => {d.pr.commits = 2;}],
]) test(`rejects ${name}`, () => assert.rejects(scenario(change)));
// Evaluate the actual metadata gate so major/unknown updates cannot reach the shell.
const gate = workflow.split('      - name: Approve and enable squash auto-merge\n')[1]
  .split('        env:')[0].replace('        if: >-\n', '').trim();
const evaluate = new Function('steps', `return ${gate.replaceAll('.update-type', "['update-type']")}`);
for (const type of ['version-update:semver-major', '', 'unknown', undefined]) {
  test(`metadata denies ${String(type)}`, () => assert.equal(evaluate({metadata: {outputs: {'update-type': type}}}), false));
}
for (const type of ['version-update:semver-minor', 'version-update:semver-patch']) {
  test(`metadata permits ${type}`, () => assert.equal(evaluate({metadata: {outputs: {'update-type': type}}}), true));
}
