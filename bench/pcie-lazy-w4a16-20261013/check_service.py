"""Post-trial feature checks on the owned Strata service; not a speed test."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import re
import time
import urllib.request

from run_matrix import CONFIG, SCRIPT, harness


def request(port, endpoint, payload):
    req = urllib.request.Request('http://127.0.0.1:%d/v1/%s' % (port, endpoint),
        data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=90) as response:
        return json.load(response)


def main(output):
    runner = harness()
    cfg = json.loads(CONFIG.read_text())
    cfg['_file'] = str(CONFIG)
    protected = (CONFIG, SCRIPT, Path(cfg['exe']), runner.PROFILE, *runner.METADATA)
    hashes = {str(path): runner.sha(path) for path in protected}
    assert not runner.miners() and runner.api('v1/status', port=30001)['activity']['in_flight'] == 0
    owner = runner.server_owner(cfg)
    assert all(os.getpgid(pid) == int(owner.name) for pid in runner.gpu_pids())
    native = runner.native(cfg['exe'])
    assert runner.sha(native / 'exe') == hashes[cfg['exe']]
    env = dict(item.split('=', 1) for item in (native / 'environ').read_bytes().decode().split('\0') if '=' in item)
    checks = []

    def record(name, response, passed):
        checks.append(dict(name=name, passed=passed, usage=response.get('usage'),
            response_sha256=hashlib.sha256(json.dumps(response, sort_keys=True).encode()).hexdigest()))
        assert passed, name + ' failed'

    model = cfg['model_name']
    chat = dict(model=model, messages=[dict(role='user', content='Reply with exactly OK.')],
                max_tokens=64, temperature=0, reasoning_effort='none', stream=False)
    response_payload = dict(model=model, input='Reply with exactly OK.', max_output_tokens=64,
                            temperature=0, reasoning_effort='none', stream=False)
    for port in (30001, 30000):
        response = request(port, 'chat/completions', chat)
        record('chat-%d' % port, response, 'OK' in (response['choices'][0]['message'].get('content') or ''))
        response = request(port, 'responses', response_payload)
        text = ''.join(part.get('text', '') for item in response.get('output', [])
                       for part in item.get('content', []) if part.get('type') == 'output_text')
        record('responses-%d' % port, response, 'OK' in text)

    import capacity_checks
    response = request(30001, 'chat/completions', dict(chat, messages=[dict(role='user', content=[
        dict(type='text', text='Name the dominant color of this image in one word.'),
        dict(type='image_url', image_url=dict(url=capacity_checks.red_png()))])], max_tokens=128))
    record('image', response, 'red' in (response['choices'][0]['message'].get('content') or '').lower())

    response = request(30001, 'chat/completions', dict(chat,
        messages=[dict(role='user', content='Call record_number with value 7.')], max_tokens=128,
        tools=[dict(type='function', function=dict(name='record_number', description='Record a number',
            parameters=dict(type='object', properties=dict(value=dict(type='integer')), required=['value'])))],
        tool_choice=dict(type='function', function=dict(name='record_number'))))
    calls = response['choices'][0]['message'].get('tool_calls', [])
    record('tool-call', response, any(call['function']['name'] == 'record_number' and
           json.loads(call['function']['arguments']).get('value') == 7 for call in calls))

    # A distinct synthetic prefix keeps these checks outside the cold speed data.
    path = runner.INPUTS / 'inputs/cold-8192-0.json'
    prefix = json.loads(path.read_text())
    prefix.update(stream=False, max_output_tokens=64)
    prefix['input'][0]['content'] = 'PCIe-lazy-final-prefix-check. ' + prefix['input'][0]['content']
    for index in range(2):
        response = request(30001, 'responses', prefix)
        metric = runner.api('metrics', port=30001)['requests'][0]
        record('prefix-%d' % index, response, response['usage']['output_tokens'] > 0)
    assert metric['reused'] > 0, 'No prefix reuse'
    checks[-1]['reused'] = metric['reused']

    log = Path(cfg['log'])
    offset = log.stat().st_size
    concurrent = dict(prefix, max_output_tokens=256)
    payloads = []
    for index in range(2):
        payload = json.loads(json.dumps(concurrent))
        payload['input'][0]['content'] += '\nBranch %d: continue listing concise numbered facts until the token limit.' % index
        payloads.append(payload)
    peak_running = 0
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(request, 30001, 'responses', payload) for payload in payloads]
        while not all(future.done() for future in futures):
            peak_running = max(peak_running, runner.api('metrics', port=30001)['live'].get('running', 0))
            time.sleep(.1)
        for index, future in enumerate(futures):
            response = future.result()
            record('concurrent-%d' % index, response, response['usage']['output_tokens'] == 256)
    with log.open('rb') as handle:
        handle.seek(offset)
        lines = handle.read().decode(errors='replace')
    admissions = [dict(slot=int(slot), held_tokens=int(tokens)) for slot, tokens in
                  re.findall(r'strata batch: slot (\d+) takes (\d+) tokens', lines)]
    assert peak_running == 2 and len({entry['slot'] for entry in admissions}) == 2
    assert all(runner.sha(Path(path)) == digest for path, digest in hashes.items())
    assert not runner.miners() and runner.api('v1/status', port=30001)['activity']['in_flight'] == 0
    runner.save(output, dict(passed=True, time=time.time(), files=hashes,
        loaded_binary_sha256=runner.sha(native / 'exe'), native_pid=int(native.name), api_pid=int(owner.name),
        engine=runner.api('metrics', port=30001)['engine'], checks=checks,
        peak_running=peak_running, native_slot_admissions=admissions,
        env={key: value for key, value in env.items() if key.startswith('STRATA_')},
        native_args=(native / 'cmdline').read_bytes().decode().split('\0'),
        scope='Post-trial API/gateway/image/tool/prefix and short two-slot correctness checks. '
              'Not throughput measurements, a new full-context capacity campaign, or general quality evaluation.'))
    print('SERVICE_CHECKS_PASSED', len(checks), 'peak_running', peak_running, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args().output)
