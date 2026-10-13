"""Rebuild only changed objects; retain the configured W4A16 binary's linked libraries."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path('/home/tc/Strata-nvfp4')
SEED = ROOT / 'bench/prefill-residency-p0-p3-20261012/candidate-b8'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--job', required=True)
    parser.add_argument('--prefill', required=True)
    parser.add_argument('--fence', required=True)
    parser.add_argument('--transport-source', help='Frozen source bundle for raw PLE transport/cache')
    args = parser.parse_args()
    job = Path(args.job).resolve()
    assert job.is_relative_to(ROOT / 'bench/ple-upload-w4a16-20261013')
    job.mkdir(exist_ok=False)
    seed = json.loads((SEED / 'build.json').read_text())
    expected = '1b827daf441e701d17b12c77a67cee9b2ead8015288d1392e191f063aa469c39'
    assert sha(SEED / 'strata') == seed['link']['binary_sha256'] == expected
    assert sha(ROOT / 'engine/strata-nvfp4-w4a16-r256') == expected
    command = seed['link']['command']
    inputs = {}
    for item in command:
        if item.endswith(('.a', '.o', '.so')):
            path = Path(item) if Path(item).is_absolute() else ROOT / 'build' / item
            inputs[str(path)] = sha(path)
    source = job / 'source'
    shutil.copytree(Path(seed['source']), source)
    for item in seed['objects']:
        assert sha(source / item['source']) == item['sha256']
    for name, digest in seed['header_sha256'].items():
        assert sha(source / name) == digest
    frozen = {'seed_build_sha256': sha(SEED / 'build.json'), 'seed_binary_sha256': expected,
              'link_inputs': inputs, 'source_origin': seed['source']}
    (job / 'baseline-provenance.json').write_text(json.dumps(frozen, indent=2) + '\n')
    # Relinking original objects must reproduce the configured production executable.
    anchor = list(command)
    anchor[anchor.index('-o') + 1] = str(job / 'baseline-relinked')
    subprocess.run(anchor, cwd=ROOT / 'build', check=True)
    assert sha(job / 'baseline-relinked') == expected, 'Linked dependencies changed since the configured build'
    prefill = source / 'src/prefill/prefill.cpp'
    prefill.write_bytes(Path(args.prefill).read_bytes())
    fence = source / 'include/strata/prefill/ple_upload.hpp'
    fence.write_bytes(Path(args.fence).read_bytes())
    transport_names = ('include/strata/kernels/ngram.hpp', 'include/strata/prefill/kernels.hpp',
                       'include/strata/ngram/raw_row_cache.hpp', 'src/kernels/ngram.cpp', 'src/prefill/ple_transport.cu')
    if args.transport_source:
        for name in transport_names:
            target = source / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((Path(args.transport_source) / name).read_bytes())
    build = job / 'build'
    build.mkdir()
    obj = build / 'prefill.cpp.o'
    original = next(item for item in seed['objects'] if item['source'] == 'src/prefill/prefill.cpp')
    compile_cmd = [item.replace(str(SEED), str(job)).replace(str(job / 'source'), str(source))
                   for item in original['command']]
    # The seed object paths are directly under its directory; keep new outputs in build/.
    for flag, value in [('-o', str(obj)), ('-MT', str(obj)), ('-MF', str(obj) + '.d')]:
        compile_cmd[compile_cmd.index(flag) + 1] = value
    subprocess.run(compile_cmd, cwd=ROOT / 'build', check=True)
    archive = build / 'libstrata_prefill.a'
    shutil.copy2(SEED / 'libstrata_prefill.a', archive)
    assert sha(archive) == seed['prefill_archive_sha256']
    subprocess.run(['ar', 'r', str(archive), str(obj)], check=True)
    objects = [{'source': 'src/prefill/prefill.cpp', 'sha256': sha(prefill),
                'object_sha256': sha(obj), 'command': compile_cmd}]
    headers = {'include/strata/prefill/ple_upload.hpp': sha(fence)}
    kernel_seed = ROOT / 'bench/upstream-ports-20261011/candidate/libstrata_kernels.a'
    kernel_archive = None
    if args.transport_source:
        headers.update({name: sha(source / name) for name in transport_names if name.startswith('include/')})
        ngram_obj = build / 'ngram.cpp.o'
        ngram_cmd = ['/usr/bin/c++', '-I' + str(source / 'include'), '-I' + str(ROOT / 'include'),
                     '-I' + str(ROOT / 'third_party/ggml'), '-isystem',
                     '/usr/local/cuda-13.2/targets/x86_64-linux/include', '-O3', '-DNDEBUG',
                     '-std=c++20', '-Wall', '-Wextra', '-o', str(ngram_obj), '-c', str(source / 'src/kernels/ngram.cpp')]
        subprocess.run(ngram_cmd, check=True)
        kernel_archive = build / 'libstrata_kernels.a'
        shutil.copy2(kernel_seed, kernel_archive)
        assert sha(kernel_archive) == inputs[str(kernel_seed)]
        subprocess.run(['ar', 'r', str(kernel_archive), str(ngram_obj)], check=True)
        transport_obj = build / 'ple_transport.cu.o'
        transport_cmd = ['/usr/local/cuda-13.2/bin/nvcc', '-I' + str(source / 'include'), '-O3',
                         '-std=c++20', '-arch=sm_89', '--ftz=false', '-o', str(transport_obj),
                         '-c', str(source / 'src/prefill/ple_transport.cu')]
        subprocess.run(transport_cmd, check=True)
        subprocess.run(['ar', 'r', str(archive), str(transport_obj)], check=True)
        for name, output, cmd in [('src/kernels/ngram.cpp', ngram_obj, ngram_cmd),
                                  ('src/prefill/ple_transport.cu', transport_obj, transport_cmd)]:
            objects.append(dict(source=name, sha256=sha(source / name), object_sha256=sha(output), command=cmd))
    link = list(command)
    link[link.index('-o') + 1] = str(build / 'strata')
    link = [str(archive) if item == str(SEED / 'libstrata_prefill.a') else item for item in link]
    if kernel_archive:
        link = [str(kernel_archive) if item == str(kernel_seed) else item for item in link]
    for path, digest in inputs.items():
        assert sha(path) == digest, 'Retained link input changed: ' + path
    subprocess.run(link, cwd=ROOT / 'build', check=True)
    result = {'source': str(source), 'objects': objects,
              'header_sha256': headers,
              'link': {'binary_sha256': sha(build / 'strata'), 'command': link,
                       'changed_archives_sha256': {str(p): sha(p) for p in (archive, kernel_archive) if p}},
              'retained_inputs': frozen, 'compiler': subprocess.check_output(['c++', '--version'], text=True).splitlines()[0]}
    (build / 'build.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'built': str(build / 'strata'), 'sha256': result['link']['binary_sha256'],
                      'baseline_relink_matches_production': True}), flush=True)


if __name__ == '__main__':
    main()
