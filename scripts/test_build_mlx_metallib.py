import os
from pathlib import Path
import subprocess
import tempfile
import unittest

SCRIPT = Path(__file__).with_name('build_mlx_metallib.sh').resolve()

class MetalCacheTests(unittest.TestCase):
    def test_targets_cache_and_failed_rebuild(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            kernels = root / 'build/checkouts/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/kernels'
            kernels.mkdir(parents=True)
            (kernels / 'test.metal').write_text('kernel source')
            release = root / 'build/release'
            release.mkdir()
            binary = root / 'xcrun'
            binary.write_text('''#!/usr/bin/env python3
import os, pathlib, sys
args=sys.argv[1:]
if '--show-sdk-path' in args: print('/mock/sdk'); sys.exit(0)
if '--show-sdk-version' in args: print(os.environ.get('MOCK_SDK','27')); sys.exit(0)
if '--version' in args: print(os.environ.get('MOCK_COMPILER','compiler1')); sys.exit(0)
with open(os.environ['MOCK_LOG'],'a') as log: log.write(' '.join(args)+'\\n')
if os.environ.get('MOCK_FAIL') == 'compile' and 'metal' in args: sys.exit(1)
if os.environ.get('MOCK_FAIL') == 'link' and 'metallib' in args:
    pathlib.Path(args[args.index('-o')+1]).write_text('partial library')
    sys.exit(1)
pathlib.Path(args[args.index('-o')+1]).write_text('artifact')
''')
            binary.chmod(0o755)
            log = root / 'calls'
            env = dict(os.environ, PATH=str(root)+os.pathsep+os.environ['PATH'],
                       BUILD_DIR=str(root/'build'), MOCK_LOG=str(log),
                       MACOSX_DEPLOYMENT_TARGET='15.0', MLX_METAL_LANGUAGE_VERSION='3.2',
                       MOCK_SDK='27', MOCK_COMPILER='compiler1', MOCK_FAIL='', SKIP_BUILD='0')
            def run(**changes):
                return subprocess.run(['bash',str(SCRIPT),'release'], env=dict(env,**changes),
                                      capture_output=True, text=True)
            self.assertEqual(run().returncode, 0)
            self.assertIn('-std=metal3.2 -mmacosx-version-min=15.0 -g0', log.read_text())
            baseline = log.read_text()
            self.assertIn('hash match', run().stdout)
            self.assertEqual(log.read_text(), baseline)
            for change in [{'MOCK_SDK':'28'}, {'MOCK_COMPILER':'compiler2'},
                           {'MACOSX_DEPLOYMENT_TARGET':'16.0'}, {'MLX_METAL_LANGUAGE_VERSION':'3.1'}]:
                self.assertEqual(run().returncode, 0)
                before = log.read_text()
                self.assertEqual(run(**change).returncode, 0)
                self.assertNotEqual(log.read_text(), before)
            self.assertEqual(run().returncode, 0)
            saved = (release/'mlx.metallib').read_bytes()
            saved_hash = (release/'.mlx.metallib.sha').read_bytes()
            (kernels/'test.metal').write_text('changed source')
            for failure in ['compile', 'link']:
                with self.subTest(failure=failure):
                    before = log.read_text()
                    self.assertNotEqual(run(MOCK_FAIL=failure).returncode, 0)
                    calls = log.read_text()[len(before):]
                    self.assertIn(' metal ', calls)
                    if failure == 'link':
                        self.assertIn(' metallib ', calls)
                    self.assertEqual((release/'mlx.metallib').read_bytes(), saved)
                    self.assertEqual((release/'.mlx.metallib.sha').read_bytes(), saved_hash)
            self.assertEqual(run().returncode, 0)
            self.assertNotEqual((release/'.mlx.metallib.sha').read_bytes(), saved_hash)
            self.assertIn('hash match', run().stdout)

if __name__ == '__main__': unittest.main()
