# Building the Metal shader library

`make build` builds the release Swift package and its MLX Metal shader library.
`make debug` does the same for the debug configuration. After a manual Swift
build, run `./scripts/build_mlx_metallib.sh release` (or `debug`). The script
writes `mlx.metallib` next to the executable and copies it into existing test
bundles.

The script defaults to macOS 15.0 and Metal language version 3.2, matching the
package’s minimum macOS version even when a newer SDK is installed. It removes
shader debug information with `-g0` while retaining runtime reflection.

For a custom build, `MACOSX_DEPLOYMENT_TARGET` overrides the minimum macOS
version and `MLX_METAL_LANGUAGE_VERSION` overrides the Metal language version.
Use a target and language version supported by the installed compiler and the
systems where the library will run. Raising the deployment target can make the
library incompatible with macOS 15; these overrides affect this script, not the
package’s declared platform requirements.

The cache includes the build configuration, compiler flags, SDK path/version,
Metal compiler version, build script, and kernel source/header paths and contents.
Unchanged inputs skip compilation; `--force` forces a rebuild. Compilation or
linking failure leaves the previous library and cache hash in place and returns
a nonzero exit status. A successful retry updates the cache.

Run the no-download regression test with:

```sh
python3 scripts/test_build_mlx_metallib.py
```

CI runs this test before building Swift. It checks target flags, cache hits and
invalidation, preservation after compilation and partial-link failures, and
recovery on the next successful build.
