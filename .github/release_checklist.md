### Release checklist

This PR changes the version of ClimaCore. Before merging:

- [ ] Check the [nightly AMIP run][amip] with the `main` branch of ClimaCore and other relevant packages.
- [ ] Check that [ClimaAtmos][atmos], [ClimaLand][land], and [ClimaCoupler][coupler] CI pass with the `main` branch of ClimaCore, and ideally open PRs that fix any issues beforehand.
- [ ] Check for unexpected performance and compilation regressions.
- [ ] Check that `NEWS.md` is updated with all relevant changes for users.

All downstream tests must be run with the `main` branch of ClimaCore.jl. This includes the tests in each package's `test` directory and everything exercised by Buildkite CI.

[amip]: https://buildkite.com/clima/climacoupler-coarse-nightly-amip
[atmos]: https://buildkite.com/clima/climaatmos-ci
[land]: https://buildkite.com/clima/climaland-ci
[coupler]: https://buildkite.com/clima/climacoupler-ci
