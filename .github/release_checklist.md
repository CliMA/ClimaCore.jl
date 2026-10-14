### Release checklist

This PR changes the version of ClimaCore. Before merging:

- [ ] Check the [nightly AMIP run][amip] with the `main` branch of ClimaCore and other relevant packages.
- [ ] Check that ClimaAtmos, ClimaLand, and ClimaCoupler CI pass with the `main` branch of ClimaCore, and ideally open PRs that fix any issues beforehand.
- [ ] Check for unexpected performance and compilation regressions.
- [ ] Check that `NEWS.md` is updated with all relevant changes for users.

[amip]: https://buildkite.com/clima/climacoupler-coarse-nightly-amip
