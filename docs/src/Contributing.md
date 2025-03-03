# Contributing Guide

Thank you for considering contributing to `ClimaCore.jl`! We hope this guide
helps you make a contribution.

## What to contribute?

- The easiest way to contribute is by running `ClimaCore.jl`, identifying
  problems and opening issues.

- You can tackle an existing issue. See our open [Issues](https://github.com/CliMA/ClimaCore.jl/issues). We try to keep a list of [good first issues](https://github.com/CLiMA/ClimaCore.jl/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22) accessible to new contributors.

- Write an example or tutorial.

- Improve documentation or comments if you found something hard to use.

- Implement a new feature for `ClimaCore.jl` and its users.

If you're interested in working on something, let us know by commenting on existing issues or
by opening a new issue. This is to make sure no one else is working on the same issue and so
we can help and guide you in case there is anything you need to know beforehand.

## How to contribute and bug reporting

The simplest way to contribute to `ClimaCore.jl` is to create or comment on issues, requesting something you think is missing or reporting something you think is not functioning properly.

The most useful issues or bug reports:

* Head over to the [issues](https://github.com/CLiMA/ClimaCore.jl/issues) page.

* Search to see if your issue already exists or has even been solved previously.

* If you indeed have a new issue or request, click the "New Issue" button and select the `Bug report` template.

* Provide an explicit code snippet of code that reproduces the bug in the latest tagged version of `ClimaCore.jl`. Please be as specific as possible. Include the version of the code you were using, as well as what operating system you are running. The output of Julia's `versioninfo()` and `] status` is helpful to include. Try your best to include a complete, ["minimal working example"](https://en.wikipedia.org/wiki/Minimal_working_example) that reproduces the issue. Reducing bug-producing code to a minimal example can dramatically decrease the time it takes to resolve an issue.

* Paste the _entire_ error received when running the code snippet, even if it's unbelievably long.

* Use triple backticks (e.g., ````` ```some_code; and_some_more_code;``` `````) to enclose code snippets, and other [markdown formatting syntax](https://docs.github.com/en/github/writing-on-github/getting-started-with-writing-and-formatting-on-github/basic-writing-and-formatting-syntax) to make your issue easy and quick to read.

* Report the `ClimaCore.jl` version, Julia version, machine (especially if using a GPU) and any other possibly useful details of the computational environment in which the bug was created.

Discussions are recommended for asking questions about (for example) the user interface, implementation details, science, and life in general.

## But I want to _code_!

* New users help write `ClimaCore.jl` code and documentation by [forking](https://docs.github.com/en/github/collaborating-with-pull-requests/working-with-forks) the ClimaCore.jl repository, [using git](https://guides.github.com/introduction/git-handbook/) to edit code and docs, and then creating a [pull request](https://docs.github.com/en/github/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request-from-a-fork). Pull requests are reviewed by `ClimaCore.jl` collaborators.

* A pull request can be merged once it is reviewed and approved by collaborators. If the pull request author has write access, they have the reponsibility of merging their pull request. Otherwise, `ClimaCore.jl` collabators will execute the merge with permission from the pull request author.

* Note: for small or minor changes (such as fixing a typo in documentation), the [GitHub editor](https://docs.github.com/en/github/managing-files-in-a-repository/managing-files-on-github/editing-files-in-your-repository) is super useful for forking and opening a pull request with a single click.

* Write your code with love and care. In particular, conform to existing `ClimaCore.jl` style and formatting conventions. For example, we love verbose and explicit variable names, use `TitleCase` for types, `snake_case` for objects, and always.put.spaces.after.commas. For formatting decisions we loosely follow the [YASGuide](https://github.com/jrevels/YASGuide). It's worth few extra minutes of our time to leave future generations with well-written, readable code.



### Using `git`

If you are unfamiliar with `git` and version control, the following guides
can be helpful:

- [Atlassian (bitbucket) `git`
  tutorials](https://www.atlassian.com/git/tutorials). A set of tips and tricks
  for getting started with `git`.
- [GitHub's `git` tutorials](https://try.github.io/). A set of resources from
  GitHub to learn `git`.

We provide a brief [git tutorial](https://github.com/CliMA/ClimaWorkshops/blob/main/intro-best-practices/intro-to-git.md) in the [Introduction to Best Practices](https://github.com/CliMA/ClimaWorkshops/tree/main/intro-best-practices) of our [ClimaWorkshops](https://github.com/CliMA/ClimaWorkshops) series.

### General coding guidelines
1. Keep the number of members of Julia structs small if possible (less than 8 members).
2. Code should reflect "human intuition" if possible. This mean abstraction should reflect how humans reason about the problem under consideration.
3. Code with small blast radius. If your code needs to be modified or extended, the resulting required changes should be as small and as localized as possible.
4. When you write code, write it with testing and debugging in mind.
5. Ideally, the lowest level structs have no defaults for their member fields. Nobody can remember all the defaults, so it is better to introduce them at the high-level API only.
6. Make sure that module imports are specific so that it is easy to trace back where functions that are used inside a module are coming from.
7. Consider naming abstract Julia types "AbstractMyType" in order to avoid confusion for the reader of your code.
8. Comments in your code should explain why the code exists and clarify if necessary, not just restate the line of code in words.
9. Be mindful of namespace issues when writing functional code, especially when writing function code that represents mathematical or physical concepts.
10. Condider using keywords in your structs to allow readers to more effectively reason about your code.

### Who is a "collaborator" and how can I become one?

* Collaborators have permissions to review pull requests and status allows a contributor to review pull requests in addition to opening them. Collaborators can also create branches in the main `ClimaCore.jl` repository.

* We ask that new contributors try their hand at forking `ClimaCore.jl`, and opening and merging a pull request before requesting collaborator status.

### Ground Rules

* Each pull request should consist of a logical collection of changes. You can
  include multiple bug fixes in a single pull request, but they should be related.
  For unrelated changes, please submit multiple pull requests.

* Do not commit changes to files that are irrelevant to your feature or bugfix
  (eg: `.gitignore`).

* Be willing to accept criticism and work on improving your code; we don't want
  to break other users' code, so care must be taken not to introduce bugs. We
  discuss pull requests and keep working on them until we believe we've done a
  good job.

* Be aware that the pull request review process is not immediate, and is
  generally proportional to the size of the pull request.


### Setting up your development environment

* Install [Julia](https://julialang.org/) on your system.

* Install `git` on your system if it is not already there (install XCode command line tools on
  a Mac or `git bash` on Windows).

* Login to your GitHub account and make a fork of the
  [`ClimaCore.jl` repository](https://github.com/CLiMA/ClimaCore.jl) by
  clicking the "Fork" button.

* Clone your fork of the `ClimaCore.jl` repository (in terminal on Mac/Linux or git shell/
  GUI on Windows) in the location you'd like to keep it.
  ```
  git clone https://github.com/your-user-name/ClimaCore.jl.git
  ```

* Navigate to that folder in the terminal or in Anaconda Prompt if you're on Windows.

* Connect your repository to the upstream (main project).
  ```
  git remote add `ClimaCore.jl` https://github.com/CLiMA/ClimaCore.jl.git
  ```

* Create the development environment by opening Julia via `julia --project` then
  typing in `] instantiate`. This will install all the dependencies in the Project.toml
  file.

* You can test to make sure `ClimaCore.jl` works by typing in `] test`. Doing so will run all
  the tests (and this can take a while).

Your development environment is now ready!

## Pull Requests

We follow the [ColPrac guide](https://github.com/SciML/ColPrac) for collaborative practices.
We ask that new contributors read that guide before submitting a pull request.

Changes and contributions should be made via GitHub pull requests against the `main` branch.

When you're done making changes, commit the changes you made. Chris Beams has written a
[guide](https://chris.beams.io/posts/git-commit/) on how to write good commit messages.

When you think your changes are ready to be merged into the main repository, push to your fork
and [submit a pull request](https://github.com/CLiMA/ClimaAtmos.jl/compare/).

**Working on your first Pull Request?** You can learn how from this _free_ video series
[How to Contribute to an Open Source Project on GitHub](https://egghead.io/courses/how-to-contribute-to-an-open-source-project-on-github), Aaron Meurer's [tutorial on the git workflow](https://www.asmeurer.com/git-workflow/), or the guide [“How to Contribute to Open Source"](https://opensource.guide/how-to-contribute/).


### Unit testing

Currently a number of checks are run per commit for a given PR.

- `JuliaFormatter` checks if the PR is formatted with `.dev/climaformat.jl`.
- `Documentation` rebuilds the documentation for the PR and checks if the docs
  are consistent and generate valid output.
- `Unit Tests` run subsets of the unit tests defined in `tests/`, using `Pkg.test()`.
  The tests are run in parallel to ensure that they finish in a reasonable time.
  The tests only run the latest commit for a PR, branch and will kill any stale jobs on push.
  These tests are only run on linux (Ubuntu LTS).

Unit tests are run against every new commit for a given PR,
the status of the unit-tests are not checked during the merge
process but act as a sanity check for developers and reviewers.
Depending on the content changed in the PR, some CI checks that
are not necessary will be skipped.  For example doc only changes
do not require the unit tests to be run.

### Integration testing

Currently a number of checks are run during integration testing before being
merged into `main`.

- `JuliaFormatter` checks if the PR is formatted with `.dev/climaformat.jl`.
- `Documentation` checks that the documentation correctly builds for the merged PR.
- `OS Unit Tests` checks that `ClimaCore.jl` package unit tests can pass
   on every OS supported with a pre-compiled system image (Linux, macOS, Windows).
- `ClimaCore CI` computationally expensive integration testing on CPU and
  GPU hardware using HPC cluster resources.

Integration tests are run when triggered by a reviewer through `bors`.
Integration tests are more computationally heavyweight than unit-tests and can
exercise tests using accelerator hardware (GPUs).

Currently HPC cluster integration tests are run using the [Buildkite CI service](https://buildkite.com/clima/climacore-ci).
Tests are parallelized and run as individual [Slurm](https://slurm.schedmd.com/documentation.html)
batch jobs on the HPC cluster and defined in `.buildkite/pipeline.yml`.

## Contributing to Documentation

Documentation is written in Julia-flavored markdown and generated from two sources:
```
$CLIMACORE_HOME/docs/src
```
And [Literate.jl](https://fredrikekre.github.io/Literate.jl/v2/) tutorials:
```
$CLIMACORE_HOME/tutorials
```

To locally build the documentation you need to create a new `docs` project
to build and install the documentation related dependencies:

```
cd $CLIMACORE_HOME
julia --project=docs/ -e 'using Pkg; Pkg.instantiate()'
julia --project=docs docs/make.jl
```

The makefile script will generate the appropriate markdown files and
static html from both the `docs/src` and `tutorials/` directories,
saving the output in `docs/src/generated`.

### How to generate a literate tutorial file

To create a tutorial using `ClimaCore.jl`, please use
[Literate.jl](https://github.com/fredrikekre/Literate.jl),
and consult the [Literate documentation](https://fredrikekre.github.io/Literate.jl/stable/)
for questions. For now, all literate tutorials are held in
the `tutorials` directory.

With Literate, all comments turn into markdown text and any
Julia code is read and run *as if it is in the Julia REPL*.
As a small caveat to this, you might need to suppress the
output of certain commands. For example, if you define and
run the following function

```
function f()
    return x = [i * i for i in 1:10]
end
x = f()
```

The entire list will be output, while

```
f();
```

does not (because of the `;`).

To show plots, you may do something like the following:

```
using Plots
plot(x)
```

Please consider writing the comments in your tutorial as if they are meant to
be read as an *article explaining the topic the tutorial is meant to explain.*
If there are any specific nuances to writing Literate documentation for `ClimaCore.jl`, please let us know!

## Credits

This contributor's guide is heavily based on the excellent [ClimateMachine.jl contributor's guide](https://clima.github.io/ClimateMachine.jl/latest/Contributing/) and [ClimaAtmos.jl contributor's guide](https://clima.github.io/ClimaAtmos.jl/dev/contributor_guide/), which is heavily based on the excellent [Oceananigans.jl contributor's guide](https://clima.github.io/OceananigansDocumentation/stable/contributing/) which, in turn, is heavily based on the excellent [MetPy contributor's guide](https://github.com/Unidata/MetPy/blob/master/CONTRIBUTING.md).
