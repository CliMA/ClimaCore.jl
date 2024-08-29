#!/bin/bash

# This script disables users from pushing directly to the main branch
# Refs:
#     https://stackoverflow.com/questions/46146491/prevent-pushing-to-master-on-github
#     https://gist.github.com/vlucas/8009a5edadf8d0ff7430

# @link https://gist.github.com/mattscilipoti/8424018
#
# Called by "git push" after it has checked the remote status,
# but before anything has been pushed.
#
# If this script exits with a non-zero status nothing will be pushed.
#
# Steps to install, from the root directory of your repo...
# 1. Copy the file into your repo at `.git/hooks/pre-push`
# 2. Set executable permissions, run `chmod +x .git/hooks/pre-push`
# 3. Or, use `rake hooks:pre_push` to install
#
# Try a push to master, you should get a message `*** [Policy] Never push code directly to...`
#
# The commands below will not be allowed...
# `git push origin main`
# `git push --force origin main`
# `git push --delete origin main`


protected_branch='main'

policy="\n\n[Policy] Never push code directly to the "$protected_branch" branch! (Prevented with pre-push hook.)\n\n See .dev/hooks/pre-push.sh for settings.\n\n"

current_branch=$(git symbolic-ref HEAD | sed -e 's,.*/\(.*\),\1,')

push_command=$(ps -ocommand= -p $PPID)

is_destructive='force|delete|\-f'

will_remove_protected_branch=':'$protected_branch

do_exit(){
  echo -e $policy
  exit 1
}

if [[ $push_command =~ $is_destructive ]] && [ $current_branch = $protected_branch ]; then
  do_exit
fi

if [[ $push_command =~ $is_destructive ]] && [[ $push_command =~ $protected_branch ]]; then
  do_exit
fi

if [[ $push_command =~ $will_remove_protected_branch ]]; then
  do_exit
fi

# Prevent ALL pushes to protected_branch
if [[ $push_command =~ $protected_branch ]] || [ $current_branch = $protected_branch ]; then
  do_exit
fi

unset do_exit

exit 0
