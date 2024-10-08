#!/bin/bash

# Syncs current folder with the folders on Ada. This makes it easier
# to develop, since both share a single source of truth.

# Use -s to ssh directly, instead of having a separate process to ssh

# NOTE: Make sure to modify this as per your convienience
# In particular, change the path of the files, as well as the shell.
# This file is written keeping my preferences in mind.

REMOTE_PATH="anlp/project"
USERNAME="monish"

# Ensures that all files, excluding the venv and artifacts, are synced
# with Ada.
rsync -chavzP --stats ./ ada:/home2/$USERNAME/$REMOTE_PATH --exclude "venv" --exclude "artifacts" --exclude '.git' --exclude '__pycache__'

case "$1" in
    -s) ssh -t ada 'cd ~/anlp/$REMOTE_PATH; zsh -l'
esac
