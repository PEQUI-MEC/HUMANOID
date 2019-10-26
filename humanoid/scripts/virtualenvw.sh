#!/bin/bash
source virtualenvwrapper.sh
workon $1
shift
python $@
deactivate
