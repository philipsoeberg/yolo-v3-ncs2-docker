#!/bin/bash
E=$@
#setup openvino
source /openvino/bin/setupvars.sh
#set FLASK in development
export FLASK_ENV=development
# Hand off to the CMD
echo "Running [$E]"
exec $E
