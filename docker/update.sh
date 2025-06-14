#!/usr/bin/env bash
control_c() {
    kill $PID
    exit
}

while :
do
    python -m etc.tool.update
    sleep 600 | while read line ; do
   PID=$!
   echo $line 
done