#!/usr/bin/env bash
control_c() {
    kill $PID
    exit
}

trap control_c SIGINT

while true ; do 
   python -m etc.tool.update
   sleep 600 | while read line ; do
   PID=$!
   echo $line 
done