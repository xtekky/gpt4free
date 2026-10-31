#!/bin/bash
trap break INT
for (( c=0; c<=1000000; c++ ))
do
echo "UPDATE: a$c"
git pull origin main
sleep 120
echo "UPDATE: b$c"
git pull origin main
sleep 120
echo "UPDATE: c$c"
git pull origin main
sleep 120
echo "UPDATE: d$c"
git pull origin main
sleep 120
echo "UPDATE: #$c"
git pull origin main
sleep 120
done
echo "STOPPED."
trap - INT
sleep 1
echo "END."