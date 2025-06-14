#!/bin/bash
trap break INT
for (( c=0; c<=1000000; c++ ))
do  
python -m etc.tool.update
sleep 600
done
echo "I have broken out of the interminably long for loop"
trap - INT
sleep 1
echo "END."