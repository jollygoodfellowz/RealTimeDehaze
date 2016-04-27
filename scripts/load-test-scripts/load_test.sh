#!/bin/bash

jobs=$1

for i in `seq 1 $jobs`;
do
	echo "Starting job $i"
	./run_one_job.sh $i &
	sleep 1s
done

pids=$(ps -e | grep -i display_image | grep -oh "^[0-9]\S*")

str=""
for pid in $pids; do
	str+=$pid
	str+=","
done

pidstat -h -I -r -u -v -p "${str::-1}" 1 > mem.txt
bash tocsv.sh mem.txt > mem.csv
