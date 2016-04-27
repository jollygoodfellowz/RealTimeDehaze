#!/bin/bash

count=0
limit=75
touch outgpu.txt

while true; do
	count=$((count+1))
	gpu=$(python -c "print 100 - $(nvidia-smi -i 0 -q -d UTILIZATION | grep Memory | grep -o '[0-9]')")
	cmp=$(echo "$cpu>$limit" | bc)

	if [ "$cmp" -eq 1 ];
	then
		echo "Hit 75% cpu usage. Now exiting," >> outgpu.txt
		exit
	else
		run_one_job.sh $count gpu &> /dev/null &
	fi
done
