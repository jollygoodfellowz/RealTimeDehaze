#!/bin/bash


outfile="out"
outfile+=$1
outfile+=$2
outcsv=$outfile
outcsv+=".csv"
outfile+=".txt"

infile="../../dehaze/vid"
infile+=$1
infile+=".avi"

touch $outfile
echo "Starting a new job," > $outfile

time1=$(date +%s)
parallel ../../dehaze/./display_image ::: $infile >> $outfile
time2=$(date +%s)

delta=$((time2 - time1))

echo $outcsv
grep -o "[0-9].[0-9]*" $outfile > $outcsv 
echo "The total time in seconds $delta" >> $outfile
