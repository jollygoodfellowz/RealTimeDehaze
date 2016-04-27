#!/bin/bash

file=$1
sed '1d;/^[#]/d;/^$/d;s/^[ ]*//;s/[ ]\+/,/g' $1 > mem.csv
