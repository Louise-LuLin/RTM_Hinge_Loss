#!/bin/sh
read -p "Enter source, mode:" source mode
echo
for (( k=5; k<=50; k+5 ))
do
    echo "$k"
#	nohup ./run -prefix ~/lab/dataset -source $source -set byUser_20k_review -crossV 5 -nuOfTopics $k -emIter 50 -topicmodel RTM -mode $mode> ./output/"$mode"_"$source"_RTM_"$k".output 2>&1 &
done