#!/bin/bash
read -p "Enter source, mode:" source mode
echo
for k in {5..50..5}
do
	nohup ./run -prefix ~/lab/dataset -source $source -set byUser_20k_review -crossV 5 -nuOfTopics $k -emIter 50 -topicmodel RTM -mode $mode> ./output/"$mode"_"$source"_RTM_"$k".output 2>&1 &
done