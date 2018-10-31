#!/bin/bash
read -p "Enter source, mode, coldstart, minTopic, maxTopic: " source mode cold min max
echo
for (( k = $min; k <= $max; k = k + 5 ))
do
	nohup ./run -prefix ~/lab/dataset -source $source -set byUser_20k_review -crossV 5 -nuOfTopics $k -emIter 50 -topicmodel RTM -mode $mode -flagColdstart $cold > ./output/"$mode"_"$cold"_"$source"_RTM_"$k".output 2>&1 &
done