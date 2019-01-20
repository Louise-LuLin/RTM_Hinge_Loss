#!/bin/bash
read -p "Enter source, model, mode, coldstart, tune, minTopic, maxTopic: " source model mode cold tune min max
echo
for (( k = $min; k <= $max; k = k + 10 ))
do
	nohup ./run -flagTune $tune -varMaxIter 3 -prefix ~/lab/dataset -source $source -set byUser_20k_review -crossV 5 -nuOfTopics $k -emIter 50 -topicmodel $model -mode $mode -flagColdstart $cold > ./output/"$mode"_"$cold"_"$source"_RTM_"$model"_"$tune"_"$k".output 2>&1 &
done
