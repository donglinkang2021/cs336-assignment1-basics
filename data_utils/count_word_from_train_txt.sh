#!/bin/bash
data_file="data/owt_train.txt"
# data_file="data/TinyStoriesV2-GPT4-train.txt"
echo "Counting character occurrences in data/owt_train.txt..."
echo "File size: $(ls -lh $data_file | awk '{print $5}')"

echo "Occurrences of '鈥檚'"
count1=$(grep -o "鈥檚" $data_file | wc -l)
echo $count1

echo "Occurrences of ''s'"
count2=$(grep -o "'s" $data_file | wc -l)
echo $count2

echo "Occurrences of '’s'"
count3=$(grep -o "’s" $data_file | wc -l)
echo $count3

echo "Occurrences of '‘s'"
count4=$(grep -o "‘s" $data_file | wc -l)
echo $count4

echo ""
echo "Counting completed!"