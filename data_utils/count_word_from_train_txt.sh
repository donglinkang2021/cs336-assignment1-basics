#!/bin/bash
data_file="data/owt_train.txt"
# data_file="data/TinyStoriesV2-GPT4-train.txt"
echo "Counting character occurrences in data/owt_train.txt..."
echo "File size: $(ls -lh $data_file | awk '{print $5}')"

echo "Occurrences of '鈥檚'"
echo $(grep -o "鈥檚" $data_file | wc -l)

echo "Occurrences of ''s'"
echo $(grep -o "'s" $data_file | wc -l)

echo "Occurrences of '’s'"
echo $(grep -o "’s" $data_file | wc -l)

echo "Occurrences of '‘s'"
echo $(grep -o "‘s" $data_file | wc -l)

echo ""
echo "Counting completed!"