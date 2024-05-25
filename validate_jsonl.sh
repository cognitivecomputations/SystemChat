#!/bin/bash
# Usage: ./validate_jsonl.sh yourfile.jsonl

filename="$1"
line_number=0

while IFS= read -r line
do
    line_number=$((line_number + 1))
    echo "$line" | jq . >/dev/null 2>&1
    if [ $? -ne 0 ]; then
        echo "Error in line $line_number: $line"
    fi
done < "$filename"
