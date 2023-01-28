in="data/cleaned.jsonl"
train="data/cleaned_train.jsonl"
test="data/cleaned_test.jsonl"

awk -v train="$train" -v test="$test" '{if(rand()<0.7) {print > train} else {print > test}}' $in