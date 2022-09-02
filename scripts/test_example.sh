REPO_FOLDER="path_to_repository_dir"
gpu="0"

prefix_name="AS-D2-0-1"
experiment="attention_full_0_1"
dataset="$REPO_FOLDER/data/ukbb/list_testing_ids.csv"
label="label_test_sample"
bash "${REPO_FOLDER}/scripts/run_test.sh" \
    $REPO_FOLDER $prefix_name $experiment $gpu $dataset $label
