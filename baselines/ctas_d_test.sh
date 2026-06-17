#!/bin/bash
# Run CTAS-D solver on all test set configs (up to 4 parallel jobs).
#
# Usage: bash baselines/ctas_d_test.sh

set -o monitor

TEST_SET="data/testSet_20A_50T_CONDET"
todo_array=($(find "./${TEST_SET}/configs" -wholename "*/planner_param.yaml"))

index=0
max_jobs=4

function add_next_job {
    if [[ $index -lt ${#todo_array[*]} ]]; then
        echo "adding job ${todo_array[$index]}"
        do_job "${todo_array[$index]}" &
        index=$(($index + 1))
    fi
}

function do_job {
    echo "Processing $1 file..."
    results="$(echo $1 | sed -e 's/planner_param/results/')"
    ./baselines/CTAS-D/build/main "$1" "$results"
    sleep 2
}

trap add_next_job CHLD

# Add initial set of jobs
while [[ $index -lt $max_jobs ]]; do
    add_next_job
done

# Wait for all jobs to complete
wait
echo "done"
