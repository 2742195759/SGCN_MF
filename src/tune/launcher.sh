#!/bin/bash
if (( $# != 3 ))
then 
    echo 'USAGE   : ./launcher.sh SKELETON_FILE TASK_FILE PARAMETER_FILE'
    echo 'EXAMPLE : ./launcher.sh tune/skeleton tune/task ./parser.py   '
    exit -1
fi

skeleton=`cat $1`
tasks=`cat $2`
parameter_default_file=$3

IFS=$'\n'
for task in $tasks
do 
    echo $task
    IFS=' '
    cnt=1
    parameter_name=
    optimal_result=0
    optimal_val=0
    for run in $task 
    do
        if (( $cnt == 1 )) 
        then 
            parameter_name=$run
            (( cnt=$cnt+1 ))
        else 
            echo "[TUNE] start task : $parameter_name = $run"
            output=$( $skeleton --$parameter_name=$run --description=$parameter_name\_$run\_ | grep -F '[XKLOG]' )
            echo $output
            maxval=$(echo "$output" | awk -f ./tune/get_max_f1_from_output.awk)
            echo $maxval
            if [ $(echo "$maxval >= $optimal_result" | bc) == 1 ] ; 
            then 
                echo change
                optimal_val=$run
                optimal_result=$maxval
            fi
        fi
    done

    echo "[TUNE] : $parameter_name > $optimal_val"     >>  ./log/tune_log
    echo "         ACCURATE        > $optimal_result"  >>  ./log/tune_log
    echo ""  >>  ./log/tune_log

    ## change the default parameter
    newfile=$(awk -f ./tune/sub_default_parameter.awk $parameter_default_file -v key=$parameter_name -v val=$optimal_val)
    echo "$newfile" > $parameter_default_file
done 
