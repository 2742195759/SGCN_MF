if (( $# != 3 ))
then 
    echo 'USAGE   : ./launcher.sh SKELETON_FILE TASK_FILE PARAMETER_NAME'
    echo 'EXAMPLE : ./launcher.sh tune/skeleton tune/task learning_rate'
    exit -1
fi

skeleton=`cat $1`
tasks=`cat $2`
target_parameter=$3

IFS=$'\n'
for task in $tasks
do 
    echo $task
    IFS=' '
    cnt=1
    parameter_name=
    for run in $task 
    do
        if (( $cnt == 1 ))
        then 
            parameter_name=$run
            (( cnt=$cnt+1 ))
        else 
            $skeleton --$parameter_name=$run --description=$parameter_name\_$run\_
        fi

        if [[ $parameter_name != $target_parameter ]] 
        then 
            break
        fi
    done
done 
