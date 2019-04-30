if (( $# != 1 )) 
then 
    exit -1
fi 

file=$1
awk -f ./tune/get_max_f1.awk $file
