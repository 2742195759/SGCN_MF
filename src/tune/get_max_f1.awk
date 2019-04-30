function max(a , b) 
{
    if (a < b) return b
    else return a
}

BEGIN {
    maxx = 0  
    start = 0 
}

$0 ~ /#####/ { 
    start = 1 
}
$0 !~ /#####/ { 
    if (start == 1) {
        maxx = max(maxx , $3)
    }
}

END {
    print(maxx ,  FILENAME)
}
