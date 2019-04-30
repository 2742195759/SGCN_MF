function max(a , b) 
{
    if (a < b) return b
    else return a
}

BEGIN {
    maxx = 0 
}

{ 
    maxx = max(maxx , $4)
}

END {
    print(maxx)
}
