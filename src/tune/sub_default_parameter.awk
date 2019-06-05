BEGIN {
    start=0
    found_line=0
    template = sprintf("parser\\.add_argument\\(\\\"\\-\\-%s\\\"," , key)
}

$0 ~ template{
    start = 1
}

$0 ~ /default ?= ?.*,/{
    if ( start == 1 ) {
        gsub("default *= *.*," , sprintf("default = \"%s\"," , val))
        start = 0
    }
}

{
    print $0
}
