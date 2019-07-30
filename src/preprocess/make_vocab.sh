for f in "$@" ;do \
    echo ${f} ; \
    cat ${f} | sed '/^$/d' | perl -pe 's/^\s+//; s/\s+\n$/\n/; s/ +/\n/g'  | \
    LC_ALL=C sort | LC_ALL=C uniq -c | LC_ALL=C sort -r -g -k1 | \
    perl -pe 's/^\s+//; ($a1,$a2)=split;
       if( $a1 >= 2 ){ $_="$a2\t$a1\n" }else{ $_="" } ' > ${f}.vocab_t3_tab ;\
done
