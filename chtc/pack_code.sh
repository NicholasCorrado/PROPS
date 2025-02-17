cd ../.. # cd just outside the repo
tar --exclude="chtc" --exclude='src/results' --exclude='results_chtc' -czvf PROPS.tar.gz PROPS
scp PROPS.tar.gz ncorrado@ap2001.chtc.wisc.edu:/staging/ncorrado
rm PROPS.tar.gz