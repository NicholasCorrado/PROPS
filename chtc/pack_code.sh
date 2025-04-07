cd ../.. # cd just outside the repo
tar --exclude="chtc" --exclude='local' --exclude='results' --exclude='.git'  -czvf PROPS.tar.gz PROPS
scp PROPS.tar.gz ncorrado@ap2001.chtc.wisc.edu:/staging/ncorrado
rm PROPS.tar.gz