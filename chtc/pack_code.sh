cd ../.. # cd just outside the repo
tar --exclude="chtc" --exclude='src/results' --exclude='results_chtc' -czvf DataAugmentationForRL.tar.gz DataAugmentationForRL
cp DataAugmentationForRL.tar.gz /staging/ncorrado
