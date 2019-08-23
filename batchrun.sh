#!/bin/bash
# example: bash batchrun.sh 065 424
simcatname="Cssos_magsim_SNR_tile_"$1"_"$2".txt"
cat2lephare=${simcatname%.*}"_toLephare.txt"
lephareout="OutFitCssos_tile_"$1"_"$2".out"
python hst814simsed_phutil_mp.py $1
python magsim2lephare.py $simcatname
export LEPHAREWORK='/mnt/work/CSSOS/lephare_dev/sim2pht-Z_'$2
echo $LEPHAREWORK
cp $cat2lephare $LEPHAREWORK
cd $LEPHAREWORK
$LEPHAREDIR/source/zphota -c $LEPHAREWORK/config/cssos_zphot_cssbands.para -CAT_IN $LEPHAREWORK/$cat2lephare -CAT_OUT $LEPHAREWORK/$lephareout -INP_TYPE M -CAT_TYPE LONG -TRANS_TYPE 1 -CAT_FMT MEME -SPEC_OUT NO -ZFIX NO
python /work/CSSOS/filter_improve/fromimg/windextract/results/zstat.py $lephareout
cd -
