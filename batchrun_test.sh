#!/bin/bash
# example: bash batchrun.sh 424
schemecode=$1
#echo $schemecode
mergelist="Cssos_magsim_toMerge.lst"
rm $mergelist
tiles=''
for tile in 053 #053 064 065 066 077
    do
        simcatname="Cssos_magsim_SNR_tile_"$tile"_"$schemecode".txt"
        python3 hst814simsed_phutil_mp_debug.py $tile
        ls $simcatname >> $mergelist
        tmp=$tiles
        tiles=$tmp$tile
    done
mergedfile="Cssos_magsim_SNR_tile"$tiles"_"$schemecode".txt"
python3 SimuCataMerge.py $mergelist $mergedfile
python3 magsim2lephare.py $mergedfile $schemecode
tolephare_merged=${mergedfile%.*}"_toLephare.txt"
lephareout="OutFitCssos_tile"$tiles"_"$schemecode".out"
export LEPHAREWORK='/mnt/work/CSSOS/lephare_dev/sim2pht-Z_'$schemecode
echo $LEPHAREWORK
cp $tolephare_merged $LEPHAREWORK
cd $LEPHAREWORK
$LEPHAREDIR/source/zphota -c $LEPHAREWORK/config/cssos_zphot_cssbands.para -CAT_IN $LEPHAREWORK/$tolephare_merged -CAT_OUT $LEPHAREWORK/$lephareout -INP_TYPE M -CAT_TYPE LONG -TRANS_TYPE 1 -CAT_FMT MEME -SPEC_OUT YES -ZFIX NO
python3 /work/CSSOS/filter_improve/fromimg/windextract/results/zstat.py $lephareout
cd /work/CSSOS/filter_improve/fromimg/windextract/
mv $LEPHAREWORK/$tolephare_merged ./results/
mv $LEPHAREWORK/$lephareout ./results/
mv $LEPHAREWORK/${lephareout%.*}".png" ./results/

# python3 snr_check.py $mergedfile