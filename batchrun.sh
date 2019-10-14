#!/bin/bash
# example: bash batchrun.sh 424
schemecode=$1
#echo $schemecode
mergelist="Cssos_magsim_toMerge.lst"
rm $mergelist
tiles=''
for tile in 053 064 065 066 077
    do
        simcatname="Cssos_magsim_SNR_tile_"$tile"_"$schemecode".txt"
        python3 hst814simsed_phutil_mp_debug.py $tile
        ls $simcatname >> $mergelist
        tmp=$tiles
        tiles=$tmp$tile
        sleep 3
    done
mergedfile="Cssos_magsim_SNR_tile"$tiles"_"$schemecode".txt"
echo "Files to be merged:"
cat $mergelist ; echo ''
python3 SimuCataMerge.py $mergelist $mergedfile
sleep 3
python3 magsim2lephare.py $mergedfile $schemecode
sleep 3
tolephare_merged=${mergedfile%.*}"_toLephare.txt"
lephareout="OutFitCssos_tile"$tiles"_"$schemecode".out"
export LEPHAREWORK='/mnt/work/CSSOS/lephare_dev/sim2pht-Z_'$schemecode
echo $LEPHAREWORK
cp $tolephare_merged $LEPHAREWORK
cd $LEPHAREWORK
$LEPHAREDIR/source/zphota -c $LEPHAREWORK/config/cssos_zphot_cssbands.para -CAT_IN $LEPHAREWORK/$tolephare_merged -CAT_OUT $LEPHAREWORK/$lephareout -INP_TYPE M -CAT_TYPE LONG -TRANS_TYPE 1 -CAT_FMT MEME -SPEC_OUT YES -ZFIX NO
sleep 3
python3 /work/CSSOS/filter_improve/fromimg/windextract/results/zstat.py $lephareout
sleep 3
cd /work/CSSOS/filter_improve/fromimg/windextract/
mv $mergelist ./results/
mv $mergedfile ./results/
mv $tolephare_merged ./results/
mv $LEPHAREWORK/$lephareout ./results/
mv $LEPHAREWORK/${lephareout%.*}".png" ./results/

# python3 snr_check.py $mergedfile