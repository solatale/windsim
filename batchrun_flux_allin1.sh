#!/bin/bash
# example: bash batchrun.sh 424
schemecode=$1
#echo $schemecode
mergelist="Cssos_FluxSim_toMerge2.lst"
if [ -e $mergelist ]
    then
        rm $mergelist
fi
tiles=''
tsleep=1
for tile in 052 053 054 064 065 066 076 077 078 #039 040 041 042 043 055 067 079 091 051 052 053 054 063 064 065 066 075 076 077 078 087 088 089 090
    do
        simcatname="Cssos_FluxSim_SNR_tile_"$tile"_allin1.txt"
        echo $tile
        python3 hst814simsed_phutil_mp_flux_allin1_debug_debkg.py $tile
        ls $simcatname >> $mergelist
        tmp=$tiles
        tiles=$tmp$tile
        sleep $tsleep
    done
mergedfile="Cssos_FluxSim_SNR_tilemrg_"$tiles"_allin1.txt"
echo "Files to be merged:"
cat $mergelist ; echo ''
python3 SimuCataMerge.py $mergelist $mergedfile
sleep $tsleep
python3 fluxsim2lephare_SNR10_allin1.py $mergedfile $schemecode
sleep $tsleep
tolephare_merged=${mergedfile%.*}"_"$schemecode"_flux_toLephare.txt"
lephareout="OutFitCssos_tilemrg_"$tiles"_"$schemecode".out"
export LEPHAREWORK='/work/CSSOS/lephare_dev/sim2pht-Z_'$schemecode
echo $LEPHAREWORK
cp $tolephare_merged $LEPHAREWORK
cd $LEPHAREWORK
$LEPHAREDIR/source/zphota -c $LEPHAREWORK/config/cssos_zphot_cssbands.para -CAT_IN $LEPHAREWORK/$tolephare_merged -CAT_OUT $LEPHAREWORK/$lephareout -INP_TYPE F -CAT_TYPE LONG -TRANS_TYPE 1 -CAT_FMT MEME -SPEC_OUT YES -ZFIX NO
sleep $tsleep
python3 /work/CSSOS/filter_improve/fromimg/windextract/results/zstat_allin1.py $lephareout
sleep $tsleep
cd /work/CSSOS/filter_improve/fromimg/windextract/
mv $LEPHAREWORK/$lephareout ./results/
mv $LEPHAREWORK/${lephareout%.*}".png" ./results/


# python3 snr_check.py $mergedfile