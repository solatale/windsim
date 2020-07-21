#!/bin/bash
# example: bash batchrun.sh 424 424uBgN 222 222uBgN 4262uBgN

echo -e "\n""Generating mock data catalog..."

mergelist="Cssos_FluxSim_toMerge.lst"
if [ -e $mergelist ]
    then
        rm $mergelist
fi
# if [ -e "csscat_merge.txt" ]
#     then
#         rm "csscat_merge.txt"
# fi

tiles=''
tsleep=1
for tile in 039 040 041 042 043 051 052 053 054 055 063 064 065 066 067 075 #076 077 078 079 087 088 089 090 091
    do
        simcatname="Cssos_FluxSim_SNR_tile_"$tile"_allin1_uBgN.txt"
        echo 'Tile: '$tile
        python3 hst814simsed_phutil_mp_flux_allin1_uBgN.py $tile
        # cat "csscat_"$tile".txt" >> csscat_merge.txt
        ls $simcatname >> $mergelist
        tmp=$tiles
        tiles=$tmp$tile
        # sleep $tsleep
    done
mergedfile="Cssos_FluxSim_SNR_tilemrg_"$tiles"_allin1.txt"
echo "Files to be merged:"
cat $mergelist ; echo ''
python3 SimuCataMerge.py $mergelist $mergedfile
sleep $tsleep

nparam=$#
for i in `seq $nparam`
do
    senariocode=${!i}
    cd /work/CSSOS/filter_improve/fromimg/windextract/
    echo -e "\nSenario "$senariocode"\n"
    python3 fluxsim2lephare_SNRcomb_allin1_uBgN.py $mergedfile $senariocode
    sleep $tsleep
    tolephare_merged=${mergedfile%.*}"_"$senariocode"_flux_toLephare.txt"
    # lephareout="OutFitCssos_tilemrg_"$tiles"_"$senariocode".out"
    lephareout=${mergedfile%.*}"_"$senariocode"_flux_OutFit.out"
    export LEPHAREWORK='/work/CSSOS/lephare_dev/sim2pht-Z_allin1'
    echo "LePhare Work in "$LEPHAREWORK
    cp $tolephare_merged $LEPHAREWORK
    cd $LEPHAREWORK
    echo ""
    $LEPHAREDIR/source/zphota -c $LEPHAREWORK/config/cssos_zphot_cssbands.para -CAT_IN $LEPHAREWORK/$tolephare_merged -CAT_OUT $LEPHAREWORK/$lephareout -INP_TYPE F -CAT_FMT MEME -CAT_TYPE LONG -TRANS_TYPE 0 -SPEC_OUT NO -ZFIX NO -Z_INTERP YES
    sleep $tsleep
    cd /work/CSSOS/filter_improve/fromimg/windextract/results/
    mv $LEPHAREWORK/$lephareout ./
    mv $LEPHAREWORK/$tolephare_merged ./
    # python3 zstat_lephout_uBgN.py $lephareout $senariocode r20 r10 ri10 griz10
    #fim ${lephareout%.*}".png" &
    sleep $tsleep
done


# python3 snr_check.py $mergedfile
