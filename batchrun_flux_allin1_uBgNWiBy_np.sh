#!/bin/bash
# example: bash batchrun.sh 424 424uBgN 222 222uBgN 4262uBgN

date "+%Y-%m-%dT%H:%M:%S"

# echo -e "\n""Get filters ready."
# cd /work/CSSOS/filter_improve/fromimg/windextract/filters
# python3 filterthrput.py filter_imaging.list
# cd /work/CSSOS/filter_improve/fromimg/windextract

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
N_Lephare=6
datetag="20200906"
#datetag=`date +%Y%m%d`
datetagmd=`date +%m%d`
export LEPHAREWORK='/work/CSSOS/lephare_dev/sim2phtz_allin1_ext'
export baseworkdir='/work/CSSOS/filter_improve/fromimg/windextract'
export resultdir=$baseworkdir'/result0906'
echo "LePhare Work in "$LEPHAREWORK

tiles=''
tsleep=1
for tile in 039 040 041 042 043 051 052 053 054 055 063 064 065 066 067 075 #076 077 078 079 087 088 089 090 091
    do
        simcatname="Cssos_FluxSim_SNR_tile_"$tile"_allin1_uBgNWiBy.txt"
        echo 'Tile: '$tile
        # python3 hst814simsed_phutil_mp_flux_allin1_uBgNWiBy.py $tile
        ls $simcatname >> $mergelist
        tmp=$tiles
        tiles=$tmp$tile
        # sleep $tsleep
    done
mergedfile="Cssos_FluxSim_SNR_tilemrg_"$tiles"_allin1_uBgNWiBy_"$datetag".txt"
echo "Files to be merged:"
cat $mergelist ; echo ''
# python3 SimuCataMerge.py $mergelist $mergedfile
sleep $tsleep

nparam=$#
for i in `seq $nparam`
do
    senariocode=${!i}
    echo -e "\nSenario "$senariocode"\n"
    cd $baseworkdir
    python3 fluxsim2lephare_SNRcomb_allin1_uBgNWiByUV_np.py $mergedfile $senariocode $N_Lephare
    sleep $tsleep

    lephoutroot=${mergedfile%.*}"_"$senariocode"_OutFit"
    for ile in `seq $N_Lephare`
    do
    {
        tolephare_merged=${mergedfile%.*}"_"$senariocode"_toLephare_"$ile".txt"
        lephareout=$lephoutroot"_"$ile".out"
        cp $tolephare_merged $LEPHAREWORK
        cd $LEPHAREWORK
        $LEPHAREDIR/source/zphota -c $LEPHAREWORK/config/cssos_zphot_cssbands.para -CAT_IN $LEPHAREWORK/$tolephare_merged -CAT_OUT $LEPHAREWORK/$lephareout -INP_TYPE F -CAT_FMT MEME -CAT_TYPE LONG -TRANS_TYPE 0 -SPEC_OUT NO -ZFIX NO -Z_INTERP YES > /tmp/"lepharefit_"$ile".log"
        cd $resultdir
        cp $baseworkdir/$mergedfile ./
        mv $LEPHAREWORK/$tolephare_merged ./
        mv $LEPHAREWORK/$lephareout ./

    }&
    done
    wait
    # sleep $tsleep
    echo -e "\nFitting cycle finished."
    cd $resultdir
    python3 $baseworkdir/zstat_lephout_uBgNWiBy_np.py $N_Lephare $lephoutroot $senariocode i20 i10 gi10 griz10
    # sleep $tsleep
done
date "+%Y-%m-%dT%H:%M:%S"
