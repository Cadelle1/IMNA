for d in 0.2
do
	PD=graph_data/douban
	TRAINRATIO=${d}
	TRAIN=${PD}/dictionaries/node,split=${TRAINRATIO}.train.dict
  TEST=${PD}/dictionaries/node,split=${TRAINRATIO}.test.dict

	python -u network_alignment.py \
    --source_dataset ${PD}/online/graphsage/ \
    --target_dataset ${PD}/offline/graphsage/ \
    --groundtruth ${PD}/dictionaries/node,split=${TRAINRATIO}.test.dict \
	NAME \
	--train_dict ${TRAIN} \
	--log \
	--cuda
done

#for d in 0.2
#do
#	PD=graph_data/allmv_tmdb
#	TRAINRATIO=${d}
#	TRAIN=${PD}/dictionaries/node,split=${TRAINRATIO}.train.dict
#    TEST=${PD}/dictionaries/node,split=${TRAINRATIO}.test.dict
#
#	python -u network_alignment.py \
#    --source_dataset ${PD}/allmv/graphsage/ \
#    --target_dataset ${PD}/tmdb/graphsage/ \
#    --groundtruth ${PD}/dictionaries/node,split=${TRAINRATIO}.test.dict \
#	NAME \
#	--train_dict ${TRAIN} \
#	--log \
#	--cuda
#done

#for d in 0.2
#do
#	PD=graph_data/fq-tw-data
#	TRAINRATIO=${d}
#	TRAIN=${PD}/dictionaries/node,split=${TRAINRATIO}.train.dict
#    TEST=${PD}/dictionaries/node,split=${TRAINRATIO}.test.dict
#
#	python -u network_alignment.py \
#    --source_dataset ${PD}/foursquare/graphsage/ \
#    --target_dataset ${PD}/twitter/graphsage/ \
#    --groundtruth ${PD}/dictionaries/node,split=${TRAINRATIO}.test.dict \
#	NAME \
#	--train_dict ${TRAIN} \
#	--log \
#	--cuda
#done

#for d in 0.2
#do
#	PD=graph_data/fb-tw-data
#	TRAINRATIO=${d}
#	TRAIN=${PD}/dictionaries/node,split=${TRAINRATIO}.train.dict
#    TEST=${PD}/dictionaries/node,split=${TRAINRATIO}.test.dict
#
#	python -u network_alignment.py \
#    --source_dataset ${PD}/facebook/graphsage/ \
#    --target_dataset ${PD}/twitter/graphsage/ \
#    --groundtruth ${PD}/dictionaries/node,split=${TRAINRATIO}.test.dict \
#	NAME \
#	--train_dict ${TRAIN} \
#	--log \
#	--cuda
#done

#for d in 0.2
#do
#	PD=graph_data/flickr_lastfm
#	TRAINRATIO=${d}
#	TRAIN=${PD}/dictionaries/node,split=${TRAINRATIO}.train.dict
#    TEST=${PD}/dictionaries/node,split=${TRAINRATIO}.test.dict
#
#	python -u network_alignment.py \
#    --source_dataset ${PD}/flickr/graphsage/ \
#    --target_dataset ${PD}/lastfm/graphsage/ \
#    --groundtruth ${PD}/dictionaries/node,split=${TRAINRATIO}.test.dict \
#	NAME \
#	--train_dict ${TRAIN} \
#	--log \
#	--cuda
#done

#for d in 0.2
#do
#	PD=graph_data/flickr_myspace
#	TRAINRATIO=${d}
#	TRAIN=${PD}/dictionaries/node,split=${TRAINRATIO}.train.dict
#    TEST=${PD}/dictionaries/node,split=${TRAINRATIO}.test.dict
#
#	python -u network_alignment.py \
#    --source_dataset ${PD}/flickr/graphsage/ \
#    --target_dataset ${PD}/myspace/graphsage/ \
#    --groundtruth ${PD}/dictionaries/node,split=${TRAINRATIO}.test.dict \
#	NAME \
#	--train_dict ${TRAIN} \
#	--log \
#	--cuda
#done

#for d in 0.2
#do
#	PD=dataspace/bn-fly-drosophila_medulla_1
#	TRAINRATIO=${d}
#	TRAIN=${PD}/permut/dictionaries/node,split=${TRAINRATIO}.train.dict
#  TEST=${PD}/permut/dictionaries/node,split=${TRAINRATIO}.test.dict
#
#	python -u network_alignment.py \
#    --source_dataset ${PD}/graphsage/ \
#    --target_dataset ${PD}/permut/graphsage/ \
#    --groundtruth ${PD}/permut/dictionaries/node,split=${TRAINRATIO}.test.dict \
#	NAME \
#	--train_dict ${TRAIN} \
#	--log
#done

#--source_dataset graph_data/douban/online/graphsage/ --target_dataset graph_data/douban/offline/graphsage/ --groundtruth graph_data/douban/dictionaries/node,split=0.01.test.dict NAME --train_dict graph_data/douban/dictionaries/node,split=0.01.train.dict --log --cuda
#--source_dataset graph_data/allmv_tmdb/allmv/graphsage/ --target_dataset graph_data/allmv_tmdb/tmdb/graphsage/ --groundtruth graph_data/allmv_tmdb/dictionaries/node,split=0.01.test.dict NAME --train_dict graph_data/allmv_tmdb/dictionaries/node,split=0.01.train.dict --log --cuda
#--source_dataset graph_data/phone_email/phone/graphsage/ --target_dataset graph_data/phone_email/email/graphsage/ --groundtruth graph_data/phone_email/dictionaries/node,split=0.1.test.dict NAME --train_dict graph_data/phone_email/dictionaries/node,split=0.1.train.dict --log --cuda