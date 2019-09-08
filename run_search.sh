for dataset in NCI1 PTC1 IMDBBINARY IMDBMULTI COLLAB MUTAG PROTEINS; do
for d in ./out/*${dataset}*; do
python search.py --data_dir $d --dataset ${dataset}
done
done
