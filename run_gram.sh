for num_mlp_layers in 1 2 3; do
for scale in uniform degree; do

# NCI1 and PTC1
for dataset in NCI1 PTC1; do
for num_layers in 10 12 14; do
out_dir=./out/dataset-${dataset}-num_layers-${num_layers}-num_mlp_layers-${num_mlp_layers}-scale-${scale}
mkdir -p ${out_dir}
python gram.py --dataset ${dataset} --num_mlp_layers ${num_mlp_layers} --num_layers ${num_layers} --scale ${scale} --out_dir ${out_dir}
done
done

# IMDBBINARY IMDBMULTI COLLAB MUTAG PROTEINS
for dataset in IMDBBINARY IMDBMULTI COLLAB MUTAG PROTEINS; do
for num_layers in 2 4; do
out_dir=./out/dataset-${dataset}-num_layers-${num_layers}-num_mlp_layers-${num_mlp_layers}-scale-${scale}
mkdir -p ${out_dir}
python gram.py --dataset ${dataset} --num_mlp_layers ${num_mlp_layers} --num_layers ${num_layers} --scale ${scale} --out_dir ${out_dir}
done
done

done
done
