export CUDA_VISIBLE_DEVICES=0

python train.py -s <DATA_DIR>  --eval \
     --port 4810 --expname 'bouncingballs' --voxelsize 0.005 

python render.py -m ./output/bouncingballs --render_type 'metrics'

python metrics.py -m ./output/bouncingballs
