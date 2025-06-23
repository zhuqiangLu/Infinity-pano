export PYTHONPATH=/srv/zhlu6105/Infinity-pano/:$PYTHONPATH

torchrun --nnodes=1 --nproc-per-node=1 --master_addr=127.0.0.1 --master_port=12432 infinity/dataset/dataset_t2i_iterable.py