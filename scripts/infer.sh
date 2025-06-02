#!/bin/bash


#!/bin/bash
# activate environment 
module load anaconda/2021.11
# module load cuda/11.7.0 
# module load compilers/gcc/11.3.0
# module load cudnn/8.6.0.163_cuda11.x
# module load ffmpeg/4.4.1-gcc11  
# source activate ai4multi_meta
# export CUDA_HOME=/home/bingxing2/apps/cuda/11.7.0
module load ffmpeg/4.4.1-gcc11  
module load anaconda/2021.11 compilers/cuda/12.1 cudnn/8.8.1.3_cuda12.x compilers/gcc/11.3.0
# source activate multi_clone_flash_attn

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=2
export TRANSFORMERS_VERBOSITY="info"

export NCCL_ALGO=Ring
export NCCL_MAX_NCHANNELS=16
export NCCL_MIN_NCHANNELS=16
export NCCL_TOPO_FILE=/home/bingxing2/apps/nccl/conf/dump.xml
export NCCL_IB_HCA=mlx5_0,mlx5_2
export NCCL_IB_GID_INDEX=3
export NCCL_IB_TIMEOUT=23
export NCCL_IB_RETRY_CNT=7
# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=ALL
export http_proxy=https://wanglintao:MEKLD6LPq5BPPDtozP0ag4ErMiYnOYwtoWKkqgjsLgmqlz4JsIqQGIaCxrST@blsc-proxy.pjlab.org.cn:13128
export https_proxy=https://wanglintao:MEKLD6LPq5BPPDtozP0ag4ErMiYnOYwtoWKkqgjsLgmqlz4JsIqQGIaCxrST@blsc-proxy.pjlab.org.cn:13128

export XFORMERS_FORCE_DISABLE_TRITON=1
# export http_proxy=http://u-cEoRwn:EDvFuZTe@172.16.4.9:3128
# export https_proxy=http://u-cEoRwn:EDvFuZTe@172.16.4.9:3128
# nodes gpus rank master_addr hosfile job_id
# nodes
NODES=$1

# gpus
NUM_GPUS=$2

# rank
NODE_RANK=$3

# Master addr
MASTER_ADDR=$4
MASTER_PORT=29501

#DHOSTFILE
DHOSTFILE=$5

# JOB_ID
JOB_ID=$6

echo $NODES
echo $NUM_GPUS
echo $NODE_RANK
echo $MASTER_ADDR
echo $MASTER_PORT
echo $DHOSTFILE

# logs
OUTPUT_LOG="./log/train_rank${NODE_RANK}_${JOB_ID}.log"
echo "nodes,gpus,node_rank,master_addr,master_port,dhostfile" >> $OUTPUT_LOG
echo "$NODES,$NUM_GPUS,$NODE_RANK,$MASTER_ADDR,$MASTER_PORT,$DHOSTFILE" >> $OUTPUT_LOG


# export CUDA_HOME=/home/bingxing2/apps/cuda/11.7.0
export LD_PRELOAD=/home/bingxing2/ailab/scxlab0109/.conda/envs/dna_ft/lib/python3.8/site-packages/sklearn/__check_build/../../scikit_learn.libs/libgomp-d22c30c5.so.1.0.0
free -h
export LD_PRELOAD=/home/bingxing2/apps/compilers/gcc/12.2.0/lib64/libstdc++.so
export HF_TOKEN=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

python -c "import triton; print(triton.__version__)" 

# set arguments for inference
pn=0.06M
model_type=infinity_2b
use_scale_schedule_embedding=0
use_bit_label=1
checkpoint_type='torch'
infinity_model_path=debug/ar-ckpt-giter012K-ep9-iter260-last.pth
vae_type=32
vae_path=weights/infinity_vae_d32reg.pth
cfg=4
tau=0.5
rope2d_normalized_by_hw=2
add_lvl_embeding_only_first_block=1
rope2d_each_sa_layer=1
text_encoder_ckpt=weights/google-flan-t5-xl
text_channels=2048
apply_spatial_patchify=0
# run inference
python3 run_infinity.py \
--cfg ${cfg} \
--tau ${tau} \
--pn ${pn} \
--model_path ${infinity_model_path} \
--vae_type ${vae_type} \
--vae_path ${vae_path} \
--add_lvl_embeding_only_first_block ${add_lvl_embeding_only_first_block} \
--use_bit_label ${use_bit_label} \
--model_type ${model_type} \
--rope2d_each_sa_layer ${rope2d_each_sa_layer} \
--rope2d_normalized_by_hw ${rope2d_normalized_by_hw} \
--use_scale_schedule_embedding ${use_scale_schedule_embedding} \
--cfg ${cfg} \
--tau ${tau} \
--checkpoint_type ${checkpoint_type} \
--text_encoder_ckpt ${text_encoder_ckpt} \
--text_channels ${text_channels} \
--apply_spatial_patchify ${apply_spatial_patchify} \
--prompt "" \
--seed 1 \
--use_flex_attn 0 \
--h_div_w_template 0.5 \
--save_file tmp.jpg 
