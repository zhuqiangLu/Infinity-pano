#!/bin/bash


python -c "import triton; print(triton.__version__)" 

# set arguments for inference
pn=0.06M
model_type=infinity_layer12
use_scale_schedule_embedding=0
use_bit_label=1
checkpoint_type='torch'
infinity_model_path=weights/infinity_125M_256x256.pth
# infinity_model_path=checkpoints/mp3d_cube_map_125m/ar-ckpt-giter003K-ep93-iter24-last.pth
vae_type=16
vae_path=weights/infinity_vae_d16.pth
cfg=4
tau=0.5
rope2d_normalized_by_hw=2
add_lvl_embeding_only_first_block=1
rope2d_each_sa_layer=1
text_encoder_ckpt=weights/flan-t5-xl
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
--prompt "an image of a room with a table" \
--seed 1 \
--use_flex_attn 0 \
--h_div_w_template 1 \
--enable_model_cache 1 \
--save_file pano
