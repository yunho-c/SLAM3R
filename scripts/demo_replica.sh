#!/bin/bash
I2P_MODEL="Image2PointsModel(pos_embed='RoPE100', img_size=(224, 224), head_type='linear', output_mode='pts3d', depth_mode=('exp', -inf, inf), conf_mode=('exp', 1, inf), \
enc_embed_dim=1024, enc_depth=24, enc_num_heads=16, dec_embed_dim=768, dec_depth=12, dec_num_heads=12, \
mv_dec1='MultiviewDecoderBlock_max',mv_dec2='MultiviewDecoderBlock_max', enc_minibatch = 11)"

L2W_MODEL="Local2WorldModel(pos_embed='RoPE100', img_size=(224, 224), head_type='linear', output_mode='pts3d', depth_mode=('exp', -inf, inf), conf_mode=('exp', 1, inf), \
enc_embed_dim=1024, enc_depth=24, enc_num_heads=16, dec_embed_dim=768, dec_depth=12, dec_num_heads=12, \
mv_dec1='MultiviewDecoderBlock_max',mv_dec2='MultiviewDecoderBlock_max', enc_minibatch = 11, need_encoder=False)"

######################################################################################
# set the path of the model weights below
######################################################################################
I2P_WEIGHT_PATH="checkpoints/slam3r_i2p.pth"
L2W_WEIGHT_PATH="checkpoints/slam3r_l2w.pth"

######################################################################################
# set the img_dir below to the directory of the set of images you want to reconstruct
# set the postfix below to the format of the rgb images in the img_dir
######################################################################################
TEST_DATASET="Seq_Data(img_dir='data/Replica/room0', postfix='.jpg', \
img_size=224, silent=False, sample_freq=1, \
start_idx=0, num_views=-1, start_freq=1, to_tensor=True)"

######################################################################################
# set the parameters for whole scene reconstruction below
# for defination of these parameters, please refer to the recon.py
######################################################################################
TEST_NAME="Replica_demo"
KEYFRAME_FREQ=20
UPDATE_BUFFER_FREQ=3
MAX_NUM_REGISTER=10
WIN_R=5
NUM_SCENE_FRAME=10
INITIAL_WINSIZE=5 
CONF_THRES_L2W=10
CONF_THRES_I2P=1.5
NUM_POINTS_SAVE=1000000

GPU_ID=-1


python recon.py \
--test_name $TEST_NAME \
--l2w_model "${L2W_MODEL}" \
--l2w_weights "${L2W_WEIGHT_PATH}" \
--dataset "${TEST_DATASET}" \
--i2p_model "${I2P_MODEL}" \
--i2p_weights "${I2P_WEIGHT_PATH}" \
--gpu_id $GPU_ID \
--keyframe_freq $KEYFRAME_FREQ \
--win_r $WIN_R \
--num_scene_frame $NUM_SCENE_FRAME \
--initial_winsize $INITIAL_WINSIZE \
--conf_thres_l2w $CONF_THRES_L2W \
--conf_thres_i2p $CONF_THRES_I2P \
--num_points_save $NUM_POINTS_SAVE \
--update_buffer_freq $UPDATE_BUFFER_FREQ \
--max_num_register $MAX_NUM_REGISTER
