CUDA_VISIBLE_DEVICES=0,1,4,5,6
export NCCL_DEBUG=WARN
export NCCL_DEBUG_SUBSYS=OFF
CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES /primus_xpfs_workspace_T04/huangshijie/miniconda3/envs/wan_s2v/bin/torchrun --nproc_per_node=5 --master_port=29102  minimal_inference/s2v_streaming_interact.py \
     --ulysses_size 1 \
     --task s2v-14B \
     --size "720*400" \
     --base_seed 420 \
     --training_config liveavatar/configs/s2v_causal_sft.yaml \
     --offload_model False \
     --convert_model_dtype \
     --prompt "A close-up, black and white video of a 60-70 year old male digital human, resembling Albert Einstein. He has a surprised and curious expression with raised eyebrows and slightly parted lips, as if about to speak. He wears a dark suit, white shirt, and dark tie. In the dimly lit background to his right, a blackboard with a chalk circle and symbols like 'L' and 'K' is visible. The scene has a thoughtful, intelligent, and slightly whimsical atmosphere, rendered in a classic black and white film style."  \
     --image "examples/demo/case1.jpg" \
     --audio "examples/demo/case1_talker.wav" \
     --infer_frames 48 \
     --load_lora \
     --lora_path "/primus_xpfs_workspace_T04/huangyubo/Causvid/checkpoints/1005_s2v_causal_sft_1_4_0/ckpt/merged_model_step_step_25000/only_lora.pt"    \
     --lora_path_dmd "/primus_xpfs_workspace_T04/huangyubo/LongLive/1027_s2v_selfforcing_1_13_2/ckptcopy/step_2500_generator_lora.pt" \
     --sample_steps 4 \
     --sample_guide_scale 0 \
     --num_clip 100 \
     --num_gpus_dit 4 \
     --sample_solver euler \
     --enable_vae_parallel \
     --ckpt_dir ckpt/Wan2.2-S2V-14B/ 
     
