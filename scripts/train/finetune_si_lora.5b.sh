export OMP_NUM_THREADS=8
export NCCL_IB_DISABLE=1
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=enp70s0
export NCCL_DEBUG=INFO
export NUM_GPUS=1
export NNODES=1
export RANK=0
export ADDR="127.0.0.1"
export PORT=29500


LLM_VERSION="lmms-lab/llava-onevision-qwen2-0.5b-si"
# for 7b model we recommend bs=1, accum=2, 16 nodes, 128 gpus, lr=1e-5, warmup=0.03
# for 72b model we recommend bs=1, accum=1, 32 nodes, 256 gpus, lr=1e-5, warmup=0.03
LLM_VERSION_CLEAN="${LLM_VERSION//\//_}"
VISION_MODEL_VERSION="google/siglip-so400m-patch14-384"
VISION_MODEL_VERSION_CLEAN="${VISION_MODEL_VERSION//\//_}"

############### Pretrain ################

BASE_RUN_NAME="llava_ov_lora" #"llavanext-google_siglip-so400m-patch14-384-Qwen_Qwen2-0.5B-Instruct-mlp2x_gelu-pretrain_blip558k_plain"
echo "BASE_RUN_NAME: ${BASE_RUN_NAME}"

############### Finetune ################

#I think i need to do my banana data in the format that Qwen expects - Review with marcel

# Stage 2
PROMPT_VERSION="qwen_2"
RUN_NAME="llava-onevision-${VISION_MODEL_VERSION_CLEAN}-${LLM_VERSION_CLEAN}-ov_stage_am9"
# PREV_STAGE_CHECKPOINT="lmms-lab/llava-onevision-qwen2-0.5b-si" # replace it with your last checkpoint training from mid stage
PREV_STAGE_CHECKPOINT="checkpoints/onevision/llava-onevision-qwen2-0.5b-si" # replace it with your last checkpoint training from mid stage
echo "PREV_STAGE_CHECKPOINT: ${PREV_STAGE_CHECKPOINT}"
echo "MID_RUN_NAME: ${RUN_NAME}"

#make sure transformers is at verison 4.46.0

ACCELERATE_CPU_AFFINITY=1 torchrun --nproc_per_node="${NUM_GPUS}" --nnodes="${NNODES}" --node_rank="${RANK}" --master_addr="${ADDR}" --master_port="${PORT}" \
    llava/train/train_mem.py \
    --lora_enable True \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --lora_r 128 --lora_alpha 256 \
    --deepspeed scripts/zero3.json \
    --model_name_or_path $PREV_STAGE_CHECKPOINT \
    --version $PROMPT_VERSION \
    --data_path formatted_training_data.json \
    --image_folder /home/edward/LLaVA-NeXT/\
    --vision_tower ${VISION_MODEL_VERSION} \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --group_by_modality_length True \
    --image_aspect_ratio anyres_max_9 \
    --mm_patch_merge_type spatial_unpad \
    --bf16 True \
    --run_name $RUN_NAME \
    --output_dir checkpoints/onevision/dtc_lora_0.5b_$RUN_NAME \
    --num_train_epochs 10  \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 1 \
    --learning_rate 1e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 32768 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to tensorboard \
    --torch_compile True \
    --torch_compile_backend "inductor" \
    --dataloader_drop_last True \
    --mm_vision_tower_lr 2e-6 \
    #--add_faster_video False \
    #--mm_tunable_parts="mm_vision_tower,mm_mlp_adapter,mm_language_model" \
    
#--image_grid_pinpoints  "(1x1),...,(6x6)" \
#--frames_upbound 32
    
exit 0;

# You can delete the sdpa attn_implementation if you want to use flash attn
#--video_folder /home/edward/LLaVA-NeXT/videos \
#--mm_tunable_parts="mm_vision_tower,mm_mlp_adapter,mm_language_model" \
#--mm_vision_tower_lr=2e-6 \
#--image_grid_pinpoints  "(1x1),...,(6x6)" \
#--frames_upbound 32