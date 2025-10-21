# Reference: The code has been modified from the LDM repo: https://https://github.com/yccyenchicheng/SDFusion

RED='\033[0;31m'
NC='\033[0m' # No Color
DATE_WITH_TIME=`date "+%Y-%m-%dT%H-%M-%S"`

logs_dir='./outputs/logs'

### set gpus ###
gpu_ids=0         # single-gpu
#gpu_ids=0,1   # multi-gpu

if [ ${#gpu_ids} -gt 1 ]; then
    # specify these two if multi-gpu
    NGPU=2
    # NGPU=3
    # NGPU=4
    PORT=17418
    echo "HERE"
fi
################

### hyper params ###
lr=1e-4
batch_size=4
####################

### model stuff ###
model="vqvae"
vq_cfg="./configs/vqvae.yaml"
#ckpt="saved checkpoint file path" # resume training from checkpoint
####################


### dataset stuff ###
max_dataset_size=10000000
dataset_mode='MP-20-Charge'
dataroot="./data"
###########################

### display & log stuff ###
display_freq=10000 # default: log every display_freq batches
print_freq=10000 # default:
total_iters=100000000
save_steps_freq=10000
###########################

today=$(date '+%m%d')
me=`basename "$0"`
me=$(echo $me | cut -d'.' -f 1)

name="${DATE_WITH_TIME}-${model}"

debug=0
if [ $debug = 1 ]; then
    printf "${RED}Debugging!${NC}\n"
	batch_size=2
	max_dataset_size=12
    total_iters=1000000
    save_steps_freq=3
	display_freq=2
	print_freq=1
    name="DEBUG-${name}"
fi

cmd="train.py --name ${name} --logs_dir ${logs_dir} --gpu_ids ${gpu_ids} --lr ${lr} --batch_size ${batch_size} \
                --model ${model} --vq_cfg ${vq_cfg}\
                --dataset_mode ${dataset_mode} --display_freq ${display_freq} --print_freq ${print_freq} --max_dataset_size ${max_dataset_size} \
                --total_iters ${total_iters} --save_steps_freq ${save_steps_freq} \
                --debug ${debug}"
 
if [ ! -z "$dataroot" ]; then
    cmd="${cmd} --dataroot ${dataroot}"
    echo "setting dataroot to: ${dataroot}"
fi

if [ ! -z "$ckpt" ]; then
    cmd="${cmd} --ckpt ${ckpt}"
    echo "continue training with ckpt=${ckpt}"
fi


multi_gpu=0
if [ ${#gpu_ids} -gt 1 ]; then
    multi_gpu=1
fi

echo "[*] Training is starting on `hostname`, GPU#: ${gpu_ids}, logs_dir: ${logs_dir}"

if [ $multi_gpu = 1 ]; then
    cmd="-m torch.distributed.launch --nproc_per_node=${NGPU} --master_port=${PORT} ${cmd}"
fi

echo "[*] Training with command: "
echo "CUDA_VISIBLE_DEVICES=${gpu_ids} python ${cmd}"

# exit

CUDA_VISIBLE_DEVICES=${gpu_ids} python ${cmd}
