This branch adds the ability to do Distributed Data Parallel (DDP) within wirecell.dnn. 

Some docs on DDP:

* https://docs.pytorch.org/docs/2.13/generated/torch.nn.parallel.DistributedDataParallel.html 
* https://docs.pytorch.org/tutorials/beginner/ddp_series_theory.html
  * this is a series of docs/presentations
* https://tech.preferred.jp/en/blog/technologies-behind-distributed-deep-learning-allreduce/
* https://docs.pytorch.org/tutorials/intermediate/dist_tuto.html

This enables the usage of multi-GPU & multi-node training within the wcpy.dnn infrastructure.
At a high level, DDP works by:
1. It duplicates models across GPUs
2. Within a forward pass, each GPU processes a batch of distinct data
3. Gradients are calculated separately
4. They are then aggregated via communication between GPU devices
5. Model weights are updated in-sync

(Refine this part) This uses `torchrun` to facilitate the launch and distribution

## Usage for single node

Minimal options
<pre>
CUDA_VISIBLE_DEVICES="{Set,Of,Devices}"  torchrun --standalone --nproc_per_node={NProcs} \
  -m wirecell.dnn train -c {configuration_file} \
  -d cuda -b {BatchSize} -e {NEpochs} 
</pre>

DDP-Relevant Option Explanations:
* NProcs: This is the number of parallel processes per node, and should correspond to the number of GPUs on that node
* Set,Of,Devices: comma-separated set of device IDs (i.e. 0,1,2) of the specific devices that would be used during training
  * The length of this must not be less than NProcs
* BatchSize: this is the batch size **per GPU**

### Possibly-necessary Extra Environment Variables
I need to refine my understanding of all of these settings, but there are some possibly-necessary extra env vars to get
the GPUs to properly communicate to each other and these are heavily dependent on their hardware.

| Env Var  | Description |
| ------------- | ------------- |
| NCCL_IB_DISABLE  | TBD  |
| NCCL_P2P_DISABLE  | TBD  |

### Known Env Vars needed for various machines/devices

#### wcgpu0.phy.bnl.gov
Devices: 2x RTX 4090s

Does not need any env vars

#### wcgpu1.phy.bnl.gov
Devices: 2x RTX 4090

I assume this is the same as wcgpu0, but have not confirmed that it does not need any env vars

#### sgpu0004.sdcc.bnl.gov
Devices: 10x L40S

Needed `NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1`
