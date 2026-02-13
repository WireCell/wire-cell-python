import torch
import contextlib
from torch import nn 
import torch.autograd.profiler as profiler
import torch.utils.checkpoint as checkpoint

def call_mp3_prod(as_cells_0, as_cells_1, as_cells_2):
    return as_cells_0 * as_cells_1 * as_cells_2
    # return torch.prod(as_cells, dim=-2)
def call_mp2_prod(as_cells_i, as_cells_j):
    return as_cells_i*as_cells_j

class CPPScatterOpModule(nn.Module):
    def __init__(
        self,
        in_features,
        # hidden_features,
        out_features,
        # ind0_set,
        # ind1_set,
        # ind2_set,
        # mix_ind_set,
        cells_chans,
        ops_library,
        reduction='sum',
        chunk_size=32
    ):
        super().__init__()

        self.chunk_size = chunk_size
        # # self.out_features = out_features
        self.in_features = in_features
        self.out_features = 2*in_features
        self.reduction = reduction
        # self.ind0_set = ind0_set
        # self.ind1_set = ind1_set
        # self.ind2_set = ind2_set
        # ind_set = [ind0_set, ind1_set, ind2_set]
        # self.mix_ind_set = mix_ind_set

        # self.cells_to_chans_0 = []
        # self.cells_to_chans_1 = []
        # self.cells_to_chans_2 = []
        # for iface in range(2):
        #     self.cells_to_chans_0.append(self.ind0_set[iface][self.mix_ind_set[iface].T[:, 0], 1])
        #     self.cells_to_chans_1.append(self.ind1_set[iface][self.mix_ind_set[iface].T[:, 1], 1])
        #     self.cells_to_chans_2.append(self.ind2_set[iface][self.mix_ind_set[iface].T[:, 2], 1])
        # self.cells_to_chans_0 = torch.cat(self.cells_to_chans_0)
        # self.cells_to_chans_1 = torch.cat(self.cells_to_chans_1)
        # self.cells_to_chans_2 = torch.cat(self.cells_to_chans_2)
        # self.cells_to_chans_0 = cells_chans[0]
        # self.cells_to_chans_1 = cells_chans[1]
        # self.cells_to_chans_2 = cells_chans[2]
        self.cells_to_chans = cells_chans
        torch.ops.load_library(ops_library)
        self.cpp_scatter_ops = {}

        self.cpp_scatter_ops = {
            0: torch.classes.my_ops.MyScatterOp(self.cells_to_chans[0]),
            1: torch.classes.my_ops.MyScatterOp(self.cells_to_chans[1]),
            2: torch.classes.my_ops.MyScatterOp(self.cells_to_chans[2]),
        }

    def send_ops(self, device):
        for k, op in self.cpp_scatter_ops.items():
            if op.device() == device:
                # print('Not sending')
                continue
            print('Device before:', op.device())
            op.to(device)
            print('Device after:', op.device())

    def send_one_index(self, ind, device):
        if ind.device != device:
            return ind.to(device)
        return ind

    def send_indices(self, device):
        self.cells_to_chans_0 = self.send_one_index(self.cells_to_chans_0, device)
        self.cells_to_chans_1 = self.send_one_index(self.cells_to_chans_1, device)
        self.cells_to_chans_2 = self.send_one_index(self.cells_to_chans_2, device)

    

    def forward(
        self,
        input_tensor: torch.Tensor,
    ):
        F_in, R, C = input_tensor.shape
        F_out = self.out_features
        output = input_tensor.new_zeros((F_out, R, C))

        
        self.send_ops(input_tensor.device)
        # self.send_indices(input_tensor.device)
        self.cells_to_chans = self.cells_to_chans.to(input_tensor.device)
       
        # print(f'input shape/dtype: {input_tensor.shape} {input_tensor.dtype}')
        for r_start in range(0, R, self.chunk_size):
            # print(f'Chunk starting at {r_start}')
            r_end = min(r_start + self.chunk_size, R)
            curr_R = r_end - r_start
            chunk = input_tensor[:, r_start:r_end, :]
            target_shape = output[:, r_start:r_end, :].shape
            print(f'Chunk {r_start}, {torch.cuda.memory_allocated(0) / (1024**2):.2f}')
            
            for i in range(2): #MP3/MP2 -- Change to 'target-plane-agnostic' for MP2
                target = output[
                    i*self.in_features:(i+1)*self.in_features, #Should really be in_features=1
                    r_start:r_end,
                ]
                # torch.cuda.synchronize()
                # mem0 = torch.cuda.memory_allocated(0) / (1024**2)
                as_cells_0 = chunk[:,:,self.cells_to_chans[0]]
                as_cells_1 = chunk[:,:,self.cells_to_chans[1]]
                as_cells_2 = chunk[:,:,self.cells_to_chans[2]]
                as_cells = [
                    as_cells_0,
                    as_cells_1,
                    as_cells_2
                ]
                # torch.cuda.synchronize()
                # mem1 = torch.cuda.memory_allocated(0) / (1024**2)
                # print(f'as_cells allocates {mem1 - mem0:.2f} MB, ({mem0:.2f}, {mem1:.2f})')
                if i == 0:
                    # mp = torch.prod(as_cells, dim=-2)
                    # torch.cuda.synchronize()
                    # mem0 = torch.cuda.memory_allocated(0) / (1024**2)
                    mp = call_mp3_prod(as_cells_0, as_cells_1, as_cells_2)
                    # mp = checkpoint.checkpoint(call_mp3_prod, as_cells_0, as_cells_1, as_cells_2)
                    # torch.cuda.synchronize()
                    # mem1 = torch.cuda.memory_allocated(0) / (1024**2)
                    # print(f'mp3_prod allocates {mem1 - mem0:.2f} MB, ({mem0:.2f}, {mem1:.2f})')
                    # print('MP cells', mp.shape)
                    for j in range(3):
                        op = self.cpp_scatter_ops[j]
                        target = target + op.forward(mp, target.shape, self.reduction)
                else:
                    mp2_indices = [(1,2), (0,2), (0,1)]
                    for j in range(3):

                        # mp = torch.prod(as_cells[:,:,mp2_indices[j]], dim=-2)
                        # mp = checkpoint.checkpoint(call_mp2_prod, as_cells, mp2_indices[j])
                        # mp = checkpoint.checkpoint(
                        #     call_mp3_prod,
                        #     (1-as_cells_0 if j == 0 else as_cells_0),
                        #     (1-as_cells_1 if j == 1 else as_cells_1),
                        #     (1-as_cells_2 if j == 2 else as_cells_2)
                        # )
                        # mp = call_mp3_prod(
                        #     (1-as_cells_0 if j == 0 else as_cells_0),
                        #     (1-as_cells_1 if j == 1 else as_cells_1),
                        #     (1-as_cells_2 if j == 2 else as_cells_2)
                        # )
                        mp = call_mp2_prod(
                            as_cells[mp2_indices[j][0]],
                            as_cells[mp2_indices[j][1]],
                        )
                        op = self.cpp_scatter_ops[j]
                        target = target + op.forward(mp, target.shape, self.reduction)
            # for i in range(3): #MP3/MP2 -- Change to 'target-plane-agnostic' for MP2
            #     torch.cuda.synchronize()
            #     memA = torch.cuda.memory_allocated(0) / (1024**2)
            #     as_cells = chunk[:,:,self.cells_to_chans[i]]
            #     torch.cuda.synchronize()
            #     memB = torch.cuda.memory_allocated(0) / (1024**2)
            #     print(f'as_cells allocates {memB - memA:.2f} MB, ({memA:.2f}, {memB:.2f})')
            #     print(as_cells.shape)
            #     print(self.in_features)
            #     count = 0
            #     for j in range(3):
            #         if i == j: continue
            #         target = output[
            #             count*self.in_features:(count+1)*self.in_features, #Should really be in_features=1
            #             r_start:r_end,
            #         ]
            #         op = self.cpp_scatter_ops[j]
            #         print(target.shape, as_cells.shape)
            #         target += op.forward(as_cells, target.shape, self.reduction)
            #         count += 1

        return output