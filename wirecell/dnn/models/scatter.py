import torch
from torch import nn 
# from torch.amp import custom_bwd, custom_fwd

class Scatter(torch.autograd.Function):
    @staticmethod
    # @custom_fwd
    # @torch.no_grad()
    def forward(ctx, input_tensor, w1, b1, w2, b2, ind0_set, ind1_set, ind2_set, mix_ind_set, chunk_size=2):
        # print('Forward')
        # input_tensor: (F_in, Rows, Cols)
        F_in, R, C = input_tensor.shape
        F_out = w2.size(0)
        output = input_tensor.new_zeros((F_out, R, C))
        # print(output.dtype)
        def project(chunk_in, ind):
            res = chunk_in.new_zeros((F_in, curr_R, len(ind)))
            res[..., ind[:,0]] = chunk_in[..., ind[:,1]]
            return res
        # def project(chunk_in, ind):
        #     res = chunk_in.new_zeros((F_in, curr_R, len(ind)))
        #     res[..., ind[:,0]] = chunk_in[..., ind[:,1]]
        #     return res
        
        # Pre-calculate mixed indices for all sets: Shape (num_sets, 3, len_mix_ind)
        # We store them in a list for easier iteration
        m_indices = []
        num_sets = len(ind0_set)
        m_indices = []
        for s in range(num_sets):
            m_ind0, m_ind1, m_ind2 = ind0_set[s][:,1][mix_ind_set[s][0]], ind1_set[s][:,1][mix_ind_set[s][1]], ind2_set[s][:,1][mix_ind_set[s][2]]
            m_indices.append([m_ind0, m_ind1, m_ind2])  
        all_masks = []
        total_masks_size = 0
        nchunks = 0
        # We chunk over the Rows (R) to save memory
        for r_start in range(0, R, chunk_size):
            nchunks += 1
            r_end = min(r_start + chunk_size, R)
            curr_R = r_end - r_start
            
            # 1. Gather & Cat
            # Slicing Columns (dim 2). Shape of each: (F_in, curr_R, len_ind)
            chunk = input_tensor[:, r_start:r_end, :]

            # print('Chunk', chunk.shape)
            out_chunk = None
            # all_masks.append([]) # Per Row Set
            set_data = []
            # with torch.no_grad():
            for s in range(num_sets):
                c1 = project(chunk, ind0_set[s])[..., mix_ind_set[s][0]]
                c2 = project(chunk, ind1_set[s])[..., mix_ind_set[s][1]]
                c3 = project(chunk, ind2_set[s])[..., mix_ind_set[s][2]]
            
                # Concatenate along Feature dim: (3*F_in, curr_R, len_ind)
                c = torch.cat([c1, c2, c3], dim=0)
                # Prepare for Linear: (curr_R * len_ind, 3*F_in)
                # Permute moves Features to the end
                c_flat = c.permute(1, 2, 0).reshape(-1, 3 * F_in)

                # 2. MLP Logic (Linear -> ReLU -> Linear)
                z1 = torch.matmul(c_flat, w1.t()) + b1
                a1 = torch.relu(z1)
                d_flat = torch.matmul(a1, w2.t()) + b2
            
                # 3. Scatter Reduce Amax
                d_for_scatter = d_flat.view(curr_R, -1, F_out)

                if out_chunk is None:
                    out_chunk = input_tensor.new_zeros((curr_R, C, F_out))

                ind0 = ind0_set[s]
                ind1 = ind1_set[s]
                ind2 = ind2_set[s]
                mix_ind = mix_ind_set[s]
                for ind_i, ind in enumerate([ind0, ind1, ind2]):
                    # Replicate indices across the chunked rows
                    to_scatter_indices = ind[:,1][mix_ind[ind_i]]
                    exp_ind = to_scatter_indices.unsqueeze(0).unsqueeze(-1).expand(curr_R, -1, F_out)
                    # print(out_chunk.dtype, d_for_scatter.dtype, w1.dtype, a1.dtype, z1.dtype)
                    out_chunk = torch.scatter_reduce(
                        out_chunk, 
                        dim=1,
                        index=exp_ind,
                        src=d_for_scatter, 
                        reduce='amax', 
                        include_self=True
                    )
                set_data.append(d_for_scatter.permute(2,0,1))
            output[:, r_start:r_end, :] = out_chunk.permute(2, 0, 1)


            for s in range(num_sets):
                ind0 = ind0_set[s]
                ind1 = ind1_set[s]
                ind2 = ind2_set[s]
                mix_ind = mix_ind_set[s]
                d_src = set_data[s]
                # all_masks[-1].append([])
                for k, m_idx in enumerate(m_indices[s]):
                    # Replicate indices across the chunked rows
                    exp_ind = m_idx.view(1, 1, -1).expand(F_out, curr_R, -1)
                    # Match d_src against the final f_out_chunk winners
                    mask = (d_src == torch.gather(out_chunk.permute(2,0,1), 2, exp_ind))
                    # all_masks[-1][-1].append(mask.to(bool))
                    all_masks.append(mask.to(bool))
                    # all_masks[-1][-1].append(torch.where(mask))
                    # all_masks[-1][-1][-1] = tuple(wi.to(torch.uint8).to('cpu') if i < 2 else wi.to('cpu') for i, wi in enumerate(all_masks[-1][-1][-1]))
        #             for w in all_masks[-1][-1][-1]:
        #                 print(w.numel()*w.element_size(), w.numel())
        #                 total_masks_size += w.numel()*w.element_size()

            # print('Total mask size:', total_masks_size)
        
        ctx.save_for_backward(input_tensor, w1, b1, w2, b2, *all_masks)
        ctx.indices = (ind0_set, ind1_set, ind2_set, mix_ind_set)
        ctx.input_shape = (F_in, R, C)
        # ctx.all_masks = all_masks
        ctx.chunk_size = chunk_size
        ctx.num_sets = num_sets
        ctx.nchunks = nchunks
        ctx.num_views = 3
        return output

    @staticmethod
    @torch.no_grad()
    # @custom_bwd
    def backward(ctx, grad_output):
        # print('CALLING BACKWARD')
        

        # all_masks = ctx.all_masks

        input_tensor, w1, b1, w2, b2 = ctx.saved_tensors[:5]
        all_masks = ctx.saved_tensors[5:]
        ind0_set, ind1_set, ind2_set, mix_ind_set = ctx.indices
        F_in, R, C = input_tensor.shape
        F_out = w2.size(0)
        def project(chunk_in, ind):
            res = chunk_in.new_zeros((F_in, curr_R, len(ind)))
            res[..., ind[:,0]] = chunk_in[..., ind[:,1]]
            return res
        
        num_sets = len(ind0_set)
        num_views = ctx.num_views
        m_indices = []
        for s in range(num_sets):
            m_ind0, m_ind1, m_ind2 = ind0_set[s][:,1][mix_ind_set[s][0]], ind1_set[s][:,1][mix_ind_set[s][1]], ind2_set[s][:,1][mix_ind_set[s][2]]
            m_indices.append([m_ind0, m_ind1, m_ind2])

        grad_input = torch.zeros_like(input_tensor)
        grad_w1, grad_b1 = torch.zeros_like(w1), torch.zeros_like(b1)
        grad_w2, grad_b2 = torch.zeros_like(w2), torch.zeros_like(b2)

        for ir, r_start in enumerate(range(0, R, ctx.chunk_size)):
            r_end = min(r_start + ctx.chunk_size, R)
            curr_R = r_end - r_start
            
            # --- RECOMPUTE ---
            chunk = input_tensor[:, r_start:r_end, :]
            f_out_chunk = input_tensor.new_zeros((F_out, curr_R, C))
            g_out_chunk = grad_output[:, r_start:r_end, :]

            set_data = []
            
            # print('Chunk', chunk.shape)
            for s in range(num_sets):
                ind0 = ind0_set[s]
                ind1 = ind1_set[s]
                ind2 = ind2_set[s]
                mix_ind = mix_ind_set[s]
                c1 = project(chunk, ind0)[..., mix_ind[0]]
                c2 = project(chunk, ind1)[..., mix_ind[1]]
                c3 = project(chunk, ind2)[..., mix_ind[2]]
                c = torch.cat([c1,c2,c3], dim=0)
                c_flat = c.permute(1, 2, 0).reshape(-1, 3 * F_in)
                z1 = torch.matmul(c_flat, w1.t()) + b1
                a1 = torch.relu(z1)
                d_flat = torch.matmul(a1, w2.t()) + b2
                total_grad_d = torch.zeros_like(d_flat).view(F_out, curr_R, -1)
                test_grad_a1 = 0
                for k, m_idx in enumerate(m_indices[s]):
                    exp_ind = m_idx.view(1, 1, -1).expand(F_out, curr_R, -1)
                    # w = all_masks[ir][s][k]
                    w = all_masks[ir*(num_sets*num_views) + s*num_views + k]
                    total_grad_d += torch.gather(g_out_chunk, 2, exp_ind) * w

                # 2. MLP Backward
                grad_d_flat = total_grad_d.permute(1, 2, 0).reshape(-1, F_out)
                grad_a1 = grad_d_flat @ w2
                # print('Test', torch.all(grad_a1 == test_grad_a1))

                grad_w2.add_(grad_d_flat.t() @ a1)
                grad_b2.add_(grad_d_flat.sum(0))
            
                grad_z1 = grad_a1 * (z1 > 0).float() #ReLU Deriv
                grad_c_flat = grad_z1 @ w1
                grad_w1.add_(grad_z1.t() @ c_flat)
                grad_b1.add_(grad_z1.sum(0))

                # 3. Indexing Backward
                grad_c = grad_c_flat.view(curr_R, -1, 3 * F_in).permute(2, 0, 1)
                gc1, gc2, gc3 = torch.split(grad_c, F_in, dim=0)
                m_ind0, m_ind1, m_ind2 = m_indices[s]
                gi_slice = grad_input[:, r_start:r_end, :]
                gi_slice.index_add_(2, m_ind0, gc1) # Wait - check indices here
                gi_slice.index_add_(2, m_ind1, gc2)
                gi_slice.index_add_(2, m_ind2, gc3)

        # print('BACKWARD DONE')
        return grad_input, grad_w1, grad_b1, grad_w2, grad_b2, None, None, None, None, None
    



class TripleScatterModule(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, chunk_size=32):
        super().__init__()
        # Initialize weights matching the cat([c1, c2, c3]) logic
        self.w1 = nn.Parameter(torch.randn(hidden_features, in_features * 3))
        self.b1 = nn.Parameter(torch.zeros(hidden_features))
        self.w2 = nn.Parameter(torch.randn(out_features, hidden_features))
        self.b2 = nn.Parameter(torch.zeros(out_features))
        self.chunk_size = chunk_size

    def forward(self, input_tensor, ind0, ind1, ind2, mix_ind):
        return Scatter.apply(
            input_tensor, self.w1, self.b1, self.w2, self.b2,
            ind0, ind1, ind2, mix_ind, self.chunk_size
        )