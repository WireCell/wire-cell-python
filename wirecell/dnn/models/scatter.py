import torch
from torch import nn 

class Scatter(torch.autograd.Function):
    @staticmethod
    @torch.no_grad() # Crucial: Don't track history inside the manual forward
    def forward(ctx, input_tensor, w1, b1, w2, b2, ind0_set, ind1_set, ind2_set, mix_ind_set, chunk_size=2):
        # input_tensor: (F_in, Rows, Cols)
        F_in, R, C = input_tensor.shape
        F_out = w2.size(0)
        output = input_tensor.new_zeros((F_out, R, C))
        # print(output.dtype)
        def project(chunk_in, ind):
            res = chunk_in.new_zeros((F_in, curr_R, len(ind)))
            res[..., ind[:,0]] = chunk_in[..., ind[:,1]]
            return res
        
        # Pre-calculate mixed indices for all sets: Shape (num_sets, 3, len_mix_ind)
        # We store them in a list for easier iteration
        m_indices = []
        num_sets = len(ind0_set)

        # We chunk over the Rows (R) to save memory
        for r_start in range(0, R, chunk_size):
            r_end = min(r_start + chunk_size, R)
            curr_R = r_end - r_start
            
            # 1. Gather & Cat
            # Slicing Columns (dim 2). Shape of each: (F_in, curr_R, len_ind)
            chunk = input_tensor[:, r_start:r_end, :]

            # print('Chunk', chunk.shape)
            out_chunk = None
            for s in range(num_sets):
                c1 = project(chunk, ind0_set[s])[..., mix_ind_set[s][0]]
                c2 = project(chunk, ind1_set[s])[..., mix_ind_set[s][1]]
                c3 = project(chunk, ind2_set[s])[..., mix_ind_set[s][2]]
            
                # Concatenate along Feature dim: (3*F_in, curr_R, len_ind)
                c = torch.cat([c1, c2, c3], dim=0)
                # print('c', c.shape)
                # Prepare for Linear: (curr_R * len_ind, 3*F_in)
                # Permute moves Features to the end
                c_flat = c.permute(1, 2, 0).reshape(-1, 3 * F_in)

                # 2. MLP Logic (Linear -> ReLU -> Linear)
                z1 = torch.matmul(c_flat, w1.t()) + b1
                a1 = torch.relu(z1)
                d_flat = torch.matmul(a1, w2.t()) + b2
            
                # 3. Scatter Reduce Amax
                # d_for_scatter shape: (curr_R, len_ind, F_out)
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
                    # print(exp_ind.shape)
                    out_chunk = torch.scatter_reduce(
                        out_chunk, 
                        dim=1,
                        index=exp_ind,
                        src=d_for_scatter, 
                        reduce='amax', 
                        include_self=True
                    )
            output[:, r_start:r_end, :] = out_chunk.permute(2, 0, 1)
            
        ctx.save_for_backward(input_tensor, w1, b1, w2, b2)
        ctx.indices = (ind0_set, ind1_set, ind2_set, mix_ind_set)
        ctx.input_shape = (F_in, R, C)
        ctx.chunk_size = chunk_size
        return output

    @staticmethod
    @torch.no_grad()
    def backward(ctx, grad_output):
        print('CALLING BACKWARD')
        def project(chunk_in, ind):
            res = chunk_in.new_zeros((F_in, curr_R, len(ind)))
            res[..., ind[:,0]] = chunk_in[..., ind[:,1]]
            return res

        input_tensor, w1, b1, w2, b2 = ctx.saved_tensors
        ind0_set, ind1_set, ind2_set, mix_ind_set = ctx.indices
        F_in, R, C = input_tensor.shape
        F_out = w2.size(0)
        num_sets = len(ind0_set)
        m_indices = []
        for s in range(num_sets):
            m_ind0, m_ind1, m_ind2 = ind0_set[s][:,1][mix_ind_set[s][0]], ind1_set[s][:,1][mix_ind_set[s][1]], ind2_set[s][:,1][mix_ind_set[s][2]]
            m_indices.append([m_ind0, m_ind1, m_ind2])

        grad_input = torch.zeros_like(input_tensor)
        grad_w1, grad_b1 = torch.zeros_like(w1), torch.zeros_like(b1)
        grad_w2, grad_b2 = torch.zeros_like(w2), torch.zeros_like(b2)

        for r_start in range(0, R, ctx.chunk_size):
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
                # print('cflat', c_flat.shape)
                z1 = torch.matmul(c_flat, w1.t()) + b1
                a1 = torch.relu(z1)
                d_flat = torch.matmul(a1, w2.t()) + b2
                d_src = d_flat.view(curr_R, -1, F_out).permute(2, 0, 1)
                # print('dsrc', d_src)
                
                # Reconstruct winners
                for m_idx in m_indices[s]:
                    exp_ind = m_idx.view(1, 1, -1).expand(F_out, curr_R, -1)
                    # print(exp_ind)
                    f_out_chunk = torch.scatter_reduce(f_out_chunk, 2, exp_ind, d_src, reduce='amax', include_self=True)
                set_data.append((c_flat, z1, a1, d_src))
            # print('Fout chunk', f_out_chunk)
            for s in range(num_sets):
                c_flat, z1, a1, d_src = set_data[s]
                # print('cflat', c_flat)
                # print('z1', z1)
                # print('a1', a1)
                # print('dsrc', d_src)
                # 1. Accumulate Gradients from 3 Scatter Paths
                total_grad_d = torch.zeros_like(d_src)
                for m_idx in m_indices[s]:
                    exp_ind = m_idx.view(1, 1, -1).expand(F_out, curr_R, -1)
                    # Match d_src against the final f_out_chunk winners
                    mask = (d_src == torch.gather(f_out_chunk, 2, exp_ind))
                    # mask = (d_src - torch.gather(f_out_chunk, 2, exp_ind)).abs() < 1e-8
                    # mask = torch.ones_like(d_src, dtype=torch.bool) # Temporary: ignore winners
                    total_grad_d += torch.gather(g_out_chunk, 2, exp_ind) * mask.float()

                # 2. MLP Backward
                grad_d_flat = total_grad_d.permute(1, 2, 0).reshape(-1, F_out)
                grad_a1 = grad_d_flat @ w2

                grad_w2.add_(grad_d_flat.t() @ a1)
                grad_b2.add_(grad_d_flat.sum(0))
            
                grad_z1 = grad_a1 * (z1 > 0).float() #ReLU Deriv
                grad_c_flat = grad_z1 @ w1
                grad_w1.add_(grad_z1.t() @ c_flat)
                grad_b1.add_(grad_z1.sum(0))

                # 3. Indexing Backward
                grad_c = grad_c_flat.view(curr_R, -1, 3 * F_in).permute(2, 0, 1)
                gc1, gc2, gc3 = torch.split(grad_c, F_in, dim=0)
                # print('GC SHAPES', gc1.shape, gc2.shape, gc3.shape, F_in)
                # print('GC Any:', grad_c.any(), gc1.any(), gc2.any(), gc3.any())
                m_ind0, m_ind1, m_ind2 = m_indices[s]
                gi_slice = grad_input[:, r_start:r_end, :]
                gi_slice.index_add_(2, m_ind0, gc1) # Wait - check indices here
                gi_slice.index_add_(2, m_ind1, gc2)
                gi_slice.index_add_(2, m_ind2, gc3)
                # print(f"Grad Output Sum: {g_out_chunk.abs().sum().item()}")
                # print(f"Total Grad D Sum: {total_grad_d.abs().sum().item()}")
                # print(f"Grad Z1 Sum: {grad_z1.abs().sum().item()}")
                # print(f"Grad C Flat Sum: {grad_c_flat.abs().sum().item()}")

        print('BACKWARD DONE')
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