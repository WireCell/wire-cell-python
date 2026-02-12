from argparse import ArgumentParser as ap
from cpp_scatter_simpler import CPPScatterOpModule
import torch
import h5py as h5
import numpy as np 

def load_file(input_file):
    if 'random' in input_file:
        return torch.randint(0,2,(600,2560))
    if '.h5' in input_file:
        filename, dset = input_file.split(':')
        with h5.File(filename) as f:
            result = torch.tensor(np.array(f[dset]))
            return (result.T[1400:2400] > 0.5).to(torch.float)

if __name__ == '__main__':

    parser = ap()
    parser.add_argument('-i', type=str, help='Input file:array name', required=True)
    parser.add_argument('--library', type=str, help='Library file', required=True)
    parser.add_argument('--cells', type=str, help='File containing cells/channel mapping', required=True)
    parser.add_argument('--device', type=str, help='Which device', default='cpu', choices=['cpu', 'cuda', 'cuda:0', 'cuda:1'])
    parser.add_argument('-o', type=str, help='Output file', default='roundabout.pt')
    parser.add_argument('--plane', '-p', type=int, default=0, help='Which plane to scatter-roundabout', choices=[0,1,2])
    parser.add_argument('--chunk', type=int, help='Chunk size', default=36)
    args = parser.parse_args()

    input_array = load_file(args.i).to(args.device)
    print(input_array.shape)

    with torch.no_grad():

        cells_chans_file = torch.load(args.cells)
        cells_chans_f0 = cells_chans_file['cells_chans_f0']
        cells_chans_f1 = cells_chans_file['cells_chans_f1']
        print(cells_chans_f0.shape, cells_chans_f1.shape)
        cells_chans = torch.cat([
            cells_chans_f0, cells_chans_f1
        ], dim=1).to(args.device)
        print(cells_chans.shape)

        result = torch.zeros_like(input_array)
        torch.ops.load_library(args.library)
        
        for plane in [0,1,2]:
            input_as_cells = input_array[:,cells_chans[plane]]
            op = torch.classes.my_ops.MyScatterOp(cells_chans[plane])
            for r_start in range(0, input_as_cells.shape[0], args.chunk):
                r_end = r_start + args.chunk
                target = result[r_start:r_end]
                print(target.shape)
                target += op.forward(input_as_cells[r_start:r_end], target.shape, 'max')

        op_module = CPPScatterOpModule(1, 1, cells_chans, args.library, reduction='max', chunk_size=args.chunk)

        mod_result = op_module(input_array.unsqueeze(0))
        torch.save({'input':input_array, 'result':result, 'mod_result':mod_result}, args.o)