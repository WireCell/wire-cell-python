from wirecell.dnn.models.scatter import TripleScatterModule, Scatter
from argparse import ArgumentParser as ap
import torch

from wirecell.util.wires import schema, persist
import torch.autograd.profiler as profiler
def verify_spatial_logic():
    # Use small dimensions so gradcheck finishes quickly
    # Use double precision (float64) for numerical stability in testing
    F_in, R, C = 2, 4, 8
    hidden, F_out = 4, 2
    chunk_size = 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.double

    # 1. Setup inputs with requires_grad=True
    input_tensor = torch.randn(F_in, R, C, device=device, dtype=dtype, requires_grad=True)
    w1 = torch.randn(hidden, F_in * 3, device=device, dtype=dtype, requires_grad=True)
    b1 = torch.randn(hidden, device=device, dtype=dtype, requires_grad=True)
    w2 = torch.randn(F_out, hidden, device=device, dtype=dtype, requires_grad=True)
    b2 = torch.randn(F_out, device=device, dtype=dtype, requires_grad=True)
    
    # 2. Setup Indices (Non-differentiable)
    ind1 = [torch.tensor([[0,0], [1,1], [2,2]], device=device, dtype=torch.long)]
    ind2 = [torch.tensor([[0,3], [1,4], [2,5]], device=device, dtype=torch.long)]
    ind3 = [torch.tensor([[0,6], [1,7]], device=device, dtype=torch.long)]
    # 012
    # 120
    # 101
    mix_ind = [torch.tensor([[0, 1, 1], [1, 2, 0], [2, 0, 1]], device=device, dtype=torch.long).T]
    print(mix_ind[0][2])
    # ind1 = [face_plane_wires_channels[0,0], face_plane_wires_channels[1,0]]
    # ind2 = [face_plane_wires_channels[0,1], face_plane_wires_channels[1,1]]
    # ind3 = [face_plane_wires_channels[0,2], face_plane_wires_channels[1,2]]
    # mix_ind = [good_indices_0.T, good_indices_1.T]
    # 3. Define a wrapper for gradcheck
    # gradcheck expects a function and a tuple of inputs
    def func(inp, weight1, bias1, weight2, bias2):
        return Scatter.apply(
            inp, weight1, bias1, weight2, bias2, 
            ind1, ind2, ind3, mix_ind, chunk_size
        )

    print("Running gradcheck...")
    # eps: step size for finite difference
    # atol: absolute tolerance
    test_passed = torch.autograd.gradcheck(func, (input_tensor, w1, b1, w2, b2), eps=1e-6, atol=1e-4)
    
    if test_passed:
        print("✅ SUCCESS: Manual backward pass matches numerical gradients!")
    else:
        print("❌ FAILED: Gradient mismatch detected.")

        
def verify_spatial_logic2(args):
    good_indices_0, good_indices_1 = make_cells(args.cells)
    # if args.cells is not None:
    #     cells_from_file = torch.load(args.cells)
    #     good_indices_0 = cells_from_file['cells_face0']
    #     good_indices_1 = cells_from_file['cells_face1']
    ## TODO -- ELSE

    chanmap = make_chanmap(args.chanmap)
    face_plane_wires_channels = make_wire_chans(args.schema, chanmap)

    # Use small dimensions so gradcheck finishes quickly
    # Use double precision (float64) for numerical stability in testing
    F_in, R, C = 1, 1, len(chanmap)
    hidden, F_out = 4, 2
    chunk_size = 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.double

    # 1. Setup inputs with requires_grad=True
    input_tensor = torch.randn(F_in, R, C, device=device, dtype=dtype, requires_grad=True)
    w1 = torch.randn(hidden, F_in * 3, device=device, dtype=dtype, requires_grad=True)
    b1 = torch.randn(hidden, device=device, dtype=dtype, requires_grad=True)
    w2 = torch.randn(F_out, hidden, device=device, dtype=dtype, requires_grad=True)
    b2 = torch.randn(F_out, device=device, dtype=dtype, requires_grad=True)
    
    # 2. Setup Indices (Non-differentiable)
    ind1 = [face_plane_wires_channels[0,0].to(device), face_plane_wires_channels[1,0].to(device)]
    ind2 = [face_plane_wires_channels[0,1].to(device), face_plane_wires_channels[1,1].to(device)]
    ind3 = [face_plane_wires_channels[0,2].to(device), face_plane_wires_channels[1,2].to(device)]
    mix_ind = [good_indices_0.to(device).T, good_indices_1.to(device).T]
    # 3. Define a wrapper for gradcheck
    # gradcheck expects a function and a tuple of inputs
    def func(inp, weight1, bias1, weight2, bias2):
        return Scatter.apply(
            inp, weight1, bias1, weight2, bias2, 
            ind1, ind2, ind3, mix_ind, chunk_size
        )

    print("Running gradcheck...")
    # eps: step size for finite difference
    # atol: absolute tolerance
    test_passed = torch.autograd.gradcheck(func, (input_tensor, w1, b1, w2, b2), eps=1e-6, atol=1e-4)
    
    if test_passed:
        print("✅ SUCCESS: Manual backward pass matches numerical gradients!")
    else:
        print("❌ FAILED: Gradient mismatch detected.")

def make_chanmap(ch):
    return {i:i  for i in range(int(ch))} ##TODO -- CHECK

def make_wire_chans(schema, chanmap):
    #Build the map to go between wire segments & channels 
    store = persist.load(schema)
    face_ids = [0, 1]
    faces = [store.faces[f] for f in face_ids]
    face_to_planes = {}
    for i, face in enumerate(faces):
        face_to_planes[i] = [store.planes[p] for p in face.planes]
    face_plane_wires_channels = {}
    for i, face in enumerate(faces):
        for jj, j in enumerate(face.planes):
            plane = store.planes[j]
            wire_chans = torch.zeros((len(plane.wires), 2), dtype=int)
            for wi in plane.wires:
                wire = store.wires[wi]
                wire_chans[wire.ident, 0] = wire.ident
                wire_chans[wire.ident, 1] = chanmap[wire.channel]
            face_plane_wires_channels[(i,jj)] = torch.tensor(wire_chans, dtype=torch.int)
            print("FPWC size", face_plane_wires_channels[(i,jj)].dtype)
    return face_plane_wires_channels
def make_cells(cells):
    if cells is not None:
            cells_from_file = torch.load(cells)
            good_indices_0 = cells_from_file['cells_face0']
            good_indices_1 = cells_from_file['cells_face1']
    return good_indices_0, good_indices_1

def check_time(args):
    good_indices_0, good_indices_1 = make_cells(args.cells)
    # if args.cells is not None:
    #     cells_from_file = torch.load(args.cells)
    #     good_indices_0 = cells_from_file['cells_face0']
    #     good_indices_1 = cells_from_file['cells_face1']
    ## TODO -- ELSE

    chanmap = make_chanmap(args.chanmap)
    face_plane_wires_channels = make_wire_chans(args.schema, chanmap)

    input = torch.randn((args.fin, args.ticks, len(chanmap)))
    print('made input of size', input.shape)

    tsm = TripleScatterModule(args.fin, args.hidden, args.fout, chunk_size=args.chunk)
    print('Made model')

    print('Checking device')
    device = args.device
    if 'gpu' == device:
        device = 'cuda'
    if 'cuda' in device:
        if torch.cuda.is_available():
            tsm = tsm.to(device)
            print('Sent to', device)
        else:
            print('WARNING CUDA NOT AVAILABLE -- DEFAULTING TO CPU')
    elif device != 'cpu':
        raise RuntimeError('Need to provide either cpu, gpu, or cuda:N to --device')
    

    ind0 = [face_plane_wires_channels[0,0].to(device), face_plane_wires_channels[1,0].to(device)]
    ind1 = [face_plane_wires_channels[0,1].to(device), face_plane_wires_channels[1,1].to(device)]
    ind2 = [face_plane_wires_channels[0,2].to(device), face_plane_wires_channels[1,2].to(device)]
    mix_ind = [good_indices_0.to(device).T, good_indices_1.to(device).T]

    input = input.to(device)
    print('Calling model forward')
    with profiler.profile(with_stack=True, profile_memory=True) as prof:
        y = tsm(input, ind0, ind1, ind2, mix_ind)
    print(prof.key_averages(group_by_stack_n=5).table(sort_by='self_cpu_time_total', row_limit=5))
    print('Done')


    loss = y.sum()
    print('Calling backward')
    with profiler.profile(with_stack=True, profile_memory=True) as prof:
        loss.backward()
    print(prof.key_averages(group_by_stack_n=5).table(sort_by='self_cpu_time_total', row_limit=5))
    print('Done')

    print('Calling model without grad')
    with torch.no_grad():
        with profiler.profile(with_stack=True, profile_memory=True) as prof:
            y = tsm(input, ind0, ind1, ind2, mix_ind)
    print(prof.key_averages(group_by_stack_n=5).table(sort_by='self_cpu_time_total', row_limit=5))
    print('Done')

if __name__ == '__main__':
    parser = ap()
    subparsers = parser.add_subparsers(dest='command')
    verify_parser = subparsers.add_parser('verify')
    
    verify2_parser = subparsers.add_parser('verify2')
    verify2_parser.add_argument('--fin', help='Input features', type=int, default=1)
    verify2_parser.add_argument('--hidden', help='Hidden features', type=int, default=8)
    verify2_parser.add_argument('--fout', help='Output features', type=int, default=1)
    verify2_parser.add_argument('--chunk', help='Chunk size', type=int, default=1)
    verify2_parser.add_argument('--ticks', help='Nticks', type=int, default=10)
    verify2_parser.add_argument('--schema', help='Schema file name', type=str, required=True)
    verify2_parser.add_argument('--chanmap', help='Chan map file OR number of wires for 1:1 map')
    verify2_parser.add_argument('--cells', help='Cells file (optional)', default=None)
    verify2_parser.add_argument('--device', help='Which device to run on', default='cpu')
    
    time_parser = subparsers.add_parser('time')
    time_parser.add_argument('--fin', help='Input features', type=int, default=1)
    time_parser.add_argument('--hidden', help='Hidden features', type=int, default=8)
    time_parser.add_argument('--fout', help='Output features', type=int, default=1)
    time_parser.add_argument('--chunk', help='Chunk size', type=int, default=1)
    time_parser.add_argument('--ticks', help='Nticks', type=int, default=10)
    time_parser.add_argument('--schema', help='Schema file name', type=str, required=True)
    time_parser.add_argument('--chanmap', help='Chan map file OR number of wires for 1:1 map')
    time_parser.add_argument('--cells', help='Cells file (optional)', default=None)
    time_parser.add_argument('--device', help='Which device to run on', default='cpu')
    args = parser.parse_args()

    if args.command == 'time':
        check_time(args)
    elif args.command == 'verify':
        verify_spatial_logic()
    elif args.command == 'verify2':
        verify_spatial_logic2(args)