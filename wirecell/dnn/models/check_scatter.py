from wirecell.dnn.models.scatter import TripleScatterModule
from argparse import ArgumentParser as ap
import torch

from wirecell.util.wires import schema, persist

if __name__ == '__main__':
    parser = ap()
    parser.add_argument('--fin', help='Input features', type=int, default=1)
    parser.add_argument('--hidden', help='Hidden features', type=int, default=8)
    parser.add_argument('--fout', help='Output features', type=int, default=1)
    parser.add_argument('--chunk', help='Chunk size', type=int, default=1)
    parser.add_argument('--ticks', help='Nticks', type=int, default=10)
    parser.add_argument('--schema', help='Schema file name', type=str, required=True)
    parser.add_argument('--chanmap', help='Chan map file OR number of wires for 1:1 map')
    parser.add_argument('--cells', help='Cells file (optional)', default=None)
    parser.add_argument('--device', help='Which device to run on', default='cpu')
    args = parser.parse_args()

    if args.cells is not None:
        cells_from_file = torch.load(args.cells)
        good_indices_0 = cells_from_file['cells_face0']
        good_indices_1 = cells_from_file['cells_face1']
    ## TODO -- ELSE


    #Build the map to go between wire segments & channels 
    store = persist.load(args.schema)
    face_ids = [0, 1]
    faces = [store.faces[f] for f in face_ids]
    face_to_planes = {}
    chanmap = {i:i  for i in range(int(args.chanmap))} ##TODO -- CHECK
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


    ind0 = [face_plane_wires_channels[0,0], face_plane_wires_channels[1,0]]
    ind1 = [face_plane_wires_channels[0,1], face_plane_wires_channels[1,1]]
    ind2 = [face_plane_wires_channels[0,2], face_plane_wires_channels[1,2]]
    mix_ind = [good_indices_0.T, good_indices_1.T]

    print('Calling model once')

    tsm(input, ind0, ind1, ind2, mix_ind)