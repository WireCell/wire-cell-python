import crossover
import schema_load
from argparse import ArgumentParser as ap

if __name__ == '__main__':
    parser = ap()
    parser.add_argument('--wires', '-w', help='Wire file', type=str)
    parser.add_argument('--file', '-f', help='Input file', type=str)
    args = parser.parse_args()

    #Load wires
    store = schema_load.StoreDB()
    schema_load.load_file(args.wires, store)

    #Load file