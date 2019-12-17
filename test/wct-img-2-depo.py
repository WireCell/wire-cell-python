#!/usr/bin/env python
import sys, os, glob

def main(filepattern):
    if (os.path.exists('depo/0')):
        print('found old depo, removing ...')
        os.system('rm -rf depo')
    os.system('mkdir -p depo/0')
    cmd = 'wirecell-img json-depos -s center -n 5000 -o depo/0/0-test.json %s' % (
        filepattern, )
    print(cmd)
    os.system(cmd)

if __name__ == "__main__":
    if (len(sys.argv)!=2):
        print("usage: python wct-img-2-depo.py 'filepattern'")
    else:
        main(sys.argv[1])
