#!/bin/env python

# Requires jq python library: https://pypi.org/project/jq/
# pip install jq

import sys, os, pprint
import jq, json
import click
pp = pprint.PrettyPrinter(indent=1, width=100, compact=True)

def overview():
    return 'sort_by(.type) | .[] | .type + ":" + .name'

def find_type(t):
    return '.[] | select(.type == "%s") | {".name": .name} + .data' % t

def find_instance(t, n):
    return f'.[] | select(.type == "{t}" and .name == "{n}") | .data'


@click.group(help='Helper commands to inspect wire-cell configurations')
def cli():
    pass


@cli.command(name='json', 
    help='''show JSON file content based on input [pattern]

    [pattern] can be either "type" or "type:name" of the component.
    If [pattern] is omitted, overview of all components will be shown.
    ''')
@click.argument('filename', type=click.Path(exists=True))
@click.argument('pattern', default='')
def json_search(filename, pattern):
    if (pattern == ''):
        jq_string = overview()
    else:
        if(':' in pattern):
            t, n = pattern.split(':')
            if (n==''):
                jq_string = find_type(t)
            else:
                jq_string = find_instance(t, n)
        else:
            jq_string = find_type(pattern)
    
    click.echo(click.style('JQ query', fg='green') + ': ' + jq_string)
    click.echo('-'*30)
    with open(filename) as f:
        json_input = json.load(f)
        # for one in iter(jq.compile(jq_string).input(json_input)):
        #     pp.pprint(one)
        #     click.echo('-'*30)

        all = jq.compile(jq_string).input(json_input).all()
        pp.pprint(all)
    click.echo('-'*30)


@cli.command(help='show $WIRECELL_PATH')
@click.argument('string', envvar='WIRECELL_PATH')
def path(string):
    pp.pprint(string.split(':'))


@cli.command(help='convert jsonnet to json and pdf')
@click.argument('filename', type=click.Path(exists=True))
@click.option('-o', '--output', default='wirecell.json', help='output json file name')
@click.option('-V', '--ext', default='', help='external variables: A=a:B=b')
def convert(filename, output, ext):
    if (not filename.endswith('.jsonnet')):
        click.secho('Error: File must end with .jsonnet', fg='red')
        return
    
    ext_variables = {
        'reality': 'data',
        'raw_input_label': 'daq',
        'tpc_volume_label': '1',
        'epoch': 'dynamic',
        'signal_output_form': 'sparse'
    }
    ext_from_cmd = []
    if (':' in ext):
        ext_from_cmd = ext.split(':')
    elif (ext != ''):
        ext_from_cmd = [ext]
    for item in ext_from_cmd:
        key, value = item.split('=')
        ext_variables.update({key: value})
    
    cmd = f'JSONNET_PATH=$WIRECELL_PATH jsonnet {filename} '
    for (key, value) in ext_variables.items():
        cmd += f'-V {key}={value} '
    
    cmd += f'-o {output}'
    
    click.echo(click.style('jonnet->json:\n', fg='green') + cmd)
    os.system(cmd)

    pdf = output.replace('.json', '.pdf')
    cmd = f'wirecell-pgraph dotify --jpath -1 --no-params {output} {pdf}'
    click.echo(click.style('json->pdf:\n', fg='green') + cmd)
    os.system(cmd)

'''
Examples:
./inspectwc.py --help
./inspectwc.py path
./inspectwc.py convert wirecell.jsonnet
./inspectwc.py json wirecell.json
./inspectwc.py json wirecell.json LfFilter
./inspectwc.py json wirecell.json OmnibusSigProc:anode121sigproc121
'''
if __name__ == '__main__':
    cli()