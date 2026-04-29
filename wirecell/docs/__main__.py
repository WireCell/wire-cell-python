#!/usr/bin/env python3

import ast
import functools
import hashlib
import json
import re
import subprocess
import sys
from datetime import date
from pathlib import Path

import click
from wirecell.util.cli import context, log


# ── repo / package roots ──────────────────────────────────────────────────────

@functools.lru_cache(maxsize=1)
def _repo_root():
    r = subprocess.run(
        ['git', 'rev-parse', '--show-toplevel'],
        capture_output=True, text=True
    )
    if r.returncode != 0:
        raise click.ClickException('Not inside a git repository.')
    return Path(r.stdout.strip())


def _pkg_root():
    """The wirecell/ package directory (parent of docs/)."""
    return Path(__file__).parent.parent


# ── git / hash helpers ────────────────────────────────────────────────────────

def _git_ls(path):
    """All git-tracked files under path (relative to repo root)."""
    r = subprocess.run(
        ['git', 'ls-files', '--', str(path)],
        capture_output=True, text=True,
        cwd=_repo_root()
    )
    return r.stdout.splitlines()


def _direct_py_files(dirpath):
    """Git-tracked .py files that live directly in dirpath (not in subdirs)."""
    rr = _repo_root()
    result = []
    for f in _git_ls(dirpath):
        p = rr / f
        if p.suffix == '.py' and p.parent == dirpath:
            result.append(p)
    return sorted(result)


def _hash_strings(strings):
    """SHA-256 (first 16 hex chars) of newline-joined strings, or None if empty."""
    if not strings:
        return None
    return hashlib.sha256('\n'.join(strings).encode()).hexdigest()[:16]


def _source_hash(dirpath):
    """Hash of git object hashes for direct .py files in dirpath."""
    py_files = _direct_py_files(dirpath)
    if not py_files:
        return None
    r = subprocess.run(
        ['git', 'hash-object', '--'] + [str(p) for p in py_files],
        capture_output=True, text=True,
        cwd=_repo_root()
    )
    return _hash_strings(r.stdout.splitlines())


# ── frontmatter ───────────────────────────────────────────────────────────────

def _parse_frontmatter(path):
    """Return (meta_dict, body_str) splitting simple YAML frontmatter."""
    text = path.read_text()
    if not text.startswith('---\n'):
        return {}, text
    end = text.find('\n---\n', 4)
    if end < 0:
        return {}, text
    meta = {}
    for line in text[4:end].splitlines():
        if ':' in line:
            k, _, v = line.partition(':')
            meta[k.strip()] = v.strip()
    return meta, text[end + 5:]


# ── directory discovery ────────────────────────────────────────────────────────

def _find_module_dirs(pkg_root=None):
    """
    All dirs under pkg_root that directly contain git-tracked .py files.
    Returned deepest-first (bottom-up), pkg_root directory last.
    """
    root = pkg_root or _pkg_root()
    rr = _repo_root()
    dirs = set()
    for f in _git_ls(root):
        p = rr / f
        if p.suffix == '.py':
            dirs.add(p.parent)
    return sorted(dirs, key=lambda p: (-len(p.parts), str(p)))


def _child_indexes(dirpath):
    """Existing index.md files in immediate subdirectories of dirpath."""
    result = []
    if not dirpath.is_dir():
        return result
    for child in sorted(dirpath.iterdir()):
        idx = child / 'index.md'
        if child.is_dir() and idx.exists():
            result.append(idx)
    return result


def _children_hash(dirpath):
    """Hash of body text (frontmatter excluded) of immediate children's index.md files."""
    indexes = _child_indexes(dirpath)
    if not indexes:
        return None
    bodies = [_parse_frontmatter(idx)[1] for idx in indexes]
    return _hash_strings(bodies)


# ── staleness check ───────────────────────────────────────────────────────────

def _check_one(dirpath):
    """Return ('OK'|'STALE'|'MISSING', reasons_list)."""
    index = dirpath / 'index.md'
    if not index.exists():
        return 'MISSING', []

    meta, _ = _parse_frontmatter(index)
    reasons = []

    sh = _source_hash(dirpath)
    if sh is not None and meta.get('source-hash') != sh:
        stored = meta.get('source-hash', 'none')
        reasons.append(f'source {stored!r}→{sh!r}')

    ch = _children_hash(dirpath)
    if ch is not None and meta.get('children-hash') != ch:
        stored = meta.get('children-hash', 'none')
        reasons.append(f'children {stored!r}→{ch!r}')

    return ('STALE', reasons) if reasons else ('OK', [])


# ── CLI extraction ────────────────────────────────────────────────────────────

def _extract_cli_commands(dirpath):
    """Return [(name, first_doc_line), ...] parsed from __main__.py via AST."""
    main_py = dirpath / '__main__.py'
    if not main_py.exists():
        return []
    try:
        tree = ast.parse(main_py.read_text())
    except SyntaxError:
        return []
    cmds = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        for dec in node.decorator_list:
            if (isinstance(dec, ast.Call)
                    and isinstance(dec.func, ast.Attribute)
                    and dec.func.attr == 'command'
                    and dec.args
                    and isinstance(dec.args[0], ast.Constant)):
                name = dec.args[0].value
                doc = ast.get_docstring(node) or ''
                first_line = doc.split('\n')[0].strip()
                cmds.append((name, first_line))
                break
    return cmds


# ── prompt building blocks ────────────────────────────────────────────────────

_RULES = """\
OUTPUT ONLY the raw Markdown file content.
No explanation. No surrounding code fences. No preamble.
The text will be saved verbatim as the index.md file."""

_FORMAT = """\
=== OUTPUT FORMAT ===
Structure the file exactly as:
  1. YAML frontmatter block — copy the hash values shown above verbatim
  2. # <module-path>       H1 heading using the relative path shown above
  3. 2-3 sentence overview of what this module/sub-package provides
  4. ## Modules            Markdown table: Module | Purpose | Key Symbols
  5. ## CLI Commands       Markdown table: Command | Description
                           (omit this section entirely if no __main__.py exists)
  6. ## Dependencies       notable wirecell.* imports consumed by this module
Keep total length under 200 lines.  Be concise and precise."""


def _frontmatter_str(dirpath):
    today = date.today().isoformat()
    sh = _source_hash(dirpath) or 'n/a'
    ch = _children_hash(dirpath)
    lines = ['---', f'generated: {today}', f'source-hash: {sh}']
    if ch:
        lines.append(f'children-hash: {ch}')
    lines.append('---')
    return '\n'.join(lines)


def _source_section(dirpath):
    """Prompt section containing direct .py source files + extracted CLI table."""
    rr = _repo_root()
    py_files = _direct_py_files(dirpath)
    if not py_files:
        return ''
    parts = ['=== SOURCE FILES ===']
    for p in py_files:
        parts.append(f'\n--- {p.relative_to(rr)} ---\n{p.read_text()}')
    cmds = _extract_cli_commands(dirpath)
    if cmds:
        parts.append('\n=== CLI COMMANDS (auto-extracted from __main__.py) ===')
        for name, doc in cmds:
            parts.append(f'  {name:<36} {doc}')
    return '\n'.join(parts)


def _children_section(dirpath):
    """Prompt section containing existing child index.md files."""
    rr = _repo_root()
    indexes = _child_indexes(dirpath)
    if not indexes:
        return ''
    parts = ['=== CHILD MODULE SUMMARIES (already-generated index.md files) ===']
    for idx in indexes:
        parts.append(f'\n--- {idx.relative_to(rr)} ---\n{idx.read_text()}')
    return '\n'.join(parts)


# ── option A: single-directory prompt ────────────────────────────────────────

def _single_prompt(dirpath):
    rr = _repo_root()
    rel = dirpath.relative_to(rr)
    parts = [
        f'Generate index.md for Python module: {rel}/',
        '',
        _RULES,
        f'Save path: {rel}/index.md',
        '',
        '=== REQUIRED FRONTMATTER (copy verbatim into your output) ===',
        _frontmatter_str(dirpath),
    ]
    src = _source_section(dirpath)
    if src:
        parts += ['', src]
    chi = _children_section(dirpath)
    if chi:
        parts += ['', chi]
    if not src and not chi:
        parts += ['', '(No source files or child summaries found for this directory.)']
    parts += ['', _FORMAT]
    return '\n'.join(parts)


# ── option B: combined prompt for all stale dirs ──────────────────────────────

def _combined_prompt(dirs):
    rr = _repo_root()
    sep = '=' * 72
    header = [
        'Generate index.md files for the wire-cell-python modules listed below.',
        '',
        'Work through each TASK in the listed order — they are sorted bottom-up',
        'so that leaf modules come before their parents.',
        '',
        'For EACH task emit exactly:',
        '  === FILE: <relative/path/to/index.md> ===',
        '  <raw markdown content>',
        '  === END FILE ===',
        '',
        'Nothing else between or outside the FILE blocks.',
        '',
        'For tasks whose child module summaries are absent below: use the FILE',
        'blocks you already generated in earlier tasks of this prompt.',
        '',
        _RULES,
        sep,
    ]
    tasks = []
    for i, dirpath in enumerate(dirs, 1):
        rel = dirpath.relative_to(rr)
        block = [
            f'TASK {i}: {rel}/index.md',
            sep,
            '',
            '=== REQUIRED FRONTMATTER ===',
            _frontmatter_str(dirpath),
        ]
        src = _source_section(dirpath)
        if src:
            block += ['', src]
        chi = _children_section(dirpath)
        if chi:
            block += ['', chi]
        block += ['', _FORMAT, '', sep]
        tasks.append('\n'.join(block))
    return '\n'.join(header) + '\n\n' + '\n\n'.join(tasks)


# ── option C: shell script piping per-dir prompts to $WCPY_LLM ───────────────

def _shell_script(dirs):
    rr = _repo_root()
    lines = [
        '#!/bin/sh',
        '# Generated by: wcpy docs prompt --script',
        '#',
        '# Set WCPY_LLM to any CLI command that reads a prompt on stdin and',
        '# writes the LLM response to stdout.  Examples:',
        '#   export WCPY_LLM="claude -p"',
        '#   export WCPY_LLM="llm prompt"',
        '#   export WCPY_LLM="ollama run mistral"',
        '',
        ': "${WCPY_LLM:?Please set WCPY_LLM to your LLM CLI command}"',
        '',
    ]
    for dirpath in dirs:
        rel = dirpath.relative_to(rr)
        out = dirpath / 'index.md'
        lines += [
            f'printf "\\n=== Generating {rel}/index.md ===\\n"',
            f'wcpy docs prompt "{rel}" | $WCPY_LLM > "{out}"',
            f'printf "  saved: {out}\\n"',
            '',
        ]
    lines.append('printf "\\nAll done.\\n"')
    return '\n'.join(lines)


# ── CLI ───────────────────────────────────────────────────────────────────────

@context("docs")
def cli(ctx):
    """Commands for generating and checking package documentation index files."""
    pass


@cli.command("check")
@click.argument("paths", nargs=-1, metavar="PATH|all")
@click.option("--porcelain", is_flag=True,
              help="Machine-readable output: STATUS<TAB>PATH per line")
@click.pass_context
def check_cmd(ctx, paths, porcelain):
    """Show which index.md files are missing or stale.

    \b
    PATH arguments are relative to the repo root (e.g. wirecell/util).
    Use the special value 'all' to check every module directory.

    Exits with status 1 if any file needs regeneration.
    """
    if not paths:
        click.echo(ctx.get_help())
        ctx.exit()

    rr = _repo_root()
    pkg_root = _pkg_root()

    if len(paths) == 1 and paths[0] == 'all':
        dirs = _find_module_dirs(pkg_root)
    else:
        dirs = []
        for p in paths:
            dirpath = Path(p)
            if not dirpath.is_absolute():
                dirpath = rr / dirpath
            if not dirpath.is_dir():
                raise click.BadParameter(f'not a directory: {p}', param_hint='PATH')
            dirs.append(dirpath)

    any_bad = False
    for d in dirs:
        status, reasons = _check_one(d)
        rel = d.relative_to(rr)
        if porcelain:
            click.echo(f'{status}\t{rel}')
        else:
            clr = {'OK': 'green', 'STALE': 'yellow', 'MISSING': 'red'}[status]
            label = click.style(f'{status:<8}', fg=clr, bold=True)
            detail = f'  ({"; ".join(reasons)})' if reasons else ''
            click.echo(f'{label} {rel}{detail}')
        if status != 'OK':
            any_bad = True
    sys.exit(1 if any_bad else 0)


@cli.command("prompt")
@click.argument("path", required=False, default=None, metavar="[PATH]")
@click.option("--monolith", is_flag=True,
              help="Emit one combined prompt for all stale/missing dirs  [option B]")
@click.option("--script", is_flag=True,
              help="Emit a shell script piping per-dir prompts to $WCPY_LLM  [option C]")
@click.option("--force", is_flag=True,
              help="Include up-to-date dirs as well as stale/missing ones")
@click.option("-o", "--output", default="-", metavar="FILE",
              help="Write output to FILE instead of stdout")
@click.pass_context
def prompt_cmd(ctx, path, monolith, script, force, output):
    """Emit an LLM prompt (or shell script) to regenerate stale index.md files.

    \b
    Three modes:
      A  wcpy docs prompt wirecell/util   single-directory prompt
      B  wcpy docs prompt --monolith      one combined prompt for all stale dirs
      C  wcpy docs prompt --script        shell script using $WCPY_LLM per dir
    """
    if not path and not monolith and not script:
        click.echo(ctx.get_help())
        ctx.exit()

    pkg_root = _pkg_root()
    rr = _repo_root()

    if path:
        # Option A — explicit single directory
        p = Path(path)
        dirpath = p if p.is_absolute() else (rr / p)
        if not dirpath.is_dir():
            raise click.BadParameter(f'not a directory: {path}', param_hint='PATH')
        text = _single_prompt(dirpath)
    else:
        # Collect stale/missing dirs in bottom-up order
        all_dirs = _find_module_dirs(pkg_root)
        dirs = all_dirs if force else [d for d in all_dirs if _check_one(d)[0] != 'OK']
        if not dirs:
            click.echo('All index.md files are up to date.', err=True)
            return
        # Option C or B
        text = _shell_script(dirs) if script else _combined_prompt(dirs)

    if output == '-':
        click.echo(text)
    else:
        out = Path(output)
        out.write_text(text)
        click.echo(f'wrote {out} ({len(text.splitlines())} lines)', err=True)


@cli.command("apply")
@click.argument("infile", required=False, default="-", metavar="[FILE]")
@click.option("--dry-run", is_flag=True,
              help="Show what would be written without touching any files")
def apply_cmd(infile, dry_run):
    """Parse combined LLM output (option B) and write each index.md file.

    Reads from FILE or stdin.  Expects FILE blocks in the format produced by
    'wcpy docs prompt' (no PATH, no --script):

    \b
        === FILE: relative/path/index.md ===
        <raw markdown content>
        === END FILE ===
    """
    rr = _repo_root()
    text = sys.stdin.read() if infile == '-' else Path(infile).read_text()

    pattern = re.compile(
        r'^=== FILE: (.+?) ===\n(.*?)^=== END FILE ===',
        re.MULTILINE | re.DOTALL
    )
    found = pattern.findall(text)
    if not found:
        raise click.ClickException(
            "No FILE blocks found.  "
            "Expected '=== FILE: path ===' / '=== END FILE ===' delimiters."
        )

    for rel_path, content in found:
        out = rr / rel_path.strip()
        if dry_run:
            click.echo(f'would write  {out}  ({len(content.splitlines())} lines)')
        else:
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(content)
            click.echo(f'wrote  {out}')

    if dry_run:
        click.echo(f'\n(dry run — {len(found)} file(s) not written)')


def _show_one(index_path, as_json):
    """Display a single index.md file."""
    meta, body = _parse_frontmatter(index_path)

    if as_json:
        click.echo(json.dumps({'meta': meta, 'body': body}, indent=2))
        return

    for line in body.splitlines():
        if line.startswith('# '):
            click.echo(click.style(line[2:], bold=True, fg='cyan'))
        elif line.startswith('## '):
            click.echo(click.style(line[3:], bold=True, fg='yellow'))
        elif line.startswith('### '):
            click.echo(click.style(line[4:], bold=True))
        elif line.startswith('|'):
            click.echo(click.style(line, fg='white'))
        else:
            click.echo(line)

    if meta:
        click.echo()
        click.echo(click.style('─' * 40, dim=True))
        for k, v in meta.items():
            click.echo(click.style(f'{k}: {v}', dim=True))


@cli.command("show")
@click.argument("paths", nargs=-1, metavar="PATH|all|top")
@click.option("--json", "as_json", is_flag=True,
              help="Emit JSON with 'meta' and 'body' keys instead of formatted text")
@click.pass_context
def show_cmd(ctx, paths, as_json):
    """Display one or more index.md files with terminal formatting (or as JSON).

    \b
    PATH may be a directory (reads its index.md) or a direct path to an index.md.
    Special values:
      top   the package-root index.md
      all   every existing index.md under the package root
    """
    if not paths:
        click.echo(ctx.get_help())
        ctx.exit()

    pkg_root = _pkg_root()
    rr = _repo_root()

    index_paths = []
    for p in paths:
        if p == 'top':
            index_paths.append(pkg_root / 'index.md')
        elif p == 'all':
            for d in _find_module_dirs(pkg_root):
                idx = d / 'index.md'
                if idx.exists():
                    index_paths.append(idx)
        else:
            pp = Path(p)
            ip = (pp / 'index.md') if pp.is_dir() else pp
            index_paths.append(ip)

    for index_path in index_paths:
        if not index_path.exists():
            raise click.ClickException(f'not found: {index_path}')
        _show_one(index_path, as_json)


def main():
    cli(obj=dict())


if '__main__' == __name__:
    main()
