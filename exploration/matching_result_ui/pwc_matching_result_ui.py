import io
import os
import json
import pprint
import re
import sqlite3
import matplotlib.pyplot as plt
from flask import Flask, render_template
from markupsafe import escape

app = Flask(__name__)
base_path = '/opt/unarXive/unarXive-2020_pwc_annot_wtriples'
db_path = '/opt/unarXive/unarXive-2020/papers/refs.db'
cite_patt = re.compile(
    (r'\{\{cite:([0-9A-F]{8}-[0-9A-F]{4}-4[0-9A-F]{3}'
     r'-[89AB][0-9A-F]{3}-[0-9A-F]{12})\}\}'),
    re.I
)


def get_full_text_html(arxiv_id):
    raw_text = get_full_text(arxiv_id)
    html = raw_text.replace('\n', '<br>')
    annotation_types = [
        'method_used',
        'dataset_used',
        'task_used',
        'method_named',
        'dataset_named',
        'task_named'
    ]
    html = prettify_cit_markers(html)
    for tag_name in annotation_types:
        html = html.replace(f'<{tag_name}>', f'<span class="{tag_name}">')
        html = html.replace(f'</{tag_name}>', f'</span>')

    return html


def get_refitem_info(refitem_uuid):
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    row = cur.execute(
        'SELECT * FROM bibitem where uuid=?',
        (refitem_uuid,)
    ).fetchone()
    if row is None:
        return None
    refitem_uuid, _, cd_mid, _, cd_aid, refstr = row
    return [refstr, cd_aid, cd_mid]


def prettify_cit_markers(html):
    newhtml = ''
    start = 0
    for m in cite_patt.finditer(html):
        end, newstart = m.span()
        newhtml += html[start:end]
        refitem_uuid = m.group(1)
        refitem = get_refitem_info(refitem_uuid)
        if refitem is None or (refitem[1] is None and refitem[2] is None):
            rep_inner = '<span title="{}">[CIT]</span>'.format(
                refitem[0]
            )
        elif refitem[1] is not None:  # has cited arXiv ID
            rep_inner = ('<a href="https://arxiv.org/abs/{}"'
                         'title="{}">[CIT]</a>').format(
                refitem[1],
                refitem[0]
            )
        elif refitem[2] is not None:  # has cited MAG ID
            rep_inner = ('<a href="https://api.semanticscholar.org/v1/'
                         'paper/MAG:{}" title="{}">[CIT]</a>').format(
                refitem[2],
                refitem[0]
            )
        rep = '<span class="cit_marker">' + rep_inner + '</span>'
        newhtml += rep
        start = newstart
    newhtml += html[start:]
    return newhtml


def get_full_text(arxiv_id):
    fpath_txt = os.path.join(base_path, arxiv_id) + '.txt'
    with open(fpath_txt) as f:
        full_text = f.read()
    return full_text


def get_debug_info(arxiv_id):
    fpath_dbg = os.path.join(base_path, arxiv_id) + '_debug.json'
    with open(fpath_dbg) as f:
        dbg_info = json.load(f)
    return dbg_info


def get_relations(arxiv_id):
    fpath_rel = os.path.join(base_path, arxiv_id) + '_rels.tsv'
    rels = []
    with open(fpath_rel) as f:
        for line in f:
            rel = line.split('\t')
            rel[2] = rel[2].strip()
            rels.append(rel)
    return rels


@app.route('/')
def index():
    arxiv_ids = [
        fn[:-9]
        for fn in os.listdir(base_path)
        if fn[-9:] == '_rels.tsv'
    ]
    debug_info = {}
    for arxiv_id in arxiv_ids:
        debug_info[arxiv_id] = get_debug_info(arxiv_id)
    debug_tuples_by_triple_num = sorted(
        ((aid, dbg) for aid, dbg in debug_info.items()),
        key=lambda tpl: len(tpl[1]['triples']),
        reverse=True
    )
    trpl_counts = [
        len(tpl[1]['triples']) for tpl in debug_tuples_by_triple_num
    ]
    num_bins = len(set(trpl_counts))
    fig = plt.figure()
    plt.yscale('log', nonposy='clip')
    plt.hist(trpl_counts, bins=num_bins)
    plt.xlabel('#triples')
    plt.ylabel('#papers (log scale)')
    imgdata = io.StringIO()
    fig.savefig(imgdata, format='svg')
    imgdata.seek(0)  # rewind the data
    trpl_plot_svg = imgdata.read()  # this is svg data

    return render_template(
        'index.html',
        trpl_plot_svg=trpl_plot_svg,
        tuples_by_triple_num=debug_tuples_by_triple_num[:75]
    )


@app.route('/paper/<arxiv_id>')
def paper(arxiv_id):
    debug_info = get_debug_info(arxiv_id)
    full_text_html = get_full_text_html(arxiv_id)
    return render_template(
        'paper.html',
        arxiv_id=arxiv_id,
        triples=debug_info['triples'],
        debug_str=pprint.pformat(debug_info, indent=2),
        full_text=full_text_html
    )


@app.route('/raw/<arxiv_id>')
def raw(arxiv_id):
    full_text = get_full_text(arxiv_id)
    return '<pre>' + escape(full_text) + '</pre>'
