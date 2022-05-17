""" From
        papers-with-abstracts.json
    and
        the unarXive references data base (refs.db)
    generate
        papers_to_papers.csv
"""

import csv
import json
import os
import sqlite3
from util import canonicalize_arxiv_id

in_dir = '/home/ls3data/datasets/paperswithcode/'
pprs_preprocessed_fn = 'papers.jsonl'
out_dir = '/home/ls3data/datasets/paperswithcode/preprocessed/'
cit_fn = 'papers_to_papers.csv'
unarXive_db = '/opt/unarXive/unarXive-2020/papers/refs.db'

# determine all relevant arXiv IDs
# and build a mapping from arXiv IDs to PWC IDs
arxiv_id_to_pwc_id = dict()
with open(os.path.join(out_dir, pprs_preprocessed_fn)) as f:
    for line in f:
        ppr = json.loads(line)
        if ppr['arxiv_id'] is not None:
            arxiv_id_to_pwc_id[ppr['arxiv_id']] = ppr['id']
pwc_arxiv_ids = set(arxiv_id_to_pwc_id.keys())

# fetch citations from unarXive inside relevant arXiv IDs
db_con = sqlite3.connect(unarXive_db)
db_cur = db_con.cursor()
db_cur.execute('''
    select
        citing_arxiv_id, cited_arxiv_id, uuid
    from
        bibitem
    where
        citing_arxiv_id not null and cited_arxiv_id not null
''')
citation_edges = []
for row in db_cur:
    aid_citing = canonicalize_arxiv_id(row[0])
    aid_cited = canonicalize_arxiv_id(row[1])
    uuid_citmarker = row[2]

    if aid_citing in pwc_arxiv_ids and aid_cited in pwc_arxiv_ids:
        citation_edges.append([
            aid_citing,
            aid_cited,
            uuid_citmarker,
            arxiv_id_to_pwc_id[aid_citing],
            arxiv_id_to_pwc_id[aid_cited]
        ])

# write to file
with open(os.path.join(out_dir, cit_fn), 'w') as f:
    csv_writer = csv.writer(
        f,
        delimiter=',',
        quoting=csv.QUOTE_NONE
    )
    csv_writer.writerow([
        'citing_arxiv_id',
        'cited_arxiv_id',
        'citation_marker_uuid',
        'citing_pwc_id',
        'cited_pwc_id'
    ])
    for edge in citation_edges:
        csv_writer.writerow(edge)
