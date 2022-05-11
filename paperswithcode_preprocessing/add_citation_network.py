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
pprs_orig_fn = 'papers-with-abstracts.json'
out_dir = '/home/ls3data/datasets/paperswithcode/preprocessed/'
cit_fn = 'papers_to_papers.csv'
unarXive_db = '/opt/unarXive/unarXive-2020/papers/refs.db'

# determine all relevant arXiv IDs
with open(os.path.join(in_dir, pprs_orig_fn)) as f:
    pprs = json.load(f)

pwc_arxiv_ids = set()
for ppr in pprs:
    if ppr['arxiv_id'] is not None:
        pwc_arxiv_ids.add(ppr['arxiv_id'])

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
            uuid_citmarker
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
        'citation_marker_uuid'
    ])
    for edge in citation_edges:
        csv_writer.writerow(edge)
