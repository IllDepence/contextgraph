import re


def url_to_slug(url):
    url_base = 'https://paperswithcode.com/'
    if url_base in url:
        slug = url[len(url_base):]
    elif url[0] == '/':
        # in some contexts PWC data contains URLs
        # w/o protocol and domain
        slug = url[1:]
    else:
        raise
    return slug


def slug_to_pwc_id(slug):
    pwc_id_prefix = 'pwc:'
    return pwc_id_prefix + slug


def url_to_pwc_id(url):
    return slug_to_pwc_id(
        url_to_slug(url)
    )


def name_to_slug(name):
    return re.sub(
        r'[^a-z0-9]',
        '-',
        name.lower()
    )


def canonicalize_arxiv_id(aid):
    """ Add slashes back to old arXiv IDs when they
        have been removed for file name friendliness.
    """

    m = re.match(r'^([a-z\-]+)([0-9\.]+$)', aid)
    if m:
        return m.group(1) + '/' + m.group(2)
    else:
        return aid
