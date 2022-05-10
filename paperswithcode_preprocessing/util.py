def url_to_slug(url):
    url_base = 'https://paperswithcode.com/'
    slug = url[len(url_base):]
    return slug


def slug_to_pwc_id(slug):
    pwc_id_prefix = 'pwc:'
    return pwc_id_prefix + slug


def url_to_pwc_id(url):
    return slug_to_pwc_id(
        url_to_slug(url)
    )
