def url_to_slug(url):
    url_base = 'https://paperswithcode.com/'
    slug = url[len(url_base):]
    return slug
