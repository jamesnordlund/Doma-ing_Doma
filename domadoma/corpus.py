from bs4 import BeautifulSoup
import os
import pickle
import shutil
import time
import urllib3


def write_page(http,
               url,
               path,
               prefix):
    """Writes target page to file
    :param http: urll3b PoolManager object for visiting websites
    :param url: target web page to visit
    :param path: directory location for where to store a local copy of the page
    :param prefix: local copies of files are named as <prefix>_#.html where prefix describes the source
    """

    # get the target page
    page = http.request('GET', url).data.decode('utf-8', errors='ignore')

    # ensure output subdirectory exists
    if not os.path.exists(path + '/' + prefix + '/'):
        os.mkdir(path + '/' + prefix + '/')

    # write page to output subdirectory
    with open(path + '/' + prefix + '/' + prefix + '_' + str(len(os.listdir(path + '/' + prefix))) + '.html',
              'w', encoding='utf-8') as f:
        f.write(page)


def scrape_doma(out_path='',
                refresh_press=False,
                refresh_resources=False,
                refresh_blog=False,
                refresh_all=False):
    """Scrapes writing from doma.com for analysis.  Hard-codes a very polite time.sleep() window between pages.
    :param out_path: path to directory that stores data
    :param refresh_press: True if re-indexing https://www.doma.com/press/press-releases/
    :param refresh_resources: True if re-indexing https://www.doma.com/resources-for-lenders/
    :param refresh_blog: True if re-indexing https://www.doma.com/category/writing-on-the-wall/
    :param refresh_all: True if re-indexing news, press, resources, blog, investors.  Supersedes other flags.
    """

    # define target URLs
    targets = {
        'press': ['https://www.doma.com/press/press-releases/', refresh_press],
        'resources': ['https://www.doma.com/resources-for-lenders/', refresh_resources],
        'blog': ['https://www.doma.com/category/writing-on-the-wall/', refresh_blog]
    }
    if refresh_all:
        targets = {k: [v[0], True] for k, v in targets.items()}

    # construct page reader object
    http = urllib3.PoolManager()

    if targets['press'][1]:

        # reload previously scanned news
        old_press = None
        if os.path.exists(out_path + '/press.p'):
            with open(out_path + '/press.p', 'rb') as fp:
                old_press = pickle.load(fp)

        # find new news
        a_tags = BeautifulSoup(http.request('GET', targets['press'][0]).data,
                               features="html.parser").find_all('a', {'class': 'invest'})
        urls = {a.get('href') for a in a_tags}
        with open(out_path + '/press.p', 'wb') as fp:
            pickle.dump(urls, fp)

        # keep only new items
        if old_press is not None:
            urls -= old_press

        # download new items
        time.sleep(1)
        for url in urls:
            write_page(http=http, url=url, path=out_path, prefix='press')
            time.sleep(1)

    if targets['resources'][1]:

        # reload previously scanned resources
        old_resources = None
        if os.path.exists(out_path + '/resources.p'):
            with open(out_path + '/resources.p', 'rb') as fp:
                old_resources = pickle.load(fp)

        # find new resources
        page = BeautifulSoup(http.request('GET', targets['resources'][0]).data, features="html.parser")
        more_links = [None] + [a.get('href') for a in page.find_all('a', {'class': 'page-numbers'})]
        all_urls = set()
        for i in range(len(more_links)):
            if i > 0:  # current page is already loaded, no need to hit the server for it again
                time.sleep(1)
                page = BeautifulSoup(http.request('GET', more_links[i]).data, features="html.parser")
            urls = [a.get('href') for a in page.find_all('a', {'class': 'doma-post-item__title-link'})]
            time.sleep(1)
            for url in urls:
                all_urls |= {url}
                if old_resources is not None and url in old_resources:
                    continue
                if i == 0:
                    continue
                write_page(http=http, url=url, path=out_path, prefix='resources')
                time.sleep(1)

        # url storage works differently here because resources are on multiple pages
        with open(out_path + '/resources.p', 'wb') as fp:
            pickle.dump(all_urls, fp)

    if targets['blog'][1]:

        # reload previously scanned resources
        old_blog = None
        if os.path.exists(out_path + '/blog.p'):
            with open(out_path + '/blog.p', 'rb') as fp:
                old_blog = pickle.load(fp)

        # find new resources
        page = BeautifulSoup(http.request('GET', targets['blog'][0]).data, features="html.parser")
        more_links = [None] + [a.get('href') for a in page.find_all('a', {'class': 'page-numbers'})]
        all_urls = set()
        for i in range(len(more_links)):
            if i > 0:  # current page is already loaded, no need to hit the server for it again
                time.sleep(1)
                page = BeautifulSoup(http.request('GET', more_links[i]).data, features="html.parser")
            urls = [a.get('href') for a in page.find_all('a', {'class': 'doma-post-item__title-link'})]
            time.sleep(1)
            for url in urls:
                all_urls |= {url}
                if old_blog is not None and url in old_blog:
                    continue
                write_page(http=http, url=url, path=out_path, prefix='blog')
                time.sleep(1)

        # url storage works differently here because resources are on multiple pages
        with open(out_path + '/blog.p', 'wb') as fp:
            pickle.dump(all_urls, fp)


def read_doma(out_path=''):
    """Reads scraped HTML files and outputs them to a text corpus.  Subdirectories appear under out_path."""

    # ensure output subdirectory exists
    if not os.path.exists(out_path + '/corpus/'):
        os.mkdir(out_path + '/corpus/')
    existing_corpus = os.listdir(out_path + '/corpus/')

    # because this is such a small project, all erroneous files are dumped to a single directory for inspection
    if not os.path.exists(out_path + '/errors/'):
        os.mkdir(out_path + '/errors/')

    # read blog posts
    for source in ['blog', 'press', 'resources']:
        if os.path.exists(out_path + '/' + source):
            f_names = os.listdir(out_path + '/' + source)
            new_files, wrote_files = 0, 0
            for name in f_names:

                # do not re-parse files already dumped to the corpus
                if name.replace('.html', '.txt') in existing_corpus:
                    continue
                new_files += 1

                # load the stored html file into bs4
                with open(out_path + '/' + source + '/' + name, 'rb') as f:
                    soup = BeautifulSoup(f, 'html.parser')

                # extract text
                try:
                    text = soup.find('div', {'class': 'entry-content blog-post__content'}).get_text()
                    text = '  '.join([line.strip() for line in text.split('\n')
                                      if len(line.strip()) > 0 and line.strip()[-1] in ['.', '?', '!']])  # drop headers
                except Exception as e:  # dump all erroneous files because the project is small
                    shutil.copyfile(out_path + '/' + source + '/' + name, out_path + '/errors/' + name)
                    continue

                # write to file
                out_name = out_path + '/corpus/' + name.replace('.html', '.txt')
                with open(out_name, 'w', encoding='utf8') as f:
                    f.write(text)
                wrote_files += 1

            print('wrote {}% of new {} data'.format(round(wrote_files/new_files*100, 2), source))

    return


def build_corpus(out_path=''):

    # scrape content from Doma's website
    scrape_doma(out_path, refresh_all=True)

    # read the scraped content
    read_doma(out_path)
