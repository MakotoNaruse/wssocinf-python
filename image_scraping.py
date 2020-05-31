# -*- encoding: utf-8 -*-

"""
Usage:
    query_bing_images.py <query>
    query_bing_images.py -h | --help
    query_bing_images.py --version

Options:
    -h --help  show this screen
    --version  show version
"""

from bs4 import BeautifulSoup
import requests
import re
import urllib3
import os
from docopt import docopt


def get_soup(url):
    return BeautifulSoup(requests.get(url).text, 'lxml')


def main():
    options = docopt(__doc__, version='1.0')
    query = options['<query>']
    image_type = query

    url = "http://www.bing.com/images/search?q=" + query + \
        "&gft=+filterui:color2-bw+filterui:imagesize-large&FORM=R5IR3"

    soup = get_soup(url)
    images = [a['src2'] for a in soup.find_all("img", {"src2": re.compile("mm.bing.net")})]
    print(images)

    for img in images:
        raw_img = urllib3.urlopen(img).read()
        cntr = len([i for i in os.listdir("images") if image_type in i]) + 1
        f = open("images/" + image_type + "_" + str(cntr) + '.jpg', 'wb')
        f.write(raw_img)
        f.close()


if __name__ == '__main__':
    main()
