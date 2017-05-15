#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: oesteban
# @Date:   2015-11-19 16:44:27
# @Last Modified by:   oesteban
# @Last Modified time: 2016-05-19 16:08:49
""" MRIQC setup script """
import os
import sys
from cogenc import (__version__, __description__, __license__,
                   __author__, __email__, __longdesc__)

REQ_LINKS = []

if os.path.exists('requirements.txt'):
    with open('requirements.txt', 'r') as rfile:
        REQUIREMENTS = [line.strip() for line in rfile.readlines()]
    for i, req in enumerate(REQUIREMENTS):
        if req.startswith('-e'):
            REQUIREMENTS[i] = req.split('=')[1]
            REQ_LINKS.append(req.split()[1])
else:
    REQUIREMENTS = []


def main():
    """ Install entry-point """
    from glob import glob
    from setuptools import setup

    setup(
        name='cogencoding',
        version=__version__,
        description=__description__,
        long_description=__longdesc__,
        author=__author__,
        author_email=__email__,
        license=__license__,
        maintainer_email=__email__,
        url='http://github.com/poldrack/cognitive-encoding-model',
 #       download_url='https://pypi.python.org/packages/source/m/mriqc/'
 #                    'mriqc-%s.tar.gz' % __version__,
#        entry_points={'console_scripts': ['mriqc=mriqc.utils.mriqc_run:main',
#                                          'mriqc_plot=mriqc.utils.mriqc_plot:main',
#                                          'abide2bids=mriqc.utils.abide2bids:main',
#                                          'fs2gif=mriqc.utils.fs2gif:main',
#                                          'dfcheck=mriqc.utils.dfcheck:main']},
        packages=['cogenc',],
                  #'pymallet.utils'],
        package_data={'cogenc': ['data/*']},
        install_requires=REQUIREMENTS,
        dependency_links=REQ_LINKS,
        zip_safe=False,
        classifiers=[
            'Development Status :: 3 - Alpha',
            'Intended Audience :: Science/Research',
            'Topic :: Scientific/Engineering :: Information Analysis',
            'License :: OSI Approved :: MIT License',
            'Programming Language :: Python :: 3.6',
        ],
    )

if __name__ == '__main__':
    LOCAL_PATH = os.path.dirname(os.path.abspath(sys.argv[0]))
    os.chdir(LOCAL_PATH)
    sys.path.insert(0, LOCAL_PATH)

    main()
