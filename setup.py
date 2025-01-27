import os
from setuptools import setup, find_packages


def get_version():
    version_file = os.path.join(os.path.dirname(
        __file__), 'openblock', 'version.py')
    with open(version_file, 'r') as f:
        exec(compile(f.read(), version_file, 'exec'))
    return locals()['__version__']


def parse_requirements(fname='requirements.txt', with_version=True):
    """Parse the package dependencies listed in a requirements file."""
    import re
    import sys
    from os.path import exists
    require_fpath = fname

    def parse_line(line):
        """Parse information from a line in a requirements text file."""
        if line.startswith('-r '):
            # Allow specifying requirements in other files
            target = line.split(' ')[1]
            for info in parse_require_file(target):
                yield info
        else:
            info = {'line': line}
            if line.startswith('-e '):
                info['package'] = line.split('#egg=')[1]
            else:
                # Remove versioning from the package
                pat = '(' + '|'.join(['>=', '==', '>']) + ')'
                parts = re.split(pat, line, maxsplit=1)
                parts = [p.strip() for p in parts]

                info['package'] = parts[0]
                if len(parts) > 1:
                    op, rest = parts[1:]
                    if ';' in rest:
                        # Handle platform specific dependencies
                        version, platform_deps = map(
                            str.strip, rest.split(';'))
                        info['platform_deps'] = platform_deps
                    else:
                        version = rest
                    info['version'] = (op, version)
            yield info

    def parse_require_file(fpath):
        with open(fpath, 'r') as f:
            for line in f.readlines():
                line = line.strip()
                if line and not line.startswith('#'):
                    for info in parse_line(line):
                        yield info

    def gen_packages_items():
        if exists(require_fpath):
            for info in parse_require_file(require_fpath):
                parts = [info['package']]
                if with_version and 'version' in info:
                    parts.extend(info['version'])
                if 'platform_deps' in info:
                    parts.append(';' + info['platform_deps'])
                yield ''.join(parts)

    packages = list(gen_packages_items())
    return packages


setup(
    name='openblock',
    version=get_version(),
    description='Efficient Vision Model Training and Optimization Platform',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Your Name',
    author_email='your.email@example.com',
    url='https://github.com/yourusername/OpenBlock',
    packages=find_packages(exclude=('tests', 'tools', 'demo')),
    include_package_data=True,
    license='Apache License 2.0',
    install_requires=parse_requirements('requirements.txt'),
    extras_require={
        'all': parse_requirements('requirements.txt'),
        'tests': [
            'pytest>=6.0',
            'pytest-cov>=2.0',
            'flake8>=3.9.0',
            'isort>=5.8.0',
            'yapf>=0.31.0',
        ],
        'optional': [
            'wandb>=0.12.0',
            'tensorboard>=2.5.0',
        ],
        'docs': [
            'sphinx>=4.0.0',
            'sphinx-rtd-theme>=0.5.0',
            'sphinx-markdown-tables>=0.0.15',
        ],
    },
    classifiers=[
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    python_requires='>=3.7',
    entry_points={
        'console_scripts': [
            'openblock=openblock.tools.cli:main',
        ],
    }
)
