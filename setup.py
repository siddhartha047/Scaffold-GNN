"""Keep review source archives independent of the builder's account identity."""

from setuptools import setup
from setuptools.command.sdist import sdist


class AnonymousSdist(sdist):
    def make_archive(self, *args, **kwargs):
        kwargs.update(owner="root", group="root")
        return super().make_archive(*args, **kwargs)


setup(cmdclass={"sdist": AnonymousSdist})
