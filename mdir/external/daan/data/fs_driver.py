"""
Classes providing a unified interface to different file systems e.g. local, or over an http api
"""

import os
import io
import re
import hashlib
import pickle
from functools import reduce
from urllib import request
import requests


class AbstractPath:
    """Base class for different file system path classes, either for a file or directory"""

    def __init__(self, path, *nested):
        """Store given path. If one or more nested folders provided, join them with path."""
        self.path = reduce(self._path_join, nested, path)

    def exists(self, *basenames):
        """Chech whether basenames exist under the stored path. Return a list of booleans, one for
            each basename. If basenames not given, return whether stored path exists."""
        if not basenames:
            return self._exists("")[0]
        return self._exists(basenames)

    def _exists(self, *basenames):
        raise NotImplementedError()

    def makedirs(self, *basenames):
        """Make dirs recursively for each basename under the stored path"""
        raise NotImplementedError()

    def rename(self, src, dst):
        """Rename the src basename to the dst basename"""
        raise NotImplementedError()

    def store(self, basename, content):
        """Store given content to the basename under the stored path"""
        raise NotImplementedError()

    def open(self, basename=""):
        """Return a handle to the open resource"""
        raise NotImplementedError()

    @staticmethod
    def _path_join(dirname, basename):
        """Join dirname and basename so, that when basename is empty, dirname is returned"""
        if not basename:
            return dirname
        elif basename == "..":
            return dirname.rsplit("/", 1)[0]

        assert basename[0] != "/"
        if dirname[-1] == "/":
            return dirname + basename
        return dirname + "/" + basename

    def load(self, basename=""):
        """Return loaded path content (currently only pkl files are loaded). Optionally, basename
            that will be joined with the stored path can be provided."""
        if not self._path_join(self.path, basename).endswith(".pkl"):
            raise NotImplementedError("Cannot load anything else than pickle at the moment")

        with self.open(basename) as handle:
            return pickle.load(handle)

    def __truediv__(self, other):
        return self.__class__(self.path, other)


class LocalPath(AbstractPath):
    """Represents a local file-system path"""

    def _exists(self, *basenames):
        return [os.path.exists(self._path_join(self.path, x)) for x in basenames]

    def makedirs(self, *basenames):
        for basename in basenames:
            os.makedirs(self._path_join(self.path, basename), exist_ok=True)

    def rename(self, src, dst):
        os.rename(self._path_join(self.path, src), self._path_join(self.path, dst))

    def store(self, basename, content):
        with open(self._path_join(self.path, basename), "wb") as handle:
            if callable(content):
                content(handle)
            else:
                handle.write(content)

    def open(self, basename=""):
        return open(self._path_join(self.path, basename), "rb")


class HttpPath(AbstractPath): # pylint: disable=abstract-method
    """Represents an http(s) resource. Currently, only open() method is supported."""

    @staticmethod
    def validate(content, path):
        """If a file name is in the format name-hash.ext, compute sha256 hash of the content
            and validate it"""
        match = re.search(r'.*-([a-f0-9]{8,})\.[a-zA-Z0-9]{2,}$', path)
        if not match:
            return

        stored_hsh = match.group(1)
        computed_hsh = hashlib.sha256(content).hexdigest()[:len(stored_hsh)]
        if computed_hsh != stored_hsh:
            raise ValueError("Computed hash '%s' is not consistent with stored hash '%s'" \
                    % (computed_hsh, stored_hsh))

    def open(self, basename=""):
        """Load remote url into RAM and validate it, return wrapping BytesIO object"""
        url = self._path_join(self.path, basename)
        with request.urlopen(url) as handle:
            loaded = io.BytesIO(handle.read())

        self.validate(loaded.getvalue(), url)
        return loaded


class ApiPath(AbstractPath):
    """Represents a remote file-system path that is accessed via http(s) api"""

    def __init__(self, url, *nested):
        """Parse given url into auth credentials, post data and api endpoint"""
        if isinstance(url, dict):
            self.url, self.auth, self.data, path = url["url"], url["auth"], url["data"], url["path"]
        else:
            prot, _, host, path = url.split("/", 3)
            self.auth = None
            if "@" in host:
                self.auth = tuple(host.split("@")[0].split(":"))
                host = host.split("@")[1]
            self.data = {}
            if "?" in path:
                self.data = dict(x.split("=") for x in path.split("?")[1].split("&"))
                path = path.split("?")[0]
            self.url = prot + "//" + host + "/" + path
            path = self.data.pop("path")

        super().__init__(path, *nested)

    def _request(self, additional, **kwargs):
        """Perform a request given additional post data and kwargs for the request.post() method"""
        data = {**self.data, **additional}
        return requests.post(self.url, json=data, auth=self.auth, **kwargs)

    def _exists(self, *basenames):
        return self._request({"command": "exists",
                              "path": [self._path_join(self.path, x) for x in basenames]}).json()

    def makedirs(self, *basenames):
        return self._request({"command": "makedirs",
                              "path": [self._path_join(self.path, x) for x in basenames]}).json()

    def rename(self, src, dst):
        return self._request({"command": "rename", "src": self._path_join(self.path, src),
                              "dst": self._path_join(self.path, dst)}).json()

    def store(self, basename, content):
        return self._request({"command": "write", "path": self._path_join(self.path, basename)},
                             files={"file": io.BytesIO(content)}).json()

    def open(self, basename=""):
        return self._request({"command": "read", "path": self._path_join(self.path, basename)},
                             stream=True).raw

    def __truediv__(self, other):
        return self.__class__({"url": self.url, "auth": self.auth, "data": self.data,
                               "path": self.path}, other)


def fs_driver(path, *nested):
    """Based on the path, determine a path class and return the initialized path object"""
    if (path.startswith("http://") or path.startswith("https://")):
        if "?" in path:
            return ApiPath(path, *nested)
        return HttpPath(path, *nested)
    return LocalPath(path, *nested)
