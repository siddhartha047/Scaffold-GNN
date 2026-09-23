"""NFS-safe exclusion using server-side atomic mkdir, not client-local flock.

Claims are never stolen on a timeout. After a killed owner, an operator must
verify that its launcher AND children have exited before removing that claim.
"""
import json
import errno
import os
from pathlib import Path
import socket
import time
import uuid
import warnings


class LockBusy(RuntimeError):
    pass


class DirectoryLock:
    def __init__(self, path, *, blocking=True, timeout=60, release_on_error=True):
        self.path = Path(path)
        self.blocking = blocking
        self.timeout = timeout
        self.release_on_error = release_on_error
        self.token = uuid.uuid4().hex
        self.owned = False

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        started = time.monotonic()
        while True:
            try:
                # MKDIR is serialized by the NFS server even with local_lock=all.
                self.path.mkdir()
                break
            except FileExistsError:
                if not self.blocking or (self.timeout is not None and time.monotonic() - started >= self.timeout):
                    raise LockBusy(f'Claim busy: {self.path}; never auto-steal stale claims') from None
                time.sleep(.2)
        self.owned = True
        owner = dict(token=self.token, hostname=socket.gethostname(), pid=os.getpid(),
                     slurm_job=os.environ.get('SLURM_JOB_ID'), created=time.time())
        # A crash before publication leaves a locked directory (fail closed).
        temporary = self.path / f'owner.{self.token}.tmp'
        with temporary.open('x') as handle:
            json.dump(owner, handle, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(self.path / 'owner.json')
        return self

    def release(self):
        if not self.owned:
            return
        owner_path = self.path / 'owner.json'
        owner = json.loads(owner_path.read_text())
        if owner['token'] != self.token:
            raise RuntimeError(f'Refusing to release a different owner: {self.path}')
        # Atomically retire our entire directory before cleanup. On NFS an
        # unlinked owner file may temporarily survive as a .nfs* silly-rename;
        # rmdir then raises ENOTEMPTY even though no owner remains. Cleanup must
        # not strand the *active* claim name and block every future report.
        retired = self.path.with_name(f'.{self.path.name}.released.{self.token}')
        self.path.rename(retired)
        self.owned = False
        (retired / 'owner.json').unlink()
        for attempt in range(5):
            try:
                retired.rmdir()  # Never recursively remove unexpected files.
                return
            except OSError as exc:
                if exc.errno not in (errno.ENOTEMPTY, errno.EEXIST):
                    raise
                if attempt < 4:
                    time.sleep(.1)
        warnings.warn(f'Claim released; retained NFS cleanup directory: {retired}', RuntimeWarning)

    def __exit__(self, exc_type, exc, traceback):
        if exc_type is None or self.release_on_error:
            self.release()
