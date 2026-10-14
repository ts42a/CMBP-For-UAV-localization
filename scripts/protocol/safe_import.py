"""Bounded ZIP staging and reversible promotion for downloaded result bundles."""
import os
import shutil
import stat
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath


def stage_zip(bundle, destination_parent, prefix, max_bytes=1024**3):
    parent=Path(destination_parent)
    parent.mkdir(parents=True,exist_ok=True)
    stage=Path(tempfile.mkdtemp(prefix='import-',dir=parent))
    try:
        with zipfile.ZipFile(bundle) as z:
            total=0
            seen=set()
            for member in z.infolist():
                name=member.filename
                p=PurePosixPath(name)
                mode=member.external_attr>>16
                if '\\' in name or ':' in name or p.is_absolute() or '..' in p.parts or not name.startswith(prefix):
                    raise ValueError(f'unexpected archive path: {name}')
                if name.casefold() in seen or stat.S_ISLNK(mode):
                    raise ValueError(f'duplicate path or link: {name}')
                seen.add(name.casefold())
                if mode and stat.S_IFMT(mode) not in (0,stat.S_IFREG,stat.S_IFDIR):
                    raise ValueError(f'nonregular archive member: {name}')
                total+=member.file_size
                if total>max_bytes:
                    raise ValueError('archive exceeds uncompressed result size limit')
            for member in z.infolist():
                target=stage/member.filename
                if member.is_dir():
                    target.mkdir(parents=True,exist_ok=True)
                else:
                    target.parent.mkdir(parents=True,exist_ok=True)
                    with z.open(member) as source, target.open('xb') as out:
                        shutil.copyfileobj(source,out)
        return stage, stage/prefix.rstrip('/')
    except BaseException:
        if not stage.resolve().is_relative_to(parent.resolve()):
            raise RuntimeError('staging cleanup escaped parent')
        shutil.rmtree(stage)
        raise


def promote(staged, destination):
    staged,destination=Path(staged),Path(destination)
    backup=None
    destination.parent.mkdir(parents=True,exist_ok=True)
    if destination.exists():
        backup=destination.with_name(destination.name+'.previous-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f'))
        os.replace(destination,backup)
    try:
        os.replace(staged,destination)
    except BaseException:
        if backup is not None:
            os.replace(backup,destination)
        raise
    return backup
