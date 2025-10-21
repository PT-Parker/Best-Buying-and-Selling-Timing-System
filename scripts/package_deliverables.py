import os
import glob
import zipfile
from pathlib import Path


def add_path(zf: zipfile.ZipFile, path: Path, arc_prefix: str = ""):
    if path.is_file():
        arcname = Path(arc_prefix) / path.name if arc_prefix else path
        zf.write(path, arcname)
    elif path.is_dir():
        for p in path.rglob('*'):
            if p.is_file():
                arcname = p.relative_to(path.parent)
                zf.write(p, arcname)


def main():
    out_dir = Path('deliverables')
    out_dir.mkdir(parents=True, exist_ok=True)
    zip_path = out_dir / 'deliverables.zip'
    with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
        # slides
        slide = Path('slides/final_20pages.pptx')
        if slide.exists():
            zf.write(slide, slide)

        # backtests
        bt = Path('backtest_out')
        if bt.exists():
            add_path(zf, bt)

        # config
        cfg = Path('config')
        if cfg.exists():
            add_path(zf, cfg)

        # n8n workflows
        for wf in glob.glob('n8n/workflow*.json'):
            p = Path(wf)
            if p.exists():
                zf.write(p, p)

    print(f"Packaged -> {zip_path}")


if __name__ == '__main__':
    main()

