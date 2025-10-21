import os
import sys
import argparse
import subprocess
from typing import Any, Dict

try:
    import yaml
except Exception as e:
    print("PyYAML 未安裝，請先執行: pip install PyYAML", file=sys.stderr)
    sys.exit(2)

TASKS_PATH = os.path.join('.specify', 'memory', 'tasks-cli.yaml')


def load_tasks() -> Dict[str, Any]:
    if not os.path.exists(TASKS_PATH):
        print(f"找不到 {TASKS_PATH}", file=sys.stderr)
        sys.exit(2)
    with open(TASKS_PATH, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f) or {}
    tasks = data.get('tasks') or {}
    if not isinstance(tasks, dict):
        print("tasks-cli.yaml 格式錯誤：缺少 'tasks' 區塊", file=sys.stderr)
        sys.exit(2)
    return tasks


def cmd_list() -> int:
    tasks = load_tasks()
    for name, spec in tasks.items():
        desc = ''
        if isinstance(spec, dict):
            desc = spec.get('description') or ''
        print(f"{name} - {desc}")
    return 0


def write_temp_script(content: str, ext: str) -> str:
    import tempfile
    fd, path = tempfile.mkstemp(suffix=ext)
    with os.fdopen(fd, 'w', encoding='utf-8', newline='\n') as f:
        if ext == '.sh' and not content.lstrip().startswith('#!'):
            f.write('#!/usr/bin/env bash\n')
        f.write(content)
        f.write('\n')
    if ext == '.sh':
        try:
            os.chmod(path, 0o755)
        except Exception:
            pass
    return path


def cmd_run(task: str, passthrough: list[str]) -> int:
    tasks = load_tasks()
    if task not in tasks:
        print(f"找不到任務: {task}", file=sys.stderr)
        print("可用任務：")
        for name in tasks.keys():
            print(f"- {name}")
        return 2

    spec = tasks[task] or {}
    run = spec.get('run') if isinstance(spec, dict) else None
    if not isinstance(run, dict):
        print(f"任務 {task} 缺少 run 區塊", file=sys.stderr)
        return 2

    is_windows = os.name == 'nt'
    key = 'windows' if is_windows else 'posix'
    script = run.get(key)
    if not script:
        print(f"任務 {task} 未提供當前平台 ({key}) 的腳本", file=sys.stderr)
        return 2

    # 將腳本寫入暫存檔，避免多行指令解析問題
    if is_windows:
        path = write_temp_script(script, '.cmd')
        # 透過 cmd /V:ON /C 來執行整段批次檔
        cmd = ['cmd', '/V:ON', '/C', path] + passthrough
    else:
        path = write_temp_script(script, '.sh')
        cmd = ['bash', path] + passthrough

    try:
        proc = subprocess.run(cmd)
        return proc.returncode
    finally:
        try:
            os.remove(path)
        except Exception:
            pass


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog='speckit', description='Minimal Runner for tasks-cli.yaml')
    sub = p.add_subparsers(dest='cmd', required=True)

    p_list = sub.add_parser('list', help='列出任務清單')
    p_list.set_defaults(func=lambda args: cmd_list())

    p_run = sub.add_parser('run', help='執行任務')
    p_run.add_argument('task', help='任務名稱')
    p_run.add_argument('sep', nargs='*', help='使用 -- 分隔後面要傳遞給腳本的參數')
    p_run.set_defaults(func=lambda args: cmd_run(args.task, args.sep))

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    # 支援 speckit run <task> -- <args...>
    if argv is None:
        argv = sys.argv[1:]
    if 'run' in argv and '--' in argv:
        idx = argv.index('--')
        known, rest = argv[:idx], argv[idx+1:]
        args = parser.parse_args(known)
        return cmd_run(getattr(args, 'task', ''), rest)
    else:
        args = parser.parse_args(argv)
        return args.func(args)


if __name__ == '__main__':
    raise SystemExit(main())

