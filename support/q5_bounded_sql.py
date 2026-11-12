"""Runtime-only replacement for compute_records: same records/metrics, killable SQL worker."""
import os
from pathlib import Path
import pickle
import subprocess
import sys
import tempfile
import time

WORKER = r'''
import concurrent.futures, pickle, sqlite3, sys, time
from pathlib import Path
queries_path, results_path, database, budget = sys.argv[1:]
deadline = time.monotonic() + float(budget)
with open(queries_path, "rb") as handle:
    queries = pickle.load(handle)
def execute(index, query):
    connection = sqlite3.connect(Path(database).resolve().as_uri() + "?mode=ro", uri=True)
    connection.set_progress_handler(lambda: int(time.monotonic() >= deadline), 10000)
    try:
        if time.monotonic() >= deadline:
            return index, [], "Query timed out"
        rows = connection.execute(query).fetchall()
        return index, rows, ""
    except Exception as error:
        if time.monotonic() >= deadline:
            return index, [], "Query timed out"
        return index, [], f"{type(error).__name__}: {error}"
    finally:
        connection.close()
with open(results_path, "wb") as output:
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as pool:
        futures = [pool.submit(execute, i, query) for i, query in enumerate(queries)]
        for future in concurrent.futures.as_completed(futures):
            pickle.dump(future.result(), output, protocol=pickle.HIGHEST_PROTOCOL)
            output.flush()
'''


def compute_records_bounded(queries, database, timeout_secs=120):
    database = Path(database).resolve()
    if not database.is_file():
        raise FileNotFoundError(database)
    records, errors = [[] for _ in queries], ["Query timed out" for _ in queries]
    directory = "/content" if Path("/content").is_dir() else None
    with tempfile.TemporaryDirectory(prefix="hw4_sql_worker_", dir=directory) as tmp:
        tmp = Path(tmp)
        script, incoming, outgoing = tmp / "worker.py", tmp / "queries.pkl", tmp / "results.pkl"
        script.write_text(WORKER)
        with incoming.open("wb") as handle:
            pickle.dump(list(queries), handle, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Executing {len(queries)} SQL queries: {timeout_secs}s batch limit, isolated worker.", flush=True)
        started = time.monotonic()
        process = subprocess.Popen([sys.executable, str(script), str(incoming), str(outgoing), str(database), str(timeout_secs)],
                                   stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        timed_out = False
        try:
            while True:
                remaining = timeout_secs - (time.monotonic() - started)
                if remaining <= 0:
                    timed_out = True
                    break
                try:
                    process.wait(timeout=min(15, remaining))
                    break
                except subprocess.TimeoutExpired:
                    print(f"SQL worker: {time.monotonic()-started:.0f}s elapsed; still within {timeout_secs}s limit.", flush=True)
            if timed_out and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
            stderr = process.stderr.read()
            if not timed_out and process.returncode:
                raise RuntimeError(f"SQL worker failed: {stderr[-2000:]}")
            completed = 0
            if outgoing.is_file():
                with outgoing.open("rb") as handle:
                    while True:
                        try:
                            index, rows, error = pickle.load(handle)
                        except (EOFError, pickle.UnpicklingError):
                            break  # A killed worker may leave one incomplete trailing record.
                        assert 0 <= index < len(queries)
                        records[index], errors[index] = rows, error
                        completed += 1
            print(f"SQL execution finished: {completed}/{len(queries)} completed; "
                  f"{len(queries)-completed} timed out. Worker exited.", flush=True)
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=5)
            process.stderr.close()
    return records, errors


def install_bounded_sql(utils_module):
    # Imported metric/save helpers resolve compute_records through this module's globals.
    # This changes only this child process, never part2/utils.py on disk.
    def bounded(queries):
        return compute_records_bounded(queries, utils_module.DB_PATH, 120)
    utils_module.compute_records = bounded
