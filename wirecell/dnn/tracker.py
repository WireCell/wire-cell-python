#+/usr/bin/env python
'''
Provide "machine learning experiment tracking".

The "tracker" is an object that can be used to record input parameters and
intermediate and final values of training.


'''
from time import time
import shutil
from pathlib import Path
from hashlib import sha256 as hasher
import torch
class fsflow():
    '''
    Mimic a portion of mlflow tracking API using the filesystem as store.

    This is very limited w.r.t. mlflow. In particular is just provides basic
    log_() methods and:

    - no autologging.
    - no state.
    - no entity return values.

    Logs: model, dataset, input example, signature, parameters and metrics
    '''
    def __init__(self, basedir=None):
        if basedir:
            self.path = Path(basedir)
        else:
            self.path = Path(".") / "fsflow"
        self.logpath = self.path / "fsflow-log.json"

    def log(self, thing):
        '''
        Primitive sink of thing to log.  Each entry is a line of text.

        Thing must be text or json-serializable.
        '''
        if not isinstance(thing, str):
            thing = json.dumps(thing)
        with open(self.logpath, "a") as fp:
            fp.write(thing + '\n')

    @property
    def now(self):
        return time()

    def set_tracking_uri(uri="flflow-log.json"):
        '''
        Tracking URI is at best a log file name.
        '''
        if uri.startswith("file://"):
            uri = uri[7:]
        if uri.startswith("//"):
            uri[1:]
        if uri.startswith("/"):
            self.logpath = Path(uri)
        else:
            self.logpath = self.path / uri
        

    def log_entry(self, kind, name, value):
        '''
        Top level structured log entry.
        '''
        dat = dict(t=self.now, kind=kind, name=name, value=value)
        self.log(dat)

    def log_param(self, name, value):
        self.log_entry("param", name, value)

    def log_params(self, params, **kwds):
        params.update(kwds)
        self.log_entry("params", "params", params)

    def log_metric(self, name, value):
        self.log_entry("metric", name, value)

    def log_input(self, dataset, context=None, tags=None):
        '''
        Dataset here means "tensor"
        '''
        path = self.path
        if context:
            path = path / context
        path = path / "input.npz"
        path.parent.mkdir(parents=True, exists_ok=True)
        numpy.savez_compressed(path, dataset)
        self.log_entry("input", "path", str(path.absolute()))

    def log_artifact(self, local_path, artifact_path=None):
        '''
        Copy local path to artifact path or default artifact directory.
        '''
        local_path = Path(local_path)
        if artifact_path:
            artifact_path = self.path / "artifacts" / artifact_path
        else:
            artifact_path = self.path / "artifacts" / local_path.name
        artifact_path.parent.mkdir(parents=True, exists_ok=True)
        shutil.copy(local_path, artifact_path)
        self.log_entry("artifact", local_path, artifact_path)

    def log_artifacts(self, local_dir, artifact_path=None):
        '''
        Copy contents of local dir to artifact path or default artifact directory.
        '''
        local_dir = Path(local_dir)
        if artifact_path:
            artifact_path = self.path / "artifacts" / artifact_path
        else:
            artifact_path = self.path / "artifacts" / local_path.name
        artifact_path.parent.mkdir(parents=True, exists_ok=True)
        shutil.copytree(local_dir, artifact_path)
        self.log_entry("artifacts", local_dir, artifact_path)

    @property
    def pytorch(self):
        '''
        Mimic per-framework mlflow object attributes
        '''
        return self

    def log_model(self, pytorch_model, artifact_path, **kwds):
        if artifact_path:
            artifact_path = self.path / "artifacts" / artifact_path
        else:
            artifact_path = self.path / "artifacts" / local_path.name
        artifact_path.parent.mkdir(parents=True, exists_ok=True)
        torch.save(pytorch_model.sate_dict(), artifact_path)

        self.log_entry("model", local_dir, artifact_path)


    def create_experiment(self, name, artifact_location=None, tags=None):
        '''
        Start an experiment and return a unique ID
        '''
        # fixme: does the location need to be unique to the experiment?
        if artifact_location:
            artifact_location = Path(artifact_location)
        else:
            artifact_location = self.path / "artifacts"
        artifact_location = artifact_location.absolute()

        tags = tags or dict()

        t = self.now
        h = hasher()
        h.update(str(t).encode())
        h.update(name.encode())
        h.update(artifact_location.encode())
        jtags = json.dumps(tags)
        h.update(jtags.encode())
        eid = h.hexdigest()

        ent = dict(eid=eid, artifact_location=artifact_location, tags=tags)
        self.log_entry("create_experiment", name, ent)

        return eid

    def set_experiment(self, experiment_name=None, experiment_id=None):
        self.log_entry("set_experiment", experiment_id or experiment_name)
                
    def start_run(self, run_id=None, experiment_id=None, run_name=None, **kwds):
        self.log_entry("start_run", run_id or run_name,
                       dict(kwds, run_id=run_id, experiment_id=experiment_id, run_name=run_name))
    def end_run(self, status = 'FINISHED'):
        self.log_entry("end_run", 'status', status)



try:
    import mlflow
    flow = mlflow
except ImportError:
    flow = fsflow()
