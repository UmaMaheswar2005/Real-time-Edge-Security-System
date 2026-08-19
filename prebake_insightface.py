from insightface.app import FaceAnalysis
import os, glob
fa = FaceAnalysis(name="buffalo_sc", providers=["CPUExecutionProvider"])
fa.prepare(ctx_id=-1, det_size=(320, 320))
model_dir = os.path.expanduser("~/.insightface/models/buffalo_sc")
files = [os.path.basename(f) for f in glob.glob(model_dir + "/*")]
print(f"InsightFace models ready: {files}")
