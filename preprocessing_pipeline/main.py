import json
import preprocessing_pipeline
from preprocessing_pipeline.preprocessing_links \
     import TransformNiftiToNpy, CreateFolderStructure, CreateBatches
import os

if __name__ == '__main__':
    with open(os.path.join(os.path.dirname(preprocessing_pipeline.__file__),"config.json")) as f:
        config = json.load(f)
    preprocessing_chain = [
        TransformNiftiToNpy(), CreateFolderStructure(), CreateBatches()]
    for link in preprocessing_chain:
        link.run(config)
