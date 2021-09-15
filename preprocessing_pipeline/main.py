import json
from preprocessing_pipeline.preprocessing_links \
    import TransformNiftiToNpy, CreateFolderStructure, CreateBatches


if __name__ == '__main__':
    with open("config.json") as f:
        config = json.load(f)
    preprocessing_chain = [
        TransformNiftiToNpy(), CreateFolderStructure(), CreateBatches()]
    for link in preprocessing_chain:
        link.run(config)
