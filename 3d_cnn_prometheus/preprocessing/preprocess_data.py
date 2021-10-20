import json

from preprocessing_links import TransformNiftiToNpy, CreateFolderStructure, CreateBatches


def preprocess_data():
    with open("config.json") as f:
        config = json.load(f)

    preprocessing_chain = [TransformNiftiToNpy(), CreateFolderStructure(), CreateBatches()]
    for link in preprocessing_chain:
        print(f'🐉 Started {type(link).__name__} ...')
        link.run(config)
        print(f'🐉 Finished {type(link).__name__}')


if __name__ == '__main__':
    preprocess_data()
