from bayesian_cnn_prometheus.analysis.masks_analyzer import MasksAnalyzer


def main():
    model_name = ''
    lesion_masks_path = '/Users/sol/Documents/3d-cnn-prometheus/bayesian_cnn_prometheus/data/masks_'
    variance_masks_path = '/Users/sol/Documents/3d-cnn-prometheus/bayesian_cnn_prometheus/data/variance'

    mask_analyzer = MasksAnalyzer(model_name, lesion_masks_path, variance_masks_path)
    mask_analyzer.perform_analysis(save_to_json=True)


if __name__ == '__main__':
    main()
