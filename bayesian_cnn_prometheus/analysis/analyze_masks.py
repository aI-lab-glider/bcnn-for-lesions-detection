from bayesian_cnn_prometheus.analysis.masks_analyzer import MasksAnalyzer


def main():
    model_name = None
    lesion_masks_path = None
    variance_masks_path = None

    mask_analyzer = MasksAnalyzer(model_name, lesion_masks_path, variance_masks_path)
    mask_analyzer.perform_analysis(save_to_json=True)


if __name__ == '__main__':
    main()
