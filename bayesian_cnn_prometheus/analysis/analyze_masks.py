from bayesian_cnn_prometheus.analysis.correlation_analyzer import CorrelationAnalyzer


def main():
    variance_masks_path = "/home/szymon/Pulpit/AILab/3d-cnn-prometheus/bayesian_cnn_prometheus/data/SEGMENTATION_VARIANCE_0003.nii.gz"
    cancer_masks_path = "/home/szymon/Pulpit/AILab/3d-cnn-prometheus/bayesian_cnn_prometheus/data/IMG_0003.nii.gz"

    correlation_analyzer = CorrelationAnalyzer(variance_masks_path, cancer_masks_path)
    correlation_analyzer.perform_analysis(print_metrics=True)


if __name__ == '__main__':
    main()
