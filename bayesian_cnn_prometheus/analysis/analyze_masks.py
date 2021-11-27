from bayesian_cnn_prometheus.analysis.correlation_analyzer import CorrelationAnalyzer


def main():
    variance_mask_path = None
    lesion_mask_path = None

    correlation_analyzer = CorrelationAnalyzer(variance_mask_path, lesion_mask_path)
    correlation_analyzer.perform_analysis(print_metrics=True)


if __name__ == '__main__':
    main()
