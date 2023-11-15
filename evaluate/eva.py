from datasets import load_metric
from evaluate import list_evaluation_modules

if __name__ == '__main__':
    # 列出评价指标
    metrics_list = list_evaluation_modules()
    print(len(metrics_list))
    print(metrics_list)

    # 加载一个评价指标
    metric = load_metric('glue', 'mrpc')
    print(metric.inputs_description)

    # 计算一个评价指标
    predictions = [0, 1, 0]
    references = [0, 1, 1]
    final_score = metric.compute(predictions=predictions, references=references)
    print(final_score)
