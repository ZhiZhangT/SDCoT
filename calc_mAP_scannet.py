# Given the log output of eval.py, calculate the respective mAP scores for base and novel classes:
def calculate_mean_ap(output_str):
    # Define the lists of base and novel types
    base_types = ['bathtub', 'bed', 'bookshelf', 'cabinet', 'chair', 'counter', 'curtain', 'desk', 'door']
    novel_types = ['otherfurniture', 'picture', 'refrigerator', 'showercurtain', 'sink', 'sofa', 'table', 'toilet', 'window']

    # Initialize dictionaries to store AP scores
    ap_scores = {}

    # Parse the output string to extract AP scores
    lines = output_str.strip().split("\n")
    for line in lines:
        parts = line.split()
        obj_type = parts[1]
        ap_score = float(parts[4])
        ap_scores[obj_type] = ap_score

    # Extract AP scores for base and novel types
    base_ap_scores = [ap_scores[typ] for typ in base_types if typ in ap_scores]
    novel_ap_scores = [ap_scores[typ] for typ in novel_types if typ in ap_scores]

    # Calculate the mean AP scores
    mean_base_ap = sum(base_ap_scores) / len(base_ap_scores) if base_ap_scores else 0.0
    mean_novel_ap = sum(novel_ap_scores) / len(novel_ap_scores) if novel_ap_scores else 0.0

    return mean_base_ap, mean_novel_ap

# Example usage
output_str = """eval bathtub Average Precision: 0.002685
eval bed Average Precision: 0.016903
eval bookshelf Average Precision: 0.003056
eval cabinet Average Precision: 0.013822
eval chair Average Precision: 0.009161
eval counter Average Precision: 0.000212
eval curtain Average Precision: 0.015142
eval desk Average Precision: 0.021226
eval door Average Precision: 0.003468
eval otherfurniture Average Precision: 0.327894
eval picture Average Precision: 0.074556
eval refrigerator Average Precision: 0.340017
eval showercurtain Average Precision: 0.609499
eval sink Average Precision: 0.546705
eval sofa Average Precision: 0.768387
eval table Average Precision: 0.617546
eval toilet Average Precision: 0.837221
eval window Average Precision: 0.336348"""

mean_base_ap, mean_novel_ap = calculate_mean_ap(output_str)
print("Mean Average Precision for base types:", mean_base_ap)
print("Mean Average Precision for novel types:", mean_novel_ap)
