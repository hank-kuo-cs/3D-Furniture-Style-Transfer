from torch.nn import L1Loss


def get_correct_num(sample_features, positive_features, negative_features):
    correct_num = 0
    batch_size = len(sample_features)

    for i in range(batch_size):
        dist_p = L1Loss()(sample_features[i], positive_features[i])
        dist_n = L1Loss()(sample_features[i], negative_features[i])

        if dist_p < dist_n:
            correct_num += 1

    return correct_num
