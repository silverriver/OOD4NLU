import numpy as np
import sklearn.metrics as metric
import bisect


def cal_roc_auc(ind_sm, ood_sm):
    ind_sm = np.max(np.array(ind_sm), axis=1)
    ood_sm = np.max(np.array(ood_sm), axis=1)
    y_true = np.concatenate([np.zeros_like(ood_sm), np.ones_like(ind_sm)])
    y_score = np.concatenate([ood_sm, ind_sm])
    return metric.roc_auc_score(y_true, y_score)


def cal_roc_auc_other(ind_logits, ood_logits):
    ind_sm = ind_logits
    ood_sm = ood_logits
    ind_sm = np.max(np.array(ind_sm), axis=1)
    ood_sm = np.max(np.array(ood_sm), axis=1)
    y_true = np.concatenate([np.zeros_like(ood_sm), np.ones_like(ind_sm)])
    y_score = np.concatenate([ood_sm, ind_sm])
    return metric.roc_auc_score(y_true, y_score)


def cal_pr(ind_sm, ood_sm):
    ind_sm = np.max(np.array(ind_sm), axis=1)
    ood_sm = np.max(np.array(ood_sm), axis=1)
    y_true_ind_pos = np.concatenate([np.zeros_like(ood_sm), np.ones_like(ind_sm)])
    y_true_ood_pos = np.concatenate([np.ones_like(ood_sm), np.zeros_like(ind_sm)])
    y_score_ind_pos = np.concatenate([ood_sm, ind_sm])
    y_score_ood_pos = np.negative(np.concatenate([ood_sm, ind_sm]))
    ood_pos_aupr = metric.average_precision_score(y_true=y_true_ood_pos, y_score=y_score_ood_pos)
    ind_pos_aupr = metric.average_precision_score(y_true=y_true_ind_pos, y_score=y_score_ind_pos)
    return ood_pos_aupr, ind_pos_aupr


def cal_brier_score(ind_sm, ood_sm):
    ind_sm = np.max(np.array(ind_sm), axis=1)
    ood_sm = np.max(np.array(ood_sm), axis=1)
    y_true = np.concatenate([np.zeros_like(ood_sm), np.ones_like(ind_sm)])
    y_score = np.concatenate([ood_sm, ind_sm])
    return metric.brier_score_loss(y_true, y_score)


def cal_ind_f1(ind_sm, ind_gt):
    y_pred = np.argmax(np.array(ind_sm), axis=1)
    return metric.f1_score(ind_gt, y_pred, average='micro')


def _find_min_index_large_value(l, value):
    """
    find the index s.t. l[index] >= value but l[index - 1] < value
    assume l is sorted [0, 0.1, ... 1]
    0 < value < 1
    :param l:
    :param value:
    :return:
    """
    return bisect.bisect_left(l, value)
    # index = -1
    # for i in range(len(l)):
    #     if l[len(l) - i -1] < value:
    #         index = len(l) - i
    #         break
    # return index


def _find_max_index_less_value(l, value):
    """
    find the index s.t. l[index] < value but l[index + 1] >= value
    assume l is sorted [0, 0.1, ... 1]
    0 < value < 1
    :param l:
    :param value:
    :return:
    """
    index = -1
    for i in range(len(l)):
        if l[i] >= value:
            index = i - 1
            break
    return index


def fpr_at_tpr95(ind_sm, ood_sm):
    ind_sm = np.max(np.array(ind_sm), axis=1)
    ood_sm = np.max(np.array(ood_sm), axis=1)
    y_true = np.concatenate([np.zeros_like(ood_sm), np.ones_like(ind_sm)])
    y_score = np.concatenate([ood_sm, ind_sm])
    fpr, tpr, thresholds = metric.roc_curve(y_true, y_score)
    index = _find_min_index_large_value(tpr, 0.95)
    if index == -1:
        return -1, -1, -1
    else:
        return fpr[index], tpr[index], thresholds[index]


def fpr_at_tprN(ind_sm, ood_sm, N):
    ind_sm = np.max(np.array(ind_sm), axis=1)
    ood_sm = np.max(np.array(ood_sm), axis=1)
    y_true = np.concatenate([np.zeros_like(ood_sm), np.ones_like(ind_sm)])
    y_score = np.concatenate([ood_sm, ind_sm])
    fpr, tpr, thresholds = metric.roc_curve(y_true, y_score)
    index = _find_min_index_large_value(tpr, N)
    if index == -1:
        return -1, -1, -1
    else:
        return fpr[index], tpr[index], thresholds[index]


def err_at_tpr95(ind_sm, ood_sm):
    ind_sm = np.max(np.array(ind_sm), axis=1)
    ood_sm = np.max(np.array(ood_sm), axis=1)
    y_true = np.concatenate([np.zeros_like(ood_sm), np.ones_like(ind_sm)])
    y_score = np.concatenate([ood_sm, ind_sm])
    fpr, tpr, thresholds = metric.roc_curve(y_true, y_score)
    index = _find_min_index_large_value(tpr, 0.95)
    if index == -1:
        return -1, -1, -1
    else:
        return (1 - tpr[index] + fpr[index]) / 2.0, tpr[index], thresholds[index]


def acc_at_fpr01(ind_sm, ood_sm):
    ind_sm = np.max(np.array(ind_sm), axis=1)
    ood_sm = np.max(np.array(ood_sm), axis=1)
    y_true = np.concatenate([np.zeros_like(ood_sm), np.ones_like(ind_sm)])
    y_score = np.concatenate([ood_sm, ind_sm])
    fpr, tpr, thresholds = metric.roc_curve(y_true, y_score)
    index = _find_max_index_less_value(fpr, 0.01)
    if index == -1:
        return -1, -1, -1
    else:
        return (tpr[index] + 1 - fpr[index]) / 2.0, fpr[index], thresholds[index]


def tpr_at_fpr01(ind_sm, ood_sm):
    ind_sm = np.max(np.array(ind_sm), axis=1)
    ood_sm = np.max(np.array(ood_sm), axis=1)
    y_true = np.concatenate([np.zeros_like(ood_sm), np.ones_like(ind_sm)])
    y_score = np.concatenate([ood_sm, ind_sm])
    fpr, tpr, thresholds = metric.roc_curve(y_true, y_score)
    index = _find_max_index_less_value(fpr, 0.01)
    if index == -1:
        return -1, -1, -1
    else:
        return tpr[index], fpr[index], thresholds[index]


def t_scaling_softmax(logits, t):
    """
    compute temperature scaled max(softmax)
    :param logits: array of shape [bs, num_cls]
    :param t: float
    :return: array in shape [bs] representing max(softmax)
    """
    logits = np.array(logits, dtype=np.float32)
    logits = logits / t
    max = np.max(logits, axis=1, keepdims=True)
    logits -= max
    exp_logits = np.exp(logits)
    softmax = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    return softmax


def _count_pos(sm, thes):
    """
    sm is sorted, count the num of element that > thes
    :param sm:
    :param thes:
    :return:
    """
    if len(sm) == 0:
        return 0
    if len(sm) == 1:
        if sm[0] > thes:
            return 1
        else:
            return 0
    mid = int(len(sm) / 2)
    if sm[mid] > thes:
        return len(sm[:mid]) + 1 + _count_pos(sm[mid+1:], thes)
    else:
        return _count_pos(sm[:mid], thes)


def cls_acc_with_ood(ind_gt, ind_sm, ood_sm):
    """
    calculate acc with different thresholds, would be same with roc curve if the classifier for ind is perfect
    :param ind_gt:
    :param ind_sm:
    :param ood_sm:
    :return:
    """
    assert ind_gt.shape[0] == ind_sm.shape[0]
    ind_max_sm = np.max(np.array(ind_sm), axis=1)
    ood_max_sm = np.max(np.array(ood_sm), axis=1)
    ind_correct = np.array(np.equal(np.argmax(ind_sm, axis=1), ind_gt), dtype=np.int32)

    max_index = np.argsort(ind_max_sm)
    ind_max_sm = ind_max_sm[max_index][::-1]
    ind_correct = ind_correct[max_index][::-1]
    ood_max_sm = np.sort(ood_max_sm)[::-1]
    acc_list = []
    fpr_list = []
    # ERROR here is a bug, thes_list would not be a sorted list, but the res will not be affected too much
    thes_list = np.unique(np.concatenate([ind_max_sm, ood_max_sm]))[::-1]

    total = len(ood_max_sm) + len(ind_max_sm)
    # sm > thes means positive, sm <= thes means negative
    for thes in thes_list:
        fp = _count_pos(ood_max_sm, thes)
        fpr = fp / len(ood_max_sm)
        ind_pos_index = _count_pos(ind_max_sm, thes)  # ind_max_sm[ind_pos_index -1 ] > thes, the value after <= thes
        acc = (len(ood_max_sm) - fp + np.sum(ind_correct[:ind_pos_index])) / total
        acc_list.append(acc)
        fpr_list.append(fpr)
    return fpr_list, acc_list, thes_list


def cal_detection_acc(ind_sm, ood_sm):
    """
    1 - min_a{P_in{q(x) <= a} * P_ind + P_out{q(x) > a} * P_out}
    :param ind_sm:
    :param ood_sm:
    :return:
    """
    ind_max_sm = np.max(np.array(ind_sm), axis=1)
    ood_max_sm = np.max(np.array(ood_sm), axis=1)

    ind_max_sm = np.sort(ind_max_sm)[::-1]
    ood_max_sm = np.sort(ood_max_sm)[::-1]

    thes_list = np.sort(np.concatenate([ind_max_sm, ood_max_sm]))[::-1]

    # sm > thes means positive, sm <= thes means negative
    acc = 1.0
    res_thes = 0
    for thes in thes_list:
        fp = _count_pos(ood_max_sm, thes)
        tp = _count_pos(ind_max_sm, thes)
        acc_ = fp / len(ood_max_sm) * 0.5 + (1 - tp / len(ind_max_sm)) * 0.5
        if acc_ < acc:
            acc = acc_
            res_thes = thes

    return 1 - acc, res_thes


def max_cls_acc(ind_gt, ind_sm, ood_sm):
    fpr, acc, thes = cls_acc_with_ood(ind_gt, ind_sm, ood_sm)
    index = 0
    for i, v in enumerate(acc):
        if v > acc[index]:
            index = i
    return acc[index], fpr[index], thes[index]


def cls_acc_at_fpr01(ind_gt, ind_sm, ood_sm):
    fpr, acc, thes = cls_acc_with_ood(ind_gt, ind_sm, ood_sm)
    index = 0

    for i, v in enumerate(fpr):
        if v >= 0.01:
            break
        else:
            if acc[i] > acc[index]:
                index = i

    return acc[index], fpr[index], thes[index]


def acc_fpr_at_thes(ind_gt, ind_sm, ood_sm, thes):
    ind_max_sm = np.max(np.array(ind_sm), axis=1)
    ood_max_sm = np.max(np.array(ood_sm), axis=1)
    ind_correct = np.array(np.equal(np.argmax(ind_sm, axis=1), ind_gt), dtype=np.int32)

    max_index = np.argsort(ind_max_sm)
    ind_max_sm = ind_max_sm[max_index][::-1]
    ind_correct = ind_correct[max_index][::-1]
    ood_max_sm = np.sort(ood_max_sm)[::-1]

    fp = _count_pos(ood_max_sm, thes)
    fpr = fp / len(ood_max_sm)
    ind_pos_index = _count_pos(ind_max_sm, thes)
    acc = (len(ood_max_sm) - fp + np.sum(ind_correct[:ind_pos_index])) / (len(ind_max_sm) + len(ood_max_sm))
    ind_acc = np.sum(ind_correct[:ind_pos_index]) / len(ind_max_sm)

    return acc, ind_acc, fpr


if __name__ == '__main__':
    import pickle
    import matplotlib.pyplot as plt
    import sklearn.metrics as metrics
    import os
    root_path = "/home/data/zhengyinhe/rejection/dl_cnn_kl_nseed/ind_20_ood_cgan_ood"
    ind_input = os.path.join(root_path, 'eval', 'test_ind_res.pkl')
    ood_input = os.path.join(root_path, 'eval', 'test_ood_res.pkl')
    print('ind_input', ind_input)
    print('ood_input', ood_input)
    with open(ind_input, 'rb') as f:
        ind_res = pickle.load(f)

    with open(ood_input, 'rb') as f:
        ood_res = pickle.load(f)

    ind_sm = np.max(ind_res['intent_softmax'], axis=1)
    ood_sm = np.max(ood_res['intent_softmax'], axis=1)
    fpr, tpr, _ = metric.roc_curve(np.concatenate([np.ones_like(ind_sm), np.zeros_like(ood_sm)]),
                                   np.concatenate([ind_sm, ood_sm]))
    fpr_list, acc_list, _ = cls_acc_with_ood(ind_res['intent_gt'], ind_res['intent_softmax'], ood_res['intent_softmax'])
    print('max acc: ', max_cls_acc(ind_res['intent_gt'], ind_res['intent_softmax'], ood_res['intent_softmax']))
    print('acc@fpr01: ', cls_acc_at_fpr01(ind_res['intent_gt'], ind_res['intent_softmax'], ood_res['intent_softmax']))

    plt.plot(fpr, tpr, label='roc')
    plt.plot(fpr_list, acc_list, label='fpr_acc')

    print('cal ing')
    roc = cal_roc_auc(ind_res['intent_softmax'], ood_res['intent_softmax'])
    pr = cal_pr(ind_res['intent_softmax'], ood_res['intent_softmax'])
    detec_acc = cal_detection_acc(ind_res['intent_softmax'], ood_res['intent_softmax'])
    fpr_tpr95 = fpr_at_tpr95(ind_res['intent_softmax'], ood_res['intent_softmax'])
    print('roc', roc)
    print('pr', pr)
    print('detect acc.', detec_acc)
    print('brier_socre', cal_brier_score(ind_res['intent_softmax'], ood_res['intent_softmax']))
    print('f1', cal_ind_f1(ind_res['intent_softmax'], ind_res['intent_gt']))
    print('fpr@tpr95', fpr_tpr95)
    print('err@tpr95', err_at_tpr95(ind_res['intent_softmax'], ood_res['intent_softmax']))
    print('acc@fpr01', acc_at_fpr01(ind_res['intent_softmax'], ood_res['intent_softmax']))
    print('tpr@fpr01', tpr_at_fpr01(ind_res['intent_softmax'], ood_res['intent_softmax']))

    print('------res:---------')
    print(roc)
    print(pr[1])
    print(pr[0])
    print(detec_acc[0])
    print(fpr_tpr95[0])

    print('temperature scaling with ', 10)
    roc = cal_roc_auc(t_scaling_softmax(ind_res['intent_logits'], 10),
                      t_scaling_softmax(ood_res['intent_logits'], 10))
    pr = cal_pr(t_scaling_softmax(ind_res['intent_logits'], 10),
                t_scaling_softmax(ood_res['intent_logits'], 10))
    detec_acc = cal_detection_acc(t_scaling_softmax(ind_res['intent_logits'], 10),
                                  t_scaling_softmax(ood_res['intent_logits'], 10))
    fpr_tpr95 = fpr_at_tpr95(t_scaling_softmax(ind_res['intent_logits'], 10),
                             t_scaling_softmax(ood_res['intent_logits'], 10))
    print('roc', roc)
    print('pr', pr)
    print('detect acc.', detec_acc)
    print('brier_socre', cal_brier_score(t_scaling_softmax(ind_res['intent_logits'], 10),
                                         t_scaling_softmax(ood_res['intent_logits'], 10)))
    print('f1', cal_ind_f1(ind_res['intent_softmax'], ind_res['intent_gt']))
    print('fpr@tpr95', fpr_tpr95)
    print('err@tpr95', err_at_tpr95(t_scaling_softmax(ind_res['intent_logits'], 10),
                                    t_scaling_softmax(ood_res['intent_logits'], 10)))
    print('acc@fpr01', acc_at_fpr01(t_scaling_softmax(ind_res['intent_logits'], 10),
                                    t_scaling_softmax(ood_res['intent_logits'], 10)))
    print('tpr@fpr01', tpr_at_fpr01(t_scaling_softmax(ind_res['intent_logits'], 10),
                                    t_scaling_softmax(ood_res['intent_logits'], 10)))

    print('------res:---------')
    print(roc)
    print(pr[1])
    print(pr[0])
    print(detec_acc[0])
    print(fpr_tpr95[0])

    ind_sm = np.max(t_scaling_softmax(ind_res['intent_logits'], 10), axis=1)
    ood_sm = np.max(t_scaling_softmax(ood_res['intent_logits'], 10), axis=1)
    fpr, tpr, _ = metric.roc_curve(np.concatenate([np.ones_like(ind_sm), np.zeros_like(ood_sm)]),
                                   np.concatenate([ind_sm, ood_sm]))
    fpr_list, acc_list, _ = cls_acc_with_ood(ind_res['intent_gt'],
                                             t_scaling_softmax(ind_res['intent_logits'], 10),
                                             t_scaling_softmax(ood_res['intent_logits'], 10))

    print('max acc: ', max_cls_acc(ind_res['intent_gt'],
                                   t_scaling_softmax(ind_res['intent_logits'], 10),
                                   t_scaling_softmax(ood_res['intent_logits'], 10)))

    print('acc@fpr01: ', cls_acc_at_fpr01(ind_res['intent_gt'],
                                          t_scaling_softmax(ind_res['intent_logits'], 10),
                                          t_scaling_softmax(ood_res['intent_logits'], 10)))

    plt.plot(fpr, tpr, label='roc_t')
    plt.plot(fpr_list, acc_list, label='fpr_acc_t')
    plt.legend()
    plt.show()

    t = list(range(1, 20))

    roc = []
    ind_pr = []
    ood_pr = []
    plt.figure(1)
    for i in t:
        ind_sm = t_scaling_softmax(ind_res['intent_logits'], i)
        ood_sm = t_scaling_softmax(ood_res['intent_logits'], i)
        temp_auroc = cal_roc_auc(ind_sm, ood_sm)
        temp_ind_pr, temp_ood_pr = cal_pr(ind_sm, ood_sm)
        roc.append(temp_auroc)
        print(temp_auroc)
        ind_pr.append(temp_ind_pr)
        ood_pr.append(temp_ood_pr)
        ind_sm = np.max(ind_sm, axis=1)
        ood_sm = np.max(ood_sm, axis=1)
        y_true = np.concatenate([np.zeros_like(ood_sm), np.ones_like(ind_sm)])
        y_score = np.concatenate([ood_sm, ind_sm])
        fpr, tpr, _ = metrics.roc_curve(y_true, y_score)
        plt.plot(fpr, tpr, label=str(i))

    plt.legend()

    plt.figure(2)
    plt.plot(t, roc, label='roc')
    plt.plot(t, ind_pr, label='ind_pr')
    plt.plot(t, ood_pr, label='ood_pr')
    plt.legend()
    plt.show()

