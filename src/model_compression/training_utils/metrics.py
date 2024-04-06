from collections import defaultdict
from sklearn.metrics import f1_score


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def acc_and_f1(preds, labels, f1_avg="binary"):
    acc = simple_accuracy(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds, average=f1_avg)
    return {
        "acc": acc,
        "f1": f1,
        "acc_and_f1": (acc + f1) / 2,
    }

def superglue_compute_metrics(task_name, preds, labels, guids=None):
    assert len(preds) == len(labels)
    if task_name == "boolq":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "cb":
        return acc_and_f1(preds, labels, f1_avg="macro")
    elif task_name == "copa":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "multirc":
        assert len(guids) == len(preds), "Different number of predictions and IDs!"
        qst2ans = defaultdict(list)
        for idx, pred, label in zip(guids, preds, labels):
            qst_idx = f"{idx[0]}-{idx[1]}"
            qst2ans[qst_idx].append((pred, label))

        f1s, ems = [], []
        for qst, preds_and_labels in qst2ans.items():
            preds, labels = zip(*preds_and_labels)
            f1 = f1_score(y_true=labels, y_pred=preds)
            f1s.append(f1)
            em = int(sum([p == l for p, l in preds_and_labels]) == len(preds_and_labels))
            ems.append(em)

        avg_f1 = sum(f1s) / len(f1s)
        avg_em = sum(ems) / len(ems)
        em_and_f1 = (avg_em + avg_f1) / 2
        return {"f1": avg_f1, "em": avg_em, "em_and_f1": em_and_f1}
    elif task_name == "rte":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "wic":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "wsc":
        return acc_and_f1(preds, labels)
    else:
        raise KeyError(task_name)
