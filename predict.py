# coding=utf-8

# knn monitor as in InstDisc https://arxiv.org/abs/1805.01978
# implementation follows http://github.com/zhirongw/lemniscate.pytorch and https://github.com/leftthomas/SimCLR
from pytorch_metric_learning import testers



    ### compute accuracy using AccuracyCalculator from pytorch-metric-learning ###
def knn_test(train_set, test_set, model, accuracy_calculator, batch_size = 64, get_mismatch = False):
    '''
    :param train_set:
    :param test_set:
    :param model:
    :param accuracy_calculator:
    :param batch_size: calculate embedding batch size
    :return:
    '''
    #convenient function from pytorch-metric-learning
    # from utils.sober_logger import SoberLogger

    tester = testers.BaseTester(batch_size=batch_size, dataloader_num_workers=16,
                                    accuracy_calculator=accuracy_calculator)

    print("Calculate train/test embedding !")
    train_embeddings, train_labels = tester.get_all_embeddings(train_set, model, eval = True)
    test_embeddings, test_labels = tester.get_all_embeddings(test_set, model, eval = True)

    print("Computing accuracy")
    accuracies = accuracy_calculator.get_accuracy(test_embeddings,
                                                       train_embeddings,
                                                        test_labels,
                                                       train_labels,
                                                       False)
    if "precision_at_k" in accuracies.keys():
        print("Test set accuracy (Precision@%d) = %.5f"%(accuracy_calculator.k, accuracies["precision_at_k"]))
    elif "precision_at_1" in accuracies.keys():
        print("Test set accuracy (Precision@1) = %.5f"%(accuracies["precision_at_1"]))

    if not get_mismatch:
        return accuracies, {'train':[train_embeddings, train_labels], 'test':[test_embeddings, test_labels]}
    else:
        return accuracies, {'train':[train_embeddings, train_labels], 'test':[test_embeddings, test_labels]}, \
               accuracy_calculator.get_mismatches()
