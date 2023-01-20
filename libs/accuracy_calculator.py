# coding=utf-8
import faiss
import torch
from pytorch_metric_learning.utils import common_functions as c_f, stat_utils
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator, get_label_match_counts, \
    get_lone_query_labels, try_getting_not_lone_labels, precision_at_k
import logging
import numpy as np
from pytorch_metric_learning.utils.stat_utils import try_gpu
from sklearn.metrics import confusion_matrix, classification_report
from utils.sober_logger import SoberLogger


class MyAccuracyCalculator(AccuracyCalculator):


    def __init__(self, include=(), exclude=(), avg_of_avgs=False, k=None, label_comparison_fn=None, distance_metric ='L2'):
        super().__init__(include, exclude, avg_of_avgs, k, label_comparison_fn)
        self.results = None
        self.distance_metric = distance_metric

    def normal_processing(self, knn_labels, query_labels, not_lone_query_mask, k = 1):
        '''
        :param knn_labels:
        :param query_labels:
        :param not_lone_query_mask:
        :param k: count and find the max number of label from k nearest query
        :return:
        '''
        knn_labels, query_labels = try_getting_not_lone_labels(
            knn_labels, query_labels, not_lone_query_mask
        )
        if knn_labels is None:
            return 0
        [query_labels, knn_labels] = [
            c_f.to_numpy(x) for x in [query_labels, knn_labels]
        ]

        pred_k_labels,soft_probabilities = [], []
        knn_labels = knn_labels[:, :k, 0] if len(knn_labels.shape) == 3 else knn_labels[:, :k]

        # for each query, find the max number of label from k nearest neighbors
        for knn_label in knn_labels:
            u, c = np.unique(knn_label, return_counts = True)
            pred_label = u[c == c.max()]
            pred_k_labels.append(pred_label[0])
        pred_k_labels = np.array(pred_k_labels).reshape((-1, ))


        return query_labels.reshape((-1,)), pred_k_labels


    def calculate_confusion_matrix(
            self, knn_labels, query_labels, not_lone_query_mask, **kwargs
    ):
        query_labels, knn_labels = self.normal_processing(knn_labels, query_labels, not_lone_query_mask, self.k)
        self.confusion_matrix_result = confusion_matrix(query_labels, knn_labels)
        return self.confusion_matrix_result


    def calculate_tp(
            self, knn_labels, query_labels, not_lone_query_mask, **kwargs
    ):

        query_labels, knn_labels = self.normal_processing(knn_labels, query_labels, not_lone_query_mask, self.k)
        return int(confusion_matrix(query_labels, knn_labels)[0][0])

    def calculate_fn(
            self, knn_labels, query_labels, not_lone_query_mask, **kwargs
    ):
        query_labels, knn_labels = self.normal_processing(knn_labels, query_labels, not_lone_query_mask, self.k)
        return - int(confusion_matrix(query_labels, knn_labels)[1][0])

    def calculate_classification_report(self, knn_labels, query_labels, not_lone_query_mask, **kwargs):
        query_labels, knn_labels = self.normal_processing(knn_labels, query_labels, not_lone_query_mask, self.k)
        return classification_report(query_labels, knn_labels)

    def requires_knn(self):
        metrics = super().requires_knn()
        metrics.extend(["classification_report", "confusion_matrix"])
        return metrics

    def _get_accuracy(self, function_dict, **kwargs):
        self.results = kwargs
        return super()._get_accuracy(function_dict, **kwargs)

    def calculate_NMI(self, query_labels, cluster_labels, **kwargs):
        query_labels = query_labels.view((-1, ))
        return super().calculate_NMI(query_labels, cluster_labels, **kwargs)

    def calculate_AMI(self, query_labels, cluster_labels, **kwargs):
        query_labels = query_labels.view((-1, ))
        return super().calculate_AMI(query_labels, cluster_labels, **kwargs)

    def get_results(self):
        return self.results

    def calculate_precision_at_k(
            self, knn_labels, query_labels, not_lone_query_mask, **kwargs
    ):
        knn_labels, query_labels = try_getting_not_lone_labels(
            knn_labels, query_labels, not_lone_query_mask
        )
        if knn_labels is None:
            return 0
        return precision_at_k(
            knn_labels,
            query_labels[:, None],
            self.k,
            self.avg_of_avgs,
            self.label_comparison_fn,
        )

    # def get_knn_labels(self):
    #     query_labels, knn_labels = self.normal_processing(self.results['knn_labels'], self.results['query_labels'],
    #                                                       self.results['not_lone_query_mask'])
    #     # self.results = None

    def get_mismatches(self):
        knn_labels, query_labels, not_lone_query_mask, distances = self.results['knn_labels'],self.results[
            'query_labels'], \
                                                      self.results['not_lone_query_mask'], self.results['knn_distances']
        # query_labels:TP, knn_labels: predict
        query_labels, knn_labels = self.normal_processing(knn_labels, query_labels, not_lone_query_mask, self.k)
        mismatch_labels = query_labels != knn_labels
        mismatch_distance = torch.mean(distances[mismatch_labels], dim=-1)
        index = torch.argsort(mismatch_distance, dim = 0, descending=True).view(-1, ).cpu().numpy()

        return (query_labels[mismatch_labels][index], knn_labels[mismatch_labels][index]), mismatch_distance[index].cpu().numpy(

        ).reshape(-1, ), np.array([i for i in range(len(query_labels))])[mismatch_labels][index]


    def get_knn(
            self, reference_embeddings, test_embeddings, k, embeddings_come_from_same_source=False
    ):

        if embeddings_come_from_same_source:
            k = k + 1
        device = reference_embeddings.device
        reference_embeddings = c_f.to_numpy(reference_embeddings).astype(np.float32)
        test_embeddings = c_f.to_numpy(test_embeddings).astype(np.float32)

        faiss.normalize_L2(reference_embeddings)
        faiss.normalize_L2(test_embeddings)

        d = reference_embeddings.shape[1]
        SoberLogger.debug("running k-nn with k=%d" % k)
        SoberLogger.debug("embedding dimensionality is %d" % d)
        cpu_index = faiss.IndexFlatIP(d)
        distances, indices = try_gpu(cpu_index, reference_embeddings, test_embeddings, k)
        distances = c_f.to_device(torch.from_numpy(distances), device=device)
        indices = c_f.to_device(torch.from_numpy(indices), device=device)
        if embeddings_come_from_same_source:
            return indices[:, 1:], distances[:, 1:]
        return indices, distances

    def get_accuracy(
            self,
            query,
            reference,
            query_labels,
            reference_labels,
            embeddings_come_from_same_source,
            include=(),
            exclude=(),
    ):
        [query, reference, query_labels, reference_labels] = [
            c_f.numpy_to_torch(x)
            for x in [query, reference, query_labels, reference_labels]
        ]

        self.curr_function_dict = self.get_function_dict(include, exclude)

        kwargs = {
            "query": query,
            "reference": reference,
            "query_labels": query_labels,
            "reference_labels": reference_labels,
            "embeddings_come_from_same_source": embeddings_come_from_same_source,
            "label_comparison_fn": self.label_comparison_fn,
        }

        if any(x in self.requires_knn() for x in self.get_curr_metrics()):
            label_counts = get_label_match_counts(
                query_labels, reference_labels, self.label_comparison_fn
            )
            lone_query_labels, not_lone_query_mask = get_lone_query_labels(
                query_labels,
                label_counts,
                embeddings_come_from_same_source,
                self.label_comparison_fn,
            )

            num_k = self.determine_k(
                label_counts[1], len(reference), embeddings_come_from_same_source
            )

            # L2
            if self.distance_metric == 'L2':
                knn_indices, knn_distances = stat_utils.get_knn(
                    reference, query, num_k, embeddings_come_from_same_source
                )
            else:
                # cosine similarity
                knn_indices, knn_distances = self.get_knn(
                    reference, query, num_k, embeddings_come_from_same_source
                )

            knn_labels = reference_labels[knn_indices]
            if not any(not_lone_query_mask):
                logging.warning("None of the query labels are in the reference set.")
            kwargs["label_counts"] = label_counts
            kwargs["knn_labels"] = knn_labels
            kwargs["knn_distances"] = knn_distances
            kwargs["lone_query_labels"] = lone_query_labels
            kwargs["not_lone_query_mask"] = not_lone_query_mask
            kwargs['knn_indices'] = knn_indices

        if any(x in self.requires_clustering() for x in self.get_curr_metrics()):
            kwargs["cluster_labels"] = self.get_cluster_labels(**kwargs)

        return self._get_accuracy(self.curr_function_dict, **kwargs)

    def get_curr_distance_metric(self):
        return self.distance_metric