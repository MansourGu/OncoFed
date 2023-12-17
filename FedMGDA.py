from federatedml.framework.homo.blocks import RandomPaddingCipherClient, RandomPaddingCipherServer, PadsCipher, \
    RandomPaddingCipherTransVar
from federatedml.framework.homo.aggregator.aggregator_base import AggregatorBaseClient, AutoSuffix, AggregatorBaseServer
import numpy as np
from federatedml.framework.weights import Weights, NumpyWeights
from federatedml.util import LOGGER
import torch as t
from typing import Union, List
from fate_arch.computing._util import is_table
from federatedml.util import consts
import cvxopt
import copy
from flgo.utils import fmodule

AGG_TYPE = ['weighted_mean', 'sum', 'mean', 'FedMGDA']


class SecureAggregatorClient(AggregatorBaseClient):

    def __init__(self, secure_aggregate=True, aggregate_type='weighted_mean', aggregate_weight=1.0,
                 communicate_match_suffix=None):

        super(SecureAggregatorClient, self).__init__(
            communicate_match_suffix=communicate_match_suffix)
        self.secure_aggregate = secure_aggregate
        self.suffix = {
            "local_loss": AutoSuffix("local_loss"),
            "agg_loss": AutoSuffix("agg_loss"),
            "local_model": AutoSuffix("local_model"),
            "agg_model": AutoSuffix("agg_model"),
            "converge_status": AutoSuffix("converge_status")
        }

        # init secure aggregate random padding:
        if self.secure_aggregate:
            self._random_padding_cipher: PadsCipher = RandomPaddingCipherClient(
                trans_var=RandomPaddingCipherTransVar(prefix=communicate_match_suffix)).create_cipher()
            LOGGER.info('initialize secure aggregator done')

        # compute weight
        assert aggregate_type in AGG_TYPE, 'aggregate type must in {}'.format(
            AGG_TYPE)
        if aggregate_type == 'weighted_mean' or aggregate_type == 'FedMGDA':
            aggregate_weight = aggregate_weight
        elif aggregate_type == 'mean':
            aggregate_weight = 1

        self.send(aggregate_weight, suffix=('agg_weight',))
        self._weight = aggregate_weight / \
                       self.get(suffix=('agg_weight',))[0]  # local weight / total weight

        if aggregate_type == 'sum' or aggregate_type == 'FedMGDA':  # reset _weight
            self._weight = 1

        self._set_table_amplify_factor = False

        LOGGER.debug('aggregate compute weight is {}'.format(self._weight))

    def _process_model(self, model):

        to_agg = None

        if isinstance(model, np.ndarray) or isinstance(model, Weights):
            if isinstance(model, np.ndarray):
                to_agg = NumpyWeights(model * self._weight)
            else:
                to_agg = model * self._weight

            if self.secure_aggregate:
                to_agg: Weights = to_agg.encrypted(
                    self._random_padding_cipher)
            return to_agg

        # is FATE distrubed Table
        elif is_table(model):
            model = model.mapValues(lambda x: x * self._weight)

            if self.secure_aggregate:
                if not self._set_table_amplify_factor:
                    self._random_padding_cipher.set_amplify_factor(
                        consts.SECURE_AGG_AMPLIFY_FACTOR)
                model = self._random_padding_cipher.encrypt_table(model)
            return model

        if isinstance(model, t.nn.Module):
            parameters = list(model.parameters())
            tmp_list = [[np.array(p.cpu().detach().tolist()) for p in parameters if p.requires_grad]]
            LOGGER.debug('Aggregate trainable parameters: {}/{}'.format(len(tmp_list[0]), len(parameters)))
        elif isinstance(model, t.optim.Optimizer):
            tmp_list = [[np.array(p.cpu().detach().tolist()) for p in group["params"]]
                        for group in model.param_groups]
        elif isinstance(model, list):
            for p in model:
                assert isinstance(
                    p, np.ndarray), 'expecting List[np.ndarray], but got {}'.format(p)
            tmp_list = [model]

        if self.secure_aggregate:
            to_agg = [
                [
                    NumpyWeights(
                        arr *
                        self._weight).encrypted(
                        self._random_padding_cipher) for arr in arr_list] for arr_list in tmp_list]
        else:
            to_agg = [[arr * self._weight for arr in arr_list]
                      for arr_list in tmp_list]

        return to_agg

    def _recover_model(self, model, agg_model):

        if isinstance(model, np.ndarray):
            return agg_model.unboxed
        elif isinstance(model, Weights):
            return agg_model
        elif is_table(agg_model):
            return agg_model
        else:
            if self.secure_aggregate:
                agg_model = [[np_weight.unboxed for np_weight in arr_list]
                             for arr_list in agg_model]

            if isinstance(model, t.nn.Module):
                for agg_p, p in zip(agg_model[0], [p for p in model.parameters() if p.requires_grad]):
                    p.data.copy_(t.Tensor(agg_p))

                return model
            elif isinstance(model, t.optim.Optimizer):
                for agg_group, group in zip(agg_model, model.param_groups):
                    for agg_p, p in zip(agg_group, group["params"]):
                        p.data.copy_(t.Tensor(agg_p))
                return model
            else:
                return agg_model

    def send_loss(self, loss, suffix=tuple()):
        suffix = self._get_suffix('local_loss', suffix)
        assert isinstance(loss, float) or isinstance(
            loss, np.ndarray), 'illegal loss type {}, loss should be a float or a np array'.format(type(loss))
        self.send(loss * self._weight, suffix)

    def send_model(self,
                   model: Union[np.ndarray,
                   Weights,
                   List[np.ndarray],
                   t.nn.Module,
                   t.optim.Optimizer],
                   suffix=tuple()):
        """Sending model to arbiter for aggregation

        Parameters
        ----------
        model : model can be:
                A numpy array
                A Weight instance(or subclass of Weights), see federatedml.framework.weights
                List of numpy array
                A pytorch model, is the subclass of torch.nn.Module
                A pytorch optimizer, will extract param group from this optimizer as weights to aggregate
        suffix : sending suffix, by default tuple(), can be None or tuple contains str&number. If None, will automatically generate suffix
        """
        suffix = self._get_suffix('local_model', suffix)
        # judge model type
        to_agg_model = self._process_model(model)
        self.send(to_agg_model, suffix)

    def get_aggregated_model(self, suffix=tuple()):
        suffix = self._get_suffix("agg_model", suffix)
        return self.get(suffix)[0]

    def get_aggregated_loss(self, suffix=tuple()):
        suffix = self._get_suffix("agg_loss", suffix)
        return self.get(suffix)[0]

    def get_converge_status(self, suffix=tuple()):
        suffix = self._get_suffix("converge_status", suffix)
        return self.get(suffix)[0]

    def model_aggregation(self, model, suffix=tuple()):
        self.send_model(model, suffix=suffix)
        agg_model = self.get_aggregated_model(suffix=suffix)
        return self._recover_model(model, agg_model)

    def loss_aggregation(self, loss, suffix=tuple()):
        self.send_loss(loss, suffix=suffix)
        converge_status = self.get_converge_status(suffix=suffix)
        return converge_status


class SecureAggregatorServer(AggregatorBaseServer):

    def __init__(self, secure_aggregate=True, communicate_match_suffix=None, aggregate_type='FedMGDA'):
        super(SecureAggregatorServer, self).__init__(
            communicate_match_suffix=communicate_match_suffix)
        self.suffix = {
            "local_loss": AutoSuffix("local_loss"),
            "agg_loss": AutoSuffix("agg_loss"),
            "local_model": AutoSuffix("local_model"),
            "agg_model": AutoSuffix("agg_model"),
            "converge_status": AutoSuffix("converge_status")
        }
        self.model = None
        self.aggregate_type = aggregate_type
        self.secure_aggregate = secure_aggregate
        self.init_algo_para({'eta': 1.0, 'epsilon': 0.1})
        if self.secure_aggregate:
            RandomPaddingCipherServer(trans_var=RandomPaddingCipherTransVar(
                prefix=communicate_match_suffix)).exchange_secret_keys()
            LOGGER.info('initialize secure aggregator done')

        agg_weights = self.collect(suffix=('agg_weight',))
        sum_weights = 0
        for i in agg_weights:
            sum_weights += i
        self.broadcast(sum_weights, suffix=('agg_weight',))

    def optim_lambda(self, grads, lambda0):
        # create H_m*m = 2J'J where J=[grad_i]_n*m
        n = len(grads)
        Jt = []
        for gi in grads:
            Jt.append((copy.deepcopy(fmodule._modeldict_to_tensor1D(gi.state_dict())).cpu()).numpy())
        Jt = np.array(Jt)
        # target function
        P = 2 * np.dot(Jt, Jt.T)

        q = np.array([[0] for i in range(n)])
        # equality constraint λ∈Δ
        A = np.ones(n).T
        b = np.array([1])
        # boundary
        lb = np.array([max(0, lambda0[i] - self.epsilon) for i in range(n)])
        ub = np.array([min(1, lambda0[i] + self.epsilon) for i in range(n)])
        G = np.zeros((2 * n, n))
        for i in range(n):
            G[i][i] = -1
            G[n + i][i] = 1
        h = np.zeros((2 * n, 1))
        for i in range(n):
            h[i] = -lb[i]
            h[n + i] = ub[i]
        res = self.quadprog(P, q, G, h, A, b)
        return res

    def quadprog(self, P, q, G, h, A, b):
        """
        Input: Numpy arrays, the format follows MATLAB quadprog function: https://www.mathworks.com/help/optim/ug/quadprog.html
        Output: Numpy array of the solution
        """
        P = cvxopt.matrix(P.tolist())
        q = cvxopt.matrix(q.tolist(), tc='d')
        G = cvxopt.matrix(G.tolist())
        h = cvxopt.matrix(h.tolist())
        A = cvxopt.matrix(A.tolist())
        b = cvxopt.matrix(b.tolist(), tc='d')
        sol = cvxopt.solvers.qp(P, q.T, G.T, h.T, A.T, b)
        return np.array(sol['x'])

    def maga_aggregate(self, suffix=None, party_idx=-1):
        # 1. 计算归一化后的模型梯度\calculate normalized gradients
        # get suffix
        suffix = self._get_suffix('local_model', suffix)
        # recv params for aggregation
        models = self.collect(suffix=suffix, party_idx=party_idx)
        agg_result = None
        grads = [self.model - w for w in models]
        for gi in grads: gi.normalize()

        # 2. 计算原始权重λ0邻域的最优权重
        # calculate λ0
        nks = self.collect(suffix=('agg_weight',))
        nt = sum(nks)
        lambda0 = [1.0 * nk / nt for nk in nks]
        # optimize lambdas to minimize ||λ'g||² s.t. λ∈Δ, ||λ - λ0||∞ <= ε
        op_lambda = self.optim_lambda(grads, lambda0)
        op_lambda = [ele[0] for ele in op_lambda]

        # 3. 使用最有权重计算全局模型更新量，并更新模型\aggregate grads
        dt = fmodule._model_average(grads, op_lambda)
        self.model = self.models[0] - dt * self.eta
        return self.model - dt * self.eta

    def aggregate_model(self, suffix=None, party_idx=-1):

        # get suffix
        suffix = self._get_suffix('local_model', suffix)
        # recv params for aggregation
        models = self.collect(suffix=suffix, party_idx=party_idx)
        agg_result = None

        # Aggregate Weights or Numpy Array
        if self.aggregate_type == 'FedMGDA':
            agg_result = self.maga_aggregate()

        elif isinstance(models[0], Weights):
            agg_result = models[0]
            for w in models[1:]:
                agg_result += w

        # Aggregate Table
        elif is_table(models[0]):
            agg_result = models[0]
            for table in models[1:]:
                agg_result = agg_result.join(table, lambda x1, x2: x1 + x2)
            return agg_result

        # Aggregate numpy groups
        elif isinstance(models[0], list):
            # aggregation
            agg_result = models[0]
            # aggregate numpy model weights from all clients
            for params_group in models[1:]:
                for agg_params, params in zip(
                        agg_result, params_group):
                    for agg_p, p in zip(agg_params, params):
                        # agg_p: NumpyWeights or numpy array
                        agg_p += p

        if agg_result is None:
            raise ValueError(
                'can not aggregate receive model, format is illegal: {}'.format(models))

        return agg_result

    def broadcast_model(self, model, suffix=tuple(), party_idx=-1):
        suffix = self._get_suffix('agg_model', suffix)
        self.broadcast(model, suffix=suffix, party_idx=party_idx)

    def aggregate_loss(self, suffix=tuple(), party_idx=-1):

        # get loss
        suffix = self._get_suffix('local_loss', suffix)
        losses = self.collect(suffix, party_idx=party_idx)
        # aggregate loss
        total_loss = losses[0]
        for loss in losses[1:]:
            total_loss += loss

        return total_loss

    def broadcast_loss(self, loss_sum, suffix=tuple(), party_idx=-1):
        suffix = self._get_suffix('agg_loss', suffix)
        self.broadcast(loss_sum, suffix=suffix, party_idx=party_idx)

    def model_aggregation(self, suffix=tuple(), party_idx=-1):
        agg_model = self.aggregate_model(suffix=suffix, party_idx=party_idx)
        self.broadcast_model(agg_model, suffix=suffix, party_idx=party_idx)
        return agg_model

    def broadcast_converge_status(self, converge_status, suffix=tuple(), party_idx=-1):
        suffix = self._get_suffix('converge_status', suffix)
        self.broadcast(converge_status, suffix=suffix, party_idx=party_idx)

    def loss_aggregation(self, check_converge=False, converge_func=None, suffix=tuple(), party_idx=-1):
        agg_loss = self.aggregate_loss(suffix=suffix, party_idx=party_idx)
        if check_converge:
            converge_status = converge_func(agg_loss)
        else:
            converge_status = False
        self.broadcast_converge_status(
            converge_status, suffix=suffix, party_idx=party_idx)
        return agg_loss, converge_status
