#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#

"""
MAML++ wrapper.
"""
import dgl
import numpy as np
import torch
import traceback
import os

import json
from learn2learn.algorithms import MAML
from torch.autograd import grad
from torchmetrics.functional import accuracy,auroc,average_precision,specificity,recall
from tqdm import tqdm
from learn2learn.algorithms.base_learner import BaseLearner
from learn2learn.utils import clone_module, update_module, clone_named_parameters

from Dti.Dti_cnn import Dti_cnn
from Dti.Dti_meta_cnn import Dti_mamlpp_mulcnn, Dti_maml_mulcnn


def maml_pp_update(model, step=None, lrs=None, grads=None):
    """

    **Description**

    Performs a MAML++ update on model using grads and lrs.
    The function re-routes the Python object, thus avoiding in-place
    operations.

    NOTE: The model itself is updated in-place (no deepcopy), but the
          parameters' tensors are not.

    **Arguments**

    * **model** (Module) - The model to update.
    * **lrs** (list) - The meta-learned learning rates used to update the model.
    * **grads** (list, *optional*, default=None) - A list of gradients for each layer
        of the model. If None, will use the gradients in .grad attributes.

    **Example**
    ~~~python
    maml_pp = l2l.algorithms.MAMLpp(Model(), lr=1.0)
    lslr = torch.nn.ParameterDict()
    for layer_name, layer in model.named_modules():
        # If the layer has learnable parameters
        if (
            len(
                [
                    name
                    for name, param in layer.named_parameters(recurse=False)
                    if param.requires_grad
                ]
            )
            > 0
        ):
            lslr[layer_name.replace(".", "-")] = torch.nn.Parameter(
                data=torch.ones(adaptation_steps) * init_lr,
                requires_grad=True,
            )
    model = maml_pp.clone() # The next two lines essentially implement model.adapt(loss)
    for inner_step in range(5):
        loss = criterion(model(x), y)
        grads = autograd.grad(loss, model.parameters(), create_graph=True)
        maml_pp_update(model, inner_step, lrs=lslr, grads=grads)
    ~~~
    """
    if grads is not None and lrs is not None:
        params = list(model.parameters())
        if not len(grads) == len(list(params)):
            msg = "WARNING:maml_update(): Parameters and gradients have different length. ("
            msg += str(len(params)) + " vs " + str(len(grads)) + ")"
            print(msg)
        # TODO: Why doesn't this work?? I can't assign p.grad when zipping like this... Is this
        # because I'm using a tuple?
        # for named_param, g in zip(
            # [(k, v) for k, l in model.named_parameters() for v in l], grads
        # ):
            # p_name, p = named_param
        it = 0
        for name, p in model.named_parameters():
            if grads[it] is not None:
                lr = None
                layer_name = name[: name.rfind(".")].replace(
                    ".", "-"
                )  # Extract the layer name from the named parameter
                lr = lrs[layer_name][step]
                assert (
                    lr is not None
                ), f"Parameter {name} does not have a learning rate in LSLR dict!"
                p.grad = grads[it]
                p._lr = lr
            it += 1

    # Update the params
    for param_key in model._parameters:
        p = model._parameters[param_key]
        if p is not None and p.grad is not None:
            model._parameters[param_key] = p - p._lr * p.grad
            p.grad = None
            p._lr = None

    # Second, handle the buffers if necessary
    for buffer_key in model._buffers:
        buff = model._buffers[buffer_key]
        if buff is not None and buff.grad is not None and buff._lr is not None:
            model._buffers[buffer_key] = buff - buff._lr * buff.grad
            buff.grad = None
            buff._lr = None

    # Then, recurse for each submodule
    for module_key in model._modules:
        model._modules[module_key] = maml_pp_update(model._modules[module_key])
    return model


class MAMLpp(BaseLearner):
    """
    [[Source]](https://github.com/learnables/learn2learn/blob/master/learn2learn/algorithms/maml.py)

    **Description**

    High-level implementation of *Model-Agnostic Meta-Learning*.

    This class wraps an arbitrary nn.Module and augments it with `clone()` and `adapt()`
    methods.

    For the first-order version of MAML (i.e. FOMAML), set the `first_order` flag to `True`
    upon initialization.

    **Arguments**

    * **model** (Module) - Module to be wrapped.
    * **lr** (float) - Fast adaptation learning rate.
    * **lslr** (bool) - Whether to use Per-Layer Per-Step Learning Rates and Gradient Directions
        (LSLR) or not.
    * **lrs** (list of Parameters, *optional*, default=None) - If not None, overrides `lr`, and uses the list
        as learning rates for fast-adaptation.
    * **first_order** (bool, *optional*, default=False) - Whether to use the first-order
        approximation of MAML. (FOMAML)
    * **allow_unused** (bool, *optional*, default=None) - Whether to allow differentiation
        of unused parameters. Defaults to `allow_nograd`.
    * **allow_nograd** (bool, *optional*, default=False) - Whether to allow adaptation with
        parameters that have `requires_grad = False`.

    **References**

    1. Finn et al. 2017. "Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks."

    **Example**

    ~~~python
    linear = l2l.algorithms.MAML(nn.Linear(20, 10), lr=0.01)
    clone = linear.clone()
    error = loss(clone(X), y)
    clone.adapt(error)
    error = loss(clone(X), y)
    error.backward()
    ~~~
    """

    def __init__(
        self,
        model,
        lr,
        lrs=None,
        adaptation_steps=1,
        first_order=False,
        allow_unused=None,
        allow_nograd=False,
    ):
        super().__init__()
        self.module = model
        self.lr = lr
        if lrs is None:
            lrs = self._init_lslr_parameters(model, adaptation_steps, lr)
        self.lrs = lrs
        self.first_order = first_order
        self.allow_nograd = allow_nograd
        if allow_unused is None:
            allow_unused = allow_nograd
        self.allow_unused = allow_unused

    def _init_lslr_parameters(
        self, model: torch.nn.Module, adaptation_steps: int, init_lr: float
    ) -> torch.nn.ParameterDict:
        lslr = torch.nn.ParameterDict()
        for layer_name, layer in model.named_modules():
            # If the layer has learnable parameters
            if (
                len(
                    [
                        name
                        for name, param in layer.named_parameters(recurse=False)
                        if param.requires_grad
                    ]
                )
                > 0
            ):
                lslr[layer_name.replace(".", "-")] = torch.nn.Parameter(
                    data=torch.ones(adaptation_steps) * init_lr,
                    requires_grad=True,
                )
        return lslr

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def adapt(self, loss, step=None, first_order=None, allow_unused=None, allow_nograd=None):
        """
        **Description**

        Takes a gradient step on the loss and updates the cloned parameters in place.

        **Arguments**

        * **loss** (Tensor) - Loss to minimize upon update.
        * **step** (int) - Current inner loop step. Used to fetch the corresponding learning rate.
        * **first_order** (bool, *optional*, default=None) - Whether to use first- or
            second-order updates. Defaults to self.first_order.
        * **allow_unused** (bool, *optional*, default=None) - Whether to allow differentiation
            of unused parameters. Defaults to self.allow_unused.
        * **allow_nograd** (bool, *optional*, default=None) - Whether to allow adaptation with
            parameters that have `requires_grad = False`. Defaults to self.allow_nograd.
        """
        if first_order is None:
            first_order = self.first_order
        if allow_unused is None:
            allow_unused = self.allow_unused
        if allow_nograd is None:
            allow_nograd = self.allow_nograd
        second_order = not first_order

        gradients = []
        if allow_nograd:
            # Compute relevant gradients
            diff_params = [p for p in self.module.parameters() if p.requires_grad]
            grad_params = grad(
                loss,
                diff_params,
                retain_graph=second_order,
                create_graph=second_order,
                allow_unused=allow_unused,
            )
            grad_counter = 0

            # Handles gradients for non-differentiable parameters
            for param in self.module.parameters():
                if param.requires_grad:
                    gradient = grad_params[grad_counter]
                    grad_counter += 1
                else:
                    gradient = None
                gradients.append(gradient)
        else:
            try:
                gradients = grad(
                    loss,
                    self.module.parameters(),
                    retain_graph=second_order,
                    create_graph=second_order,
                    allow_unused=allow_unused,
                )
            except RuntimeError:
                traceback.print_exc()
                print(
                    "learn2learn: Maybe try with allow_nograd=True and/or allow_unused=True ?"
                )

        # Update the module
        assert step is not None, "step cannot be None when using LSLR!"
        self.module = maml_pp_update(self.module, step, lrs=self.lrs, grads=gradients)

    def clone(self, first_order=None, allow_unused=None, allow_nograd=None):
        """
        **Description**

        Returns a `MAMLpp`-wrapped copy of the module whose parameters and buffers
        are `torch.clone`d from the original module.

        This implies that back-propagating losses on the cloned module will
        populate the buffers of the original module.
        For more information, refer to learn2learn.clone_module().

        **Arguments**

        * **first_order** (bool, *optional*, default=None) - Whether the clone uses first-
            or second-order updates. Defaults to self.first_order.
        * **allow_unused** (bool, *optional*, default=None) - Whether to allow differentiation
        of unused parameters. Defaults to self.allow_unused.
        * **allow_nograd** (bool, *optional*, default=False) - Whether to allow adaptation with
            parameters that have `requires_grad = False`. Defaults to self.allow_nograd.

        """
        if first_order is None:
            first_order = self.first_order
        if allow_unused is None:
            allow_unused = self.allow_unused
        if allow_nograd is None:
            allow_nograd = self.allow_nograd
        return MAMLpp(
            clone_module(self.module),
            lr=self.lr,
            lrs=clone_named_parameters(self.lrs),
            first_order=first_order,
            allow_unused=allow_unused,
            allow_nograd=allow_nograd,
        )


def _testing_step(support, query, learner, steps, loss,maml):
    (support_drug, support_prot, support_labels) = support
    (query_drug, query_prot, query_labels) = query


    support_prot = support_prot.cuda()
    support_drug = support_drug.to(support_prot.device)
    support_labels = support_labels.long().cuda()

    query_prot = query_prot.cuda()
    query_drug = query_drug.to(support_prot.device)
    query_labels = query_labels.long().cuda()

    # Adapt the model on the support set
    if maml:
        for step in range(steps):
            # forward + backward + optimize
            pred, _ = learner(support_drug, support_prot, False, step)
            support_loss = loss(pred, support_labels)
            learner.adapt(support_loss, step=step)
        with torch.no_grad():
            # Evaluate the adapted model on the query set
            q_pred, _ = learner(query_drug, query_prot, False, steps - 1)
            query_loss = loss(q_pred, query_labels).detach()
    else:
        for step in range(steps):
            logit, _ = learner(support_drug, support_prot, False)
            train_error = loss(logit, support_labels)
            learner.adapt(train_error)
        with torch.no_grad():
            q_pred, _ = learner(query_drug, query_prot, False)
            query_loss = loss(q_pred, query_labels).detach()
    return q_pred


def test(
        trainer,
        lightning_module,
        model_state_dict,
        datamodule,
        model_config,
        train_config,
):
    if model_config.muti_out:
        if train_config.mamlpp:
            model = Dti_mamlpp_mulcnn(None, None, model_config, train_config.adaptation_steps)
        else:

            model = Dti_maml_mulcnn(None, None, model_config)
    else:
        model = Dti_cnn(None, None, model_config)
    #model = Dti_mamlpp_mulcnn(None, None, model_config,train_config.adaptation_steps)
    load = torch.load(model_state_dict)['state_dict']

    load = {i.replace('model.module.',''):j for i,j in load.items()}

    model.load_state_dict(load,strict=False)
    model.cuda()
    loss = lightning_module.loss
    loss.cuda()

    if train_config.mamlpp:
        maml = MAMLpp(
            model,
            lr=lightning_module.adaptation_lr,
            adaptation_steps=lightning_module.adaptation_steps,
            first_order=False,
            allow_nograd=True,
        )
    else:
        maml = MAML(
            model,
            lr=lightning_module.adaptation_lr,
            first_order=False,
            allow_nograd=True,
        )
    opt = torch.optim.AdamW(maml.parameters(), lightning_module.lr, betas=(0, 0.999))

    learner = maml.clone()
    learner.train()
    meta_pred = torch.tensor([]).cuda()
    meta_labels = torch.tensor([]).cuda()
    for i,batch in enumerate(tqdm(datamodule.test_dataloader())):

        # data, labels = batch
        drug, protein, labels, trg_cluster, drug_cluster, index = batch
        labels = labels.long().cuda()
        # Separate data into adaptation and evaluation sets
        support_indices = np.zeros(labels.size(0), dtype=bool)
        selection = np.arange(lightning_module.train_ways) * (lightning_module.train_shots + lightning_module.train_queries)
        for offset in range(lightning_module.train_shots):
            support_indices[selection + offset] = True
        query_indices = torch.from_numpy(~support_indices)
        support_indices = torch.from_numpy(support_indices)
        support_drug = dgl.batch([drug[i] for i in range(len(drug)) if support_indices[i]])
        # support_drug = (drug[support_indices])
        support_prot = protein[support_indices]
        support_labels = labels[support_indices]

        query_drug = dgl.batch([drug[i] for i in range(len(drug)) if query_indices[i]])
        query_prot = protein[query_indices]
        query_labels = labels[query_indices]

        pred = _testing_step((support_drug, support_prot, support_labels), (query_drug, query_prot, query_labels),
                             maml.clone(), lightning_module.adaptation_steps, loss,train_config.mamlpp)
        # meta_pred.append(pred)
        meta_pred = torch.concat([meta_pred, pred], dim=0)
        meta_labels = torch.concat([meta_labels, query_labels.long()], dim=0)
        if i > len(datamodule.test_dataloader()):
            break

    metrics_auroc = auroc(meta_pred, meta_labels.long(), task='multiclass', num_classes=2)
    metrics_acc = accuracy(meta_pred, meta_labels.long(), task='multiclass', num_classes=2)
    metrics_ap = average_precision(meta_pred, meta_labels.long(), task='multiclass', num_classes=2)
    metrics_recall = recall(meta_pred, meta_labels.long(), task='multiclass', num_classes=2)
    metrics_spec = specificity(meta_pred, meta_labels.long(), task='multiclass', num_classes=2)

    main_metrice = metrics_auroc
    # 修改文件名，将测试集损失信息添加到文件名中
    metrics = {'auroc': metrics_auroc.item(),'auprc':metrics_ap.item(),'acc':metrics_acc.item(),'recall':metrics_recall.item(),'spec':metrics_spec.item()}
    original_dir_name = trainer.logger.log_dir
    components = original_dir_name.split("/")
    last_components = components[-1].split('\\')
    components = components[:-1] + last_components
    # new_path = os.path.join("/".join(components[:-1]), f"{trainer.model.main_metrice}:{main_metrice :.4f}")
    new_path = os.path.join("/".join(components[:-1]), f"{trainer.model.main_metrice}_{main_metrice :.4f}")
    new_path = new_path.replace('.', '_')
    f = open(os.path.join(trainer.logger.log_dir, 'test_metrics.json'), 'w')
    json.dump(metrics, f)
    f.close()

    def rename_folder(folder_path):
        # 检查文件夹是否存在
        if not os.path.exists(folder_path):
            return folder_path
        else:
            # 文件夹已存在，尝试在文件夹名字后添加序号
            base_folder_path = folder_path
            index = 1
            while os.path.exists(folder_path):
                folder_path = f"{base_folder_path}({index})"
                index += 1
            return folder_path

    os.rename(original_dir_name, rename_folder(new_path))  # 重命名文件夹或文件路径
    print(f"{metrics}")
