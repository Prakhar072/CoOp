import os.path as osp

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from loss_functions import get_loss_function

_tokenizer = _Tokenizer()


def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())

    return model


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.COOP.N_CTX
        ctx_init = cfg.TRAINER.COOP.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init

        else:
            # random initialization
            if cfg.TRAINER.COOP.CSC:
                print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            else:
                print("Initializing a generic context")
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS

        self.register_buffer("vocab_embeddings", clip_model.token_embedding.weight.detach().type(dtype))

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = cfg.TRAINER.COOP.CLASS_TOKEN_POSITION

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix

        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,     # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )

        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i_half1 = ctx[i : i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i : i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,     # (1, 1, dim)
                        ctx_i_half1,  # (1, n_ctx//2, dim)
                        class_i,      # (1, name_len, dim)
                        ctx_i_half2,  # (1, n_ctx//2, dim)
                        suffix_i,     # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i = ctx[i : i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        class_i,   # (1, name_len, dim)
                        ctx_i,     # (1, n_ctx, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError

        return prompts


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

    def encode_image(self, image):
        return self.image_encoder(image.type(self.dtype))

    def encode_text_features(self):
        prompts = self.prompt_learner()
        tokenized_prompts = self.tokenized_prompts
        text_features = self.text_encoder(prompts, tokenized_prompts)
        return text_features

    def forward(self, image):
        image_features = self.image_encoder(image.type(self.dtype))

        prompts = self.prompt_learner()
        tokenized_prompts = self.tokenized_prompts
        text_features = self.text_encoder(prompts, tokenized_prompts)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()

        return logits
    
    def forward_with_label_graph(self, image, labels):
        image_features = self.encode_image(image)
        
        # Get the learned context vectors directly from PromptLearner
        # These are the actual learnable parameters (shape: n_ctx, dim) or (n_cls, n_ctx, dim)
        ctx = self.prompt_learner.ctx
        
        # If ctx is shared across classes (2D), expand to match labels
        if ctx.dim() == 2:
            # ctx shape: (n_ctx, dim)
            # Average across context dimension to get a single feature vector
            prompt_features = ctx.mean(dim=0, keepdim=True).expand(labels.shape[0], -1)
        else:
            # ctx shape: (n_cls, n_ctx, dim) - class-specific contexts
            # Select contexts based on labels and average across context dimension
            batch_ctx = ctx[labels]  # (batch_size, n_ctx, dim)
            prompt_features = batch_ctx.mean(dim=1)  # (batch_size, dim)
        
        # Normalize
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        prompt_features = prompt_features / prompt_features.norm(dim=-1, keepdim=True)
        
        # Concatenate image embeddings with learned context embeddings
        alpha = 1.0  # tunable — amplifies prompt's influence on graph
        concat_features = torch.cat([image_features, alpha * prompt_features], dim=-1)
        concat_features = concat_features / concat_features.norm(dim=-1, keepdim=True)
        
        return concat_features


# Dataset-specific hyperparameters for GLoss
DATASET_HYPERPARAMS = {
    "caltech101": {"sigma": 2.0, "gamma": 0.9},
    "dtd": {"sigma": 1.5, "gamma": 0.85},
    "fgvc_aircraft": {"sigma": 2.5, "gamma": 0.9},
    "oxford_flowers": {"sigma": 2.0, "gamma": 0.9},
    "oxford_pets": {"sigma": 1.5, "gamma": 0.85},
    "ucf101": {"sigma": 3.0, "gamma": 0.9},
}


@TRAINER_REGISTRY.register()
class CoOp(TrainerX):
    """Context Optimization (CoOp).

    Learning to Prompt for Vision-Language Models
    https://arxiv.org/abs/2109.01134
    """

    def check_cfg(self, cfg):
        assert cfg.TRAINER.COOP.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        
        if cfg.TRAINER.COOP.PREC == "fp32" or cfg.TRAINER.COOP.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model)

        print("Turning off gradients in both the image and the text encoder")
        for name, param in self.model.named_parameters():
            if "prompt_learner" not in name:
                param.requires_grad_(False)

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.prompt_learner, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        # NOTE: only give prompt_learner to the op timizer
        self.optim = build_optimizer(self.model.prompt_learner, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("prompt_learner", self.model.prompt_learner, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.COOP.PREC == "amp" else None

        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)

        #store state
        if self.batch_idx == 0 and self.epoch == 0:
            with torch.no_grad():
                self._initial_ctx = self.model.prompt_learner.ctx.data.clone()
            print(f"\n[Epoch 0, Batch 0] Stored initial prompt state")
        
        # Get the loss function based on configuration
        loss_fn = get_loss_function(self.cfg.TRAINER.LOSS_FUNCTION, self.cfg)
        
        prec = self.cfg.TRAINER.COOP.PREC
        if prec == "amp":
            with autocast():
                output = self.model(image)
                loss = loss_fn(output, label)
            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
        
        else:
            loss_fn_name = self.cfg.TRAINER.LOSS_FUNCTION
            
            if loss_fn_name == "cross_entropy":
                # Pure CE baseline
                output = self.model(image)
                loss = F.cross_entropy(output, label)
            else:  # Assume it's "custom" or "ce_gloss"
                # CE+GLoss hybrid
                output = self.model(image)
                ce_loss = F.cross_entropy(output, label)
                concat_emb = self.model.forward_with_label_graph(image, label)
                n_cls = self.model.prompt_learner.n_cls
                
                # Get dataset-specific hyperparameters
                dataset_name = self.cfg.DATASET.NAME.lower()
                hyperparams = DATASET_HYPERPARAMS.get(dataset_name, {"sigma": 2.0, "gamma": 0.9})
                sigma = hyperparams["sigma"]
                gamma = hyperparams["gamma"]
                
                g_loss = loss_fn(concat_emb, label, n_cls, sigma=sigma, gamma=gamma)
                loss = 0.2 * ce_loss + 0.8 * g_loss
            
            self.model_backward_and_update(loss)

        loss_summary = {
            "loss": loss.item(),
            "acc": compute_accuracy(output, label)[0].item(),
        }

        #if (self.batch_idx + 1) == self.num_batches:
        #    self.update_lr()
        
        #report
        if self.batch_idx % 5 == 0:  # Check every 5 batches
            print(f"\n[Epoch {self.epoch}, Batch {self.batch_idx}] Diagnostics:")
            
            # 1. Check gradients
            has_grads = False
            total_grad_norm = 0.0
            for name, param in self.model.prompt_learner.named_parameters():
                if param.grad is not None:
                    grad_norm = param.grad.norm().item()
                    total_grad_norm += grad_norm
                    has_grads = True
                    print(f"  ✓ {name}: grad_norm={grad_norm:.8f}")
                else:
                    print(f"  ✗ {name}: NO GRADIENT")
            
            if not has_grads:
                print(f"  ❌ CRITICAL: NO GRADIENTS REACHING PROMPTS")
            else:
                print(f"  ✓ Total grad norm: {total_grad_norm:.8f}")
            
            # 2. Check prompt change
            with torch.no_grad():
                current_ctx = self.model.prompt_learner.ctx.data.clone()
            
            if hasattr(self, '_initial_ctx'):
                ctx_change = (current_ctx - self._initial_ctx).abs().max().item()
                print(f"  Prompt change since epoch start: {ctx_change:.8f}")
                if ctx_change < 1e-8:
                    print(f"  ⚠️  Prompts NOT changing!")
                else:
                    print(f"  ✓ Prompts ARE changing")
    
    # ============ END OF EPOCH REPORT ============
        if (self.batch_idx + 1) == self.num_batches:
            with torch.no_grad():
                final_ctx = self.model.prompt_learner.ctx.data.clone()
            
            if hasattr(self, '_initial_ctx'):
                epoch_ctx_change = (final_ctx - self._initial_ctx).abs().max().item()
                print(f"\n[Epoch {self.epoch} END] Prompt change over entire epoch: {epoch_ctx_change:.8f}")
                if epoch_ctx_change < 1e-8:
                    print(f"  ❌ Prompts NOT changing at all")
                else:
                    print(f"  ✓ Prompts changed: {epoch_ctx_change:.8f}")
            
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "token_prefix" in state_dict:
                del state_dict["token_prefix"]

            if "token_suffix" in state_dict:
                del state_dict["token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)
