"""Radar-language model: cached radar BEV features -> projector -> visual tokens in front of the question -> small LLM answers.

    [ 256 radar tokens ][ Question: ...  Answer: ] -> the LLM writes the answer
The radar tokens come from a 16 x 16 grid over the BEV map (each token = a 6.4 m x 6.4 m patch, 64 numbers from the radar network).
The LLM is trained with LoRA (small add-on layers) together with the projector; the radar network itself is frozen (its features are cached).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer


class RadarLLM(nn.Module):
    def __init__(self, llm_dir, use_vision=True, lora_r=16, feat_dim=64, side=16, use_camera=False, cam_dim=768, cam_tokens=49,
                 use_motion=False, motion_dim=8, motion_slots=8):
        super().__init__()
        self.use_vision, self.side = use_vision, side
        self.use_camera, self.cam_tokens = use_camera, cam_tokens
        self.use_motion, self.motion_slots = use_motion, motion_slots
        self.tok = AutoTokenizer.from_pretrained(llm_dir)
        llm = AutoModelForCausalLM.from_pretrained(llm_dir, dtype=torch.float32)
        if lora_r:
            cfg = LoraConfig(r=lora_r, lora_alpha=2 * lora_r, lora_dropout=0.05, task_type="CAUSAL_LM",
                             target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
            llm = get_peft_model(llm, cfg)
        self.llm = llm
        hidden = llm.config.hidden_size
        self.projector = nn.Sequential(nn.LayerNorm(feat_dim), nn.Linear(feat_dim, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        nn.init.normal_(self.projector[-1].weight, std=0.01)
        self.pos = nn.Parameter(torch.randn(side * side, hidden) * 0.02)      # which patch of the map a token comes from
        if use_camera:
            self.cam_projector = nn.Sequential(nn.LayerNorm(cam_dim), nn.Linear(cam_dim, hidden), nn.GELU(), nn.Linear(hidden, hidden))
            nn.init.normal_(self.cam_projector[-1].weight, std=0.01)
            self.cam_pos = nn.Parameter(torch.randn(cam_tokens, hidden) * 0.02)     # which patch of the image a token comes from
        if use_motion:
            self.motion_projector = nn.Sequential(nn.LayerNorm(motion_dim), nn.Linear(motion_dim, hidden), nn.GELU(), nn.Linear(hidden, hidden))
            nn.init.normal_(self.motion_projector[-1].weight, std=0.01)
            self.motion_pos = nn.Parameter(torch.randn(motion_slots, hidden) * 0.02)   # which "rank" (1st, 2nd, ... nearest object) a token is

    def device(self):
        return next(self.parameters()).device

    def visual_tokens(self, feats):
        """(B, 64, 32, 32) cached radar features -> (B, side*side, hidden)."""
        x = F.adaptive_avg_pool2d(feats.float(), self.side).flatten(2).transpose(1, 2)      # (B, 256, 64)
        return self.projector(x) + self.pos

    def camera_tokens(self, cam_feats):
        """(B, cam_tokens, cam_dim) cached CLIP patch features -> (B, cam_tokens, hidden)."""
        return self.cam_projector(cam_feats.float()) + self.cam_pos

    def object_tokens(self, motion_feats):
        """(B, motion_slots, motion_dim) cached per-object [class, x, y, vx, vy, speed, has_points] -> (B, motion_slots, hidden)."""
        return self.motion_projector(motion_feats.float()) + self.motion_pos

    def _n_prefix(self):
        return ((self.side ** 2 if self.use_vision else 0) + (self.cam_tokens if self.use_camera else 0) +
                (self.motion_slots if self.use_motion else 0))

    def _prefix_tokens(self, feats, cam_feats, dev, motion_feats=None):
        parts = []
        if self.use_vision:
            parts.append(self.visual_tokens(feats.to(dev)))
        if self.use_camera:
            parts.append(self.camera_tokens(cam_feats.to(dev)))
        if self.use_motion:
            parts.append(self.object_tokens(motion_feats.to(dev)))
        return torch.cat(parts, dim=1) if parts else None

    def _embed(self, ids):
        return self.llm.get_input_embeddings()(ids)

    def _prompts(self, questions):
        return [self.tok(f"Question: {q}\nAnswer:", add_special_tokens=False).input_ids for q in questions]

    def loss(self, questions, answers, feats, cam_feats=None, motion_feats=None):
        dev = self.device()
        prompts = self._prompts(questions)
        answers = [self.tok(" " + a, add_special_tokens=False).input_ids + [self.tok.eos_token_id] for a in answers]
        n_vis = self._n_prefix()
        length = max(len(p) + len(a) for p, a in zip(prompts, answers)) + n_vis
        B, H = len(prompts), self.pos.shape[1]
        embeds = torch.zeros(B, length, H, device=dev)
        mask = torch.zeros(B, length, dtype=torch.long, device=dev)
        labels = torch.full((B, length), -100, dtype=torch.long, device=dev)
        vis = self._prefix_tokens(feats, cam_feats, dev, motion_feats) if n_vis else None
        for b, (p, a) in enumerate(zip(prompts, answers)):
            ids = torch.tensor(p + a, device=dev)
            n = n_vis + len(ids)
            if n_vis:
                embeds[b, :n_vis] = vis[b]
            embeds[b, n_vis:n] = self._embed(ids)
            mask[b, :n] = 1
            labels[b, n_vis + len(p):n] = torch.tensor(a, device=dev)               # only the answer is scored
        # The vocabulary is ~152k words, so full logits for every position would not fit in memory.
        # Run the transformer, then apply the output layer only at the positions that predict answer tokens.
        inner = self.llm.get_base_model() if hasattr(self.llm, "get_base_model") else self.llm
        with torch.autocast("cuda", dtype=torch.bfloat16):
            hidden = inner.model(inputs_embeds=embeds, attention_mask=mask).last_hidden_state       # (B, L, H)
            target = labels[:, 1:]
            keep = target != -100
            logits = inner.lm_head(hidden[:, :-1][keep])                                            # (n_answer_tokens, vocab)
        return F.cross_entropy(logits.float(), target[keep])

    @torch.no_grad()
    def answer(self, questions, feats, cam_feats=None, motion_feats=None, max_new_tokens=6):
        """Greedy answers as lower-case strings. Prompts are left-padded so a whole batch can be generated at once."""
        dev = self.device()
        prompts = self._prompts(questions)
        n_vis = self._n_prefix()
        length = max(len(p) for p in prompts) + n_vis
        B, H = len(prompts), self.pos.shape[1]
        embeds = torch.zeros(B, length, H, device=dev)
        mask = torch.zeros(B, length, dtype=torch.long, device=dev)
        vis = self._prefix_tokens(feats, cam_feats, dev, motion_feats) if n_vis else None
        for b, p in enumerate(prompts):
            ids = torch.tensor(p, device=dev)
            n = n_vis + len(ids)
            start = length - n
            if n_vis:
                embeds[b, start:start + n_vis] = vis[b]
            embeds[b, start + n_vis:] = self._embed(ids)
            mask[b, start:] = 1
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = self.llm.generate(inputs_embeds=embeds, attention_mask=mask, max_new_tokens=max_new_tokens, do_sample=False,
                                    pad_token_id=self.tok.pad_token_id or self.tok.eos_token_id)
        return [t.strip().lower() for t in self.tok.batch_decode(out, skip_special_tokens=True)]
