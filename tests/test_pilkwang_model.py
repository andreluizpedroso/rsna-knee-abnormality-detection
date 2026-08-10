import torch
import torch.nn as nn

from src.rsna_knee.pilkwang.model import PilkwangModel, SlotHead


class _Backbone(nn.Module):
    def __init__(self, hidden=8, tokens=5):
        super().__init__()
        self.hidden = hidden
        self.tokens = tokens

    def forward(self, pixel_values):
        batch = pixel_values.shape[0]
        vals = pixel_values.mean(dim=(1, 2, 3)).view(batch, 1, 1)
        hidden = vals.repeat(1, self.tokens, self.hidden)
        return type("Output", (), {"last_hidden_state": hidden})()


def test_slot_head_output_shape():
    head = SlotHead(dim=8, n_slot=6, n_out=12, prior=True)
    x = torch.randn(2, 6, 8)
    mask = torch.ones(2, 6)

    assert head(x, mask).shape == (2, 12)


def test_pilkwang_model_accepts_slot_bag_uint8():
    model = PilkwangModel(_Backbone(hidden=8), dim=8, pool="cls_mean")
    imgs = torch.randint(0, 255, (2, 6, 3, 16, 16), dtype=torch.uint8)
    mask = torch.ones(2, 6)

    out = model(imgs, mask, img_size=14)

    assert out.shape == (2, 12)

