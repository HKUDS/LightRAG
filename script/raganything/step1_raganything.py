# 這是 RagAnything 原生風格 (但在處理 Mineru 環境問題時可能會很痛苦)
from raganything import RagAnythingConfig, RagAnythingParser

config = RagAnythingConfig(mode="local", device="cuda")
parser = RagAnythingParser(config)

# 它內部其實也是去 call Mineru，但你失去了對 subprocess 的細微控制權
parser.parse_file("input.pdf", "output_dir")