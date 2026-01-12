from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional, Tuple

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
from flask import Flask, flash, render_template, request
from tokenizers import Tokenizer

from config.config import get_config
from src.core.dataset import casual_mask
from src.core.model import build_transformer

app = Flask(__name__, template_folder=str(Path(__file__).parent.parent.parent / "templates"))
app.config["SECRET_KEY"] = "change-me"  # Replace for production

config = get_config()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer_src: Optional[Tokenizer] = None
tokenizer_tgt: Optional[Tokenizer] = None
model: Optional[torch.nn.Module] = None
checkpoint_path: Optional[Path] = None


def safe_token_id(tokenizer: Tokenizer, token: str, default: int = 0) -> int:
	token_id = tokenizer.token_to_id(token)
	return token_id if token_id is not None else default


def load_tokenizers() -> List[str]:
	global tokenizer_src, tokenizer_tgt

	status: List[str] = []
	if tokenizer_src is None:
		src_path = Path(config["tokenizer_file"].format(lang=config["source_lang"]))
		if src_path.is_file():
			tokenizer_src = Tokenizer.from_file(str(src_path))
			status.append(f"Loaded source tokenizer from {src_path}")
		else:
			status.append("Source tokenizer missing. Run training once to generate it.")

	if tokenizer_tgt is None:
		tgt_path = Path(config["tokenizer_file"].format(lang=config["target_lang"]))
		if tgt_path.is_file():
			tokenizer_tgt = Tokenizer.from_file(str(tgt_path))
			status.append(f"Loaded target tokenizer from {tgt_path}")
		else:
			status.append("Target tokenizer missing. Run training once to generate it.")

	return status


def locate_checkpoint() -> Optional[Path]:
	search_root = Path(config["model_folder"]) / config["experiment_name"]
	if not search_root.exists():
		return None

	candidates = list(search_root.rglob("*.pt"))
	if not candidates:
		return None

	return max(candidates, key=lambda path: path.stat().st_mtime)


def build_model_for_inference() -> List[str]:
	global model, checkpoint_path

	status: List[str] = []
	if model is not None:
		return status

	checkpoint_path = locate_checkpoint()
	if checkpoint_path is None:
		status.append("No checkpoints found yet. Keep training to generate a .pt file.")
		return status

	if tokenizer_src is None or tokenizer_tgt is None:
		status.append("Tokenizers are not loaded; cannot build model yet.")
		return status

	model = build_transformer(
		src_vocab_size=len(tokenizer_src.get_vocab()),
		tgt_vocab_size=len(tokenizer_tgt.get_vocab()),
		src_seq_len=config["seq_length"],
		tgt_seq_len=config["seq_length"],
		d_model=config["d_model"],
		d_ff=config["dim_feedforward"],
		num_heads=config["nhead"],
		num_encoder_layers=config["num_encoder_layers"],
		num_decoder_layers=config["num_decoder_layers"],
		dropout=config["dropout"],
	).to(device)

	checkpoint = torch.load(checkpoint_path, map_location=device)
	if "model_state_dict" in checkpoint:
		model.load_state_dict(checkpoint["model_state_dict"])
	else:
		model.load_state_dict(checkpoint)

	model.eval()
	status.append(f"Loaded checkpoint: {checkpoint_path.name} on {device}")
	return status


def prepare_encoder_inputs(text: str) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
	if tokenizer_src is None or tokenizer_tgt is None:
		return None, None

	tokens = tokenizer_src.encode(text).ids
	max_len = config["seq_length"]
	tokens = tokens[: max_len - 2]

	pad_id = safe_token_id(tokenizer_src, "[PAD]")
	sos_id = safe_token_id(tokenizer_src, "[SOS]")
	eos_id = safe_token_id(tokenizer_src, "[EOS]")

	num_padding = max_len - len(tokens) - 2
	encoder_input = torch.tensor(
		[[sos_id] + tokens + [eos_id] + [pad_id] * num_padding],
		dtype=torch.long,
		device=device,
	)
	encoder_mask = (encoder_input != pad_id).unsqueeze(1).unsqueeze(1)
	return encoder_input, encoder_mask


def greedy_decode(model, encoder_input, encoder_mask, tokenizer_tgt, max_len, device):
	sos_idx = safe_token_id(tokenizer_tgt, "[SOS]")
	eos_idx = safe_token_id(tokenizer_tgt, "[EOS]")

	encoder_output = model.encode(encoder_input, encoder_mask)
	decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(encoder_input).to(device)

	while True:
		if decoder_input.size(1) >= max_len:
			break

		decoder_mask = casual_mask(decoder_input.size(1)).type_as(encoder_input).to(device)
		decoder_output = model.decode(decoder_input, encoder_output, encoder_mask, decoder_mask)
		projected_output = model.project(decoder_output)

		_, next_word = torch.max(projected_output[:, -1, :], dim=1)
		next_word = next_word.item()

		decoder_input = torch.cat(
			[
				decoder_input,
				torch.empty(1, 1).fill_(next_word).type_as(encoder_input).to(device),
			],
			dim=1,
		)

		if next_word == eos_idx:
			break

	return decoder_input.squeeze(0)


def translate(text: str) -> Optional[str]:
	if model is None:
		return None

	encoder_input, encoder_mask = prepare_encoder_inputs(text)
	if encoder_input is None or encoder_mask is None:
		return None

	with torch.no_grad():
		tokens = greedy_decode(
			model,
			encoder_input,
			encoder_mask,
			tokenizer_tgt,
			config["seq_length"],
			device,
		)
	return tokenizer_tgt.decode(tokens.tolist())


def collect_status() -> List[str]:
	status: List[str] = []
	status.extend(load_tokenizers())
	status.extend(build_model_for_inference())
	return status


@app.route("/", methods=["GET", "POST"])
def index():
	status = collect_status()
	ready = model is not None and tokenizer_src is not None and tokenizer_tgt is not None

	translated_text: Optional[str] = None
	input_text = ""

	if request.method == "POST":
		input_text = request.form.get("text", "").strip()
		if not input_text:
			flash("Please enter text to translate.", "warning")
		elif not ready:
			flash("Model is not ready yet. Wait for training to finish and a checkpoint to appear.", "warning")
		else:
			translated_text = translate(input_text)
			if translated_text is None:
				flash("Translation failed. Check server logs for details.", "danger")

	return render_template(
		"index.html",
		status=status,
		ready=ready,
		translated_text=translated_text,
		input_text=input_text,
		checkpoint_name=checkpoint_path.name if checkpoint_path else None,
		device=device.type,
	)


if __name__ == "__main__":
	app.run(host="0.0.0.0", port=5000, debug=True)
