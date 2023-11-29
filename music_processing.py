from miditok import REMI, TokenizerConfig
from pathlib import Path


PITCH_RANGE = (21, 109)
BEAT_RES = {(0, 1): 8, (1, 2): 4, (2, 4): 2, (4, 8): 1}
NB_VELOCITIES = 24
SPECIAL_TOKENS = ["PAD", "MASK", "BOS", "EOS"]
USE_CHORDS = False
USE_RESTS = False
USE_TEMPOS = True
USE_TIME_SIGNATURE = False
USE_PROGRAMS = False
NB_TEMPOS = 32
TEMPO_RANGE = (50, 200)  # (min_tempo, max_tempo)
TOKENIZER_PARAMS = {
    "pitch_range": PITCH_RANGE,
    "beat_res": BEAT_RES,
    "nb_velocities": NB_VELOCITIES,
    "special_tokens": SPECIAL_TOKENS,
    "use_chords": USE_CHORDS,
    "use_rests": USE_RESTS,
    "use_tempos": USE_TEMPOS,
    "use_time_signatures": USE_TIME_SIGNATURE,
    "use_programs": USE_PROGRAMS,
    "nb_tempos": NB_TEMPOS,
    "tempo_range": TEMPO_RANGE,
}
config = TokenizerConfig(**TOKENIZER_PARAMS)

# Creates the tokenizer convert MIDIs to tokens
tokens_path = Path('midi_dataset_tokens_no_bpe')
tokenizer = REMI(config)  # REMI

midi_paths = list(Path('midi_dataset').glob('**/*.mid')) + list(Path('midi_dataset').glob('**/*.midi'))

print(midi_paths, tokens_path)
tokenizer.tokenize_midi_dataset(midi_paths, tokens_path)

# Learn and apply BPE to data we just tokenized
tokens_bpe_path = Path('midi_dataset_tokens_bpe')
tokens_bpe_path.mkdir(exist_ok=True, parents=True)
tokenizer.learn_bpe(
    vocab_size=2000,
    tokens_paths=list(tokens_path.glob("**/*.json")),
    start_from_empty_voc = True,
)
tokenizer.save_params("midi_dataset_tokenizer_bpe.conf")

"""tokenizer.apply_bpe_to_dataset(
    tokens_path,
    tokens_bpe_path,
)"""