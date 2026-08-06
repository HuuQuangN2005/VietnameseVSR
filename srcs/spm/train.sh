python -m srcs.spm.spm_train --input "$1" --out_dir "$(dirname "$0")/unigram" --vocab_size "${2:-5000}"
