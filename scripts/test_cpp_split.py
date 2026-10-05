import sys, json, tokenizers
sys.stdout.reconfigure(encoding='utf-8')

tok = tokenizers.Tokenizer.from_file('examples/real_model/DeepSeek-V4.1-Flash/hf/tokenizer.json')

with open('examples/real_model/DeepSeek-V4.1-Flash/hf/tokenizer.json', encoding='utf-8') as f:
    tj = json.load(f)

vocab = tj['model']['vocab']
merges_list = tj['model']['merges']
merge_ranks = {}
for rank, m in enumerate(merges_list):
    parts = m.split(' ')
    if len(parts) == 2:
        merge_ranks[(parts[0], parts[1])] = rank

def bytes_to_unicode():
    bs = list(range(ord('!'), ord('~') + 1)) + list(range(ord('¡'), ord('¬') + 1)) + list(range(ord('®'), ord('ÿ') + 1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    return dict(zip(bs, [chr(n) for n in cs]))

b2u = bytes_to_unicode()

def bpe_word(w):
    word = list(w)
    while len(word) > 1:
        min_rank = float('inf')
        min_idx = -1
        for i in range(len(word) - 1):
            pair = (word[i], word[i+1])
            r = merge_ranks.get(pair, float('inf'))
            if r < min_rank:
                min_rank = r
                min_idx = i
        if min_rank == float('inf'):
            break
        word = word[:min_idx] + [word[min_idx] + word[min_idx+1]] + word[min_idx+2:]
    return word

import regex
pat = regex.compile(r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+""")

def encode_text(text):
    tokens = []
    for match in pat.finditer(text):
        m = match.group(0)
        b_str = ''.join(b2u[b] for b in m.encode('utf-8'))
        subwords = bpe_word(b_str)
        for sw in subwords:
            if sw in vocab:
                tokens.append(vocab[sw])
    return tokens

prompts = [
    'Hello',
    'Hello world',
    'DeepSeek',
    'Explain what a neural network is in simple terms.',
    'What is artificial intelligence?',
    'Write a short Python function that adds two numbers.',
    'Explain gravity in simple terms.'
]

for p in prompts:
    hf_ids = tok.encode(p).ids
    my_ids = encode_text(p)
    print(f'Prompt: {p!r}')
    print('  HF:    ', hf_ids)
    print('  Custom:', my_ids)
    print('  Match: ', hf_ids == my_ids)
    assert hf_ids == my_ids, f'Mismatch for {p}'

print('\nALL PROMPT ENCODINGS MATCH HF 100% BIT-EXACT!')
