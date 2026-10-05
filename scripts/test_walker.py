import sys, tokenizers
sys.stdout.reconfigure(encoding='utf-8')

tok = tokenizers.Tokenizer.from_file('examples/real_model/DeepSeek-V4.1-Flash/hf/tokenizer.json')

import json
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

def is_whitespace(c):
    return c in ' \t\r\n'

def is_ascii_letter(c):
    return ('a' <= c <= 'z') or ('A' <= c <= 'Z')

def is_digit(c):
    return '0' <= c <= '9'

def is_unicode_char(c):
    return ord(c) >= 128

def is_letter_or_unicode(c):
    return is_ascii_letter(c) or is_unicode_char(c)

def cpp_tokenize_chunks(text):
    chunks = []
    i = 0
    n = len(text)
    while i < n:
        # 1. Contractions
        if text[i] == "'" and i + 1 < n:
            rest = text[i:].lower()
            cont = None
            for c in ["'re", "'ve", "'ll", "'s", "'t", "'m", "'d"]:
                if rest.startswith(c):
                    cont = text[i:i+len(c)]
                    break
            if cont:
                chunks.append(cont)
                i += len(cont)
                continue

        # Optional single leading space: ' ?'
        start = i
        has_space = False
        if text[i] == ' ':
            has_space = True
            i += 1
            if i == n:
                chunks.append(' ')
                break

        # If followed by another whitespace, then it was a whitespace run!
        if i < n and is_whitespace(text[i]):
            while i < n and is_whitespace(text[i]):
                i += 1
            chunks.append(text[start:i])
            continue

        if i < n and is_letter_or_unicode(text[i]):
            while i < n and is_letter_or_unicode(text[i]):
                i += 1
            chunks.append(text[start:i])
            continue

        if i < n and is_digit(text[i]):
            d_count = 0
            while i < n and is_digit(text[i]) and d_count < 3:
                i += 1
                d_count += 1
            chunks.append(text[start:i])
            continue

        if i < n and not is_whitespace(text[i]):
            while i < n and not (is_letter_or_unicode(text[i]) or is_digit(text[i]) or is_whitespace(text[i])):
                i += 1
            chunks.append(text[start:i])
            continue

        if has_space and i == start + 1:
            chunks.append(' ')

    return chunks

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
    my_ids = []
    for c in cpp_tokenize_chunks(p):
        b_str = ''.join(b2u[b] for b in c.encode('utf-8'))
        for sw in bpe_word(b_str):
            if sw in vocab:
                my_ids.append(vocab[sw])
    print(f'Prompt: {p!r}')
    print('  HF:    ', hf_ids)
    print('  Custom:', my_ids)
    print('  Match: ', hf_ids == my_ids)
    assert hf_ids == my_ids
print('\nPERFECT 100% BIT-EXACT MATCH ON ALL PROMPTS!')
