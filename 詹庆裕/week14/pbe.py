def corpus(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = f.read()
    text = data.split('\n')
    text = ''.join(text)
    return text

def utf_8_encode(text):
    ids = text.encode('utf-8')
    ids = list(map(int, ids))
    return ids

def ids_merges(ids):
    merges_tuple = {}
    for start, end in zip(ids, ids[1:]):
        if (start, end) not in merges_tuple:
            merges_tuple[start, end] = 1
        else:
            merges_tuple[start, end] += 1
    return merges_tuple


def merger_replace(ids, top_merge, idx):
    new_ids = []
    i = 0
    while i < len(ids):
        if ids[i] == top_merge[0] and ids[i+1] == top_merge[1] and i < len(ids)-1:
            new_ids.append(idx)
            i += 2
        else:
            new_ids.append(ids[i])
            i += 1
    return new_ids


def merger_vocab(ids, num_merges):
    merger = {}
    for i in range(num_merges):
        merges_tuple = ids_merges(ids)
        top_merger = max(merges_tuple, key=merges_tuple.get)
        idx = i + 256
        ids = merger_replace(ids, top_merger, idx)
        merger[top_merger] = idx
    return merger

def peb_encode(text, merger):
    token = text.encode('utf-8')
    token = list(map(int, token))
    for merger_tuple, idx in merger.items():
        token = merger_replace(token, merger_tuple, idx)
    return token


def peb_decode(ids, merger):
    vocab = {idx: bytes([idx]) for idx in range(256)}
    for (p0, p1), idx in merger.items():
        vocab[idx] = vocab[p0] + vocab[p1]
    tokens = b"".join(vocab[idx] for idx in ids)
    text = tokens.decode("utf-8", errors="replace")
    return text


if __name__ == '__main__':
    article = corpus('train.txt')
    ids = utf_8_encode(article)
    merger = merger_vocab(ids, 10)
    with open("test", "r", encoding="utf-8") as f:
        test_text = f.read()
    test_ids = peb_encode(test_text, merger)
    test_text = peb_decode(test_ids, merger)
    if test_text == test_text:
        print("ok")
