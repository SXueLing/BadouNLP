def all_cut(sentence, Dict):
    def cut_helper(s, start, cuts, target):
        if start == len(s):
            target.append(cuts)
            return
        for end in range(start + 1, len(s) + 1):
            word = s[start:end]
            if word in Dict:
                cut_helper(s, end, cuts + [word], target)

    target = []
    cut_helper(sentence, 0, [], target)
    return target

def main():
    sentence = "经常有意见分歧"
    Dict = {"经常": 0.1,
            "经": 0.05,
            "有": 0.1,
            "常": 0.001,
            "有意见": 0.1,
            "歧": 0.001,
            "意见": 0.2,
            "分歧": 0.2,
            "见": 0.05,
            "意": 0.05,
            "见分歧": 0.05,
            "分": 0.1
            }
    print(all_cut(sentence, Dict))
    return

main()
