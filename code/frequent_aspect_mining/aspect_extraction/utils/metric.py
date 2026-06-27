import math


def get_spans(tags, length, token_range, span_type, ignore_index=-1):
    """Find contiguous token spans of the given span_type in the tag matrix."""
    spans = []
    start = -1
    for i in range(length):
        l, r = token_range[i]
        if tags[l][l] == ignore_index:
            continue
        elif tags[l][l] == span_type:
            if start == -1:
                start = i
        else:
            if start != -1:
                spans.append([start, i - 1])
                start = -1
    if start != -1:
        spans.append([start, length - 1])
    return spans


def find_triplet(tags, aspect_spans, opinion_spans, token_ranges, probs):
    """Find Aspect-Opinion-Sentiment triplets from span pairs."""
    triplets = []
    for al, ar in aspect_spans:
        for pl, pr in opinion_spans:
            tag_num  = [0] * 6
            prob_lst = []
            for i in range(al, ar + 1):
                for j in range(pl, pr + 1):
                    a_start = token_ranges[i][0]
                    o_start = token_ranges[j][0]
                    if al < pl:
                        tag_num[int(tags[a_start][o_start])] += 1
                        prob_lst.append(float(probs[a_start][o_start]))
                    else:
                        tag_num[int(tags[o_start][a_start])] += 1
                        prob_lst.append(float(probs[o_start][a_start]))

            if sum(tag_num[3:]) == 0:
                continue

            sentiment_prob = sum(prob_lst) / len(prob_lst)

            if tag_num[4] > tag_num[3] and tag_num[4] > tag_num[5]:
                sentiment = "positive"
            elif tag_num[3] > tag_num[4] and tag_num[3] > tag_num[5]:
                sentiment = "negative"
            else:
                sentiment = "neutral"

            triplets.append([al, ar, pl, pr, sentiment, sentiment_prob])

    return triplets
