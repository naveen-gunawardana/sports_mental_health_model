#!/bin/bash
cd "C:/Users/navee_xqu8e3o/OneDrive/Documents/programming/AM"
PY=./.venv/Scripts/python.exe
echo "===== ARM 1/3: matched 2018-2022 ====="
$PY code/classify_corpus.py "comments/2018-2022/MS_comments_2018_2022/MS_comments_2018_2022/matched" data/classified/matched_2018_2022
echo "===== ARM 2/3: matched 2023 ====="
$PY code/classify_corpus.py "comments/2023/comments_2023/filtered_subreddit_keywords/matched" data/classified/matched_2023
echo "===== ARM 3/3: baseline 2018-2022 ====="
$PY code/classify_corpus.py "comments/2018-2022/MS_comments_2018_2022/MS_comments_2018_2022/baseline" data/classified/baseline_2018_2022
echo "===== ALL ARMS DONE ====="
