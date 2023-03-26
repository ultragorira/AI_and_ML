import praw
import json

client_id = ""
client_secret = ""
subreddit = "Advice"

reddit = praw.Reddit(
    client_id=client_id,
    client_secret=client_secret,
    user_agent="python.finetune.Nearby-Landscape7357:v1 (fine-tuning ChatGPT)",
)

data = []

submission_ids = []

for submission in reddit.subreddit(subreddit).top(limit=500):
    submission_ids.append(submission.id)
    
for id in submission_ids:
    submission = reddit.submission(id)
    title = submission.title
    print(title)
    submission.comments.replace_more(limit=0)
    data.append({
        'prompt': submission.selftext,
        'completion': submission.comments[1].body
    })

with open("reddit_data.jsonl", 'w') as f:
    for item in data:
        f.write(json.dumps(item) + "\n")