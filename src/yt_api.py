import requests
from pprint import pprint


def get_comments(video_id, max_comment_size, api_key):
    """Yields all `commentThreads` from a YouTube video in batches."""

    # Get comments from the first page
    response = get_response(video_id, api_key, max_results=100)
    comment_list = response_to_comments(response, max_comment_size)

    # Get comments from the other pages
    while 'nextPageToken' in response.keys():
        response = get_response(
            video_id, api_key, page_token=response['nextPageToken'])
        comment_list += (response_to_comments(response, max_comment_size))

    return comment_list


def get_response(video_id, api_key, page_token=None, max_results=100):
    """Gets the response from YouTube API and converts it to JSON."""
    url = 'https://youtube.googleapis.com/youtube/v3/commentThreads'
    payload = {
        'videoId': video_id,
        'key': api_key,
        'maxResults': max_results,
        'part': 'snippet',
        'pageToken': page_token,
    }
    response = requests.get(url, params=payload)
    return response.json()


def response_to_comments(response, max_comment_size):
    """Converts JSON response to `comment_list` dict."""
    comment_list = []
    for full_comment in response['items']:
        comment = full_comment['snippet']
        can_reply = comment['canReply']
        total_reply_count = comment['totalReplyCount']
        comment = comment['topLevelComment']
        comment_id = comment['id']
        comment = comment['snippet']

        # Skip if comment is too long
        if len(comment['textDisplay']) > max_comment_size:
            continue
        try:
            comment_list.append({
                'comment_id': comment_id,
                'video_id': comment['videoId'],
                'channel_id': comment['authorChannelId']['value'],
                'author_display_name': comment['authorDisplayName'],
                'text_original': comment['textOriginal'],
                'text_display': comment['textDisplay'],
                'published_at': comment['publishedAt'].replace('T', ' ')[:-1],
                'updated_at': comment['updatedAt'].replace('T', ' ')[:-1],
                'like_count': comment['likeCount'],
                'can_reply': can_reply,
                'total_reply_count': total_reply_count,
            })
        except Exception as e:
            print(f"Error: {e}\nComment:")
            pprint(full_comment)

    return comment_list
