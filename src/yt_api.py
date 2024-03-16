import requests
from pprint import pprint


class YouTubeAPI():
    def __init__(self, api_key, max_comment_size):
        self.api_key = api_key
        self.url = 'https://youtube.googleapis.com/youtube/v3/commentThreads'
        self.max_comment_size = max_comment_size

    def get_comments(self, video_id):
        """Returns a list of all `commentThreads` from a YouTube video."""

        # Get comments from the first page
        response = self.get_response(video_id, max_results=100)
        comment_list = self.response_to_comments(response)

        # Get comments from the other pages
        while 'nextPageToken' in response.keys():
            response = self.get_response(
                video_id, page_token=response['nextPageToken'])
            comment_list += (self.response_to_comments(response))

        return comment_list

    def get_response(self, video_id, page_token=None, max_results=100):
        """Gets the response from YouTube API and converts it to JSON."""
        url = 'https://youtube.googleapis.com/youtube/v3/commentThreads'
        payload = {
            'videoId': video_id,
            'key': self.api_key,
            'maxResults': max_results,
            'part': 'snippet',
            'pageToken': page_token,
        }
        response = requests.get(url, params=payload)
        return response.json()

    def response_to_comments(self, response):
        """Converts JSON response to `comment_list` list."""
        comment_list = []
        for full_comment in response['items']:
            comment = full_comment['snippet']
            can_reply = comment['canReply']
            total_reply_count = comment['totalReplyCount']
            comment = comment['topLevelComment']
            comment_id = comment['id']
            comment = comment['snippet']

            # Skip if comment is too long
            if len(comment['textDisplay']) > self.max_comment_size:
                continue
            try:
                published_at = comment['publishedAt']
                comment_list.append({
                    'comment_id': comment_id,
                    'video_id': comment['videoId'],
                    'channel_id': comment['authorChannelId']['value'],
                    'author_display_name': comment['authorDisplayName'],
                    'text_original': comment['textOriginal'],
                    'text_display': comment['textDisplay'],
                    'published_at': published_at.replace('T', ' ')[:-1],
                    'updated_at': comment['updatedAt'].replace('T', ' ')[:-1],
                    'like_count': comment['likeCount'],
                    'can_reply': can_reply,
                    'total_reply_count': total_reply_count,
                })
            except KeyError as e:
                print(f"Error: {e}\nComment:")
                pprint(full_comment)

        return comment_list
