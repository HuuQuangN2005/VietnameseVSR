from srcs.datasets.transform import VideoTransform
from srcs.datasets.utils import load_video, pad_seq, to_text


class Collator:
    def __init__(self, split, text_transform):
        self.video_transform = VideoTransform(split)
        self.text_transform = text_transform

    def __call__(self, items):
        videos = []
        labels = []

        for item in items:
            start_time = item.get("start_time", 0.0)
            end_time = item.get("end_time")
            video = load_video(item["video"], start_time, end_time)
            videos.append(self.video_transform(video))

            if "label" in item:
                labels.append(self.text_transform.encode(to_text(item["label"])))

        video_batch, video_lengths = pad_seq(videos)
        batch = {"videos": video_batch, "video_lengths": video_lengths}

        if labels:
            label_batch, label_lengths = pad_seq(
                labels,
                self.text_transform.ignore_id,
            )
            batch["labels"] = label_batch
            batch["label_lengths"] = label_lengths

        return batch


class PhonemeCollator(Collator):
    pass
