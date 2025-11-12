from pyannote.database import registry, FileFinder
from pyannote.audio.tasks import OverlappedSpeechDetection
from pyannote.audio.core.model import Model
from pyannote.audio import Inference
import os
import numpy as np
import torchaudio

registry.load_database("./AMI-diarization-setup/pyannote/database.yml")

class OSDExecutive(object):
    def __init__(self, device="cuda"):
        self.protocol = registry.get_protocol("AMI.SpeakerDiarization.mini", preprocessors={"audio": FileFinder()})
        self.osd = OverlappedSpeechDetection(self.protocol, duration=2., batch_size=16)
        self.pretrained_model = Model.from_pretrained("pyannote/segmentation-3.0", use_auth_token=os.environ["HF_TOKEN"])
        self.pretrained_model.task = self.osd
        self.device = device
        self.pretrained_model.eval().to(self.device)
        self.inference_obj = Inference(self.pretrained_model, window="whole")
        self.classes = ["non-speech", "single", "single", "single", "mixed", "mixed", "mixed"]

    def __call__(self, audio_path):
        assert os.path.exists(audio_path), "FileNotFound: " + audio_path
        outputs = self.inference_obj(audio_path)
        outputs = np.exp(outputs)
        predictions = np.argmax(outputs, axis=1)
        predictions = predictions.tolist()
        num_frames = len(predictions)
        duration = self.get_duration(audio_path)
        mapping_frame_to_timestamp = duration / num_frames
        mapping_predictions = self.find_equal_runs(predictions)
        results = []
        for element in mapping_predictions:
            results.append({
                "start" : mapping_frame_to_timestamp * element["start"],
                "end" : mapping_frame_to_timestamp * element["end"],
                "class" : self.classes[element["value"]]
            })
        return results
    
    def get_duration(self, audio_path):
        info = torchaudio.info(audio_path)
        return info.num_frames / info.sample_rate

    def find_equal_runs(self, nums):
        """
        Find consecutive runs of equal integers in a list.

        Args:
            nums (list[int]): input list (can be empty).

        Returns:
            list[dict]: each dict has keys:
                - "start": start index (inclusive, 0-based)
                - "end":   end index (inclusive)
                - "value": the integer value of the run

        Example:
            >>> find_equal_runs([2,2,2,2,3,3,3,4,2,2,2])
            [
            {'start': 0, 'end': 3, 'value': 2},
            {'start': 4, 'end': 6, 'value': 3},
            {'start': 7, 'end': 7, 'value': 4},
            {'start': 8, 'end': 10, 'value': 2},
            ]
        """
        runs = []
        if not nums:
            return runs

        start = 0
        curr = nums[0]

        for i in range(1, len(nums)):
            if nums[i] != curr:
                runs.append({"start": start, "end": i - 1, "value": curr})
                start = i
                curr = nums[i]

        # append final run
        runs.append({"start": start, "end": len(nums) - 1, "value": curr})
        return runs