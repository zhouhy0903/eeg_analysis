from eegprocess import get_epoch_eeg
from score import get_score
from state import get_state

data=get_epoch_eeg(1)
data_score=get_score(1)
data_state=get_state(1,5)
print(data_score)
print(data)
print(data_state["rightEyeTargetPosition"])