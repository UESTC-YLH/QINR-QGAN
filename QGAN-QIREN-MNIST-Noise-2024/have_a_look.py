from sound.data_process import AudioFile
import matplotlib.pyplot as plt
a = AudioFile('./sound/gt_bach.wav')
plt.plot(a.timepoints)
plt.show()
debug = 1