
data, fs = sf.read('Music/1.wav', dtype='float32')  
sd.play(data, fs)
input("Write anything to stop.")
sd.stop()